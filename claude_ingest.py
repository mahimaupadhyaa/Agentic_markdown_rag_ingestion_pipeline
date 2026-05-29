"""
Agentic Markdown RAG Ingestion Pipeline
========================================
Intelligently chunks markdown files for RAG, with special handling for:
  - Tables       → always kept as a single, atomic chunk (never split)
  - Images       → extracted with alt-text, captioning hook, and raw path stored
  - Code blocks  → kept atomic per fence
  - Prose text   → split semantically, respecting sentence boundaries
  - Headings     → never emitted as standalone chunks; used only as context breadcrumbs

Usage:
    python ingest.py --input docs/          # process a folder
    python ingest.py --input README.md      # process a single file
    python ingest.py --input docs/ --output chunks.json --max-tokens 512

Output:
    A JSON file containing a list of chunk dicts, each with:
      - text          : the chunk content (what gets embedded)
      - block_type    : "table" | "image" | "code" | "text"
      - source_file   : path to the original markdown file
      - section_path  : "H1 > H2 > H3" breadcrumb at the point of this chunk
      - chunk_index   : sequential index within this file
      - token_count   : approximate token count of `text`
      - metadata      : block-type-specific extras (image_path, language, etc.)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import marko
import marko.ast_renderer
import marko.block
import marko.inline
from marko.ext.gfm import GFM  # GitHub Flavored Markdown — required for proper table nodes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IngestionConfig:
    """All tunable parameters in one place — easy to adjust or pass via CLI."""
    max_tokens: int = 512          # Target max tokens per text chunk
    min_tokens: int = 50           # Don't emit chunks smaller than this (merge forward)
    overlap_sentences: int = 1     # Sentences of overlap between consecutive text chunks
    token_chars_ratio: float = 4.0 # Approx chars per token (used when tiktoken is absent)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single RAG chunk ready for embedding and storage."""
    text: str
    block_type: str                   # "text" | "table" | "code" | "image"
    source_file: str
    section_path: str                 # e.g. "Introduction > Key Concepts"
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Approximate token count without an external tokenizer.

    tiktoken gives exact counts but requires network install.
    This heuristic (chars / 4) is accurate to within ~10% for English text,
    which is fine for chunking decisions.
    """
    return max(1, int(len(text) / chars_per_token))


# ---------------------------------------------------------------------------
# Heading breadcrumb tracker
# ---------------------------------------------------------------------------

class SectionTracker:
    """
    Maintains a stack of heading labels (H1 → H2 → H3) as we walk the AST.

    When a new heading is encountered, all deeper-level headings are popped,
    giving each block an accurate "where am I in the document" label.

    Example:
        # Intro           → stack: ["Intro"]
        ## Overview       → stack: ["Intro", "Overview"]
        ### Details       → stack: ["Intro", "Overview", "Details"]
        ## Summary        → stack: ["Intro", "Summary"]   ← Details was popped
    """

    def __init__(self):
        # List of (level, title) tuples, e.g. [(1,"Intro"), (2,"Overview")]
        self._stack: list[tuple[int, str]] = []

    def update(self, level: int, title: str):
        """Call this when a heading node is encountered."""
        # Pop anything at the same level or deeper
        self._stack = [(lvl, t) for lvl, t in self._stack if lvl < level]
        self._stack.append((level, title))

    @property
    def path(self) -> str:
        """Human-readable breadcrumb string, e.g. 'Intro > Key Concepts'."""
        if not self._stack:
            return ""
        return " > ".join(title for _, title in self._stack)


# ---------------------------------------------------------------------------
# AST helpers — extract plain text from inline nodes
# ---------------------------------------------------------------------------

def inline_to_text(node) -> str:
    """
    Recursively collapse an inline node tree into a plain string.

    marko's inline nodes (RawText, CodeSpan, Link, etc.) can be nested,
    so we recurse through children until we hit raw string content.
    """
    # Base case: raw text leaf node
    if isinstance(node, marko.inline.RawText):
        return node.children  # .children is a str on leaf nodes

    # Base case: plain Python string (some renderers return strings directly)
    if isinstance(node, str):
        return node

    # Nodes whose text is in a `target` attribute (e.g. links display text)
    if isinstance(node, marko.inline.Link):
        return "".join(inline_to_text(c) for c in node.children)

    # Recursive case: node has children list
    if hasattr(node, "children") and isinstance(node.children, list):
        return "".join(inline_to_text(c) for c in node.children)

    return ""


def block_to_text(node) -> str:
    """Collapse a block node (paragraph, list item, etc.) to plain text."""
    if isinstance(node, str):
        return node
    if hasattr(node, "children"):
        if isinstance(node.children, list):
            return "".join(block_to_text(c) for c in node.children)
        if isinstance(node.children, str):
            return node.children
    return ""


# ---------------------------------------------------------------------------
# Block handlers — one function per content type
# ---------------------------------------------------------------------------

def handle_table(node, section_path: str, source_file: str, chunk_index: int,
                 config: IngestionConfig) -> Chunk:
    """
    Tables are ALWAYS emitted as a single atomic chunk — never split.

    We produce two representations:
      1. A markdown table string (preserves structure)
      2. A natural-language summary prepended as context

    The text field contains both so the embedding captures the semantics
    while retrieval can still re-render the markdown form.
    """
    rows = []

    # GFM table structure: Table > TableRow > TableCell
    # (no separate TableHead/TableBody wrappers in marko's GFM extension)
    for row in node.children:
        row_type = type(row).__name__
        if row_type == "TableRow":
            cells = []
            for cell in row.children:
                cell_text = block_to_text(cell).strip()
                cells.append(cell_text)
            rows.append(cells)

    if not rows:
        # Fallback: render raw markdown from source if AST walk gave nothing
        return Chunk(
            text="[Table — could not be parsed]",
            block_type="table",
            source_file=source_file,
            section_path=section_path,
            chunk_index=chunk_index,
            token_count=10,
            metadata={"row_count": 0}
        )

    # --- PDF header-row fix ---
    # When Marker converts a PDF table, it sometimes emits a blank first row
    # because the PDF's visual column headers aren't recognised as a markdown
    # header row. The real column names then appear as the second data row.
    #
    # Detection: first row is all blank AND second row has actual text content.
    first_row_blank = all(cell == "" for cell in rows[0])
    second_row_has_content = len(rows) > 1 and any(cell != "" for cell in rows[1])
    if first_row_blank and second_row_has_content:
        rows = rows[1:]  # drop the blank row; row[1] becomes the header

    # Build markdown table string
    header_row = rows[0] if rows else []
    body_rows = rows[1:] if len(rows) > 1 else []

    md_lines = []
    if header_row:
        md_lines.append("| " + " | ".join(header_row) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
    for row in body_rows:
        md_lines.append("| " + " | ".join(row) + " |")

    markdown_table = "\n".join(md_lines)

    # Build a plain-text summary to front-load semantics for the embedder
    col_names = ", ".join(header_row) if header_row else "unknown columns"
    plain_summary = (
        f"[Table with {len(body_rows)} rows and columns: {col_names}]\n"
        f"Located in section: {section_path or 'document root'}\n\n"
    )

    full_text = plain_summary + markdown_table

    return Chunk(
        text=full_text,
        block_type="table",
        source_file=source_file,
        section_path=section_path,
        chunk_index=chunk_index,
        token_count=estimate_tokens(full_text, config.token_chars_ratio),
        metadata={
            "row_count": len(body_rows),
            "column_names": header_row,
            "markdown_table": markdown_table,  # raw form stored for re-rendering
        }
    )


def handle_image(node, section_path: str, source_file: str, chunk_index: int,
                 config: IngestionConfig) -> Chunk:
    """
    Images are atomic chunks. We extract:
      - alt text  (from the AST, usually already in the markdown)
      - image URL / path

    In production, you would call a VLM here to generate a richer caption.
    We provide a `vlm_caption_hook` comment to show exactly where to plug that in.
    """
    # marko represents ![alt](url) as Image with children = [RawText(alt)]
    alt_text = inline_to_text(node).strip()
    image_url = getattr(node, "dest", "") or ""

    # --- VLM Caption Hook ---
    # To add AI-generated captions, replace the line below with:
    #
    #   caption = call_vlm(image_url, prompt="Describe this image concisely.")
    #
    # Where call_vlm() calls OpenAI GPT-4V, Claude, LLaVA, etc.
    # The caption string then becomes the primary chunk text for embedding.
    caption = ""  # empty until VLM is wired in

    # Build the chunk text: use caption if available, fall back to alt text
    description = caption or alt_text or "No description available"
    chunk_text = (
        f"[Image] {description}\n"
        f"Section: {section_path or 'document root'}\n"
        f"Source: {image_url}"
    )

    return Chunk(
        text=chunk_text,
        block_type="image",
        source_file=source_file,
        section_path=section_path,
        chunk_index=chunk_index,
        token_count=estimate_tokens(chunk_text, config.token_chars_ratio),
        metadata={
            "image_path": image_url,
            "alt_text": alt_text,
            "vlm_caption": caption,  # empty until VLM is wired in
        }
    )


def handle_code_block(node, section_path: str, source_file: str, chunk_index: int,
                      config: IngestionConfig) -> Chunk:
    """
    Fenced code blocks are atomic — one fence = one chunk.

    We prepend the language and section so the embedder understands context,
    then store the raw code in metadata for retrieval rendering.
    """
    language = getattr(node, "lang", "") or "unknown"
    code_body = node.children if isinstance(node.children, str) else block_to_text(node)

    chunk_text = (
        f"[Code block — language: {language}]\n"
        f"Section: {section_path or 'document root'}\n\n"
        f"```{language}\n{code_body.strip()}\n```"
    )

    return Chunk(
        text=chunk_text,
        block_type="code",
        source_file=source_file,
        section_path=section_path,
        chunk_index=chunk_index,
        token_count=estimate_tokens(chunk_text, config.token_chars_ratio),
        metadata={
            "language": language,
            "raw_code": code_body.strip(),
        }
    )


def handle_text_block(paragraphs: list[str], section_path: str, source_file: str,
                      start_chunk_index: int, config: IngestionConfig) -> list[Chunk]:
    """
    Prose paragraphs are collected and then split into token-bounded chunks.

    Strategy:
      1. Split each paragraph into sentences (regex, good enough for most docs)
      2. Greedily pack sentences into chunks up to `max_tokens`
      3. When a chunk would overflow, emit it and start a new one
      4. Add `overlap_sentences` of the previous chunk to the new one
         so context isn't lost at boundaries
      5. Discard chunks smaller than `min_tokens` by merging forward

    Every chunk is prefixed with the section breadcrumb for context continuity.
    """
    # Flatten all paragraphs into a single sentence stream
    all_sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Simple sentence splitter: split on ". ", "! ", "? " followed by uppercase
        # Good enough for technical docs; swap for nltk.sent_tokenize if needed
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', para)
        all_sentences.extend(s.strip() for s in sentences if s.strip())

    if not all_sentences:
        return []

    section_prefix = f"[Section: {section_path}]\n\n" if section_path else ""
    prefix_tokens = estimate_tokens(section_prefix, config.token_chars_ratio)

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens: int = prefix_tokens
    chunk_index = start_chunk_index

    for sentence in all_sentences:
        sentence_tokens = estimate_tokens(sentence + " ", config.token_chars_ratio)

        # If adding this sentence would overflow, emit the current chunk first
        if current_tokens + sentence_tokens > config.max_tokens and current_sentences:
            chunk_text = section_prefix + " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                block_type="text",
                source_file=source_file,
                section_path=section_path,
                chunk_index=chunk_index,
                token_count=estimate_tokens(chunk_text, config.token_chars_ratio),
            ))
            chunk_index += 1

            # Carry over the last N sentences as overlap for the next chunk
            overlap = current_sentences[-config.overlap_sentences:] if config.overlap_sentences > 0 else []
            current_sentences = overlap
            current_tokens = prefix_tokens + sum(
                estimate_tokens(s + " ", config.token_chars_ratio) for s in overlap
            )

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Emit whatever remains
    if current_sentences:
        chunk_text = section_prefix + " ".join(current_sentences)
        if estimate_tokens(chunk_text, config.token_chars_ratio) >= config.min_tokens:
            chunks.append(Chunk(
                text=chunk_text,
                block_type="text",
                source_file=source_file,
                section_path=section_path,
                chunk_index=chunk_index,
                token_count=estimate_tokens(chunk_text, config.token_chars_ratio),
            ))
        elif chunks:
            # Chunk is too small — merge into the previous one
            chunks[-1].text += " " + " ".join(current_sentences)
            chunks[-1].token_count = estimate_tokens(chunks[-1].text, config.token_chars_ratio)

    return chunks


# ---------------------------------------------------------------------------
# Core AST walker
# ---------------------------------------------------------------------------

def walk_ast(document_node, source_file: str, config: IngestionConfig) -> list[Chunk]:
    """
    Walk the marko AST top-level children and route each node to the
    appropriate handler based on its type.

    Key design decisions:
      - Headings never become chunks; they only update the breadcrumb tracker.
      - Consecutive prose paragraphs are BUFFERED and passed together to
        handle_text_block so the semantic splitter can work across them.
      - Non-prose blocks (table, image, code) FLUSH the prose buffer first,
        then are handled atomically.
    """
    section = SectionTracker()
    chunks: list[Chunk] = []

    # Buffer for consecutive prose paragraphs — flushed when a non-prose block arrives
    prose_buffer: list[str] = []

    def flush_prose():
        """Emit all buffered prose paragraphs as semantic text chunks."""
        nonlocal prose_buffer
        if not prose_buffer:
            return
        new_chunks = handle_text_block(
            paragraphs=prose_buffer,
            section_path=section.path,
            source_file=source_file,
            start_chunk_index=len(chunks),
            config=config,
        )
        chunks.extend(new_chunks)
        prose_buffer = []

    def get_heading_text(heading_node) -> str:
        """Extract plain text from a heading node's inline children."""
        if hasattr(heading_node, "children") and isinstance(heading_node.children, list):
            return "".join(inline_to_text(c) for c in heading_node.children).strip()
        return block_to_text(heading_node).strip()

    def find_images_in_node(node) -> list:
        """Recursively find all Image inline nodes within any block."""
        images = []
        if isinstance(node, marko.inline.Image):
            images.append(node)
        if hasattr(node, "children") and isinstance(node.children, list):
            for child in node.children:
                images.extend(find_images_in_node(child))
        return images

    # Walk top-level AST nodes
    for node in document_node.children:
        node_type = type(node).__name__

        # ── Heading ────────────────────────────────────────────────────────
        if node_type == "Heading":
            flush_prose()
            level = node.level
            title = get_heading_text(node)
            section.update(level, title)
            # Headings are context only — NOT emitted as chunks

        # ── Table ──────────────────────────────────────────────────────────
        elif node_type == "Table":
            flush_prose()
            chunk = handle_table(
                node=node,
                section_path=section.path,
                source_file=source_file,
                chunk_index=len(chunks),
                config=config,
            )
            chunks.append(chunk)

        # ── Fenced code block ──────────────────────────────────────────────
        elif node_type == "FencedCode":
            flush_prose()
            chunk = handle_code_block(
                node=node,
                section_path=section.path,
                source_file=source_file,
                chunk_index=len(chunks),
                config=config,
            )
            chunks.append(chunk)

        # ── Paragraph (may contain inline images) ─────────────────────────
        elif node_type == "Paragraph":
            # Check for inline images first
            images = find_images_in_node(node)
            if images:
                # If the paragraph is *only* an image, treat it as an image chunk
                para_text = block_to_text(node).strip()
                is_image_only = len(images) == 1 and len(para_text.strip("![]()").strip()) < 5

                if is_image_only:
                    flush_prose()
                    for img in images:
                        chunk = handle_image(
                            node=img,
                            section_path=section.path,
                            source_file=source_file,
                            chunk_index=len(chunks),
                            config=config,
                        )
                        chunks.append(chunk)
                else:
                    # Mixed paragraph: emit images separately, buffer the text
                    flush_prose()
                    for img in images:
                        chunk = handle_image(
                            node=img,
                            section_path=section.path,
                            source_file=source_file,
                            chunk_index=len(chunks),
                            config=config,
                        )
                        chunks.append(chunk)
                    # Also buffer the prose portion
                    text_only = block_to_text(node).strip()
                    if text_only:
                        prose_buffer.append(text_only)
            else:
                # Pure prose — add to buffer
                para_text = block_to_text(node).strip()
                if para_text:
                    prose_buffer.append(para_text)

        # ── Lists ──────────────────────────────────────────────────────────
        elif node_type in ("List", "ListItem"):
            # Treat lists as prose — flatten to text and buffer
            list_text = block_to_text(node).strip()
            if list_text:
                prose_buffer.append(list_text)

        # ── BlockQuote ─────────────────────────────────────────────────────
        elif node_type == "Quote":
            flush_prose()
            quote_text = block_to_text(node).strip()
            if quote_text:
                # Treat blockquotes as atomic prose chunks
                chunk_text = f"[Section: {section.path}]\n\n> {quote_text}" if section.path else f"> {quote_text}"
                chunks.append(Chunk(
                    text=chunk_text,
                    block_type="text",
                    source_file=source_file,
                    section_path=section.path,
                    chunk_index=len(chunks),
                    token_count=estimate_tokens(chunk_text, config.token_chars_ratio),
                ))

        # ── Thematic break (---) ───────────────────────────────────────────
        elif node_type == "ThematicBreak":
            # A horizontal rule acts as a section boundary — flush prose
            flush_prose()

        # ── Anything else ──────────────────────────────────────────────────
        else:
            # Fallback: try to extract text and buffer it
            fallback_text = block_to_text(node).strip()
            if fallback_text:
                prose_buffer.append(fallback_text)

    # Flush any remaining prose at end of document
    flush_prose()

    return chunks


# ---------------------------------------------------------------------------
# File-level ingestion
# ---------------------------------------------------------------------------

def ingest_file(path: Path, config: IngestionConfig) -> list[Chunk]:
    """
    Parse a single markdown file and return all its chunks.

    Steps:
      1. Read raw text
      2. Parse to AST via marko
      3. Walk AST → route blocks → produce chunks
      4. Re-index chunk_index to be sequential within this file
    """
    raw_text = path.read_text(encoding="utf-8")

    # --- Pre-processing: clean up common Marker PDF conversion artifacts ---
    #
    # 1. Strip bare [Image] placeholders that Marker emits when it finds an
    #    image in the PDF but has no alt-text or URL to attach to it.
    #    These become noisy zero-information chunks if left in.
    #    We replace them with a note so the section context is preserved.
    raw_text = re.sub(
        r'\[Image\](\s*\[Image\])*',
        '[Image — no caption available]',
        raw_text
    )

    # 2. Strip page-number artifacts that Marker sometimes emits as standalone
    #    lines, e.g. "ANNEX 1\nPage 3" or "MSC.4/Circ.133ANNEX 1Page 7"
    #    These are headers baked into the PDF that bleed into the text layer.
    raw_text = re.sub(r'\bANNEX\s+\d+\s*\n?Page\s+\d+\b', '', raw_text)
    raw_text = re.sub(r'MSC\.\d+/Circ\.\d+ANNEX\s*\d+Page\s*\d+', '', raw_text)

    # Parse to AST with GFM extension (required for proper table node detection)
    parser = marko.Markdown(extensions=[GFM])
    document = parser.parse(raw_text)

    chunks = walk_ast(
        document_node=document,
        source_file=str(path),
        config=config,
    )

    # --- Post-processing: drop artifact chunks ---
    #
    # PDF-to-markdown conversion (Marker) occasionally produces tiny text chunks
    # that are pure layout artifacts: page headers, annex labels, lone circular
    # reference numbers, etc. We detect these by two rules:
    #
    # Rule A — "looks like a layout artifact":
    #   Short chunk (<= 15 words) made up almost entirely of uppercase words,
    #   numbers, and reference codes (no normal sentence structure).
    #
    # Rule B — "below minimum token threshold and no semantic content":
    #   Fewer than min_tokens AND contains no sentence-ending punctuation,
    #   meaning it's probably a stray label rather than a real sentence.
    def is_artifact(chunk: Chunk) -> bool:
        if chunk.block_type != "text":
            return False  # never drop tables, images, or code blocks
        text = chunk.text
        # Strip the section prefix before analysing
        body = re.sub(r'^\[Section:[^\]]+\]\s*', '', text).strip()
        words = body.split()
        if not words:
            return True

        # Rule A: very short + mostly non-lowercase (uppercase labels / codes)
        # e.g. "ANNEX 3 REPORTED ACTS OF PIRACY..." style PDF section headers
        if len(words) <= 30:
            non_lower = sum(1 for w in words if len(w) > 1 and w == w.upper() and w.isalpha())
            if non_lower / len(words) > 0.65 and not re.search(r'[.!?]', body):
                return True

        # Rule B: looks like stray table rows that Marker failed to detect as a table.
        # Pattern: lines that are mostly "DATE SHIPNAME CODE REF" with no punctuation
        # and match the date+IMO+circular reference pattern throughout.
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        if lines:
            date_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')
            circ_pattern = re.compile(r'MSC\.\d+/Circ\.\d+')
            lines_with_date = sum(1 for l in lines if date_pattern.search(l))
            lines_with_circ = sum(1 for l in lines if circ_pattern.search(l))
            # If most lines look like table rows but this wasn't detected as a table
            if len(lines) >= 2 and (lines_with_date + lines_with_circ) / len(lines) > 0.5:
                return True

        # Rule C: below min threshold and no sentence punctuation
        if chunk.token_count < config.min_tokens:
            if not re.search(r'[.!?]', body):
                return True

        return False

    chunks = [c for c in chunks if not is_artifact(c)]

    # Re-number chunk indices to be file-scoped sequential integers
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    return chunks


def ingest_directory(directory: Path, config: IngestionConfig) -> list[Chunk]:
    """
    Recursively find all .md / .markdown files in a directory and ingest them.
    Returns a flat list of all chunks across all files.
    """
    all_chunks: list[Chunk] = []
    md_files = sorted(directory.rglob("*.md")) + sorted(directory.rglob("*.markdown"))

    if not md_files:
        print(f"[warn] No markdown files found in {directory}", file=sys.stderr)
        return all_chunks

    for md_file in md_files:
        print(f"  → Ingesting {md_file} ...", end=" ", flush=True)
        file_chunks = ingest_file(md_file, config)
        print(f"{len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Agentic markdown RAG ingestion — intelligent chunking with table preservation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a markdown file OR a directory containing markdown files."
    )
    parser.add_argument(
        "--output", "-o", default="chunks.json",
        help="Output JSON file path (default: chunks.json)."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Target max tokens per text chunk (default: 512)."
    )
    parser.add_argument(
        "--min-tokens", type=int, default=50,
        help="Minimum tokens to emit a chunk; smaller chunks are merged (default: 50)."
    )
    parser.add_argument(
        "--overlap", type=int, default=1,
        help="Number of sentences to overlap between consecutive text chunks (default: 1)."
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the output JSON (useful for inspection)."
    )

    args = parser.parse_args()

    config = IngestionConfig(
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        overlap_sentences=args.overlap,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Markdown RAG Ingestion Pipeline ===")
    print(f"Input  : {input_path}")
    print(f"Config : max_tokens={config.max_tokens}, min_tokens={config.min_tokens}, overlap={config.overlap_sentences}")
    print()

    if input_path.is_dir():
        chunks = ingest_directory(input_path, config)
    else:
        print(f"  → Ingesting {input_path} ...", end=" ", flush=True)
        chunks = ingest_file(input_path, config)
        print(f"{len(chunks)} chunks")

    # Serialize to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dicts = [c.to_dict() for c in chunks]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, indent=2 if args.pretty else None, ensure_ascii=False)

    # Summary statistics
    type_counts = {}
    for c in chunks:
        type_counts[c.block_type] = type_counts.get(c.block_type, 0) + 1

    total_tokens = sum(c.token_count for c in chunks)
    print(f"\n=== Summary ===")
    print(f"Total chunks     : {len(chunks)}")
    print(f"Total tokens (≈) : {total_tokens:,}")
    for block_type, count in sorted(type_counts.items()):
        print(f"  {block_type:<10} : {count} chunks")
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()