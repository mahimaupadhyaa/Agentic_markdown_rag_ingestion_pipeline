"""
Hierarchical RAG Ingestion Pipeline

Combines:
- Docling for best-in-class PDF → structured markdown conversion
- Hierarchical chunking strategy:
    L0 → Document summary (LLM-generated)
    L1 → Section summaries (LLM-generated)
    L2 → Fine chunks (text split)
    L2 → Tables (never split, always intact)

Each chunk is stored with level, section, doc_id, section_id metadata
for multi-granularity retrieval.
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Union
import re
import requests

from config import config, ConfigLoader

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from qdrant_store import QdrantStore as MilvusStore


# -----------------------
# Config
# -----------------------

OLLAMA_URL = config.get("ollama", "url", default="http://localhost:11434/api/generate")
OLLAMA_MODEL = config.get("ollama", "model", default="qwen2.5:14b")

CHUNK_SIZE = config.get("document", "chunk_size", default=1200)
CHUNK_OVERLAP = config.get("document", "chunk_overlap", default=150)

DOCUMENT_SUMMARY_PROMPT = (
    "You are an expert document analyst. "
    "Write a concise summary (3-5 sentences) of the following document. "
    "Focus on the main topic, key findings, and purpose."
)

SECTION_SUMMARY_PROMPT = (
    "You are an expert document analyst. "
    "Write a concise summary (2-3 sentences) of the following section. "
    "Focus on the key points and findings in this section."
)

TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:-]+\|[\s|:-]*$")


# -----------------------
# Summarization
# -----------------------

def summarize_with_ollama(text: str, instruction: str) -> str:
    """Call local Ollama to generate a summary."""
    prompt = f"{instruction}\n\nTEXT:\n{text}".strip()
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        print(f"Ollama summarization failed: {e}")
        return text[:500] + "..."


# -----------------------
# Docling Conversion
# -----------------------

def get_document_converter(
    pdf_pipeline_options: Optional[PdfPipelineOptions] = None,
) -> DocumentConverter:
    """Create Docling converter with PDF pipeline options."""
    pdf_pipeline_options = pdf_pipeline_options or config.get_pdf_pipeline_options()
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options
            )
        }
    )


def convert_pdf_to_markdown(file_path: Path, converter: DocumentConverter) -> str:
    """Convert PDF to markdown using Docling."""
    print(f"  Converting PDF with Docling: {file_path.name}")
    result = converter.convert(str(file_path))
    markdown = result.document.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
    )
    return markdown


# -----------------------
# Table Extraction
# -----------------------

def is_table_separator(line: str) -> bool:
    return bool(TABLE_SEPARATOR_RE.match(line))


def extract_tables_from_markdown(text: str):
    """
    Extract markdown tables, replacing them with __TABLE_N__ placeholders.
    Returns (text_without_tables, list_of_table_strings).
    """
    lines = text.splitlines()
    output = []
    tables = []
    i = 0

    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            tbl = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i]:
                tbl.append(lines[i])
                i += 1
            tables.append("\n".join(tbl))
            output.append(f"__TABLE_{len(tables) - 1}__")
        else:
            output.append(lines[i])
            i += 1

    return "\n".join(output), tables


# -----------------------
# Core Processing
# -----------------------

def process_file_hierarchical(
    file_path: Path,
    converter: DocumentConverter,
    namespace: str,
    use_summaries: bool = True,
) -> List[Document]:
    """
    Process a single PDF into hierarchical LangChain Documents.

    Levels:
    - document : LLM summary of the whole doc
    - section  : LLM summary of each ## section
    - table    : intact table (never split)
    - chunk    : fine text chunk
    """
    print(f"\nProcessing: {file_path.name}")

    markdown = convert_pdf_to_markdown(file_path, converter)
    doc_id = str(uuid.uuid4())
    all_docs = []

    # ---- L0: Document summary ----
    if use_summaries:
        print("  Generating document summary...")
        doc_summary = summarize_with_ollama(markdown, DOCUMENT_SUMMARY_PROMPT)
    else:
        doc_summary = markdown[:500] + "..."

    all_docs.append(Document(
        page_content=doc_summary,
        metadata={
            "source": str(file_path),
            "namespace": namespace,
            "level": "document",
            "doc_id": doc_id,
            "section": "",
            "section_id": "",
            "chunk_index": 0,
        }
    ))

    # ---- Split into sections ----
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "section")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = header_splitter.split_text(markdown)
    print(f"  Sections found: {len(nodes)}")

    for node in nodes:
        section_name = node.metadata.get("section") or node.metadata.get("h1") or ""
        section_id = str(uuid.uuid4()) if section_name else ""
        node_text = node.page_content

        text_without_tables, tables = extract_tables_from_markdown(node_text)

        # ---- L1: Section summary ----
        if section_name and use_summaries:
            print(f"  Section summary: {section_name[:60]}")
            section_summary = summarize_with_ollama(node_text, SECTION_SUMMARY_PROMPT)
            all_docs.append(Document(
                page_content=section_summary,
                metadata={
                    "source": str(file_path),
                    "namespace": namespace,
                    "level": "section",
                    "doc_id": doc_id,
                    "section": section_name,
                    "section_id": section_id,
                    "chunk_index": 0,
                }
            ))

        # ---- L2: Tables (never split) ----
        for idx, table_text in enumerate(tables):
            all_docs.append(Document(
                page_content=table_text,
                metadata={
                    "source": str(file_path),
                    "namespace": namespace,
                    "level": "table",
                    "doc_id": doc_id,
                    "section": section_name,
                    "section_id": section_id,
                    "chunk_index": idx,
                }
            ))

        # ---- L2: Fine text chunks ----
        for idx, chunk in enumerate(size_splitter.split_text(text_without_tables)):
            if chunk.strip():
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "namespace": namespace,
                        "level": "chunk",
                        "doc_id": doc_id,
                        "section": section_name,
                        "section_id": section_id,
                        "chunk_index": idx,
                    }
                ))

    print(f"  Total chunks: {len(all_docs)}")
    return all_docs


# -----------------------
# Directory Ingestion
# -----------------------

def process_and_ingest_directory(
    directory_path: Union[str, Path],
    drop_existing: bool = False,
    namespace: str = None,
    file_extensions: List[str] = None,
    use_summaries: bool = True,
    config: Optional["ConfigLoader"] = None,
):
    directory_path = Path(directory_path)
    config_to_use = config if config else globals()["config"]

    namespace = namespace or config_to_use.get("database", "namespace", default="default")
    file_extensions = file_extensions or config_to_use.get(
        "document", "supported_file_types", default=[".pdf"]
    )

    print("\nStarting hierarchical ingestion")
    print(f"Directory  : {directory_path}")
    print(f"Namespace  : {namespace}")
    print(f"Summaries  : {'enabled (requires Ollama)' if use_summaries else 'disabled'}")

    if not directory_path.exists():
        raise FileNotFoundError(directory_path)

    files = [
        f for ext in file_extensions
        for f in directory_path.glob(f"*{ext}")
        if f.name != ".DS_Store"
    ]
    print(f"Files found: {len(files)}")

    if not files:
        print("No files found.")
        return

    converter = get_document_converter()
    store = MilvusStore(drop_old=drop_existing, namespace=namespace)

    all_docs = []
    for file in files:
        try:
            docs = process_file_hierarchical(file, converter, namespace, use_summaries)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    if not all_docs:
        print("No documents created.")
        return

    print(f"\nIndexing {len(all_docs)} chunks into Qdrant...")
    ids = store.add_documents(all_docs)
    print(f"\nDone. Indexed {len(ids)} documents.")


# -----------------------
# CLI
# -----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical RAG ingestion pipeline")
    parser.add_argument("directory", help="Directory containing documents")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection first")
    parser.add_argument("--namespace", help="Namespace for this ingestion")
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip LLM summary generation (faster, no Ollama needed)"
    )
    args = parser.parse_args()

    process_and_ingest_directory(
        directory_path=args.directory,
        drop_existing=args.drop,
        namespace=args.namespace,
        use_summaries=not args.no_summaries,
    )