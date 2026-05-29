"""
Semantic Hierarchical RAG Ingestion Pipeline

Strategy:
1. Docling converts PDF → clean markdown + extracts tables natively
2. Tables extracted directly from doc.tables (never split, never missed)
3. Semantic chunking on text — splits on meaning shifts, not token count
4. Context injection — every chunk gets doc title + section heading prepended
5. Parent-child storage — parent chunk for generation, propositions for retrieval
6. (Optional) Propositionalization via Ollama
"""

import os
import uuid
import re
import requests
from pathlib import Path
from typing import List, Optional, Union

from config import config, ConfigLoader

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from qdrant_store import QdrantStore as MilvusStore

from dotenv import load_dotenv
load_dotenv()


# -----------------------
# Config
# -----------------------

OLLAMA_URL = config.get("ollama", "url", default="http://localhost:11434/api/generate")
OLLAMA_MODEL = config.get("ollama", "model", default="qwen2.5:14b")

PROPOSITION_PROMPT = """Decompose the following text into clear, atomic factual statements.
Each statement must:
- Be self-contained and independently understandable
- Contain exactly one fact
- Be a complete sentence

Return ONLY the statements, one per line, no numbering, no preamble.

TEXT:
{text}"""


# -----------------------
# Docling
# -----------------------

def get_document_converter(
    pdf_pipeline_options: Optional[PdfPipelineOptions] = None,
) -> DocumentConverter:
    pdf_pipeline_options = pdf_pipeline_options or config.get_pdf_pipeline_options()
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )


def convert_pdf(file_path: Path, converter: DocumentConverter):
    """
    Convert PDF and return (markdown_text, [table_markdown_strings]).
    Tables are extracted directly from Docling's native table objects —
    not from markdown regex — so they are never missed or corrupted.
    The markdown export uses PLACEHOLDER for tables so they don't
    appear as garbled text in the main body.
    """
    print(f"  Converting with Docling: {file_path.name}")
    result = converter.convert(str(file_path))
    doc = result.document

    # Extract all tables via Docling's native API
    tables = []
    for table in doc.tables:
        try:
            md = table.export_to_markdown()
            if md.strip():
                tables.append(md.strip())
        except Exception as e:
            print(f"  Warning: could not export table: {e}")

    print(f"  Tables found by Docling: {len(tables)}")

    # Export body text — tables appear as <!-- image --> placeholders,
    # keeping the prose clean with no table fragments
    markdown = doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

    # Strip any residual placeholder lines so they don't pollute chunks
    clean_lines = [
        line for line in markdown.splitlines()
        if "<!-- image -->" not in line.lower()
    ]
    markdown = "\n".join(clean_lines)

    return markdown, tables


# -----------------------
# Propositionalization
# -----------------------

def propositionalize(text: str) -> List[str]:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": PROPOSITION_PROMPT.format(text=text), "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        lines = [l.strip() for l in r.json()["response"].strip().splitlines()]
        return [l for l in lines if l]
    except Exception as e:
        print(f"    Propositionalization failed: {e}")
        return [text]


# -----------------------
# Context Injection
# -----------------------

def inject_context(chunk_text: str, doc_title: str, section: str) -> str:
    parts = []
    if doc_title:
        parts.append(f"Document: {doc_title}")
    if section:
        parts.append(f"Section: {section}")
    return "\n".join(parts) + "\n\n" + chunk_text if parts else chunk_text


# -----------------------
# Core Pipeline
# -----------------------

def process_file(
    file_path: Path,
    converter: DocumentConverter,
    embeddings: OpenAIEmbeddings,
    namespace: str,
    use_propositions: bool = True,
) -> List[Document]:

    print(f"\nProcessing: {file_path.name}")

    markdown, tables = convert_pdf(file_path, converter)
    doc_id = str(uuid.uuid4())
    doc_title = file_path.stem
    all_docs = []

    # ---- Store all tables as intact documents ----
    for idx, table_md in enumerate(tables):
        context_text = inject_context(table_md, doc_title, "")
        all_docs.append(Document(
            page_content=context_text,
            metadata={
                "source": str(file_path),
                "namespace": namespace,
                "level": "table",
                "doc_id": doc_id,
                "section": "",
                "section_id": "",
                "parent_id": "",
                "chunk_index": idx,
            }
        ))

    # ---- Split prose into sections ----
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "section")],
        strip_headers=False,
    )
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    nodes = header_splitter.split_text(markdown)
    print(f"  Sections found: {len(nodes)}")

    for node in nodes:
        section_name = node.metadata.get("section") or node.metadata.get("h1") or ""
        section_id = str(uuid.uuid4()) if section_name else ""
        clean_text = node.page_content.strip()

        if not clean_text:
            continue

        try:
            semantic_chunks = semantic_splitter.split_text(clean_text)
        except Exception as e:
            print(f"  Semantic split failed, using full section: {e}")
            semantic_chunks = [clean_text]

        print(f"  '{section_name[:50]}': {len(semantic_chunks)} semantic chunks")

        for idx, chunk_text in enumerate(semantic_chunks):
            if not chunk_text.strip():
                continue

            chunk_id = str(uuid.uuid4())
            parent_text = inject_context(chunk_text, doc_title, section_name)

            # Parent chunk — full context for generation
            all_docs.append(Document(
                page_content=parent_text,
                metadata={
                    "source": str(file_path),
                    "namespace": namespace,
                    "level": "parent",
                    "doc_id": doc_id,
                    "section": section_name,
                    "section_id": section_id,
                    "parent_id": chunk_id,
                    "chunk_index": idx,
                }
            ))

            if use_propositions:
                propositions = propositionalize(chunk_text)
                print(f"    Chunk {idx}: {len(propositions)} propositions")
                for p_idx, prop in enumerate(propositions):
                    all_docs.append(Document(
                        page_content=inject_context(prop, doc_title, section_name),
                        metadata={
                            "source": str(file_path),
                            "namespace": namespace,
                            "level": "proposition",
                            "doc_id": doc_id,
                            "section": section_name,
                            "section_id": section_id,
                            "parent_id": chunk_id,
                            "chunk_index": p_idx,
                        }
                    ))
            else:
                # Index chunk directly alongside parent
                all_docs.append(Document(
                    page_content=parent_text,
                    metadata={
                        "source": str(file_path),
                        "namespace": namespace,
                        "level": "chunk",
                        "doc_id": doc_id,
                        "section": section_name,
                        "section_id": section_id,
                        "parent_id": chunk_id,
                        "chunk_index": idx,
                    }
                ))

    print(f"  Total documents: {len(all_docs)} "
          f"({len(tables)} tables + {len(all_docs) - len(tables)} text chunks)")
    return all_docs


# -----------------------
# Directory Ingestion
# -----------------------

def process_and_ingest_directory(
    directory_path: Union[str, Path],
    drop_existing: bool = False,
    namespace: str = None,
    file_extensions: List[str] = None,
    use_propositions: bool = True,
    config: Optional["ConfigLoader"] = None,
):
    directory_path = Path(directory_path)
    config_to_use = config if config else globals()["config"]

    namespace = namespace or config_to_use.get("database", "namespace", default="default")
    file_extensions = file_extensions or config_to_use.get(
        "document", "supported_file_types", default=[".pdf"]
    )

    print("\n=== Semantic Hierarchical RAG Ingestion ===")
    print(f"Directory    : {directory_path}")
    print(f"Namespace    : {namespace}")
    print(f"Propositions : {'enabled (requires Ollama)' if use_propositions else 'disabled'}")

    if not directory_path.exists():
        raise FileNotFoundError(directory_path)

    files = [
        f for ext in file_extensions
        for f in directory_path.glob(f"*{ext}")
        if f.name != ".DS_Store"
    ]
    print(f"Files found  : {len(files)}")

    if not files:
        print("No files found.")
        return

    converter = get_document_converter()
    embeddings = OpenAIEmbeddings(
        model=config_to_use.get("model", "embeddings", default="text-embedding-3-small"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    store = MilvusStore(drop_old=drop_existing, namespace=namespace)

    all_docs = []
    for file in files:
        try:
            docs = process_file(file, converter, embeddings, namespace, use_propositions)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    if not all_docs:
        print("No documents created.")
        return

    print(f"\nIndexing {len(all_docs)} documents into Qdrant...")
    ids = store.add_documents(all_docs)
    print(f"\nDone. Indexed {len(ids)} documents.")


# -----------------------
# CLI
# -----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Hierarchical RAG ingestion")
    parser.add_argument("directory", help="Directory containing documents")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection first")
    parser.add_argument("--namespace", help="Namespace for this ingestion")
    parser.add_argument(
        "--no-propositions",
        action="store_true",
        help="Skip propositionalization (faster, no Ollama needed)"
    )
    args = parser.parse_args()

    process_and_ingest_directory(
        directory_path=args.directory,
        drop_existing=args.drop,
        namespace=args.namespace,
        use_propositions=not args.no_propositions,
    )