"""
Document ingestion module.

This script:

1) Reads documents from a directory
2) Converts them using Docling
3) Chunks them
4) Converts them to LangChain Documents
5) Indexes them into Qdrant
"""

import os 
from pathlib import Path
from typing import List, Optional, Union

from config import config, ConfigLoader


from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider
)
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
    MarkdownParams
)
from docling_core.types.doc import ImageRefMode
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from transformers import AutoTokenizer
from qdrant_store import QdrantStore as MilvusStore  # drop-in replacement

def get_document_converter(
    pdf_pipeline_options: Optional[PdfPipelineOptions] = None,
) -> DocumentConverter:
    """
    Creates the Docling document converter.

    Responsible for converting:

    PDF → structured document
    """

    pdf_pipeline_options = pdf_pipeline_options or config.get_pdf_pipeline_options()

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options
            )
        }
    )

    return converter

def get_chunker(
    config: Optional["ConfigLoader"] = None
) -> HybridChunker:
    """
    Creates document chunker.
    """

    config_to_use = config if config else globals()["config"]

    tokenizer_model = config_to_use.get(
        "model", "tokenizer", default="sentence-transformers/all-MiniLM-L6-v2"
    )

    max_tokens = config_to_use.get("document", "max_tokens", default=1024)

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_model),
        max_tokens=max_tokens,
    )

    class CustomSerializerProvider(ChunkingSerializerProvider):

        def get_serializer(self, doc):

            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(
                    image_mode=ImageRefMode.PLACEHOLDER,
                    image_placeholder="",
                    mark_annotations=True,
                    include_annotations=True,
                )
            )
    
    chunker = HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=CustomSerializerProvider()
    )

    return chunker

def process_file(
    file_path: Union[str, Path],
    converter: DocumentConverter,
    chunker: HybridChunker,
    namespace: str
) -> List[Document]:
    """
    Process a single document file.
    """
    
    file_path = Path(file_path)

    print(f"\nProcessing file: {file_path}")

    loader = DoclingLoader(
        file_path=file_path,
        converter=converter,
        chunker=chunker,
        export_type=ExportType.DOC_CHUNKS
    )

    docs = loader.load()

    processed_docs = []

    for doc in docs:
        metadata = doc.metadata
        
        new_metadata = {
            "source": str(metadata["source"]),
            "page_no": metadata["dl_meta"]["doc_items"][0]["prov"][0]["page_no"],
            "namespace": namespace,
        }

        processed_doc = Document(
            page_content=doc.page_content,
            metadata=new_metadata
        )

        processed_docs.append(processed_doc)

    print(f"Chunks created: {len(processed_docs)}")
    
    return processed_docs

def process_and_ingest_directory(
    directory_path: Union[str, Path],
    drop_existing: bool = False,
    namespace: str = None,
    file_extensions: List[str] = None,
    config: Optional["ConfigLoader"] = None
):
    """
    Process an entire directory and ingest documents.
    """

    directory_path = Path(directory_path)

    config_to_use = config if config else globals()["config"]

    namespace = namespace or config_to_use.get(
        "database", "namespace", default="Default_"
    )

    file_extensions = file_extensions or config_to_use.get(
        "document", "supported_file_types", default=[".pdf"]
    )

    print("\nStarting ingestion")
    print("Directory:", directory_path)

    if not directory_path.exists():
        raise FileNotFoundError(directory_path)

    files = []

    for ext in file_extensions:
        files.extend(directory_path.glob(f"*{ext}"))

    print("Files found:", len(files))

    if not files:
        print("No files found")
        return

    converter = get_document_converter()
    chunker = get_chunker(config=config)

    milvus_store = MilvusStore(
        drop_old=drop_existing,
        namespace=namespace,
    )

    all_docs = []

    for file in files:

        if file.name == ".DS_Store":
            continue
            
        try:

            docs = process_file(
                file,
                converter,
                chunker,
                namespace,
            )

            all_docs.extend(docs)

        except Exception as e:
            print(f"Error processing file: {file}")
            print(e)

    if not all_docs:
        print("No documents created")
        return

    print("\nIndexing documents into Qdrant")

    ids = milvus_store.add_documents(all_docs)

    print(f"\nSuccessfully indexed {len(ids)} documents")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        help="Directory containing documents",
    )

    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing collection",
    )

    parser.add_argument(
        "--namespace",
        help="Namespace",
    )

    args = parser.parse_args()

    process_and_ingest_directory(
        directory_path=args.directory,
        drop_existing=args.drop,
        namespace=args.namespace,
    )