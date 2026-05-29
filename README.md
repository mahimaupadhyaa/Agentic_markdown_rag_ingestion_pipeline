# Agentic Markdown RAG Ingestion Pipeline

An intelligent markdown ingestion and chunking pipeline for Retrieval-Augmented Generation (RAG) systems.

This project parses markdown documents into semantically meaningful chunks while preserving structural integrity for tables, code blocks, images, and hierarchical sections.

Designed specifically for production-grade RAG pipelines where naive chunking breaks document semantics.

---

# Features

## Intelligent Semantic Chunking

* Splits prose using sentence-aware chunking
* Maintains semantic continuity with configurable overlap
* Respects token boundaries

## Table Preservation

* Tables are always kept atomic
* Never splits rows across chunks
* Preserves markdown table structure
* Generates semantic table summaries for better embeddings

## Image Extraction

* Extracts:

  * image paths
  * alt text
  * section context
* Includes hooks for Vision-Language Model (VLM) captioning

## Code Block Preservation

* Keeps fenced code blocks atomic
* Preserves programming language metadata
* Prevents broken code chunking

## Hierarchical Context Tracking

* Tracks document structure using heading breadcrumbs

Example:

```text
Introduction > Architecture > Embedding Layer
```

## PDF-to-Markdown Artifact Cleanup

Special handling for noisy markdown generated from PDFs:

* removes page-number artifacts
* removes broken annex labels
* removes malformed table remnants
* filters low-signal chunks

## Production-Oriented Output

Each chunk contains:

* chunk text
* block type
* token count
* source file
* section path
* metadata

---

# Supported Chunk Types

| Block Type  | Handling Strategy              |
| ----------- | ------------------------------ |
| Text        | Semantic sentence chunking     |
| Tables      | Atomic preservation            |
| Code Blocks | Atomic preservation            |
| Images      | Atomic extraction              |
| Lists       | Flattened into prose           |
| Quotes      | Preserved as standalone chunks |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/mahimaupadhyaa/Agentic_markdown_rag_ingestion_pipeline.git

cd Agentic_markdown_rag_ingestion_pipeline
```

## Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install marko
```

---

# Usage

## Process a Single Markdown File

```bash
python ingest.py --input README.md
```

## Process a Directory Recursively

```bash
python ingest.py --input docs/
```

## Specify Output File

```bash
python ingest.py --input docs/ --output chunks.json
```

## Configure Chunking Parameters

```bash
python ingest.py \
  --input docs/ \
  --output chunks.json \
  --max-tokens 512 \
  --min-tokens 50 \
  --overlap 1
```

## Pretty Print JSON Output

```bash
python ingest.py --input docs/ --pretty
```

---

# Output Schema

Each chunk is emitted as structured JSON.

Example:

```json
{
  "text": "[Section: Introduction]\\n\\nThis is a semantic chunk...",
  "block_type": "text",
  "source_file": "docs/intro.md",
  "section_path": "Introduction > Overview",
  "chunk_index": 0,
  "token_count": 184,
  "metadata": {}
}
```

---

# Example Chunk Types

## Text Chunk

```json
{
  "block_type": "text"
}
```

## Table Chunk

```json
{
  "block_type": "table",
  "metadata": {
    "row_count": 12,
    "column_names": ["Name", "Age", "Country"]
  }
}
```

## Image Chunk

```json
{
  "block_type": "image",
  "metadata": {
    "image_path": "images/architecture.png",
    "alt_text": "System architecture"
  }
}
```

## Code Chunk

```json
{
  "block_type": "code",
  "metadata": {
    "language": "python"
  }
}
```

---

# Architecture

```text
Markdown File
      ↓
Marko AST Parsing
      ↓
AST Walker
      ↓
Block-Type Routing
 ├── Tables
 ├── Images
 ├── Code Blocks
 └── Prose
      ↓
Semantic Chunking
      ↓
Artifact Cleanup
      ↓
Structured JSON Output
```

---

# Why This Exists

Most RAG ingestion pipelines use naive fixed-size chunking.

That causes:

* broken tables
* fragmented code
* lost section context
* poor retrieval quality
* hallucinations during generation

This project solves those issues with structure-aware chunking.

---

# Future Improvements

* tiktoken integration for exact token counting
* multimodal image captioning
* vector database ingestion
* LangChain integration
* LlamaIndex integration
* HTML/PDF ingestion
* metadata enrichment
* adaptive chunk sizing
* embedding pipelines
* async ingestion

---

# Tech Stack

* Python
* Marko
* GitHub Flavored Markdown (GFM)
* AST-based parsing

---

# Repository Structure

```text
.
├── ingest.py
├── requirements.txt
├── README.md
└── chunks.json
```

---

# Example Use Cases

* Enterprise RAG systems
* Documentation search
* Knowledge base ingestion
* Technical manuals
* Compliance documents
* Research paper indexing
* PDF-to-RAG pipelines
* AI assistants
* Agentic retrieval systems

---

# Contributing

Contributions are welcome.

Possible areas:

* better semantic splitting
* tokenizer integrations
* multimodal support
* benchmark evaluations
* vector DB adapters



# Author

Mahima Upadhyaa

GitHub:
https://github.com/mahimaupadhyaa
