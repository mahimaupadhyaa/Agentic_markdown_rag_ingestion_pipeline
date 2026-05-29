"""
PDF → Markdown Converter (powered by Marker)
=============================================
Converts one PDF file or a whole folder of PDFs into clean markdown files,
ready to be fed into ingest.py for RAG chunking.

Marker handles:
  - Digital PDFs  (text layer extracted directly — fast, no OCR needed)
  - Scanned PDFs  (OCR via surya — pass --force-ocr)
  - Tables        (preserved as markdown tables)
  - Images        (saved as .png files alongside the .md output)
  - Math/formulas (converted to LaTeX when --use-llm is set)
  - Code blocks   (detected and fenced)

Installation (run once):
    pip install marker-pdf

    # CPU-only machine (no GPU):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install marker-pdf

GPU note:
    Marker auto-detects CUDA/MPS. Default batch sizes use ~3 GB VRAM.
    Pass --batch-multiplier 2 to double throughput if you have more VRAM.

LLM-assisted mode (highest accuracy, requires a Gemini API key):
    export GOOGLE_API_KEY=your_key_here
    python convert_pdf.py --input doc.pdf --use-llm

Usage examples:
    # Convert a single PDF
    python convert_pdf.py --input report.pdf

    # Convert a whole folder
    python convert_pdf.py --input ./pdfs/ --output ./markdown/

    # Scanned/image PDFs — force OCR
    python convert_pdf.py --input scan.pdf --force-ocr

    # Multi-language document
    python convert_pdf.py --input doc.pdf --langs English,Hindi

    # Then run the ingestion pipeline on the output:
    python ingest.py --input ./markdown/ --output chunks.json --pretty
"""

import argparse
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency check — fail early with a helpful message
# ---------------------------------------------------------------------------

def check_marker_installed():
    """
    Marker is a large ML package. Rather than silently failing at import time,
    we check up front and print clear installation instructions.
    """
    try:
        import marker  # noqa: F401
        return True
    except ImportError:
        print(
            "\n[error] marker-pdf is not installed.\n"
            "\nInstall it with:\n"
            "    pip install marker-pdf\n"
            "\nOn a CPU-only machine, install torch first:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "    pip install marker-pdf\n",
            file=sys.stderr,
        )
        return False


# ---------------------------------------------------------------------------
# Single-file conversion
# ---------------------------------------------------------------------------

def convert_single(
    pdf_path: Path,
    output_dir: Path,
    langs: list[str],
    force_ocr: bool,
    batch_multiplier: int,
    max_pages: int | None,
    use_llm: bool,
) -> Path | None:
    """
    Convert one PDF file to markdown using Marker's Python API.

    Returns the path to the generated .md file, or None on failure.

    How Marker works internally:
      1. Layout detection  — identifies text blocks, headings, tables, images
      2. OCR (if needed)   — surya OCR runs on image-based pages
      3. Structure rebuild — reassembles reading order, merges columns
      4. Markdown render   — emits clean .md with fenced tables and code blocks
      5. LLM cleanup       — optional Gemini pass fixes cross-page tables and math
    """
    # Import here so the check above can give a nice error before we reach this point
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser

    print(f"  Converting: {pdf_path.name}")
    start = time.time()

    # Build Marker's config dict from our CLI args
    # Full list of options: https://github.com/datalab-to/marker
    config = {
        "langs": ",".join(langs) if langs else None,
        "force_ocr": force_ocr,
        "batch_multiplier": batch_multiplier,
        "use_llm": use_llm,
        "output_format": "markdown",
    }
    if max_pages is not None:
        config["max_pages"] = max_pages

    # Remove None values — Marker uses its own defaults for missing keys
    config = {k: v for k, v in config.items() if v is not None}

    try:
        # Load all required ML models once (layout, OCR, table, text cleanup)
        # This is the slow step on first run — models are cached to disk after
        config_parser = ConfigParser(config)
        model_dict = create_model_dict()

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        # Run the full conversion pipeline
        rendered = converter(str(pdf_path))

        # Extract markdown text and images from the rendered output
        markdown_text, images, metadata = text_from_rendered(rendered)

        # Write the .md file
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / (pdf_path.stem + ".md")
        md_path.write_text(markdown_text, encoding="utf-8")

        # Save extracted images alongside the markdown
        if isinstance(images, dict) and images:
            img_dir = output_dir / pdf_path.stem
            img_dir.mkdir(exist_ok=True)

            for img_name, img_data in images.items():
                img_path = img_dir / img_name

                if hasattr(img_data, "save"):
                    img_data.save(str(img_path))
                else:
                    img_path.write_bytes(img_data)

            print(f"    Saved {len(images)} image(s) → {img_dir}/")

        elapsed = time.time() - start
        page_count = metadata.get("pages", "?")
        print(f"    Done — {page_count} pages in {elapsed:.1f}s → {md_path}")
        return md_path

    except Exception as exc:
        print(f"    [error] Failed to convert {pdf_path.name}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Batch conversion (folder of PDFs)
# ---------------------------------------------------------------------------

def convert_directory(
    input_dir: Path,
    output_dir: Path,
    langs: list[str],
    force_ocr: bool,
    batch_multiplier: int,
    max_pages: int | None,
    use_llm: bool,
    skip_existing: bool,
) -> list[Path]:
    """
    Find all PDFs in input_dir (recursively) and convert each one.

    Output mirrors the input folder structure under output_dir so that:
        input_dir/research/paper.pdf  →  output_dir/research/paper.md

    Pass --skip-existing to resume an interrupted batch without re-converting
    files that already have a corresponding .md in the output folder.
    """
    pdf_files = sorted(input_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"[warn] No PDF files found in {input_dir}", file=sys.stderr)
        return []

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}\n")

    converted: list[Path] = []
    skipped = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        # Mirror subdirectory structure in output
        relative = pdf_path.relative_to(input_dir)
        file_output_dir = output_dir / relative.parent

        expected_md = file_output_dir / (pdf_path.stem + ".md")
        if skip_existing and expected_md.exists():
            print(f"  [{i}/{len(pdf_files)}] Skipping (already converted): {pdf_path.name}")
            skipped += 1
            converted.append(expected_md)
            continue

        print(f"  [{i}/{len(pdf_files)}]", end=" ", flush=True)
        md_path = convert_single(
            pdf_path=pdf_path,
            output_dir=file_output_dir,
            langs=langs,
            force_ocr=force_ocr,
            batch_multiplier=batch_multiplier,
            max_pages=max_pages,
            use_llm=use_llm,
        )

        if md_path:
            converted.append(md_path)
        else:
            failed += 1

    print(f"\nBatch complete — {len(converted)} converted, {skipped} skipped, {failed} failed.")
    return converted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF(s) to markdown using Marker, ready for ingest.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a single PDF file OR a directory containing PDF files."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=(
            "Where to write .md files. "
            "Defaults to ./markdown/ for directories, or same folder as the PDF for single files."
        )
    )
    parser.add_argument(
        "--langs", default=None,
        help="Comma-separated OCR language hints, e.g. 'English,Hindi'. "
             "Optional for digital PDFs; helps accuracy on scanned docs."
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Force OCR on every page, even if text is already embedded. "
             "Use this when you see garbled/corrupt text in the output."
    )
    parser.add_argument(
        "--batch-multiplier", type=int, default=2,
        help="Multiply default VRAM batch sizes by this factor. "
             "Higher = faster but uses more VRAM. Default: 2 (~3 GB VRAM)."
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Stop after converting this many pages. Useful for testing large PDFs."
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use Gemini LLM for higher accuracy (merges cross-page tables, fixes math). "
             "Requires GOOGLE_API_KEY env variable to be set."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="(Batch mode only) Skip PDFs that already have a .md in the output folder."
    )

    args = parser.parse_args()

    # Validate marker installation before doing anything else
    if not check_marker_installed():
        sys.exit(1)

    # Validate --use-llm has the required API key
    if args.use_llm and not os.environ.get("GOOGLE_API_KEY"):
        print(
            "[error] --use-llm requires a Gemini API key.\n"
            "Set it with:  export GOOGLE_API_KEY=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    langs = [l.strip() for l in args.langs.split(",")] if args.langs else []

    print(f"\n=== PDF → Markdown Converter (Marker) ===")
    print(f"Input          : {input_path}")
    print(f"Force OCR      : {args.force_ocr}")
    print(f"LLM assist     : {args.use_llm}")
    print(f"Batch multiplier: {args.batch_multiplier}x")
    if langs:
        print(f"Languages      : {', '.join(langs)}")
    print()

    if input_path.is_dir():
        output_dir = Path(args.output) if args.output else Path("./markdown")
        convert_directory(
            input_dir=input_path,
            output_dir=output_dir,
            langs=langs,
            force_ocr=args.force_ocr,
            batch_multiplier=args.batch_multiplier,
            max_pages=args.max_pages,
            use_llm=args.use_llm,
            skip_existing=args.skip_existing,
        )
    else:
        output_dir = Path(args.output) if args.output else input_path.parent / "markdown"
        convert_single(
            pdf_path=input_path,
            output_dir=output_dir,
            langs=langs,
            force_ocr=args.force_ocr,
            batch_multiplier=args.batch_multiplier,
            max_pages=args.max_pages,
            use_llm=args.use_llm,
        )

    print(f"\nNext step — run the RAG ingestion pipeline:")
    out_display = args.output or ("./markdown" if input_path.is_dir() else str(input_path.parent / "markdown"))
    print(f"    python ingest.py --input {out_display} --output chunks.json --pretty\n")


if __name__ == "__main__":
    main()