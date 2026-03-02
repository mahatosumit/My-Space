# Document AI Pipeline

A CPU-optimized, production-ready Document AI pipeline utilizing IBM Granite Docling technology for document understanding.

## Architecture

* **Input**: PDF files, scanned documents, mixed layouts placed in `input_docs/`.
* **Processor Core**: `processor/docling_parser.py` wraps the IBM Docling `DocumentConverter` with CPU-optimized logic.
* **Extraction**: 
  * `processor/text_exporter.py` extracts clean, readable plain text.
  * `processor/structured_exporter.py` extracts rich layout-aware Markdown and dictionary structures (JSON) suited for downstream RAG embeddings.
* **Outputs**: Clean text goes to `outputs/text/` and structured formats go to `outputs/structured/`.
* **Configuration**: `config.py` enforces CPU execution by neutralizing GPU flags and locking `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.

## Installation

1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the pipeline on an individual file or an entire directory.

**Process a single file:**
```bash
python main.py input_docs/sample.pdf
```

**Process all documents in the default input directory (`input_docs/`):**
```bash
python main.py
```

## How to Extend
- **Custom Exporters**: Add new export logic (like chunking) by creating new modules in `processor/` and integrating them into `main.process_file`.
- **Advanced OCR Tuning**: Modify `PdfPipelineOptions` within `processor/docling_parser.py` if custom Tesseract dictionaries or non-CPU backends become available.
