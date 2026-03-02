import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat

import config

logger = logging.getLogger(__name__)

class DoclingParser:
    def __init__(self, device_info: dict):
        self.device_info = device_info
        
        # Configure pipeline options for offline Granite models
        pipeline_options = PdfPipelineOptions()
        
        # Enforce local models path
        pipeline_options.artifacts_path = config.DOCLING_MODEL_DIR
        
        # Hardware backend translation
        backend = device_info.get("backend", "cpu")
        accelerator = device_info.get("device", "cpu")
        
        # Explicitly configure Granite's layout engine to use the detected hardware
        # E.g., apply CUDA or CPU appropriately via openvino/directml
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=config.DOCLING_CPU_THREADS, 
            device=accelerator
        )
        
        # Disable OCR since we want to avoid RapidOCR and rely purely on 
        # the native PDF embedded text + IBM Granite layout models.
        pipeline_options.do_ocr = False
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info(f"Initialized DoclingParser for {backend} inference.")

    def parse(self, file_path: Path):
        """Extracts structural and semantic information from the document."""
        logger.info(f"Parsing document: {file_path}")
        result = self.converter.convert(file_path)
        return result
