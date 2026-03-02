import logging
from pathlib import Path
from hardware.device_manager import get_device
from processor.docling_parser import DoclingParser
from processor.exporters import Exporters

logger = logging.getLogger(__name__)

class DocumentPipeline:
    """
    Orchestrates the Document Intelligence pipeline.
    Responsible for routing files through the Granite Docling VLM
    based on available hardware, and producing exported output.
    """
    def __init__(self):
        # 1. Hardware Detection
        self.device_info = get_device()
        logger.info(f"Initializing DocumentPipeline with: {self.device_info}")

        # 2. Setup Parsers
        self.parser = DoclingParser(self.device_info)
        
    def process(self, file_path: Path):
        """Processes a single document end-to-end."""
        logger.info(f"Pipeline started for document: {file_path.name}")
        
        try:
            # 1. Parse Document using Docling (Hardware Adaptive)
            docling_result = self.parser.parse(file_path)
            
            # 2. Export to Structured Forms (Markdown + JSON)
            Exporters.to_structured(docling_result, file_path.stem)
            
            # 3. Export to Plain Text
            Exporters.to_text(docling_result, file_path.stem)
            
            logger.info(f"Pipeline completed successfully for: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed for {file_path.name}: {e}")
            return False
