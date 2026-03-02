import logging
from pathlib import Path
import config

logger = logging.getLogger(__name__)

def export_to_text(docling_result, base_filename: str):
    """Exports the parsed document to plain text."""
    try:
        text_content = docling_result.document.export_to_text()
        output_path = config.TEXT_OUTPUT_DIR / f"{base_filename}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        logger.info(f"Saved plain text output to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to export text for {base_filename}: {e}")
        return None
