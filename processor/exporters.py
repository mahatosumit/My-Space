import logging
from pathlib import Path
import json

import config

logger = logging.getLogger(__name__)

class Exporters:
    """Handles exporting of parsed Document structural information into various formats."""

    @staticmethod
    def to_structured(docling_result, base_filename: str) -> dict:
        """Exports the parsed document to Markdown and JSON formats."""
        outputs = {}
        try:
            # Export Markdown
            md_content = docling_result.document.export_to_markdown()
            md_output_path = config.STRUCTURED_OUTPUT_DIR / f"{base_filename}.md"
            with open(md_output_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Saved structured Markdown to: {md_output_path}")
            outputs['markdown'] = md_output_path

            # Export JSON (Dictionary representation)
            dict_content = docling_result.document.export_to_dict()
            json_output_path = config.STRUCTURED_OUTPUT_DIR / f"{base_filename}.json"
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(dict_content, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved structured JSON to: {json_output_path}")
            outputs['json'] = json_output_path

        except Exception as e:
            logger.error(f"Failed to export structured data for {base_filename}: {e}")
        
        return outputs

    @staticmethod
    def to_text(docling_result, base_filename: str) -> Path | None:
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
