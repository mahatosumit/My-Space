import os
from pathlib import Path
from typing import List

def get_supported_documents(directory: Path) -> List[Path]:
    """Retrieves a list of supported document paths from the given directory."""
    supported_extensions = {".pdf", ".docx", ".pptx", ".md", ".html"}
    files = []
    if not directory.exists() or not directory.is_dir():
        return files
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in supported_extensions:
            files.append(file)
    return files
