import os
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DOCLING_MODEL_DIR = MODELS_DIR / "granite_docling"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

def download_models():
    logger.info("Starting offline model downloads...")
    
    # Ensure directories exist
    os.makedirs(DOCLING_MODEL_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # 1. Download SentenceTransformer
    embed_model = "all-MiniLM-L6-v2"
    logger.info(f"Downloading {embed_model}...")
    snapshot_download(
        repo_id=f"sentence-transformers/{embed_model}",
        local_dir=EMBEDDINGS_DIR / embed_model,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    # 2. Download Cross-Encoder
    cross_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    logger.info(f"Downloading {cross_model}...")
    snapshot_download(
        repo_id=cross_model,
        local_dir=EMBEDDINGS_DIR / "cross-encoder" / "ms-marco-MiniLM-L-6-v2",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    # 3. Download IBM Granite Docling Models
    docling_repo = "ds4sd/docling-models"
    logger.info(f"Downloading IBM Granite Docling Models from {docling_repo}...")
    snapshot_download(
        repo_id=docling_repo,
        local_dir=DOCLING_MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    logger.info("All models downloaded successfully for OFFLINE execution!")

if __name__ == "__main__":
    download_models()
