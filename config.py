import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DOCS_DIR = BASE_DIR / "input_docs"
OUTPUTS_DIR = BASE_DIR / "outputs"
TEXT_OUTPUT_DIR = OUTPUTS_DIR / "text"
STRUCTURED_OUTPUT_DIR = OUTPUTS_DIR / "structured"

# Models Directories
MODELS_DIR = BASE_DIR / "models"
DOCLING_MODEL_DIR = MODELS_DIR / "granite_docling"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

# RAG DB Directories
RAG_DB_DIR = BASE_DIR / "db"
FAISS_INDEX_PATH = str(RAG_DB_DIR / "faiss_index.bin")
SQLITE_DB_PATH = str(RAG_DB_DIR / "rag_metadata.db")
BM25_CORPUS_PATH = str(RAG_DB_DIR / "bm25_corpus.pkl")

# Local Models for Offline RAG
EMBEDDING_MODEL_NAME = str(EMBEDDINGS_DIR / "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = str(EMBEDDINGS_DIR / "cross-encoder" / "ms-marco-MiniLM-L-6-v2")
LLM_MODEL_PATH = "Phi-3-mini-4k-instruct-q4.gguf"

# Network & Inference Constraints
# Only block Hugging Face external network requests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Docling specific configuration for CPU optimization
DOCLING_CPU_THREADS = max(1, os.cpu_count() - 1)
os.environ["OMP_NUM_THREADS"] = str(DOCLING_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(DOCLING_CPU_THREADS)

# Ensure all directories exist
os.makedirs(INPUT_DOCS_DIR, exist_ok=True)
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
os.makedirs(STRUCTURED_OUTPUT_DIR, exist_ok=True)
os.makedirs(RAG_DB_DIR, exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCLING_MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
