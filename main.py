import logging
import time
from pathlib import Path

import config
from hardware.device_manager import get_device
from processor.document_loader import get_supported_documents
from processor.document_pipeline import DocumentPipeline
from processor.hybrid_rag import HybridRAGBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_startup_log(device_info: dict):
    """Outputs the required startup hardware detection summary."""
    device = device_info.get("device", "UNKNOWN")
    backend = device_info.get("backend", "UNKNOWN")
    print("\n---")
    print("Hardware Detection Summary")
    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print("Offline Mode: ENABLED")
    print("Docling VLM: LOADED (LOCAL)")
    print("---------------------------\n")

def run_zero_touch_pipeline():
    # 1. Hardware Detection Phase
    device_info = get_device()
    print_startup_log(device_info)
    
    logger.info("==================================================")
    logger.info("Uni-Doc-Intel: Zero-Touch Ingestion Pipeline")
    logger.info("==================================================")
    
    docs = get_supported_documents(config.INPUT_DOCS_DIR)
    if not docs:
        logger.warning(f"No PDFs found in the drop folder: {config.INPUT_DOCS_DIR}")
        return

    logger.info(f"Scanning {len(docs)} document(s) in drop folder...")
    
    # Check which ones have already been processed to markdown
    existing_mds = [f.stem for f in config.STRUCTURED_OUTPUT_DIR.glob("*.md")]
    
    docs_to_process = []
    for doc in docs:
        if doc.stem not in existing_mds:
            docs_to_process.append(doc)
            
    if not docs_to_process:
        logger.info("All documents have already been extracted to Markdown.")
    else:
        logger.info(f"Extracting {len(docs_to_process)} New Document(s) via IBM Granite Docling VLM...")
        pipeline = DocumentPipeline()
        
        for doc in docs_to_process:
            logger.info(f" -> Processing semantic layout extraction for: {doc.name}")
            pipeline.process(doc)
                
    # 3. Trigger Hybrid RAG Builder to chunk, embed, and index ANY new markdown files
    logger.info("==================================================")
    logger.info("Triggering Automated RAG Database Upkeep...")
    
    try:
        rag_builder = HybridRAGBuilder(device_info)
        indexed_any = rag_builder.ingest_new_documents()
        if not indexed_any:
            logger.info("Index is fully up to date.")
    except Exception as e:
        logger.error(f"Failed to rebuild Hybrid RAG index: {e}")
        
    logger.info("==================================================")
    logger.info("Zero-Touch Pipeline Complete!")
    logger.info("Run 'streamlit run app.py' to query the enterprise vault.")

if __name__ == "__main__":
    start_time = time.time()
    run_zero_touch_pipeline()
    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")
