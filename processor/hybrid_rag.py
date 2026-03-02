import os
import glob
import sqlite3
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

import faiss
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

import config

logger = logging.getLogger(__name__)

class HybridRAGBuilder:
    def __init__(self, device_info: dict):
        self.device_info = device_info
        backend = device_info.get("device", "cpu")
        
        logger.info(f"Initializing Embedding and CrossEncoder models on: {backend}")
        
        # Load local embedding models explicitly
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, 
            device=backend,
            local_files_only=True
        )
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        self.cross_encoder = CrossEncoder(
            config.CROSS_ENCODER_MODEL_NAME, 
            device=backend,
            local_files_only=True
        )
        
        self.index = self._load_or_create_faiss()
        self.bm25_corpus, self.docs_list = self._load_or_create_bm25()
        self._init_sqlite()

    def _init_sqlite(self):
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER,
                filename TEXT,
                header TEXT,
                content TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _load_or_create_faiss(self):
        if os.path.exists(config.FAISS_INDEX_PATH):
            logger.info("Loading existing FAISS index...")
            return faiss.read_index(config.FAISS_INDEX_PATH)
        else:
            logger.info("Creating fresh FAISS index...")
            return faiss.IndexFlatL2(self.dimension)

    def _load_or_create_bm25(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            logger.info("Loading existing BM25 corpus...")
            with open(config.BM25_CORPUS_PATH, "rb") as f:
                data = pickle.load(f)
                return data["bm25"], data["docs_list"]
        else:
            return None, []

    def save_indices(self):
        faiss.write_index(self.index, config.FAISS_INDEX_PATH)
        if self.docs_list:
            # We rebuild the BM25 index fully on save because rank_bm25 doesn't support append easily
            tokenized_corpus = [doc['content'].split(" ") for doc in self.docs_list]
            self.bm25_corpus = BM25Okapi(tokenized_corpus)
            with open(config.BM25_CORPUS_PATH, "wb") as f:
                pickle.dump({"bm25": self.bm25_corpus, "docs_list": self.docs_list}, f)
        logger.info("Indices successfully saved to disk.")

    def ingest_new_documents(self):
        """Scans the structured outputs and ingests only new files."""
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Get already processed files
        cursor.execute("SELECT filename FROM processed_files")
        processed = set([row[0] for row in cursor.fetchall()])
        
        md_files = glob.glob(str(config.STRUCTURED_OUTPUT_DIR / "*.md"))
        new_files = [f for f in md_files if Path(f).name not in processed]
        
        if not new_files:
            logger.info("No new documents to ingest into RAG index.")
            conn.close()
            return False

        logger.info(f"Initiating RAG chunking for {len(new_files)} new document(s)...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150,
            length_function=len
        )

        total_new_chunks = 0
        
        for file_path in new_files:
            filename = Path(file_path).name
            logger.info(f"Chunking {filename}...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Naive section split by markdown header heuristics
            # A robust enterprise pipeline uses the structured JSON, but falling back to MD chunks works well too.
            split_texts = splitter.split_text(content)
            
            # Embed all chunks
            embeddings = self.embedding_model.encode(split_texts, batch_size=32, show_progress_bar=False)
            
            # Add to FAISS
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Add to Database & Chunk list
            for i, text in enumerate(split_texts):
                chunk_id = self.index.ntotal - len(split_texts) + i
                doc_record = {
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "header": "Section", # Placeholder, can be extracted from markdown text
                    "content": text
                }
                self.docs_list.append(doc_record)
                
                cursor.execute(
                    "INSERT INTO chunks (chunk_id, filename, header, content) VALUES (?, ?, ?, ?)",
                    (chunk_id, filename, "Section", text)
                )
            
            # Mark file processed
            cursor.execute("INSERT INTO processed_files (filename) VALUES (?)", (filename,))
            total_new_chunks += len(split_texts)
            logger.info(f" -> Added {len(split_texts)} shards for {filename}.")
        
        conn.commit()
        conn.close()
        
        # Save indices back to disk
        self.save_indices()
        logger.info(f"Successfully digested {total_new_chunks} total shards into offline FAISS/BM25.")
        return True

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """Hybrid Search: Merges FAISS and BM25, then Re-ranks via CrossEncoder."""
        if self.index.ntotal == 0 or not self.bm25_corpus:
            return []

        # 1. FAISS Search
        query_emb = self.embedding_model.encode([query])
        faiss_top_k = min(top_k * 3, self.index.ntotal)
        _, faiss_indices = self.index.search(np.array(query_emb).astype('float32'), faiss_top_k)
        faiss_results = [self.docs_list[idx] for idx in faiss_indices[0]]

        # 2. BM25 Search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_corpus.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:faiss_top_k]
        bm25_results = [self.docs_list[idx] for idx in bm25_top_indices if bm25_scores[idx] > 0]

        # 3. Deduplicate
        seen_ids = set()
        combined_candidates = []
        for doc in faiss_results + bm25_results:
            if doc['chunk_id'] not in seen_ids:
                seen_ids.add(doc['chunk_id'])
                combined_candidates.append(doc)

        if not combined_candidates:
            return []

        # 4. Re-Ranking (Cross-Encoder)
        cross_inp = [[query, doc['content']] for doc in combined_candidates]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Sort by cross encoder score
        scored_candidates = sorted(zip(cross_scores, combined_candidates), key=lambda x: x[0], reverse=True)
        
        # Return exact Top-K
        final_results = [item[1] for item in scored_candidates[:top_k]]
        return final_results
