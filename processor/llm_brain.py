import os
import json
import logging
from pathlib import Path
from llama_cpp import Llama
import config

logger = logging.getLogger(__name__)

class LLMBrain:
    def __init__(self, model_path: str = "Phi-3-mini-4k-instruct-q4.gguf"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            logger.error(f"Model file not found at {self.model_path}. Please ensure the .gguf file is present.")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        logger.info(f"Loading LLM Brain from {self.model_path} (CPU optimized)")
        
        # CPU-optimized loading for Phi-3 (using threads defined in config)
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=4096,           # 4k context window as per model name
            n_threads=config.DOCLING_CPU_THREADS, 
            n_gpu_layers=0        # Force CPU
        )
        logger.info("LLM Brain loaded successfully.")

    def run_inference_on_document(self, structured_json_path: Path, prompt_instruction: str):
        """
        Reads the structured Docling output and runs inference.
        """
        if not structured_json_path.exists():
            logger.error(f"Structured document not found at {structured_json_path}")
            return None

        with open(structured_json_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        # Naive extraction of text from Docling JSON layout
        # (This can be modified to construct a highly optimized prompt based on semantic blocks)
        texts = []
        if "texts" in doc_data:
            for item in doc_data["texts"]:
                texts.append(item.get("text", ""))
        
        document_text = "\n".join(texts)[:3000] # truncate to fit context window safely

        prompt = f"""<|system|>
You are an intelligent knowledge extraction assistant. Answer the user based ONLY on the provided document context.
<|user|>
Context:
{document_text}

Instruction: {prompt_instruction}
<|assistant|>"""

        logger.info("Generating response...")
        output = self.llm(
            prompt,
            max_tokens=512,
            stop=["<|user|>", "<|system|>", "<|end|>"],
            echo=False
        )
        
        result = output["choices"][0]["text"].strip()
        logger.info("Response generated.")
        return result

    def run_rag_inference(self, prompt_instruction: str, contexts: list):
        """
        Runs inference using strictly provided RAG contexts.
        """
        if not contexts:
            document_text = "No relevant documents found."
        else:
            # Join the content of the retrieved chunks
            texts = [f"--- Document Content ---\n{doc['content']}" for doc in contexts]
            document_text = "\n\n".join(texts)[:3000] # Truncate to fit safely
            
        prompt = f"""<|system|>
You are a corporate Enterprise assistant representing Uni-Doc-Intel. Answer the user based ONLY on the provided context retrieved from company files. If the context does not contain the answer, say "I do not have enough information."
<|user|>
Retrieved Context:
{document_text}

Query: {prompt_instruction}
<|assistant|>"""

        logger.info("Generating RAG inference response...")
        output = self.llm(
            prompt,
            max_tokens=512,
            stop=["<|user|>", "<|system|>", "<|end|>"],
            echo=False
        )
        
        result = output["choices"][0]["text"].strip()
        logger.info("Response generated.")
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage (Requires the model to be downloaded to project root)
    try:
        brain = LLMBrain()
        # Find the first JSON in structured outputs
        json_files = list(config.STRUCTURED_OUTPUT_DIR.glob("*.json"))
        if json_files:
            sample_file = json_files[0]
            print(f"\nRunning test query on {sample_file.name}...\n")
            response = brain.run_inference_on_document(
                structured_json_path=sample_file,
                prompt_instruction="Summarize the primary purpose or subject of this document in 3 sentences."
            )
            print("\n=== LLM OUTPUT ===")
            print(response)
            print("==================\n")
        else:
            print("No structured JSON documents found to process. Please run main.py first.")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
