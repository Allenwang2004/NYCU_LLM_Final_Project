import torch

class Config:
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    VECTOR_DB_PATH = "faiss_index.bin"
    TOP_K = 20 # 每次檢索幾份文件
    TOP_N = 3
    ALPHA = 0.75

    #RAG config
    PDF_DIR = [
    "./documents/Computer/",
    "./documents/Physics/",
    "./documents/Probability/"
]
    TXT_DIR = "./text"
    CHUNK_DIR = "./chunks"
    EMBED_DIR = "./embeddings"
    FAISS_INDEX = "./faiss_index.index"
    
    # Fine-tuning config
    OUTPUT_DIR = "./llama-dialogue-finetuned"
    NUM_EPOCHS = 3
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 512
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Prompt Template
    # Topic -> Context (RAG) -> Generation
    PROMPT_TEMPLATE = """You are a helpful assistant capable of generating coherent dialogues based on a topic and context.

### Topic:
{topic}

### Context (Reference):
{context}

### Dialogue:
"""
