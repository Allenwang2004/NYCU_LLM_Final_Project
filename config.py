import torch

class Config:
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "faiss_index.bin"
    TOP_K_RETRIEVAL = 2 # 每次檢索幾份文件
    
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