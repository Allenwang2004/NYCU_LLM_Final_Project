# main.py
# Pipeline：Topic -> RAG Retrieval -> Fine-tuned LLM Generation -> Output

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag import RAGSystem
from data_loader import generate_mock_data
from config import Config
import sacrebleu
from pdf_processor import PDFProcessor

def load_finetuned_model():
    print("Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load fine-tuned Adapter
    print(f"Loading LoRA Adapter from {Config.OUTPUT_DIR} ...")
    try:
        model = PeftModel.from_pretrained(base_model, Config.OUTPUT_DIR)
    except Exception as e:
        print(f"Unable to load Adapter: {e}")
        print("Will use Base Model for demonstration.")
        model = base_model
        
    return model, tokenizer

def generate_dialogue(model, tokenizer, rag, topic):
    """
    Topic -> RAG -> Prompt -> Generation
    """
    # 1. RAG
    retrieved_docs = rag.retrieve(topic)
    context_str = "\n".join(retrieved_docs)
    
    # 2. 組合 Prompt
    prompt = Config.PROMPT_TEMPLATE.format(
        topic=topic,
        context=context_str
    )
    
    # 3. Model Generation
    inputs = tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 僅提取 Dialogue (去除 Prompt)
    generated_dialogue = full_output.split("### Dialogue:")[-1].strip()
    
    return generated_dialogue, context_str

def evaluate_bleu(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score

def main():
    pdf_path = "Filename.pdf"
    
    import os
    use_real_pdf = os.path.exists(pdf_path)
    
    rag = RAGSystem()
    
    if use_real_pdf:
        print(f"{pdf_path} file pre-processing")
        processor = PDFProcessor(chunk_size=400, chunk_overlap=50)
        knowledge_base = processor.load_and_chunk(pdf_path)
    else:
        print("No PDF file found")
        return

    
    rag.create_index(knowledge_base)

    model, tokenizer = load_finetuned_model()
    model.eval()
    
    print("-" * 30)
    print("System ready. Type 'exit' to quit.")
    print("Enter a topic (e.g., 'Computer Science', 'Physics', 'Probability'):")
    
    while True:
        user_topic = input("\nEnter a topic: ")
        if user_topic.lower() == 'exit':
            break
            
        print(f"Retrieving data and generating dialogue for topic '{user_topic}'...")
        
        dialogue, context = generate_dialogue(model, tokenizer, rag, user_topic)
        
        print("\n" + "="*20 + " RAG Results " + "="*20)
        print(f"Retrieved Context:\n{context}")
        print("-" * 50)
        print(f"Generated Dialogue:\n{dialogue}")
        print("="*50)

    # --- Simple automatic evaluation demo (Optional) ---
    print("\nRunning automatic evaluation (Demo BLEU Score)...")
    test_topic = "Computer Science"
    pred, _ = generate_dialogue(model, tokenizer, rag, test_topic)
    ref = "A: What is computer program B: A set of instructions for a computer to follow" 

    score = evaluate_bleu([pred], [ref])
    print(f"Topic: {test_topic}")
    print(f"Prediction: {pred}")
    print(f"Reference: {ref}")
    print(f"BLEU Score: {score:.2f}")

if __name__ == "__main__":
    main()