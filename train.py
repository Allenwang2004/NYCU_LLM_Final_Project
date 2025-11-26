# PEFT/LoRA fine-tuning main script
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from config import Config
from data_loader import format_instruction, load_daily_dialog

def train():
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # Llama 需要設定 pad_token
    
    # 2. Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
    
    # 3. LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Datatsets
    print("Load Daily Dialog dataset for training...")
    dataset = load_daily_dialog(split="train", limit=1000)
    dataset = dataset.map(format_instruction)

    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
        max_length=Config.MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Start training")
    trainer.train()
    
    print(f"Training completed, model saved at {Config.OUTPUT_DIR}")
    trainer.save_model(Config.OUTPUT_DIR)

if __name__ == "__main__":
    train()