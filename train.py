# PEFT/LoRA fine-tuning main script
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from config import Config
from data_loader import format_instruction, load_daily_dialog
import wandb

def train():
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
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
        r=32,
        lora_alpha=64,
        lora_dropout=0.3,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Datatsets
    print("Load Daily Dialog dataset for training...")
    dataset = load_daily_dialog(split="train")
    dataset = dataset.map(format_instruction)
    val_dataset = load_daily_dialog(split="validation")
    val_dataset = val_dataset.map(format_instruction)

    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        max_grad_norm=Config.MAX_GRAD_NORM,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        greater_is_better=False,
        gradient_checkpointing=True,
        report_to="wandb",
        save_total_limit=1,
        group_by_length=False,
        max_length=Config.MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )


    print("Start training")
    trainer.train()
    
    print(f"Training completed, model saved at {Config.OUTPUT_DIR}")
    trainer.save_model(Config.OUTPUT_DIR)

if __name__ == "__main__":

    wandb.init(
        project="lora-dialogue-finetune",
        name=f"LLaMA-3.2-1B-Instruct-DailyDialog-LoRA",
        config={
            "model": Config.MODEL_NAME,
            "lora_rank": Config.LORA_RANK,
            "lora_alpha": Config.LORA_ALPHA,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "max_grad_norm": Config.MAX_GRAD_NORM,
            "epochs": Config.NUM_EPOCHS,
        }
    )

    train()