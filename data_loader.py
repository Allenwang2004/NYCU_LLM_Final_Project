import random
from datasets import Dataset, load_dataset

def load_daily_dialog(split="train", limit=None):
    dataset = load_dataset("roskoN/dailydialog", split=split)

    if limit:
        print(f"取前 {limit} 筆資料")
        dataset = dataset.select(range(limit))

    def transform_data(example):
        # 處理對話：原始資料是 list
        # 為了區分說話者，我們可以簡單加上 A: B: 或者是直接合併
        # 這裡示範簡單合併，中間用空白隔開
        full_dialogue = " ".join(example['utterances'])
        
        topic = "Daily Life Conversation" 
        
        # 處理 Context (RAG part)
        # 由於這是純對話資料集，沒有附帶「知識文件」，
        # 為了不讓程式報錯，我們先填入通用 Context，讓模型學會格式。
        context = "Ordinary conversation about daily life events."
        
        return {
            "topic": topic,
            "context": context,
            "dialogue": full_dialogue
        }

    # 使用 map 進行轉換，remove_columns 把舊的欄位 (dialog, act, emotion) 丟掉
    processed_dataset = dataset.map(
        transform_data, 
        remove_columns=dataset.column_names
    )
    
    return processed_dataset

def format_instruction(sample):
    """
    將資料轉換為模型輸入格式
    Input: Topic + Context
    Target: Dialogue
    """
    from config import Config
    
    prompt = Config.PROMPT_TEMPLATE.format(
        topic=sample['topic'],
        context=sample['context']
    )
    
    return {"text": prompt + sample['dialogue']}