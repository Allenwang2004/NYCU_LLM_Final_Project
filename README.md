# NYCU_LLM_Final_Project



## 1. Environment

Python 3.11

```bash
pip install torch transformers peft datasets==3.6.0 trl bitsandbytes sentence-transformers faiss-cpu scikit-learn accelerate sacrebleu
```
For conda:
```bash
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu
```


## 2\. LoRA Fine-tuning

```bash
python train.py
```

-----

## 3\. RAG + Inference

Pipeline：Topic -> RAG Retrieval -> Fine-tuned LLM Generation -> Output

```bash
python main.py
```
