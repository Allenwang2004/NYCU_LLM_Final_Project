import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

class RAGSystem:
    def __init__(self):
        # 初始化 Embedding 模型
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.documents = [] # 儲存原始文本

    def create_index(self, documents):
        """
        建立 FAISS 索引庫
        documents: List[str] - 知識庫文件列表
        """
        print("正在為 RAG 建立索引...")
        self.documents = documents
        embeddings = self.encoder.encode(documents, convert_to_numpy=True)
        
        # 建立 L2 距離索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"索引建立完成，共 {len(documents)} 筆資料。")

    def retrieve(self, query, k=Config.TOP_K_RETRIEVAL):
        """
        根據 Query 檢索最相關的 k 筆文件 [cite: 44]
        """
        if self.index is None:
            return ["No index found."]
            
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results

# 測試用
if __name__ == "__main__":
    rag = RAGSystem()
    docs = ["Apples are red.", "Bananas are yellow.", "The sky is blue."]
    rag.create_index(docs)
    print(rag.retrieve("What color is a banana?"))