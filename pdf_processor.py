import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

class PDFProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        初始化處理器
        chunk_size: 每個片段的最大字符數
        chunk_overlap: 片段間的重疊字符數 (保持上下文連貫)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""] # 優先在段落、換行處切分
        )

    def clean_text(self, text):
        """
        清理文本：移除多餘空白、頁碼常見格式等
        """
        # 1. 將多個換行合併為一個
        text = re.sub(r'\n+', '\n', text)
        # 2. 移除多餘的空格
        text = re.sub(r'\s+', ' ', text)
        # 3. (選用) 移除類似 "Page 1 of 10" 的頁碼資訊
        # text = re.sub(r'Page \d+ of \d+', '', text)
        return text.strip()

    def load_and_chunk(self, pdf_path):
        """
        讀取 PDF 並回傳切分好的文本列表
        """
        print(f"正在處理 PDF: {pdf_path} ...")
        full_text = ""
        
        # 使用 pdfplumber 讀取 (比 pypdf 更擅長處理表格與佈局)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 提取文字
                page_text = page.extract_text()
                if page_text:
                    # 可以在這裡加入邏輯：加上 [Page X] 標籤
                    full_text += f"\n[Page {page.page_number}]\n" + page_text

        # 清理文本
        cleaned_text = self.clean_text(full_text)
        
        # 執行切分 (Chunking)
        chunks = self.text_splitter.split_text(cleaned_text)
        
        print(f"處理完成：原始長度 {len(cleaned_text)} 字，切分為 {len(chunks)} 個片段。")
        return chunks

# 測試用
if __name__ == "__main__":
    # 建立一個測試用的 dummy pdf (實際執行時請換成你的檔案路徑)
    processor = PDFProcessor()
    # chunks = processor.load_and_chunk("your_lecture_note.pdf")
    # print(chunks[0])