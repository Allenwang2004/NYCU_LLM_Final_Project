import fitz
import re
import os

pdf_dirs = [
    "./documents/Computer/",
    "./documents/Physics/",
    "./documents/Probability/"
]

output_dir = "./text"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[👉●•👻😎]', '', text)
    text = re.sub(r'\b\d{4}/\d{1,2}/\d{1,2}\b', '', text)
    text = re.sub(r'\s*,?\s*(Chien-Nan Liu|NCTUEE|Chien-Nan Liu, NCTUEE)', '', text)
    text = re.sub(r'\b\d+-\d+\b', '', text)
    
    return text.strip()

def extract_text(pdf_path, skip_first_page=False):
    doc = fitz.open(pdf_path)
    full_text = ""
    for i, page in enumerate(doc):
        if skip_first_page and i == 0:
            continue  
        page_text = page.get_text()
        if page_text:
            page_text = clean_text(page_text)
            full_text += f"\n[Page {i + 1}]\n" + page_text
    return full_text

def save_text(text, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved cleaned text: {out_path}")

if __name__ == "__main__":
    root_out_dir = "./text"

    for pdf_dir in pdf_dirs:
        category = os.path.basename(os.path.normpath(pdf_dir))  
        out_dir = os.path.join(root_out_dir, category)
        skip_first = category in ["Computer", "Physics"]  

        for file in os.listdir(pdf_dir):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, file)
                text = extract_text(pdf_path, skip_first_page=skip_first)
                filename = os.path.splitext(file)[0] + ".txt"
                save_text(text, out_dir, filename)