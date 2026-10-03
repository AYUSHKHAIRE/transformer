import os
from tqdm import tqdm

def preprocess_gutenberg(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    text = text[start:end].strip()
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text

def get_dataset_text(raw_data_root):
    all_text = []
    all_raw_files = os.listdir(raw_data_root)
    all_file_paths = [ f"{raw_data_root}{f}" for f in all_raw_files ]
    for fp in tqdm(all_file_paths,desc="Reading the files "):
        text = preprocess_gutenberg(fp)
        all_text.append(text)
    return all_text