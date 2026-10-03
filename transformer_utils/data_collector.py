import os
import requests
from tqdm import tqdm

DATASOURCE = {
    "memoirs_of_grant": "https://www.gutenberg.org/ebooks/4367.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "sleepy_hollow": "https://www.gutenberg.org/ebooks/41.txt.utf-8",
    "origin_of_species": "https://www.gutenberg.org/ebooks/2009.txt.utf-8",
    "makers_of_many_things": "https://www.gutenberg.org/ebooks/28569.txt.utf-8",
    "common_sense": "https://www.gutenberg.org/ebooks/147.txt.utf-8",
    "economic_peace": "https://www.gutenberg.org/ebooks/15776.txt.utf-8",
    "the_great_war_3": "https://www.gutenberg.org/ebooks/29265.txt.utf-8",
    "elements_of_style": "https://www.gutenberg.org/ebooks/37134.txt.utf-8",
    "problem_of_philosophy": "https://www.gutenberg.org/ebooks/5827.txt.utf-8",
    "nights_in_london": "https://www.gutenberg.org/ebooks/23605.txt.utf-8",
}

for filename, url in tqdm(DATASOURCE.items(),desc="Downloading text data "):
    os.makedirs("data",exist_ok=True)
    os.makedirs("data/raw",exist_ok=True)
    if not os.path.exists(f"data/raw/{filename}.txt"):
        response = requests.get(url)
        with open(f"data/raw/{filename}.txt", "wb") as f:
            f.write(response.content)
            
print("Data collection complete .")