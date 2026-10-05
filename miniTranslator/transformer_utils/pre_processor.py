import pandas as pd
from textblob import Word
import nltk
import re
import contractions
import json

nltk.download('wordnet')

def remove_html_tags(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    return text

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

def get_english_short_texts_map():
    with open("data/eng-shorts.json", "r") as f:
        data = json.load(f)
    return data

def short_form_conversion(text):
    short_words = get_english_short_texts_map()
    new_text=[]
    for w in text.split():
        if w.upper() in short_words:
            new_text.append(short_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def preprocess_text(text, language='english'):
    if not isinstance(text, str):
        return text
    if language == 'english':
        pattern = re.compile(r'[^a-zA-Z0-9\s]')
        return pattern.sub(r'', text)
    elif language == 'hindi':
        pattern = re.compile(r'[^\u0900-\u097F\s]')
        return pattern.sub(r'', text)
    else:
        raise ValueError("Unsupported Language, Supported languages are 'english' and 'hindi'")

def pre_process_df(df):
    df = df.drop_duplicates()
    df = df.dropna()

    df['English'] = df['English'].str.lower()
    df['English'] = df['English'].apply(remove_html_tags)
    df['English'] = df['English'].apply(remove_url)
    df['English'] = df['English'].apply(short_form_conversion)
    df['English'] = df['English'].apply(remove_emoji)
    df['English'] = df['English'].apply(expand_contractions)
    df['English'] = df['English'].apply(lambda x: preprocess_text(x, language='english'))
    
    df['Hindi'] = df['Hindi'].apply(remove_emoji)
    df['Hindi'] = df['Hindi'].apply(remove_url)
    df['Hindi'] = df['Hindi'].apply(lambda x: preprocess_text(x, language='hindi'))

    return df

df = pd.read_csv("data/Dataset_English_Hindi.csv")
df = pre_process_df(df)
df.to_csv("data/Dataset_English_Hindi_Clean.csv",index=False)

print("Processed data and saved to data/Dataset_English_Hindi_Clean.csv")