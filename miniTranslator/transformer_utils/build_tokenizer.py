import tokenizers
import os
import pandas as pd

os.makedirs("artifacts/tokenizers/",exist_ok=True)

def build_tokenizer(lang,lang_df):
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE()
    )
    tokenizer.pre_tokenizer = tokenizers.\
        pre_tokenizers.\
            ByteLevel(
        add_prefix_space=True
    )
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    VOCAB_SIZE = 8000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[start]", "[end]", "[pad]"],
        show_progress=True
    )
    texts = lang_df[l].dropna().astype(str)
    tokenizer.train_from_iterator(
        [x for x in texts], trainer=trainer
    )
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[pad]"), 
        pad_token="[pad]"
    )
    tokenizer.save(
        f"artifacts/tokenizers/{lang}_tokenizer.json", 
        pretty=True
    )
    print(f"Saved tokenizer as artifacts/tokenizers/{lang}_tokenizer.json .")
    
def load_tokenizer(path):
    return tokenizers.Tokenizer.from_file(
        path
    )

if __name__ == "__main__":
    lang_df = pd.read_csv("data/Dataset_English_Hindi_Clean.csv")
    languages = lang_df.columns.tolist()

    for l in languages:
        print(f"Building tokenizer for {l}")
        build_tokenizer(l,lang_df)    