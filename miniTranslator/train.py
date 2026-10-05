import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from transformer_utils.build_tokenizer import load_tokenizer
from transformer_utils.transformer import (
    Transformer,
    create_padding_mask,
    create_causal_mask
)

from sklearn.model_selection import train_test_split

import pandas as pd

import random
from tqdm import tqdm
import os

# Tokenizers

en_tokenizer = load_tokenizer("artifacts/tokenizers/English_tokenizer.json")
hn_tokenizer = load_tokenizer("artifacts/tokenizers/Hindi_tokenizer.json")

# Settings

model_config = {
    "num_layers": 4,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 128,
    "max_seq_len": 768,
    "vocab_size_src": len(en_tokenizer.get_vocab()),
    "vocab_size_tgt": len(hn_tokenizer.get_vocab()),
    "dropout": 0.1,
}

N_EPOCHS = 50 # inc later
LR = 3e-4
WARMUP_STEPS = 1000
CLIP_NORM = 5.0
N_SAMPLES = 5
MAX_LEN = 60
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
BATCH_SIZE = 16 if not ON_KAGGLE else 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model

model = Transformer(**model_config).to(device)

# Dataset

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text_pairs = list(
            zip(df["English"], df["Hindi"])
        )

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, hin = self.text_pairs[idx]
        hin = f"[start] {hin} [end]"
        return eng, hin
    
def collate_fn(batch):
    en_str, hn_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(
        en_str, add_special_tokens=True
    )
    hn_enc = hn_tokenizer.encode_batch(
        hn_str, add_special_tokens=True
    )
    en_ids = [enc.ids for enc in en_enc]
    hn_ids = [enc.ids for enc in hn_enc]
    return torch.tensor(en_ids), torch.tensor(hn_ids)

df = pd.read_csv("data/Dataset_English_Hindi_Clean.csv")

df = df.dropna(subset=["English", "Hindi"]).copy()
df["English"] = df["English"].astype(str).str.strip()
df["Hindi"] = df["Hindi"].astype(str).str.strip()
df = df[
    (df["English"] != "") &
    (df["Hindi"] != "")
]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_dataset = TranslationDataset(train_df)
val_dataset = TranslationDataset(val_df)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# Training

loss_fn = nn.CrossEntropyLoss(
    ignore_index=hn_tokenizer.token_to_id("[pad]")
)

optimizer = optim.Adam(
    model.parameters(), 
    lr=LR
)
warmup_scheduler = optim.\
    lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=WARMUP_STEPS
    )
cosine_scheduler = optim.\
        lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=N_EPOCHS * len(train_dataloader) - WARMUP_STEPS, 
        eta_min=0
    )
scheduler = optim.\
    lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[
            warmup_scheduler, 
            cosine_scheduler
        ], 
        milestones=[
            WARMUP_STEPS
        ]
    )

best_loss = float("inf")
for epoch in range(N_EPOCHS):
    model.train()
    train_epoch_loss = 0.0
    for en_ids, hn_ids in tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{N_EPOCHS} Training"
    ):
        en_ids = en_ids.to(device)
        hn_ids = hn_ids.to(device)
        # Source padding mask
        src_mask = create_padding_mask(
            en_ids,
            en_tokenizer.token_to_id("[pad]")
        )
        # Target causal + padding mask
        tgt_mask = create_causal_mask(
            hn_ids.shape[1],
            device
        ).unsqueeze(0)
        tgt_mask = tgt_mask + create_padding_mask(
            hn_ids,
            hn_tokenizer.token_to_id("[pad]")
        )
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            en_ids,
            hn_ids,
            src_mask,
            tgt_mask
        )
        loss = loss_fn(
            outputs[:, :-1, :].reshape(
                -1,
                outputs.shape[-1]
            ),
            hn_ids[:, 1:].reshape(-1)
        )
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CLIP_NORM,
            error_if_nonfinite=False
        )
        optimizer.step()
        scheduler.step()
        train_epoch_loss += loss.item()

    train_loss = train_epoch_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_epoch_loss = 0.0

    with torch.no_grad():
        for en_ids, hn_ids in tqdm(
            val_dataloader,
            desc=f"Epoch {epoch + 1}/{N_EPOCHS} Validation"
        ):
            en_ids = en_ids.to(device)
            hn_ids = hn_ids.to(device)
            src_mask = create_padding_mask(
                en_ids,
                en_tokenizer.token_to_id("[pad]")
            )
            tgt_mask = create_causal_mask(
                hn_ids.shape[1],
                device
            ).unsqueeze(0)
            tgt_mask = tgt_mask + create_padding_mask(
                hn_ids,
                hn_tokenizer.token_to_id("[pad]")
            )
            outputs = model(
                en_ids,
                hn_ids,
                src_mask,
                tgt_mask
            )
            loss = loss_fn(
                outputs[:, :-1, :].reshape(
                    -1,
                    outputs.shape[-1]
                ),
                hn_ids[:, 1:].reshape(-1)
            )
            val_epoch_loss += loss.item()

    eval_loss = val_epoch_loss / len(val_dataloader)
    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Eval Loss: {eval_loss:.4f}"
    )
    if eval_loss < best_loss:
        best_loss = eval_loss
        os.makedirs("artifacts/model", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"artifacts/model/transformer-epoch-{epoch + 1}.pth"
        )
        print(
            f"Saved best model "
            f"(eval loss: {eval_loss:.4f})"
        )
        
# Test a few samples

def get_hindi_translation(
    model,
    input_text,
    en_tokenizer,
    hn_tokenizer,
    device,
    max_len=60
):
    model.eval()
    start_id = hn_tokenizer.token_to_id("[start]")
    end_id = hn_tokenizer.token_to_id("[end]")
    pad_id = hn_tokenizer.token_to_id("[pad]")

    with torch.no_grad():
        en_ids = torch.tensor(
            en_tokenizer.encode(input_text).ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        src_mask = create_padding_mask(
            en_ids,
            en_tokenizer.token_to_id("[pad]")
        )
        x = model.src_embedding(en_ids)
        for encoder in model.encoders:
            x = encoder(
                x,
                src_mask,
                model.rope
            )
        enc_out = x
        hn_ids = torch.tensor(
            [[start_id]],
            dtype=torch.long,
            device=device
        )
        for _ in range(max_len):
            tgt_mask = create_causal_mask(
                hn_ids.shape[1],
                device
            ).unsqueeze(0)
            tgt_mask = tgt_mask + create_padding_mask(
                hn_ids,
                pad_id
            )
            x = model.tgt_embedding(hn_ids)
            for decoder in model.decoders:
                x = decoder(
                    x,
                    enc_out,
                    tgt_mask,
                    model.rope
                )
            logits = model.out(x)
            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True
            )
            hn_ids = torch.cat(
                [hn_ids, next_token],
                dim=1
            )
            if next_token.item() == end_id:
                break
        # Decode Hindi
        translation = hn_tokenizer.decode(
            hn_ids[0].tolist(),
            skip_special_tokens=True
        )
    return translation

for en, true_hn in random.sample(
    val_dataset.text_pairs,
    min(N_SAMPLES, len(val_dataset.text_pairs))
):
    predicted_hn = get_hindi_translation(
        model,
        en,
        en_tokenizer,
        hn_tokenizer,
        device
    )
    print(f"English:   {en}")
    print(f"Actual:    {true_hn}")
    print(f"Predicted: {predicted_hn}")
    print()