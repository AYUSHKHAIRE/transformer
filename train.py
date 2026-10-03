from transformer_utils.preprocess_text import get_dataset_text
from transformer_utils.dataset import GutenbergDataset , load_tokenizer
from transformer_utils.tansformer import TextGenerationModel , create_causal_mask

import torch
import torch.nn as nn
import torch.optim as optim

import tqdm

BATCH_SIZE = 32

text = "\n".join(get_dataset_text("data/raw/"))
tokenizer = load_tokenizer("artifacts/gutenberg_tokenizer.json")

model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 768,
    "moe_experts": 8,
    "moe_topk": 3,
    "max_seq_len": 512,
    "vocab_size": len(tokenizer.get_vocab()),
    "dropout": 0.1,
}

dataset = GutenbergDataset(
    text, 
    tokenizer, 
    seq_len=model_config["max_seq_len"]
)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

model = TextGenerationModel(**model_config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).to(torch.bfloat16)

N_EPOCHS = 2
LR = 0.0005
WARMUP_STEPS = 2000
CLIP_NORM = 6.0

optimizer = optim.AdamW(
    model.parameters(), lr=LR , foreach=False
)
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.token_to_id("[pad]")
)

# Learning rate scheduling
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=0.01, 
    end_factor=1.0, 
    total_iters=WARMUP_STEPS
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, 
    eta_min=0
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, 
    schedulers=[
        warmup_scheduler, 
        cosine_scheduler
    ],
    milestones=[WARMUP_STEPS]
)

num_params = sum(
    p.numel()
    for p in model.parameters()
)
print(
    f"Parameters: {num_params:,} "f"({num_params / 1e6:.2f}M)"
)

print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")
best_loss = float('inf')

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm.tqdm(
        dataloader, 
        desc=f"Epoch {epoch+1}/{N_EPOCHS}"
    )
    for x, y in progress_bar:
        x = x.to(device)
        y = y.to(device)

        # Create causal mask
        mask = create_causal_mask(
            x.shape[1], 
            device, 
            torch.bfloat16
        )

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x, mask.unsqueeze(0))

        # Compute loss
        loss = loss_fn(
            outputs.view(
                -1, 
                outputs.shape[-1]
            ), y.view(-1)
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            CLIP_NORM, 
            error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

        # Show loss in tqdm
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}")

    # Save checkpoint if loss improved
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "textgen_model.pth")