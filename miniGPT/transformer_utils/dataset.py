import torch
from tokenizers import Tokenizer

class GutenbergDataset(
    torch.utils.data.Dataset
):
    def __init__(
        self, 
        text, 
        tokenizer, 
        seq_len=512
    ):
        self.seq_len = seq_len
        # Encode the entire text
        self.encoded = tokenizer.encode(text).ids
 
    def __len__(self):
        return len(self.encoded) - self.seq_len
 
    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]  # +1 for target
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y
    
def load_tokenizer(path):
    return Tokenizer.from_file(
        path
    )