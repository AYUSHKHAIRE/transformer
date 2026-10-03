from .preprocess_text import get_dataset_text

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, decoders, trainers

import os


tokenizer = Tokenizer(
    BPE(
        unk_token="[unk]"
    )
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True
)

tokenizer.decoder = decoders.ByteLevel()


VOCAB_SIZE = 10_000

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "[pad]",
        "[unk]",
        "[bos]",
        "[eos]"
    ],
    show_progress=True
)


text = get_dataset_text(
    raw_data_root = "data/raw/"
)

tokenizer.train_from_iterator(
    text,
    trainer=trainer
)


tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[pad]"),
    pad_token="[pad]"
)


os.makedirs(
    "artifacts",
    exist_ok=True
)

tokenizer.save(
    "artifacts/gutenberg_tokenizer.json",
    pretty=True
)