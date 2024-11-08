from pathlib import Path
import numpy as np
import torch.nn as nn
import torch
from dataclasses import dataclass
from pathlib import Path
import os
import miditok
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DataCollator, DatasetMIDI
from torch.utils.data import DataLoader
from miditok import REMI, TokenizerConfig
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm


def train(model, optim, dataloader, ctriterion, epochs):
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        epoch_loss = 0

        for data in progress_bar:
            optim.zero_grad()

            input_ids = data["input_ids"]
            mask = data["attention_mask"]
            out = model(input_ids, attention_mask=mask)

            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = ctriterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss / len(dataloader):.4f}")


def sample(model, config, tokenizer, save_path):
    prompt = torch.tensor([[
        tokenizer.vocab['Bar_None'],
    ]])
    tokens = model.generate(prompt, max_length=config.max_seq_length)
    tokenizer.decode(tokens)
    score = tokenizer.decode(tokens)
    score.dump_midi(save_path)


def split_training_set():
    miditok.utils.split_files_for_training(
        list(Path("/Users/stevenkao/workspace/music-hw-3/pop1k7/midi_analyzed").glob("**/*.mid")),
        tokenizer,
        save_dir=Path("split"),
        max_seq_len=config.max_seq_length,
    )


@dataclass
class ModelConfig:
    vocab_size: int = -1
    d_model: int = 128
    n_head: int = 2
    n_layers: int = 2
    batch_size: int = 32
    max_seq_length: int = 1024
    # midi_dir = Path("pop1k7/midi_analyzed")
    train_midi_dir = Path("split")


config = ModelConfig()
tkn_config = TokenizerConfig(
    use_tempos=True,
    use_pitchdrum_tokens=False,
    beat_res={(0, 4): 16, (4, 12): 8},
)
tokenizer = REMI(tkn_config)
config.vocab_size = tokenizer.vocab_size
gpt_config = GPT2Config(
    vocab_size=config.vocab_size,
    n_positions=config.max_seq_length,
    n_embd=config.d_model,
    n_layer=config.n_layers,
    n_head=config.n_head,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
model = GPT2LMHeadModel(gpt_config)

# dataset setup
dataset = DatasetMIDI(
    files_paths=list(config.train_midi_dir.glob("**/*.mid")),
    tokenizer=tokenizer,
    max_seq_len=config.max_seq_length,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)

collator = DataCollator(tokenizer.pad_token_id)
dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collator)

# training setup
optim = torch.optim.Adam(model.parameters())
ctriterion = nn.CrossEntropyLoss(ignore_index=tokenizer["PAD_None"])

# train(model, optim=optim, dataloader=dataloader, ctriterion=ctriterion, epochs=1)