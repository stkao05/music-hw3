from pathlib import Path
import numpy as np
import torch.nn as nn
import torch
from dataclasses import dataclass
from pathlib import Path

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

            tokens = data["input_ids"]
            mask = data["attention_mask"]
            out = model(tokens, attention_mask=mask)
            logits = out.logits

            n_batch, n_seq, vocab_size = logits.shape
            logits = logits.view(n_batch * n_seq, vocab_size)
            tokens = tokens.flatten()

            loss = ctriterion(logits, tokens)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":

    @dataclass
    class ModelConfig:
        vocab_size: int = -1
        d_model: int = 128
        n_head: int = 2
        num_encoder_layers: int = 2
        dim_feedforward: int = 128
        batch_size: int = 32
        max_seq_length: int = 512

    config = ModelConfig()
    tokenizer = REMI(TokenizerConfig())
    config.vocab_size = tokenizer.vocab_size
    bos_token = tokenizer["BOS_None"]
    eos_token = tokenizer["EOS_None"]
    pad_token = tokenizer["PAD_None"]

    gpt_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_seq_length,
        n_embd=config.d_model,
        n_layer=config.num_encoder_layers,
        n_head=config.n_head,
        bos_token_id=bos_token,
        eos_token_id=eos_token,
    )
    model = GPT2LMHeadModel(gpt_config)

    # init dataset
    midi_dir = Path("pop1k7/midi_analyzed")
    dataset = DatasetMIDI(
        files_paths=list(midi_dir.glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_length,
        bos_token_id=bos_token,
        eos_token_id=eos_token,
    )
    collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collator)

    # init model
    optim = torch.optim.Adam(model.parameters())
    ctriterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    train(model, optim=optim, dataloader=dataloader, ctriterion=ctriterion, epochs=1)
