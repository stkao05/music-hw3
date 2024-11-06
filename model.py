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



class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, **kwarg):
        return memory


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, d_model):
        super(PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
    def forward(self, x):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0)
        positional_embeddings = self.position_embedding(position_ids)

        return positional_embeddings

class PopTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
    ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            custom_decoder=DummyDecoder(),
        )

        self.linear = nn.Linear(d_model, vocab_size)
        nn.init.zeros_(self.linear.bias)

        self.position_embedding = PositionalEmbedding(max_seq_length, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, tokens, padding_mask):
        n_batch, n_seq = tokens.shape
        casual_mask = self.transformer.generate_square_subsequent_mask(n_seq)

        # n_batch, n_seq, n_embed
        y = self.token_embedding(tokens) + self.position_embedding(tokens)
        y = y.permute(1, 0, 2)  # n_seq, n_batch, n_embed
        y = self.transformer(
            src=y,
            tgt=y,
            src_mask=casual_mask,
            src_key_padding_mask=padding_mask.float(),
            src_is_causal=True,
            memory_is_causal=True,
        )
        y = y.permute(1, 0, 2)  # n_batch, n_seq, d_model
        y = self.linear(y)

        return y


def sample():
    sample_size = 1
    bos_token = tokenizer['BOS_None']
    eos_token = tokenizer['EOS_None']
    pad_token = tokenizer['PAD_None']
    tokens = torch.tensor([bos_token]).reshape(sample_size, 1) # (batch_n, seq_len)

    for _ in range(100):
        padding_mask = torch.tensor([pad_token] * tokens.shape[1]).reshape(sample_size, -1)
        logits = model(tokens, padding_mask) # (batch_n, seq_len, vocab_size)
        next_token = logits.argmax(dim=2) # (batch_n, seq_len)
        next_token = next_token[:,-1].view(sample_size, 1)
        tokens = torch.cat((tokens, next_token), dim=1)

        if next_token[0, -1] == eos_token:
            break

    return tokens


def train(model):
    optim = torch.optim.Adam(model.parameters())
    ctriterion = nn.CrossEntropyLoss(ignore_index=tokenizer["PAD_None"])
    data = next(iter(dataloader))

    for _ in range(5):
        model.train()
        optim.zero_grad()

        tokens = data["input_ids"]
        mask = data["attention_mask"]

        logits = model(tokens, mask)
        n_batch, n_seq, vocab_size = logits.shape
        logits = logits.view(n_batch * n_seq, vocab_size)
        tokens = tokens.flatten()
        loss = ctriterion(logits, tokens)

        loss.backward()
        optim.step()

        print(loss.item())


# def play():
#     midi_dir = Path("pop1k7/midi_analyzed")

#     # midi_path = "pop1k7/midi_transcribed/src_001/3.midi"
#     midi_path = "tutorial/000.mid"
#     MIDIPlayer(midi_path, 400)  


if __name__ == "__main__":

    config = ModelConfig()
    tokenizer = REMI(TokenizerConfig())
    config.vocab_size = tokenizer.vocab_size

    midi_dir = Path("pop1k7/midi_analyzed")
    dataset = DatasetMIDI(
        files_paths=list(midi_dir.glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_length,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collator)


    # model = PopTransformer(
    #     vocab_size=config.vocab_size,
    #     d_model=config.d_model,
    #     nhead=config.n_head,
    #     num_encoder_layers=config.num_encoder_layers,
    #     dim_feedforward=config.dim_feedforward,
    #     max_seq_length=config.max_seq_length
    # )


    data = next(iter(dataloader))
    tokens = data['input_ids'].tolist()
    score = tokenizer.decode(tokens)
    score


