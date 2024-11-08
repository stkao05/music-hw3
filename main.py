import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import miditok
import numpy as np
import torch
import torch.nn as nn
import wandb
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DataCollator, DatasetMIDI
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass
class ModelConfig:
    vocab_size: int = -1
    d_model: int = 128
    n_head: int = 2
    n_layers: int = 2
    batch_size: int = 32
    max_seq_length: int = 1024
    max_sample_length: int = 3072
    split_midi_dir = Path("split")
    sample_dir = Path("samples")
    checkpoint_dir = Path("checkpoints")
    epochs: int = 100
    device: str = ""
    debug: bool = False


def train(model, optim, dataloader, ctriterion, config:ModelConfig):
    for epoch in range(config.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=True)
        epoch_loss = 0

        i = 0
        for data in progress_bar:
            model.train()
            optim.zero_grad()

            input_ids = data["input_ids"].to(config.device)
            mask = data["attention_mask"].to(config.device)
            out = model(input_ids, attention_mask=mask)

            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = ctriterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            i += 1
            if i == 1 and config.debug:
                break

        avg_epoch_loss = epoch_loss / len(dataloader)
        wandb.log({"loss": avg_epoch_loss})
        checkpoint_save(model, optim, epoch, avg_epoch_loss, config)
        print(f"Epoch {epoch + 1} completed with average loss: {avg_epoch_loss:.4f}")

        sample(model, config, tokenizer, epoch)


def checkpoint_save(model, optim, epoch, loss, config: ModelConfig):
    checkpoint_path = config.checkpoint_dir / f"cp_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    wandb.save(checkpoint_path)


def checkpoint_load(checkpoint_path, model, optim):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    print("load checkpoint", checkpoint_path)


@torch.no_grad()
def sample(model, config: ModelConfig, tokenizer, epoch):
    model.eval()
    save_path = config.sample_dir / f"sample_{epoch}.mid"
    tokens = torch.tensor(
        [
            [
                tokenizer.vocab["Bar_None"],
            ]
        ],
        device=config.device,
    )

    while tokens.size(1) < config.max_sample_length:
         # generate one token at a time
        input_context = tokens[:, -(config.max_seq_length-1):]
        output = model.generate(
            input_context,
            attention_mask=torch.ones(input_context.shape, device=config.device),
            max_length=input_context.size(1) + 1,
            do_sample=True,
        )
        new_token = output[:, -1:]
        tokens = torch.cat((tokens, new_token), dim=1)
        if new_token.item() == tokenizer["EOS_None"]:
            break

    print("sample token len", tokens.size(1))
    tokens = tokens.cpu()
    tokenizer.decode(tokens)
    score = tokenizer.decode(tokens)
    score.dump_midi(save_path)
    wandb.save(save_path)


def split_training_set(midi_dir, config: ModelConfig, tokenizer):
    miditok.utils.split_files_for_training(
        list(Path(midi_dir).glob("**/*.mid")),
        tokenizer,
        save_dir=config.split_midi_dir,
        max_seq_len=config.max_seq_length,
    )


# python main.py --device=cuda:7 --d_model=128 --n_head=2 --n_layers=2 --batch_size=32 --max_seq_length=1024
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model checkpoint.")
    parser.add_argument("--cp", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_sample_length", type=int, default=3072)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--split", type=str, default=False)
    parser.add_argument("--apikey", type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("available gpu num: ", torch.cuda.device_count())
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    tkn_config = TokenizerConfig(
        use_tempos=True,
        use_pitchdrum_tokens=False,
        beat_res={(0, 4): 16, (4, 12): 8},
    )
    tokenizer = REMI(tkn_config)
    config = ModelConfig(
        device=device,
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        debug=args.debug
    )
    print(config)

    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if args.split:
        os.makedirs(config.split_midi_dir, exist_ok=True)
        split_training_set(args.split, config, tokenizer)
        os._exit(1)

    # dataset setup
    dataset = DatasetMIDI(
        files_paths=list(config.split_midi_dir.glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_length,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=collator, shuffle=True
    )

    # training setup
    if args.apikey:
        wandb.login(key=args.apikey)

    wandb.init(
        project="pop-transformer",
        config=config,
        mode=None if args.apikey else "disabled",
    )
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
    model.generation_config.pad_token_id = tokenizer["PAD_None"]
    model.to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.Adam(model.parameters())
    ctriterion = nn.CrossEntropyLoss(ignore_index=tokenizer["PAD_None"])
    if args.cp:
        checkpoint_load(args.cp, model, optim)

    train(
        model,
        optim=optim,
        dataloader=dataloader,
        ctriterion=ctriterion,
        config=config
    )
    wandb.finish()
