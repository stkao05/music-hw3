# %% 
import argparse
import sys

sys.path.append("musdr")

import os
from pathlib import Path

import torch
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel

from eval_metrics import eval_dir
from main import ModelConfig, checkpoint_load


def sample(
    sample_size: int,
    batch_size: int,
    config: ModelConfig,
    out_dir: str,
    model,
    tokenizer: REMI,
    gen_config: GenerationConfig,
    prompt,
):
    sample_dir = Path(out_dir)
    os.makedirs(sample_dir, exist_ok=True)

    iter_num = sample_size // batch_size
    for i in range(iter_num):
        if prompt:
            tokens = torch.tensor(
                [prompt] * batch_size,
                device=config.device,
            )  # (batch_n, seq_n)
        else:
            tokens = torch.tensor(
                [[tokenizer.vocab["Bar_None"]]] * batch_size,
                device=config.device,
            )  # (batch_n, seq_n)

        with tqdm(
            total=config.max_sample_length, desc="Generating tokens", unit="token"
        ) as pbar:
            pbar.update(tokens.size(1))
            while tokens.size(1) < config.max_sample_length:
                # Generate one token at a time
                input_context = tokens[:, -(config.max_seq_length - 1) :]
                output = model.generate(
                    input_context,
                    attention_mask=torch.ones(
                        input_context.shape, device=config.device
                    ),
                    generation_config=gen_config,
                    max_length=input_context.size(1) + 1,
                )
                new_token = output[:, -1:]
                tokens = torch.cat((tokens, new_token), dim=1)
                pbar.update(1)

                # check if all batch has ended
                all_end = (tokens == tokenizer["EOS_None"]).any(dim=1).all()
                if all_end:
                    break

        tokens = tokens.cpu()
        for i in range(tokens.size(0)):
            score = tokenizer.decode(tokens[i : i + 1, :])
            score.dump_midi(sample_dir / f"{iter_num * batch_size + i}.mid")

    print("evaluating", sample_dir)
    eval_result = eval_dir(
        sample_dir, tokenizer, result_path=sample_dir / "eval_result.csv"
    )
    print(eval_result)


def main(**args):
    if torch.cuda.is_available():
        print("available gpu num: ", torch.cuda.device_count())
        device = torch.device(args["device"])
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
        n_embd=512,
        n_head=8,
        n_layers=12,
        batch_size=8,
        max_seq_length=1024,
        max_sample_length=args["sample_len"],
    )

    gpt_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_seq_length,
        n_embd=config.n_embd,
        n_layer=config.n_layers,
        n_head=config.n_head,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    model = GPT2LMHeadModel(gpt_config)
    model.generation_config.pad_token_id = tokenizer["PAD_None"]
    model.to(config.device)
    checkpoint_load(args["cp"], model, config)
    model.eval()
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def run_exp(name: str, gen_config: GenerationConfig):
        print("--------------")
        print(f"config: {name}")
        out_dir = Path(args["out_dir"]) / name
        sample(
            args["sample_size"],
            args["batch_size"],
            config,
            out_dir,
            model,
            tokenizer,
            gen_config,
        )

    # run_exp("greedy", GenerationConfig(num_beams=1, do_sample=False))
    # run_exp("multinomial_sampling", GenerationConfig(num_beams=1, do_sample=True))
    # run_exp(
    #     "multinomial_sampling_temp",
    #     GenerationConfig(num_beams=1, do_sample=True, temperature=1.5),
    # )
    # run_exp(
    #     "beam-search_multinomial_sampling",
    #     GenerationConfig(num_beams=5, do_sample=True),
    # )

    def sample_prompt(promp_name, config_name, gen_config):
        print("promp and config:", promp_name, config_name)
        path = Path("./prompt_song") / (promp_name + ".mid")
        prompt = tokenizer.encode(path)[0]
        out_dir = Path(args["out_dir"]) / promp_name + "_" + config_name
        sample(
            args["sample_size"],
            args["batch_size"],
            config,
            out_dir,
            model,
            tokenizer,
            gen_config,
            prompt
        )

    sample_prompt("song_1", "greedy", GenerationConfig(num_beams=1, do_sample=False))
    sample_prompt("song_1", "multinomial_sampling", GenerationConfig(num_beams=1, do_sample=True))
    sample_prompt("song_1", "beam-search_multinomial_sampling", GenerationConfig(num_beams=5, do_sample=True))
    # sample_prompt("song_2")
    # sample_prompt("song_3")
    # tokens = tokenizer.encode("prompt_song/song_1.mid")[0]
    # print(len(tokens))



# main(cp="checkpoints/cp_82.pt", sample_size=4, batch_size=4, sample_len=3072, out_dir="sample_out")

# %% 

# python sample.py --cp=checkpoints/cp_82.pt --sample_size=4 --sample_len=64 --out_dir=sample_out
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model checkpoint.")
    parser.add_argument("--cp", type=str, default="./")
    parser.add_argument("--device", type=str)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--sample_len", type=int)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    main(**vars(args))
