import argparse
import itertools
import os
import random
from glob import glob

from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
from miditok import REMI, TokenizerConfig, MusicTokenizer
from tqdm import tqdm

from musdr.side_utils import (
    compute_histogram_entropy,
    get_bars_crop,
    get_onset_xor_distance,
    get_pitch_histogram,
)


def compute_piece_pitch_entropy(
    piece_ev_seq, window_size, bar_ev_id, pitch_evs, verbose=False
):
    """
    Computes the average pitch-class histogram entropy of a piece.
    (Metric ``H``)

    Parameters:
      piece_ev_seq (list): a piece of music in event sequence representation.
      window_size (int): length of segment (in bars) involved in the calc. of entropy at once.
      bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
      pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
      verbose (bool): whether to print msg. when a crop contains no notes.

    Returns:
      float: the average n-bar pitch-class histogram entropy of the input piece.
    """
    # remove redundant ``Bar`` marker
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    n_bars = piece_ev_seq.count(bar_ev_id)
    if window_size > n_bars:
        print(
            "[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.".format(
                window_size
            )
        )
        window_size = n_bars

    # compute entropy of all possible segments
    pitch_ents = []
    for st_bar in range(0, n_bars - window_size + 1):
        seg_ev_seq = get_bars_crop(
            piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id
        )

        pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs=pitch_evs)
        if pitch_hist is None:
            if verbose:
                print(
                    "[Info] No notes in this crop: {}~{} bars.".format(
                        st_bar, st_bar + window_size - 1
                    )
                )
            continue

        pitch_ents.append(compute_histogram_entropy(pitch_hist))

    return np.mean(pitch_ents)


def compute_piece_groove_similarity(
    piece_ev_seq, bar_ev_id, pos_evs, pitch_evs, max_pairs=1000
):
    """
    Computes the average grooving pattern similarity between all pairs of bars of a piece.
    (Metric ``GS``)

    Parameters:
      piece_ev_seq (list): a piece of music in event sequence representation.
      bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
      pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
      pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
      max_pairs (int): maximum #(pairs) considered, to save computation overhead.

    Returns:
      float: 0~1, the average grooving pattern similarity of the input piece.
    """
    # remove redundant ``Bar`` marker
    if piece_ev_seq[-1] == bar_ev_id:
        piece_ev_seq = piece_ev_seq[:-1]

    # get every single bar & compute indices of bar pairs
    n_bars = piece_ev_seq.count(bar_ev_id)
    bar_seqs = []
    for b in range(n_bars):
        bar_seqs.append(get_bars_crop(piece_ev_seq, b, b, bar_ev_id))
    pairs = list(itertools.combinations(range(n_bars), 2))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    # compute pairwise grooving similarities
    grv_sims = []
    for p in pairs:
        grv_sims.append(
            1.0
            - get_onset_xor_distance(
                bar_seqs[p[0]], bar_seqs[p[1]], bar_ev_id, pos_evs, pitch_evs=pitch_evs
            )
        )

    return np.mean(grv_sims)


def eval_dir(midi_dir: str, tokenizer: MusicTokenizer, result_path: str):
    bar_id = tokenizer["Bar_None"]
    pos_ids = [v for k, v in tokenizer.vocab.items() if "Position" in k]
    pitch_ids = [v for k, v in tokenizer.vocab.items() if "Pitch_" in k]

    test_pieces = sorted(glob(os.path.join(midi_dir, "*.mid")))
    result_dict = {"piece_name": [], "H1": [], "H4": [], "GS": []}

    for p in tqdm(test_pieces):
        result_dict["piece_name"].append(p.replace("\\", "/").split("/")[-1])
        tokens = tokenizer.encode(p)
        # encode output list of miditok.TokSequence for multi-track
        tokens = list(tokens[0])

        h1 = compute_piece_pitch_entropy(
            tokens, 1, bar_ev_id=bar_id, pitch_evs=pitch_ids, verbose=False
        )
        result_dict["H1"].append(h1)
        h4 = compute_piece_pitch_entropy(
            tokens, 4, bar_ev_id=bar_id, pitch_evs=pitch_ids, verbose=False
        )
        result_dict["H4"].append(h4)
        gs = compute_piece_groove_similarity(
            tokens, bar_ev_id=bar_id, pos_evs=pos_ids, pitch_evs=pitch_ids
        )
        result_dict["GS"].append(gs)

    if len(result_dict):
        df = pd.DataFrame.from_dict(result_dict)
        df.to_csv(result_path, index=False, encoding="utf-8")

    def filter(arr):
        return [x for x in arr if x is not None and not np.isnan(x)]
    
    return {
        "H1": np.mean(filter(result_dict["H1"])).item(),
        "H4": np.mean(filter(result_dict["H4"])).item(),
        "GS": np.mean(filter(result_dict["GS"])).item()
    }


"""
PYTHONPATH="musdr" python eval_metrics.py --midi_dir=pop1k7/midi_analyzed/src_001
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, help="", required=True)
    args = parser.parse_args()

    tokenizer = REMI(TokenizerConfig())
    eval_dir(args.midi_dir, tokenizer)

    # tokenizer = REMI(TokenizerConfig())
    # bar_id = tokenizer["Bar_None"]
    # pos_ids = [v for k, v in tokenizer.vocab.items() if "Position" in k]
    # pitch_ids = [v for k, v in tokenizer.vocab.items() if "Pitch_" in k]

    # test_pieces = sorted(glob(os.path.join(args.midi_dir, "*.mid")))
    # result_dict = {"piece_name": [], "H1": [], "H4": [], "GS": []}

    # for p in tqdm(test_pieces):
    #     result_dict["piece_name"].append(p.replace("\\", "/").split("/")[-1])
    #     tokens = tokenizer.encode(p)
    #     # encode output list of miditok.TokSequence for multi-track
    #     tokens = list(tokens[0])

    #     h1 = compute_piece_pitch_entropy(
    #         tokens, 1, bar_ev_id=bar_id, pitch_evs=pitch_ids, verbose=False
    #     )
    #     result_dict["H1"].append(h1)
    #     h4 = compute_piece_pitch_entropy(
    #         tokens, 4, bar_ev_id=bar_id, pitch_evs=pitch_ids, verbose=False
    #     )
    #     result_dict["H4"].append(h4)
    #     gs = compute_piece_groove_similarity(
    #         tokens, bar_ev_id=bar_id, pos_evs=pos_ids, pitch_evs=pitch_ids
    #     )
    #     result_dict["GS"].append(gs)

    # if len(result_dict):
    #     df = pd.DataFrame.from_dict(result_dict)
    #     df.to_csv("pop1k7.csv", index=False, encoding="utf-8")
