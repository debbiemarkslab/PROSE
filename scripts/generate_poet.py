import argparse
import itertools
import string
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
import pandas as pd
from poet.alphabets import Uniprot21
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet import PoET
from poet.fasta import parse_stream

import pickle
from train import PromoterModel, ATCG, PromoterDataset
from poet.alphabets import Uniprot21

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()
PBAR_POSITION = 1


T = TypeVar("T", np.ndarray, torch.Tensor)


def append_startstop(x: T, alphabet: Uniprot21) -> T:
    x_ndim = x.ndim
    assert x_ndim in {1, 2}
    if x_ndim == 1:
        x = x[None, :]

    if isinstance(x, torch.Tensor):
        empty_func = torch.empty
    else:
        empty_func = np.empty
    x_ = empty_func((x.shape[0], x.shape[1] + 2), dtype=x.dtype)
    x_[:, 0] = alphabet.start_token
    x_[:, -1] = alphabet.stop_token
    x_[:, 1:-1] = x
    if x_ndim == 1:
        x_ = x_.flatten()
    return x_


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> List:
    return [
            append_startstop(alphabet.encode(s.encode('utf-8')
                                             .translate(None, delete=ASCII_LOWERCASE_BYTES)), 
                                             alphabet)
            for s in msa_sequences ]


def get_seqs_from_fastalike(filepath: Path) -> list[bytes]:
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=False)]



def sample_msa_sequences(
    get_sequence_fn: Callable[[int], bytes],
    sample_idxs: Sequence[int],
    max_tokens: int,
    alphabet: Uniprot21,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    truncate: bool = True,
) -> list[np.ndarray]:
    assert alphabet.start_token != -1
    assert alphabet.stop_token != -1
    if not shuffle:
        assert shuffle_seed is None

    seqs, total_tokens = [], 0
    for idx in sample_idxs:
        next_sequence = get_sequence_fn(idx)
        seqs.append(append_startstop(alphabet.encode(next_sequence), alphabet=alphabet))
        total_tokens += len(seqs[-1])
        if total_tokens > max_tokens:
            break

    # shuffle order and truncate to max tokens
    if shuffle:
        rng = (
            np.random.default_rng(shuffle_seed)
            if shuffle_seed is not None
            else np.random
        )
        final_permutation = rng.permutation(len(seqs))
    else:
        final_permutation = np.arange(len(seqs))
    final_seqs, total_tokens = [], 0
    for seq in [seqs[i] for i in final_permutation]:
        if truncate and (total_tokens + len(seq) > max_tokens):
            seq = seq[: max_tokens - total_tokens]
        total_tokens += len(seq)
        final_seqs.append(seq)
        if total_tokens >= max_tokens:
            break
    return final_seqs


def _get_sample_fast(
    memory: Optional[list[PackedTensorSequences]],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
) -> np.ndarray:
    
    sample_xs, sample_scores = model.sample_given_memory(memory = memory, 
                                                         temperature = 1, # try changing it
                                                         top_k = None,
                                                         top_p = None,
                                                         minlen = 500, 
                                                         maxlen = 1000,
                                                         alphabet = alphabet,
                                                         batch_size = batch_size)
    
    return [alphabet.decode(x.cpu().numpy()).decode("utf-8")[1:-1] for x in sample_xs], sample_scores.cpu().tolist()



def get_sample_fast(
    msa_sequences: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).cuda()
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).cuda()
      
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
        )
    else:
        memory = None

    return _get_sample_fast(
        memory=memory,
        model=model,
        batch_size=batch_size,
        alphabet=alphabet,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", 
                        type=str, 
                        default="/n/groups/marks/users/erik/Promoter_Poet_private/data/poet.ckpt")

    parser.add_argument("--output_csv_path",
                        type=str,
                        default="data/sampled_seqs_greedy.csv",
    )

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=16)
    parser.add_argument("--seed", 
                        type=int, 
                        default=188257)
 
    parser.add_argument("--hits_path",
                        type=str,
                        default="data/hits.pkl"
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load model
    ckpt = torch.load(args.ckpt_path)
    model = PoET(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
    model.load_state_dict(
        {k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()}
    )
    del ckpt
    model = model.cuda().half().eval()
    alphabet = Uniprot21(
        include_gap=True, include_startstop=True, distinct_startstop=True
    )

    # process msa
    msa_sequences = get_seqs_from_fastalike(args.msa_a3m_path)
    msa = get_encoded_msa_from_a3m_seqs(msa_sequences=msa_sequences, alphabet=alphabet)
    msa_sequences = [
        # np.array(dataset.get_inference_seqs(v, id)) for (id, v) in zip(names, variants)
    ]

    all_samples = []
    all_scores = []

    torch.cuda.empty_cache()

    with torch.cuda.amp.autocast():
        for prompt, _ in tqdm(zip(msa_sequences, names), total=len(names)):
            prompt = get_encoded_msa_from_a3m_seqs(msa_sequences=prompt, alphabet=alphabet)
            samples, scores = get_sample_fast(prompt, model, args.batch_size, alphabet)
            all_samples.append(samples)
            all_scores.append(scores)

    print("-------saving output--------")

    df = pd.DataFrame()
    df['GENE'] = names
    df['samples'] = all_samples
    df['scores'] = all_scores
    df.to_csv("data/generated_promoters.csv")
    print("-------finished-------")
    
if __name__ == "__main__":
    main()
