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
from poet.fasta import parse_stream
from poet.models.modules.packed_sequence import PackedTensorSequences
from poet.models.poet import PoET

import pickle
from train import PromoterModel, ATCG, PromoterDataset
from poet.alphabets import Alphabet, Uniprot21
from poet.models.poet import PoET

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


def get_seqs_from_fastalike(filepath: Path) -> list[bytes]:
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=False)]


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> List:
    return [
            append_startstop(alphabet.encode(s.encode('utf-8')
                                             .translate(None, delete=ASCII_LOWERCASE_BYTES)), 
                                             alphabet)
            for s in msa_sequences
        ]
    


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


def jit_warmup(embedding_model: PoET, alphabet: Uniprot21):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().cuda()
    segment_sizes = torch.tensor(segment_sizes).long().cuda()
    _ = embedding_model.embed(x.unsqueeze(0), segment_sizes.unsqueeze(0))


def _get_logps_tiered_fast(
    memory: Optional[list[PackedTensorSequences]],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    max_variant_length = max(len(v) for v in variants)
    memory = model.logits_allocate_memory(
        memory=memory,
        batch_size=batch_size,
        length=max_variant_length - 1,  # discount stop token
    )
    criteria = nn.CrossEntropyLoss(ignore_index=alphabet.mask_token, reduction="none")
    logps = []
    if pbar_position is not None:
        pbar = trange(
            0,
            len(variants),
            batch_size,
            desc=f"[{pbar_position}] decoding",
            leave=False,
            position=pbar_position,
        )
    else:
        pbar = range(0, len(variants), batch_size)
    for start_idx in pbar:
        
        this_variants = variants[start_idx : start_idx + batch_size]
        this_variants = pad_sequence(
            [torch.from_numpy(v).long() for v in this_variants],
            batch_first=True,
            padding_value=alphabet.mask_token,
        )
        if this_variants.size(1) < max_variant_length:
            this_variants = F.pad(
                this_variants,
                (0, max_variant_length - this_variants.size(1)),
                value=alphabet.mask_token,
            )
        assert (this_variants == alphabet.gap_token).sum() == 0
        this_variants = this_variants.cuda()
        logits = model.logits(this_variants[:, :-1], memory, preallocated_memory=True)
        targets = this_variants[:, 1:]
        score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
        logps.append(score.cpu().numpy())
        
    return np.hstack(logps)


def get_logps_tiered_fast(
    msa_sequences: Sequence[np.ndarray],
    variants: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    pbar_position: Optional[int] = None,
) -> np.ndarray:
    if len(msa_sequences) > 0:
        segment_sizes = torch.tensor([len(s) for s in msa_sequences]).cuda()
        msa_sequences: torch.Tensor = torch.cat(
            [torch.from_numpy(s).long() for s in msa_sequences]
        ).cuda()
      
        memory = model.embed(
            msa_sequences.unsqueeze(0),
            segment_sizes.unsqueeze(0),
            pbar_position=pbar_position,
        )
    else:
        memory = None

    return _get_logps_tiered_fast(
        memory=memory,
        variants=variants,
        model=model,
        batch_size=batch_size,
        alphabet=alphabet,
        pbar_position=pbar_position,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", 
                        type=str, 
                        default="model/epoch=9-val_loss=0.00-other_metric=0.00.ckpt")

    parser.add_argument("--variants_path",
                        type=str,
                        default="data/eQTL_with_pip.csv",
    )

    parser.add_argument("--output_npy_path",
                        type=str,
                        default="data/scored_variants.npy",
    )
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=2)
    parser.add_argument("--seed", 
                        type=int, 
                        default=188257)
 
    parser.add_argument("--hits_path",
                        type = str,
                        default="data/hits.pkl"
    )
    
    args = parser.parse_args()
    args.output_npy_path = Path(args.output_npy_path)
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    print("-------loading data--------")

    variants_df = pd.read_csv(args.variants_path)
    wt = variants_df["WT"].to_numpy()
    variants = variants_df["VAR"].to_numpy()
    names = variants_df["GENE"] + "_" + variants_df["CHROM"]

    with open(args.hits_path, "rb") as f:
        hits = pickle.load(f)

    
    print("-------loading model--------")

    model = PromoterModel()
    model.load_from_checkpoint(args.ckpt_path)
    model = model.model
    alphabet = ATCG()
    model = model.cuda().half().eval()
    dataset = PromoterDataset(hits, {}, alphabet, max_length = 16000 )

    
    # get homologs to score
    print("-------generating prompt--------")
    msa_sequences = [
        # np.array(dataset.get_inference_seqs(v, id)) for (id, v) in zip(names, variants)
    ]
    for id, v in tqdm(zip(names, variants), total=len(names)):
        curr = np.array(dataset.get_inference_seqs(v, id))
        if curr.size == 0:
            print(f'{id} has no hits!')
        
        msa_sequences.append(curr)
    
    variants = variants = [
        append_startstop(alphabet.encode(v.encode('utf-8')), alphabet=alphabet)
        for v in variants
    ]

    wt =  [
        append_startstop(alphabet.encode(v.encode('utf-8')), alphabet=alphabet)
        for v in wt
    ]

    # score the variants
    logps = []
    print("-------scoring variants--------")
    torch.cuda.empty_cache()

    for i, (w ,var) in tqdm(enumerate(zip(wt, variants)), total=len(wt)):
      
        msa = msa_sequences[i]
        # skip unknown genes
        if not msa.size:
            logps.append(None)
            continue

        msa = get_encoded_msa_from_a3m_seqs(msa_sequences=msa, alphabet=alphabet)

        forward_logps = get_logps_tiered_fast(
            msa_sequences=msa,
            variants=[np.array(var), np.array(w)],
            model=model,
            batch_size=args.batch_size,
            alphabet=alphabet,
            pbar_position=PBAR_POSITION,
        )
       
        backward_logps = get_logps_tiered_fast(
            msa_sequences=[np.ascontiguousarray(s[::-1]) for s in msa],
            variants=[np.ascontiguousarray(var[::-1]), np.ascontiguousarray(w[::-1])],
            model=model,
            batch_size=args.batch_size,
            alphabet=alphabet,
            pbar_position=PBAR_POSITION,
        )

        curr_logps = (forward_logps[0] + backward_logps[0] - forward_logps[1] - backward_logps[1]) / 2
        logps.append(curr_logps)

    print("-------saving output--------")
    np.save(args.output_npy_path, logps)
    variants_df['POET'] = logps
    variants_df.to_csv('data/scored_eQTL.csv')
    print("-------finished--------")
if __name__ == "__main__":
    main()
