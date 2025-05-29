import pickle
import pandas as pd
import random 
import matplotlib.pyplot as plt

def create_sanity_check():

    human_path = '/n/groups/marks/users/erik/Promoter_Poet_private/data/query.pkl'
        
    with open(human_path, 'rb') as f:
        human_seq = pickle.load(f)

    def mutate_dna(dna_string, mutation_rate=0.25):
        """
        Randomly mutates a percentage of nucleotides in a DNA string.
        
        Args:
            dna_string (str): A string of DNA sequence
            mutation_rate (float): Percentage of nucleotides to mutate (default: 0.25 for 25%)
            
        Returns:
            str: DNA string with random mutations
        """
        nucleotides = list(dna_string)
        positions_to_mutate = random.sample(range(len(nucleotides)), int(len(nucleotides) * mutation_rate))
        
        for position in positions_to_mutate:
            current = nucleotides[position].upper()
            # Get available nucleotides for mutation (excluding the current one)
            options = [n for n in "ATGC" if n != current]
            # Randomly select one of the alternative nucleotides
            nucleotides[position] = random.choice(options)
        
        return ''.join(nucleotides)


    df = pd.DataFrame()
    df['GENE'] = list(human_seq.keys())

    df['WT'] = list(human_seq.values())
    df['GC'] = df['WT'].map(lambda x: ''.join(char for char in x if char not in "GCgc"))
    df['RANDOM'] = df['WT'].map(mutate_dna)
    # df = df[df['GENE'].str.endswith('_chr19')]
    df = df.head(1000)
    df.to_csv('data/sanity_check.csv')


import argparse
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

from train import PromoterModel, ATCG, PromoterDataset
from poet.alphabets import Uniprot21
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


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> List:
    return [
            append_startstop(alphabet.encode(s.encode("ascii")), alphabet)
            for s in msa_sequences
        ]

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
                        default="/n/groups/marks/users/erik/Promoter_Poet_private/model/1e-3_lr_reversed_random_no_dropout_small_epoch=11-val_loss=0.00-other_metric=0.00.ckpt")
    parser.add_argument("--output_csv_path",
                        type=str,
                        default="data/val_scored_sanity_check.csv",)
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=3)
    parser.add_argument("--seed", 
                        type=int, 
                        default=188257)
    parser.add_argument("--hits_path",
                        type = str,
                        default="data/hits.pkl")
    
    args = parser.parse_args()
    args.output_csv_path = Path(args.output_csv_path)
    return args


@torch.inference_mode()
def main(context_length):
    variants_df = pd.read_csv('data/sanity_check.csv')

    wt = variants_df["WT"].to_numpy()
    gc_var = variants_df["GC"].to_numpy()
    names = variants_df["GENE"].to_numpy()
    random_var = variants_df["RANDOM"].to_numpy()

    dataset = PromoterDataset(sequences = hits, 
                              queries = {}, 
                              alphabet = alphabet, 
                              max_length = context_length)

    # get homologs to score
    print("-------generating prompt--------")

    msa_sequences = [
        # np.array(dataset.get_inference_seqs(v, id)) for (id, v) in zip(names, variants)
    ]
    for id, v in tqdm(zip(names, wt), total=len(names)):
        curr = np.array(dataset.get_inference_seqs(v, id))
        if curr.size == 0:
            print(f'{id} has no hits!')
        msa_sequences.append(curr)
    
    gc_var = [
        append_startstop(alphabet.encode(v.encode("ascii")), alphabet=alphabet)
        for v in gc_var
    ]
    random_var = [
        append_startstop(alphabet.encode(v.encode("ascii")), alphabet=alphabet)
        for v in random_var
    ]
    wt =  [
        append_startstop(alphabet.encode(v.encode("ascii")), alphabet=alphabet)
        for v in wt
    ]

    # score the variants
    wt_scores = []
    gc_scores = []
    random_scores = []

    print("-------scoring variants--------")

    torch.cuda.empty_cache()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i, (w ,gc, random) in tqdm(enumerate(zip(wt, gc_var, random_var)), total=len(wt)):
        
            msa = msa_sequences[i]
            # msa = []
            msa = get_encoded_msa_from_a3m_seqs(msa_sequences=msa, alphabet=alphabet)

            forward_logps = get_logps_tiered_fast(
                msa_sequences=msa,
                variants=[np.array(gc), np.array(random), np.array(w)],
                model=model,
                batch_size=args.batch_size,
                alphabet=alphabet,
                pbar_position=PBAR_POSITION,
            )
        
            backward_logps = get_logps_tiered_fast(
                msa_sequences=[np.ascontiguousarray(s[::-1]) for s in msa],
                variants=[np.ascontiguousarray(gc[::-1]), np.ascontiguousarray(random[::-1]) ,np.ascontiguousarray(w[::-1])],
                model=model,
                batch_size=args.batch_size,
                alphabet=alphabet,
                pbar_position=PBAR_POSITION,
            )

            # curr_logps = (forward_logps[0] + backward_logps[0] - forward_logps[1] - backward_logps[1]) / 2
            
            gc_scores.append((forward_logps[0] + backward_logps[0]) / (2 * (len(gc) - 2)))  # -2 for start/stop tokens))
            random_scores.append((forward_logps[1] + backward_logps[1]) / (2 * (len(random) - 2)))  # -2 for start/stop tokens)
            wt_scores.append((forward_logps[2] + backward_logps[2] )/ (2 * (len(w) - 2)))  # -2 for start/stop tokens)
            # gc_scores.append((forward_logps[0] ) / ((len(gc) - 2)))  # -2 for start/stop tokens))
            # random_scores.append((forward_logps[1] ) / ((len(random) - 2)))  # -2 for start/stop tokens)
            # wt_scores.append((forward_logps[2] )/ ((len(w) - 2)))  # -2 for start/stop tokens)
            # logps.append(curr_logps)

    print("-------saving output--------")

    # np.save(args.output_npy_path, logps)
    # variants_df['GC_scores'] = gc_scores
    # variants_df['random_scores'] = random_scores
    # variants_df['WT_scores'] = wt_scores

    gc_scores = np.array(gc_scores)
    random_scores = np.array(random_scores)
    wt_scores = np.array(wt_scores)
    # variants_df.to_csv(args.output_csv_path)
    
    print("-------result-------")
    print(f'WT average: {wt_scores.mean()}')
    print(f'GC average: {gc_scores.mean()}')
    print(f'Random average: {random_scores.mean()}')
    print("-------finished-------")

    return wt_scores.mean(), gc_scores.mean(), random_scores.mean()
    
if __name__ == "__main__":
    # main()

    create_sanity_check()
    args = parse_args()
    print("-------loading data--------")

    # variants_df = pd.read_csv(args.variants_path

    with open(args.hits_path, "rb") as f:
        hits = pickle.load(f)

    print("-------loading model--------")

    model = PromoterModel.load_from_checkpoint(args.ckpt_path)
    model = model.model
    alphabet = ATCG()
    model = model.cuda().eval()

    ctx_length = 2 ** np.arange(19)
    
    all_wt = []
    all_gc = []
    all_rand = []

    # for ctx in ctx_length:
    for ctx in [32800]:

        wts, gcs, rands = main(ctx)
        all_wt.append(wts)
        all_gc.append(gcs)
        all_rand.append(rands)
        
    breakpoint()
    plt.figure(figsize=(10, 6))
    
    plt.plot(ctx_length, all_wt, 'o-', label='WT')
    plt.plot(ctx_length, all_gc, 's-', label='GC')
    plt.plot(ctx_length, all_rand, '^-', label='Random')
    
    plt.xscale('log', base=2)  # Use log scale with base 2 for x-axis
    plt.xlabel('Context Length (Powers of 2)')
    plt.ylabel('Value')
    plt.title('Performance vs Context Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Add x-ticks at powers of 2
    plt.xticks(ctx_length, [str(int(x)) for x in ctx_length], rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("context_lengths.png", dpi=1200)



