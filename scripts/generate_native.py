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
            append_startstop(alphabet.encode(s.encode("ascii")
                                             .translate(None, delete=ASCII_LOWERCASE_BYTES)), 
                                             alphabet)
            for s in msa_sequences ]


def _get_sample_fast(
    memory: Optional[list[PackedTensorSequences]],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    temperature: float
) -> np.ndarray:
    
    sample_xs, sample_scores = model.sample_given_memory(memory = memory, 
                                                         temperature = temperature, # try changing it
                                                         top_k = None,
                                                         top_p = None,
                                                        #  minlen = 500, 
                                                         maxlen = 1000,
                                                         alphabet = alphabet,
                                                         batch_size = batch_size)
    
    return [alphabet.decode(x.cpu().numpy()).decode("utf-8")[1:-1] for x in sample_xs], sample_scores.cpu().tolist()



def get_sample_fast(
    msa_sequences: Sequence[np.ndarray],
    model: PoET,
    batch_size: int,
    alphabet: Uniprot21,
    temperature: float
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
        temperature = temperature
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", 
                        type=str, 
                        default="/n/groups/marks/users/erik/Promoter_Poet_private/model/reversed_big_30%_human_greedy_0.1_dropout_epoch=2-val_loss=0.00-other_metric=0.00.ckpt")

    parser.add_argument("--output_csv_path",
                        type=str,
                        default="data/reversed_sampled_seqs_greedy.csv",
    )

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=2)
    parser.add_argument("--seed", 
                        type=int, 
                        default=188257)
 
    parser.add_argument("--hits_path",
                        type=str,
                        default="data/hits.pkl"
    )
    parser.add_argument("--temp",
                        type=float,
                        default=1
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("-------loading data--------")

    with open(args.hits_path, "rb") as f:
        hits = pickle.load(f)

    names = [i for i in hits.keys() if i.endswith('_chr19')]
    # names = names[:3000]
   
    print("-------loading model--------")

    model = PromoterModel.load_from_checkpoint(args.ckpt_path)
    model = model.model
    alphabet = ATCG()
    model = model.cuda().eval()
    dataset = PromoterDataset(sequences = hits, 
                              queries = {}, 
                              alphabet = alphabet, 
                              max_length = 32000)
    
    # get homologs to score
    print("-------generating prompt--------")

    msa_sequences = [
        # np.array(dataset.get_inference_seqs(v, id)) for (id, v) in zip(names, variants)
    ]

    for id in tqdm(names, total=len(names)):
        curr = np.array(dataset.get_inference_seqs(variant = "", id = id))
        msa_sequences.append(curr)

    print("-------generating sequences--------")
    all_samples = []
    all_scores = []

    torch.cuda.empty_cache()

    with torch.cuda.amp.autocast():
        for prompt, _ in tqdm(zip(msa_sequences, names), total=len(names)):
            prompt = get_encoded_msa_from_a3m_seqs(msa_sequences=prompt, alphabet=alphabet)
            samples, scores = get_sample_fast(prompt, model, args.batch_size, alphabet, args.temp)
            # print(samples)
            print(len(samples[0]), len(samples[1]))
            all_samples.append(samples)
            all_scores.append(scores)

    print("-------saving output--------")

    df = pd.DataFrame()
    df['GENE'] = names
    df['samples'] = all_samples
    df['scores'] = all_scores
    df.to_csv("data/1k_val_big_greedy_mid_temp_generated_promoters.csv")
    print("-------finished-------")
    
if __name__ == "__main__":
    main()
