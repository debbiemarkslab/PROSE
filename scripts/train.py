import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import lightning
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_optimizer import Adafactor
import pickle
from poet.alphabets import Alphabet, Uniprot21
from poet.models.poet import PoET
import random 

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.grads import grad_norm
from lightning.pytorch.callbacks import ModelCheckpoint
import math
from Levenshtein import distance as levenshtein_distance
IGNORE_INDEX = -100


class ATCG(Alphabet):
    """
    tokenize DNA sequence 
    """

    def __init__(
        self,
        mask=True,
        include_gap=True,
        include_startstop=True,
        distinct_startstop=True,
    ):
        chars = b"ATCG"
        gap_token = start_token = stop_token = -1
        if include_gap:
            chars = chars + b"-"
            gap_token = len(chars) - 1
        if include_startstop:
            chars = chars + b"*"
            start_token = stop_token = len(chars) - 1
        if distinct_startstop:
            chars = chars + b"$"
            stop_token = len(chars) - 1
        mask_token = len(chars)
        encoding = np.arange(len(chars))
        missing = mask_token

        super(ATCG, self).__init__(chars, encoding=encoding, mask=mask, missing=missing)

        self.gap_token = gap_token
        self.start_token = start_token
        self.stop_token = stop_token
        self.mask_token = mask_token


class PromoterDataset(Dataset):
    """
    Dataloader for promoter sets from Zoonomia
    """

    def __init__(self, sequences: dict, queries: dict, alphabet, max_length):
        self.alphabet = alphabet
        self.sequences = {k : list(v) for k, v in sequences.items()}
        self.queries = queries
        self.num_special_characters = 2
        self.ids = list(sequences.keys())
        self.sampling_weights = [len(v) for k, v in self.sequences.items()]
        total = sum(self.sampling_weights)
        self.sampling_weights = [i/ total for i in self.sampling_weights]
        self.max_len = max_length
        self.iters = len(self.ids) * 5

    def __getitem__(self, idx):
        '''
        randomly sample a gene with replacement
        weighted by the number of sequences the gene has   
        '''
        id = np.random.choice(self.ids, p = self.sampling_weights)
        sampled_set = self.sample_and_fit_sequences(id, weights = None, max_individual_seq_length=None, max_seq_length=self.max_len)
        # sampled_set = self.diverse_sample_and_fit_sequences(id, max_seq_length=self.max_len)
        return self.pack_inputs(sampled_set, self.alphabet, self.max_len)

    def get_inference_seqs(self, variant: str,id: str):
        try:
            sampled_set = self.sample_and_fit_sequences(id, sequence=variant, max_seq_length=self.max_len)

            # sampled_set = self.diverse_sample_and_fit_sequences(id, max_seq_length=self.max_len, sequence=variant)
        except KeyError:
            return []
        return sampled_set["passages"]

    def sample_query(self, id: str, p_human: float) -> str:
        '''
        sample a query sequence for a given gene, with a chosen probability of 
        sampling a human sequence 
        
        Args:
        id: gene name 
        p_human: percent probability of sampling human 

        Returns:
        single sampled DNA sequence as a query
        '''
        human = np.random.choice([True, False], p=[p_human, 1 - p_human])
        if human:
            return self.queries[id]
        else:
            return np.random.choice(self.sequences[id])
        
    
    def diverse_sample_and_fit_sequences(
        self,
        query_id_or_sequence: Optional[str] = None,  # Changed parameter name to be more explicit
        max_seq_length: int = 1024,
        max_individual_seq_length: Optional[int] = None,
        include_query: bool = True,
        truncate: bool = False,
        p_human: float = 0.3,
        sequence = None
    ) -> Dict[str, Any]:
        """
        refer to docs of sample_and_fit_sequences

        uses k-DPP sampling to maximize diversity within a subset based on hamming distance 
        """
        if sequence is None:
            query_sequence = self.sample_query(query_id_or_sequence, p_human) # P(human sequence) = 0.3
        else:
            query_sequence  = sequence

        query_length = len(query_sequence) + self.num_special_characters if query_sequence else None
        
        # Calculate effective length for passages
        effective_length = max_seq_length - (query_length if query_length is not None else 0)
        
        total_tokens = query_length if query_length is not None else 0
        leftover = effective_length

        max_iters = 200
        beta = 0.2
        rounds = 0
        S = {}
        score = 0

        while leftover > 0 and rounds < max_iters:
            rounds += 1
            if random.random() < 0.5:
            # Uniformly randomly select an element from V
                s = random.choice(self.sequences[query_id_or_sequence])
                if s not in S:
                    # Attempt to add s
                    new_score = 0
                    for t in S.keys():
                        new_score += levenshtein_distance(t, s) / max(len(t), len(s))
                    new_score /= (len(S) + 1)**2
                    
                    p_add = math.exp(beta * (score+new_score)) / (math.exp(beta * score) + math.exp(beta * (score+new_score)))
                    if random.random() < p_add:
                        S[s] = len(s)
                        l = (len(s) + self.num_special_characters)
                        score += new_score
                        leftover -= l
                        total_tokens += l
                        
                elif s in S:
                    # Attempt to delete s
                    new_score = 0
                    for t in S.keys():
                        new_score += levenshtein_distance(t, s) / max(len(t), len(s))
                    new_score /= (len(S) + 1)**2
                    p_delete = math.exp(beta * (score - new_score)) / (math.exp(beta * score) + math.exp(beta * (score - new_score)))
                    if random.random() < p_delete:
                        del S[s]
                        l = (len(s) + self.num_special_characters)
                        leftover += l
                        total_tokens -= l
                        score -= new_score
           
        #                                  
        return {
            "id": query_id_or_sequence if include_query else None,
            "sequence": query_sequence,
            "passages": list(S.keys()),
            "passage_lengths": list(S.values()),
            "query_length": query_length,
            "total_tokens": total_tokens
        }
        

    def sample_and_fit_sequences(
        self,
        query_id_or_sequence: Optional[str] = None,  # Changed parameter name to be more explicit
        weights: Optional[List[float]] = None,
        max_seq_length: int = 1024,
        max_individual_seq_length: Optional[int] = None,
        include_query: bool = True,
        sample: bool = True,
        truncate: bool = False,
        chunk_size: int = 2,
        max_skips: int = 2,
        sequence = None
    ) -> Dict[str, Any]:
        """
        Packs sequences until max length is reached. Can exclude query and use deterministic ordering.
        
        Args:
            query_id_or_sequence: Either a UniRef ID to lookup in fasta_dataset, or the actual sequence
            member_ids: List of UniRef IDs or sequences for passages
            weights: Optional sampling weights for passages
            fasta_dataset: Optional map from UniRef IDs to sequences. If None, assumes inputs are sequences
            max_seq_length: Maximum length of all sequences combined
            max_individual_seq_length: Optional int to limit the length of individual sequences.
            include_query: Whether to include and process query sequence
            sample: Whether to sample randomly (True) or use sequential order (False)
            truncate: Whether to truncate the last sequence if it doesn't fit
            chunk_size: Number of IDs to process at a time for sampling
            max_skips: Max chunks allowed to add no passages before giving up

        Returns:
            Dict with id, sequence, passages, passage_lengths, query_length, and total_tokens
        """

        def truncate_sequence(seq: str, max_len: int) -> str:
            """Helper to truncate a sequence to max length while accounting for special tokens"""
            if len(seq) + self.num_special_characters > max_len:
                return seq[:max_len - self.num_special_characters]
            return seq
        
        if sequence is not None:
            query_sequence = sequence
        else:
            query_sequence = self.sample_query(query_id_or_sequence, 0.3) # P(human sequence) = 0.3
        
        
        # Apply individual sequence length limit if specified
        if max_individual_seq_length and query_sequence:
            query_sequence = truncate_sequence(query_sequence, max_individual_seq_length)
        
        query_length = len(query_sequence) + self.num_special_characters if query_sequence else None
        
        # Calculate effective length for passages
        effective_length = max_seq_length - (query_length if query_length is not None else 0)
        
        passages = []
        passage_lengths = []
        total_tokens = query_length if query_length is not None else 0
        leftover = effective_length

        
        member_ids = self.sequences[query_id_or_sequence]

        if sample:
            member_weights = np.array(weights if weights is not None else [1.0 / len(member_ids)] * len(member_ids))
            member_weights /= member_weights.sum()
            
            skip_rounds = 0
            while leftover > 0 and skip_rounds < max_skips:
                sampled_ids = np.random.choice(member_ids, size=chunk_size, replace=True, p=member_weights)
                added = False
                
                for seq in sampled_ids:
                    if not seq:
                        continue
                    
                    # Apply individual sequence length limit if specified
                    if max_individual_seq_length:
                        seq = truncate_sequence(seq, max_individual_seq_length)
                    
                    seq_len = len(seq) + self.num_special_characters
                    if seq_len <= leftover:
                        passages.append(seq)
                        passage_lengths.append(seq_len)
                        leftover -= seq_len
                        total_tokens += seq_len
                        added = True
                    elif truncate and leftover > 0:
                        # Truncate to fit remaining space
                        trunc_len = leftover - self.num_special_characters
                        seq = seq[:trunc_len]
                        passages.append(seq)
                        passage_lengths.append(leftover)
                        total_tokens += leftover
                        leftover = 0
                        added = True
                        break
                    
                    if leftover <= 0:
                        break
                        
                skip_rounds = 0 if added else skip_rounds + 1
           
                        
        return {
            "id": query_id_or_sequence if include_query else None,
            "sequence": query_sequence,
            "passages": passages,
            "passage_lengths": passage_lengths,
            "query_length": query_length,
            "total_tokens": total_tokens
        }

    def pack_inputs(
        self, 
        sample, 
        tokenizer,  # DNA tokenizer instance
        max_seq_length, 
        padding_idx=IGNORE_INDEX,
        reverse_sequence=False,
        skip_padding=True
    ):
        """
        Packs a single sample's query and passages into a single sequence, using Uniprot21 tokenization.
        Each sequence is wrapped with start/stop tokens before concatenation.

        Args:
            sample (Dict[str, Any]): A dictionary containing:
                - "id": Query ID
                - "sequence": Query sequence string
                - "passages": List of passage sequence strings
                - "passage_lengths": List of passage lengths (including special tokens)
                - "query_length": Length of query sequence (including special tokens)
            tokenizer: Uniprot21 tokenizer instance
            max_seq_length (int): Maximum length of the packed sequence
            padding_idx (int): Token ID to use for padding
            cls_token (str): Start token character ($ for Uniprot21)
            eos_token (str): Stop token character (* for Uniprot21) 
            reverse_sequence (bool): If True, reverse sequences before tokenization
            skip_padding (bool): If True, don't pad sequence to max_seq_length

        Returns:
            Dict[str, torch.Tensor]: Packed inputs including:
                - "tokens": (seq_len,) LongTensor of token IDs
                - "seq_lens": List[int] containing length of each sequence chunk
                - "labels": (seq_len,) LongTensor same as tokens but with padding masked to -100

        Example:
            For sequence ABC, passages [DEF, GH] and max_seq_length=14:
            
            Normal order (reverse_sequence=False):
                tokens:  [$ D E F * $ G H * $ A B C * <pad>]
                seq_lens: [5, 4, 5]  # Each sequence length includes its start/stop tokens
                labels:  [$ D E F * $ G H * $ A B C * -100]
                
            Reversed (reverse_sequence=True):
                tokens:  [* F E D $ * H G $ * C B A $ <pad>]
                seq_lens: [5, 4, 5]  # Lengths stay same, just content is reversed
                labels:  [* F E D $ * H G $ * C B A $ -100]

            Note: $ = start token, * = stop token, <pad> = padding token
                Actual tokens will be Uniprot21 token IDs, shown here as characters for clarity
                Sequences are concatenated: passages first, then query
        """
        """
    [previous docstring remains the same]
    """
        def tokenize_sequence(text: str, reverse_sequence: bool) -> np.ndarray:
            if reverse_sequence:
                text = text[::-1]
            if isinstance(text, str):
                text_bytes = text.encode('ascii')
            else:
                text_bytes = text  # Already bytes
            tokens = tokenizer.encode(text_bytes)
            if reverse_sequence:
                return np.concatenate([[tokenizer.stop_token], tokens, [tokenizer.start_token]])
            else:
                return np.concatenate([[tokenizer.start_token], tokens, [tokenizer.stop_token]])

        # Initialize sequences and lengths
        all_sequences = []
        all_lengths = sample["passage_lengths"].copy()  # Use pre-calculated passage lengths
        
        # Process passages
        if sample["passages"]:
            for seq in sample["passages"]:
                if seq is not None:
                    tokens = tokenize_sequence(seq, reverse_sequence)
                    all_sequences.append(tokens)
        
        
        # Process query if it exists
        query_sequence = sample["sequence"]
        if query_sequence is not None:
            tokens = tokenize_sequence(query_sequence, reverse_sequence)
            all_sequences.append(tokens)
            # Use pre-calculated query length
            all_lengths.append(sample["query_length"])
        
        # Concatenate all sequences
        if all_sequences:
            tokens = np.concatenate(all_sequences)
            tokens = torch.from_numpy(tokens).long()
        else:
            tokens = torch.empty(0, dtype=torch.long)
        
        # Add padding if needed
        current_length = sum(all_lengths)
        if not skip_padding and current_length < max_seq_length:
            padding_length = max_seq_length - current_length
            all_lengths.append(padding_length)
            padding = torch.full((padding_length,), padding_idx, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding])
        
        # Create labels (same as tokens but with padding masked)

        return {
            "tokens": tokens,
            "seq_lens": all_lengths,
        }

    def padded_collate_packed(
        self,
        batch: List[Dict[str, Any]],
        pad_mode: Optional[str] = None,
        fixed_length: Optional[int] = None,
        padding_idx: int = IGNORE_INDEX,
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of packed samples into a single dictionary.

        Args:
            batch: List of dictionaries containing packed sequences.
            fabric: Fabric instance for moving data to device.
            pad_mode: Can be "largest", "fixed", or None. If None, behavior is the default ("largest").
            fixed_length: If pad_mode=="fixed", pad each sample to this length.
            padding_idx: The token ID to use for padding.

        Returns:
            Dict with:
                - "tokens": (bsz, seq_len) LongTensor (padded)
                - "labels": (bsz, seq_len) LongTensor (padded)
                - "segment_sizes": (bsz, max_segments) LongTensor (padded with zeros)
        """
        # If pad_mode is None, default to "largest"
        if pad_mode is None:
            pad_mode = "largest"
        
        # Determine target length for tokens
        token_lengths = [x["tokens"].size(0) for x in batch]
        if pad_mode == "largest":
            target_length = max(token_lengths)
        elif pad_mode == "fixed":
            if fixed_length is None:
                raise ValueError(
                    "fixed_length must be provided when pad_mode is 'fixed'"
                )
            target_length = fixed_length
        else:
            raise ValueError("Unknown pad_mode. Use 'largest', 'fixed', or None.")

        padded_tokens = []
   
        for x in batch:
            token = x["tokens"]
            pad_len = target_length - token.size(0)
            if pad_len > 0:
                token = F.pad(token, (0, pad_len), value=padding_idx)
            padded_tokens.append(token)
        tokens_tensor = torch.stack(padded_tokens, dim=0)


        # Collate segment sizes (each sample may have several segments).
        max_segments = max(len(x["seq_lens"]) for x in batch)
        bsz = len(batch)
        segment_sizes = torch.zeros((bsz, max_segments), dtype=torch.int32)
        for i, x in enumerate(batch):
            seq_lens = torch.tensor(x["seq_lens"], dtype=torch.int32)
            segment_sizes[i, : len(seq_lens)] = seq_lens

        collated_batch = {
            "tokens": tokens_tensor,
            "segment_sizes": segment_sizes,
        }
       
        collated_batch = self._to_cuda(collated_batch)
        return collated_batch

    def __len__(self):
        return self.iters

    def _to_cuda(self, tok_dict):
        return {k: v.long().cuda() for k, v in tok_dict.items()}
    
    @staticmethod
    def train_validation_split(chr: int, sequences: dict, queries: dict) -> Tuple[Dict, Dict, Dict, Dict]:
        return sequences, queries, sequences, queries
    
    @staticmethod
    def _train_validation_split(chr: int, sequences: dict, queries: dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        splits two dictionaries based on the chromosome number of the keys

        """
        train_seq = {}
        train_query = {}
        val_seq = {}
        val_query = {}
        chr_pattern = f"_chr{chr}"
        for key in sequences.keys():
            if key not in queries:
                continue
            if chr_pattern in key:
                val_seq[key] = sequences[key]
                val_query[key] = queries[key]
            else:
                train_seq[key] = sequences[key]
                train_query[key] = queries[key]

        print(f"training with {len(train_query)} sequences")
        print(f"validating with {len(val_query)} sequences")

        return train_seq, train_query, val_seq, val_query   



def jit_warmup(embedding_model: PoET, alphabet: Uniprot21):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().cuda()
    segment_sizes = torch.tensor(segment_sizes).long().cuda()
    x = embedding_model(x.unsqueeze(0), segment_sizes.unsqueeze(0))
    print("warmup", x)


class SqrtDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    ''' 
    Used in original PoET Training
    '''
    def __init__(self, 
                 optimizer,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 scaling_factor: float = 1.0,
                 offset: int = 0):
        
        self.scaling_factor = scaling_factor
        self.offset = offset
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> list[float]:
        """Compute scaled learning rate using sqrt decay"""
        return [base_lr * self.scaling_factor / math.sqrt(self.last_epoch + 1 + self.offset)
                for base_lr in self.base_lrs]

    def theoretical_bound(self, 
                         total_steps: int,
                         gradient_bound: float) -> float:
        """
        Theoretical convergence bound for convex functions with L-Lipschitz gradients:
        
        f(x̄_T) - f(x^*) ≤ (2D²L + D√(2σ²T)) / √T
        
        Where:
        - D: Diameter of feasible set
        - L: Lipschitz constant
        - σ²: Gradient variance bound
        - T: Total number of steps
        
        Returns expected optimization error bound
        """
        return (2 * self.scaling_factor**2 * gradient_bound**2 + 
                self.scaling_factor * math.sqrt(2 * gradient_bound**2 * total_steps)) / torch.sqrt(total_steps)

class PromoterModel(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        # ckpt = torch.load(ckpt_path)
        init = {
            "n_vocab": 24,
            "hidden_dim": 1024,
            "num_layers": 12,
            "nhead": 16,
            "dropout": 0,
            "use_multi_rotary": True,
            "norm": True,
        }
        self.model = PoET(**init).cuda()

    def forward(self, xs: torch.Tensor, segment_sizes: torch.Tensor) -> torch.Tensor:
        if (torch.isnan(xs).any() or torch.isnan(segment_sizes).any()):
            raise Exception("NAN in INPUT!!")
        
        # print(xs.shape, segment_sizes.shape)
        return self.model(xs, segment_sizes)

    def training_step(self, batch, batch_idx):
        xs = batch["tokens"]
        segment_sizes = batch["segment_sizes"]
        logits = self.model(xs, segment_sizes)

        # Calculate loss (next token prediction)
        targets = xs[:, 1:].contiguous()  # Shift targets by 1
        logits = logits[:, :-1, :].contiguous()  # Remove last logit
        # logits = self._clamp(logits)
        # loss = F.cross_entropy(
        #     self._clamp(logits.view(-1, logits.size(-1))),
        #     self._clamp(targets.view(-1)),
        #     ignore_index=IGNORE_INDEX,
        # )
        # logits = torch.nn.functional.softmax(logits, dim=-1)
        # print(logits.sum(dim=-1))
        sequence_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=IGNORE_INDEX
        )
        
        # # Reshape loss and apply mask
        # loss = loss.reshape(logits.size(0), -1)
        # valid_positions = (targets != -100).float()
        # sequence_loss = ((loss * valid_positions).sum(dim=1) / (valid_positions.sum(dim=1))).mean()
        
        # Handle any NaN or Inf values
        if not torch.isfinite(sequence_loss).all():
            print(sequence_loss)
            # self.log(f'nan_inf_detected', 1.0, sync_dist=True)
            sequence_loss = torch.where(
                torch.isfinite(sequence_loss), 
                sequence_loss, 
                torch.tensor(0.0, device=sequence_loss.device)
            )
            self.zero_grad()

        perplexity = torch.exp(sequence_loss)
        self.log("train_loss", sequence_loss)
        self.log("train_perplexity", perplexity)
        return sequence_loss
    
    def validation_step(self, batch, batch_idx):
        xs = batch["tokens"]
        segment_sizes = batch["segment_sizes"]
        logits = self(xs, segment_sizes)

        # Calculate loss (next token prediction)
        targets = xs[:, 1:].contiguous()  # Shift targets by 1
        logits = logits[:, :-1, :].contiguous()  # Remove last logit
      
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction='mean'
        )
        perplexity = torch.exp(loss)

        self.log("validation_loss", loss)
        self.log("validation_perplexity", perplexity)

        return loss

    def configure_optimizers(self):
        opt =  Adafactor(
            self.model.parameters(),
            lr=1e-4,
        )
        # scheduler = SqrtDecayScheduler(opt)
        return [opt]#, [scheduler]
        # return torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def initialize_model(self):
        self.model.train()
        for param in self.model.parameters():
            if param.dim() > 1:
                torch.nn.init.kaiming_uniform_(param)


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms = grad_norm(self.model.layer, norm_type=2)
        # print(norms)
        pass

    def _clamp(self, x: torch.Tensor, minimum=0.0001) -> torch.Tensor:
        return x.clamp(min=torch.Tensor([minimum]).to(x.dtype).cuda())

    
def compute_similarity_matrix(strings):
    """
    Compute a similarity matrix from dissimilarity scores.
    
    Args:
        strings: List of strings
        dissimilarity_func: Function that takes two strings and returns a dissimilarity score
    
    Returns:
        A similarity matrix (kernel matrix)
    """

    def hamming_distance(seq1, seq2):
        """
        Calculate the Hamming distance between two DNA sequences.
        
        Args:
            seq1 (str): First DNA sequence
            seq2 (str): Second DNA sequence
            
        Returns:
            int: Hamming distance (number of differing positions)
            
        Raises:
            ValueError: If sequences have different lengths
        """
        if len(seq1) != len(seq2):
            raise ValueError(f"{len(seq1)} and {len(seq2)} are not equal length")
        
        distance = 0
        for i in range(len(seq1)):
            if seq1[i] != seq2[i]:
                distance += 1
                
        return distance 

    n = len(strings)
    # Initialize similarity matrix
    L = np.zeros((n, n))
    # Compute pairwise dissimilarities and convert to similarities
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i, j] = 1.0  # Self-similarity is 1
            else:
                # Convert dissimilarity to similarity (higher dissimilarity = lower similarity)
                dissimilarity = levenshtein_distance(strings[i], strings[j])
                # Use RBF kernel to convert dissimilarity to similarity
                L[i, j] = np.exp(-dissimilarity)
    
    return L

def elementary_symmetric_polynomial(eigenvalues, k):
    """
    Compute elementary symmetric polynomials (e_k) using recursion.
    
    Args:
        eigenvalues: Array of eigenvalues
        k: Size of the subset
    
    Returns:
        Array of e_0, e_1, ..., e_k
    """
    n = len(eigenvalues)
    E = np.zeros((k + 1, n + 1))
    E[0, :] = 1
    
    for l in range(1, k + 1):
        for n_idx in range(1, n + 1):
            E[l, n_idx] = E[l, n_idx - 1] + eigenvalues[n_idx - 1] * E[l - 1, n_idx - 1]
    
    return E[:, n]

def sample_kdpp(L, k):
    """
    Sample a subset of size k using k-DPP.
    
    Args:
        L: Similarity matrix (kernel matrix)
        k: Size of the subset to sample
        
    Returns:
        List of indices representing the sampled subset
    """
    n = L.shape[0]
    
    # Compute eigendecomposition of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute elementary symmetric polynomials
    E = elementary_symmetric_polynomial(eigenvalues, k)
    
    # Phase 1: Select k eigenvectors
    selected_indices = []
    remaining = list(range(n))
    
    for i in range(k, 0, -1):
        # Compute probabilities
        probs = np.zeros(len(remaining))
        for j, idx in enumerate(remaining):
            # print(E.shape, eigenvalues.shape, probs.shape)
            # breakpoint()
            probs[j] = eigenvalues[idx] * E[i-1] / E[i]
  
        # Normalize probabilities
        probs = probs / np.sum(probs)
        
        # Sample index
        j = np.random.choice(len(remaining), p=probs)
        selected_indices.append(remaining[j])
        remaining.pop(j)
    
    # Phase 2: Convert from eigenvector indices to item indices
    V = eigenvectors[:, selected_indices]
    
    # Orthonormalize V
    Y = np.zeros((n, k))
    for i in range(k):
        Y[:, i] = V[:, i]
        for j in range(i):
            Y[:, i] = Y[:, i] - np.dot(Y[:, i], Y[:, j]) * Y[:, j]
        Y[:, i] = Y[:, i] / np.linalg.norm(Y[:, i])
    
    # Sample items
    selected_set = set()
    remaining_items = list(range(n))
    
    for i in range(k, 0, -1):
        # Compute probabilities
        probs = np.zeros(len(remaining_items))
        for j, idx in enumerate(remaining_items):
            probs[j] = np.sum(Y[idx, :i]**2)
        
        # Normalize probabilities
        probs = probs / np.sum(probs)
        
        # Sample item
        j = np.random.choice(len(remaining_items), p=probs)
        selected_set.add(remaining_items[j])
        
        # Update Y
        e_j = Y[remaining_items[j], :i] / np.linalg.norm(Y[remaining_items[j], :i])
        Y_new = np.zeros((n, i-1))
        
        for l in range(n):
            for m in range(i-1):
                Y_new[l, m] = Y[l, m] - e_j[m] * Y[l, i-1]
        
        Y = Y_new
        remaining_items.pop(j)
    
    return list(selected_set)

def kdpp_select_diverse_seqs(sequences, k):
    """
    Select a diverse subset of sequences using k-DPP.
    
    Args:
        strings: List of strings
        dissimilarity_func: Function that takes two strings and returns a dissimilarity score
        k: Size of the subset to select
        
    Returns:
        List of selected strings
    """
    # Compute similarity matrix
    L = compute_similarity_matrix(sequences)
    # Sample subset
    selected_indices = sample_kdpp(L, k)
    # Return selected strings
    return [sequences[i] for i in selected_indices]

def main():
    parser = argparse.ArgumentParser(description="Train a PromoterModel")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_len", type=int, default=16000, help="Maximum sequence length"
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=1e-2,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="data/proet.ckpt",
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    h =  {
    'OXKSZ': {'ACAGAGTAACTGC', 'CACGCAAGCGACTA', 'GGGCGGGTAGTACCC'},
    'GHXGF': {'AAAATGGTCTGTC', 'AATAAGTGAG', 'CGCGACGCCATAGTT'},
    'XSVNO': {'CCGATCCCGCCCGC', 'GCGGGCCGGCATGCC', 'TTTAGTTGTGTT'},
    'LPTSW': {'ATTGCTGGACA', 'GCAGTGCCAGTTTC', 'GTCATCGTGCACAT'},
    'SAPJM': {'AGAGGTACCC', 'CGGGTGAAATT', 'CTGGTCACGAGTT'},
    'KNGBV': {'AGCACGCACG', 'GAGCGTATCGCAGC'},
    'YTWYS': {'AATGCAACTGGTT', 'ATGAAATTATT', 'GTACAGACCC'},
    'IMXVE': {'TAGTGCAACC', 'TGAGACGGACGAT',},
    'QDKFG': {'CAGGTTTGGCCTGT', 'CCCAGTTACCA', 'GTTTGACTGC'},
    'GDLYH': {'AACGACAAGCATG', 'TACCCGAGTGTAT', 'TGGGATCTCA'},
    'XQIRL': {'CACGTGGCGCCGCTT', 'CCGTTTTATTG', 'GTCCTTACATGCCCC'},
    'BSKUP': {'GAGCCTCTTGCG', 'GAGCGTATCGCAGC'},
    }
    q =  {
    'OXKSZ': 'ACAGAGTAACTGC' ,
    'GHXGF': 'AAAATGGTCTGTC',
    'XSVNO': 'CCGATCCCGCCCGC',
    'LPTSW': 'ATTGCTGGACA',
    'SAPJM': 'CTGGTCACGAGTT',
    'KNGBV': 'CGGGTAGTTGCGAAC',
    'YTWYS': 'GTACAGACCC',
    'IMXVE': 'TAGTGCAACC',
    'QDKFG': 'GTTTGACTGC',
    'GDLYH': 'TGGGATCTCA',
    'XQIRL': 'GTCCTTACATGCCCC',
    'BSKUP': 'TCGACGAATG',
    }

    with open("data/hits.pkl", "rb") as f:
        h = pickle.load(f)
    with open("data/query.pkl", "rb") as f:
        q = pickle.load(f)

    # example, example_query, val_example, val_query = PromoterDataset.train_validation_split( chr= 19, sequences=example, queries=example_query)

    train_seq, train_query, val_seq, val_query  = PromoterDataset._train_validation_split(19, h, q)
    # train_seq, train_query, val_seq, val_query  = PromoterDataset.train_validation_split(19, h, q)


    alphabet = ATCG(
        mask=True, include_gap=True, include_startstop=True, distinct_startstop=True
    )

    model = PromoterModel()
    
    model.initialize_model()

    logger = WandbLogger(project = "poet")
    train_dataset = PromoterDataset(train_seq, train_query, alphabet, args.max_len)
    val_dataset = PromoterDataset(val_seq, val_query, alphabet, args.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.padded_collate_packed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.padded_collate_packed,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='model/',
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
        save_top_k = -1,
        every_n_epochs = 1

    )
    
    trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # accelerator="cpu",
        # devices="auto",
        logger=logger,
        detect_anomaly=True,
        log_every_n_steps=10,
        precision="bf16",
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value"
        strategy="ddp",
        devices=2,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback]
    )
    # uniprot = Uniprot21()
    # jit_warmup(model, uniprot)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders= val_loader,
    )
    trainer.save_checkpoint("0.3_human_random.ckpt")
   

if __name__ == "__main__":
    main()
