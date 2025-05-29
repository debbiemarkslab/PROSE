import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import lightning
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_optimizer import Adafactor
import pickle
from poet.alphabets import Alphabet
from poet.models.poet import PoET
# from bert import DNA
import random

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.grads import grad_norm
from lightning.pytorch.callbacks import ModelCheckpoint
from collections import defaultdict

IGNORE_INDEX = -100
# IGNORE_INDEX = 3


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

    def __init__(self, sequences: list, alphabet, max_length):
        self.alphabet = alphabet
        self.sequences = sequences
        self.max_len = max_length
        self.iters = len(self.sequences) * 5
        self.num_special_characters = 2



    def __getitem__(self, idx):
        """
        randomly sample a gene with replacement
        weighted by the number of sequences the gene has
        """
       
        
        sampled_set = self.sample_and_fit_sequences(
            sequence = random.choice(self.sequences),
            max_seq_length=self.max_len,
        )
        
        # sampled_set = self.diverse_sample_and_fit_sequences(id, max_seq_length=self.max_len)
        return self.pack_inputs(sampled_set, 
                                self.alphabet, 
                                self.max_len, 
                                reverse_sequence = np.random.rand() > 0.5) # Prob(reverse) = 0.5

    def get_inference_seqs(self, variant: str, id: str):
        try:
            sampled_set = self.greedy_sample_and_fit_sequences(
                id, sequence=variant, max_seq_length=self.max_len
            )
        except KeyError:
            return []
        return sampled_set["passages"]

    def sample_and_fit_sequences(
        self,
        query_id_or_sequence: Optional[
            str
        ] = None,  # Changed parameter name to be more explicit
        weights: Optional[List[float]] = None,
        max_seq_length: int = 1024,
        max_individual_seq_length: Optional[int] = None,
        include_query: bool = True,
        sample: bool = True,
        truncate: bool = True,
        chunk_size: int = 1,
        max_skips: int = 2,
        sequence = None,
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
                return seq[: max_len - self.num_special_characters]
            return seq

        if sequence is not None:
            query_sequence = sequence
        else:
            query_sequence = self.sample_query(
                query_id_or_sequence, 
                p_human = 0.05 # Prob(human sequence) 
            )  
        # print(query_sequence)
        # breakpoint()
        # Apply individual sequence length limit if specified
        if max_individual_seq_length and query_sequence:
            query_sequence = truncate_sequence(
                query_sequence, max_individual_seq_length
            )

        query_length = (
            len(query_sequence) + self.num_special_characters
            if query_sequence
            else None
        )

        # Calculate effective length for passages
        effective_length = max_seq_length - (
            query_length if query_length is not None else 0
        )

        passages = []
        passage_lengths = []
        total_tokens = query_length if query_length is not None else 0
        leftover = effective_length

        if sample:
          
            skip_rounds = 0
            while leftover > 0 and skip_rounds < max_skips:

                seq = query_sequence

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
            "total_tokens": total_tokens,
        }
    
    def pack_inputs(
        self,
        sample,
        tokenizer,  # DNA tokenizer instance
        max_seq_length,
        padding_idx=IGNORE_INDEX,
        reverse_sequence=False,
        skip_padding=True,
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

        def tokenize_sequence(text: str, reverse_sequence: bool) -> np.ndarray:
            
            if reverse_sequence:
                text = text[::-1]
            if isinstance(text, str):
                text_bytes = text.encode("ascii")
            else:
                text_bytes = text  # Already bytes
            tokens = tokenizer.encode(text_bytes)
            if tokens.ndim == 0:
                tokens = tokens.unsqueeze(0)
            if reverse_sequence:
                # return torch.cat(
                #     [torch.Tensor([tokenizer.stop_token]), tokens, torch.Tensor([tokenizer.start_token])]
                # )
                return np.concatenate(
                    [[tokenizer.stop_token], tokens, [tokenizer.start_token]]
                )
            else:
                # return torch.cat(
                #     [torch.Tensor([tokenizer.start_token]), tokens, torch.tensor([tokenizer.stop_token])]
                # )
                return np.concatenate(
                    [[tokenizer.start_token], tokens, [tokenizer.stop_token]]
                )

        # Initialize sequences and lengths
        all_sequences = []
        all_lengths = []
        all_lengths = sample[
            "passage_lengths"
        ].copy()  # Use pre-calculated passage lengths

        # Process passages
        if sample["passages"]:
            for seq in sample["passages"]:
                if seq is not None:
                    tokens = tokenize_sequence(seq, reverse_sequence)
                    all_sequences.append(tokens)
                    # all_lengths.append(len(tokens))

        # Process query if it exists
        query_sequence = sample["sequence"]
        if query_sequence is not None:
            tokens = tokenize_sequence(query_sequence, reverse_sequence)
            all_sequences.append(tokens)
            # Use pre-calculated query length
            all_lengths.append(sample["query_length"])
            # all_lengths.append(len(tokens))


        # Concatenate all sequences
        if all_sequences:
            # tokens = torch.cat(all_sequences).long()
            tokens = torch.from_numpy(tokens).long()
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
            "labels": tokens.clone()
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
        padded_labels = []

        for x in batch:
            token = x["tokens"]
            label = x["labels"]
            pad_len = target_length - token.size(0)
            if pad_len > 0:
                token = F.pad(token, (0, pad_len), value=padding_idx)
                label = F.pad(label, (0, pad_len), value=padding_idx)
            padded_labels.append(label)
            padded_tokens.append(token)
        tokens_tensor = torch.stack(padded_tokens, dim=0)
        labels_tensor = torch.stack(padded_labels, dim=0)

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
            "labels": labels_tensor,
        }

        collated_batch = self._to_cuda(collated_batch)
        return collated_batch

    def __len__(self):
        return self.iters

    def _to_cuda(self, tok_dict):
        return {k: v.long().cuda() for k, v in tok_dict.items()}

    @staticmethod
    def train_validation_split(
        chr: int, sequences: dict, queries: dict
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        return sequences, queries, sequences, queries

    @staticmethod
    def _train_validation_split(
        chr: int, sequences: dict, queries: dict
    ) -> Tuple[Dict, Dict, Dict, Dict]:
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


class PromoterModel(lightning.LightningModule):
    def __init__(self, config = None):
        super().__init__()
        # ckpt = torch.load(ckpt_path)
        small = {
            "n_vocab": 8,
            "hidden_dim": 768,
            "num_layers": 6,
            "nhead": 12,
            "dropout": 0,
            "use_multi_rotary": True,
            "norm": True,
        }

        big = {
        "n_vocab": 8,
        "hidden_dim": 1024,
        "num_layers": 12,
        "nhead": 16,
        "dropout": 0.1,
        "use_multi_rotary": True,
        "norm": True
        }
        if config is None:
            config = small
        self.model = PoET(**config).cuda()
        # print(self.model.n_vocab)

    def forward(self, xs: torch.Tensor, segment_sizes: torch.Tensor) -> torch.Tensor:
        if torch.isnan(xs).any() or torch.isnan(segment_sizes).any():
            raise Exception("NAN in INPUT!!")

        # print(xs.shape, segment_sizes.shape)
        return self.model(xs, segment_sizes)

    def training_step(self, batch, batch_idx):
        
        xs = batch["tokens"]
        # print(xs.max(), xs.min())
        # breakpoint()
        segment_sizes = batch["segment_sizes"]
        label = batch["labels"]
        logits = self.model(xs, segment_sizes)

        # Calculate loss (next token prediction)
        targets = label[:, 1:].contiguous()  # Shift targets by 1
        logits = logits[:, :-1, :].contiguous()  # Remove last logit

        query_positions = self.compute_last_segment_positions(segment_sizes)
        query_loss = self.calculate_individual_losses(
            logits, targets, query_positions=query_positions
        )

        sequence_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
            ignore_index=IGNORE_INDEX,
        )

        # Handle any NaN or Inf values
        if not torch.isfinite(sequence_loss).all():
            print("NaN occurred, loss: ", sequence_loss)
            sequence_loss = torch.where(
                torch.isfinite(sequence_loss),
                sequence_loss,
                torch.tensor(0.0, device=sequence_loss.device),
            )
            self.zero_grad()

        perplexity = torch.exp(sequence_loss)
        self.log("train_loss", sequence_loss)
        self.log("train_perplexity", perplexity)
        self.log("train_query_loss", query_loss.mean())

        return sequence_loss

    def validation_step(self, batch, batch_idx):
        xs = batch["tokens"]
        # print(xs.max(), xs.min())
        # breakpoint()
        segment_sizes = batch["segment_sizes"]
        label = batch["labels"]

        logits = self(xs, segment_sizes)

        # Calculate loss (next token prediction)
        targets = label[:, 1:].contiguous()  # Shift targets by 1
        logits = logits[:, :-1, :].contiguous()  # Remove last logit

        query_positions = self.compute_last_segment_positions(segment_sizes)
        query_loss = self.calculate_individual_losses(
            logits, targets, query_positions=query_positions
        )

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        perplexity = torch.exp(loss)

        self.log("validation_loss", loss)
        self.log("validation_perplexity", perplexity)
        self.log("validation_query_loss", query_loss.mean())

        return loss

    def configure_optimizers(self):
        opt = Adafactor(
            self.model.parameters(),
            lr=1e-4,
        )
        return opt

    def calculate_individual_losses(
        self, shift_logits, shift_labels, padding_idx=IGNORE_INDEX, query_positions=None
    ):
        """
        Calculate loss for each sequence in the batch, exactly matching HuggingFace's implementation.
        Not calculating log of outputs
        """
        batch_size, _, vocab_size = shift_logits.size()

        # Flatten the tokens
        flat_shift_logits = shift_logits.view(-1, vocab_size)
        flat_shift_labels = shift_labels.view(-1)

        # Get per-token losses (with reduction='none' to get per-token values)
        token_losses = F.cross_entropy(
            flat_shift_logits,
            flat_shift_labels,
            reduction="none",
            ignore_index=padding_idx,
        )
        token_losses = token_losses.view(batch_size, -1)
        # Create mask for valid positions (non-padding)
        mask = (shift_labels != padding_idx).float()

        # If query positions provided, only calculate loss for query tokens
        if query_positions is not None:
            query_mask = torch.zeros_like(mask)
            for i, (start, end) in enumerate(query_positions):
                # Adjust positions to account for the shift
                query_mask[i, (start - 1) : (end - 1)] = 1.0
            mask = mask * query_mask

        # Calculate average loss per sequence
        seq_lengths = mask.sum(dim=1).clamp(min=1)
        individual_losses = (token_losses * mask).sum(dim=1) / seq_lengths

        return individual_losses

    def compute_last_segment_positions(self, segment_sizes):
        """
        Computes the start and end positions of the last valid segment for each item in the batch.

        Args:
            segment_sizes: Tensor of shape [batch_size, max_num_segments] containing the sizes
                        of segments for each batch item. Padded with zeros for shorter sequences.

        Returns:
            List of tuples (start_pos, end_pos) for each batch item
        """
        batch_size = segment_sizes.size(0)
        valid_segments_mask = segment_sizes > 0

        last_valid_segment_idx = (
            torch.sum(valid_segments_mask, dim=1) - 1
        )  # [batch_size]

        query_positions = []

        for i in range(batch_size):
            # Get the index of the last valid segment
            last_idx = last_valid_segment_idx[i].item()

            # Calculate start position by summing all previous segment sizes
            start_pos = torch.sum(segment_sizes[i, :last_idx]).item()

            # Calculate end position by adding the size of the last segment
            end_pos = start_pos + segment_sizes[i, last_idx].item()
            query_positions.append((start_pos, end_pos))

        return query_positions


def main():
    parser = argparse.ArgumentParser(description="Train a PromoterModel")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=12, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_len", type=int, default=32768, help="Maximum sequence length"
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="data/proet.ckpt",
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    # with open("data/hits.pkl", "rb") as f:
    #     h = pickle.load(f)
    # with open("data/query.pkl", "rb") as f:
    #     q = pickle.load(f)   

    # train_seq, train_query, val_seq, val_query = (
    #     PromoterDataset._train_validation_split(19, h, q)
    # )

    alphabet = ATCG(
        mask=True, include_gap=True, include_startstop=True, distinct_startstop=True
    )
    
    # alphabet = DNA(mask=True)
    cfg = {
            "n_vocab": 8,
            "hidden_dim": 768,
            "num_layers": 6,
            "nhead": 12,
            "dropout": 0,
            "use_multi_rotary": True,
            "norm": True,
        }
    
    model = PromoterModel(cfg)

    logger = WandbLogger(project="poet")

    def generate_dna_sequence(length):
        """
        Generate a random DNA sequence of the specified length.
        
        Parameters:
        length (int): The desired length of the DNA sequence
        
        Returns:
        str: A random DNA sequence consisting of A, T, G, and C
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Length must be a positive integer")
        
        # DNA nucleotide bases
        bases = ['A', 'T', 'G', 'C']
        
        # Generate random sequence
        dna_sequence = ''.join(random.choice(bases) for _ in range(length))
        
        return dna_sequence


    seqs = [generate_dna_sequence(1000) for i in range(2000)]
    with open("data/random.pkl", "wb") as fp:   
        pickle.dump(seqs, fp)

    train_dataset = PromoterDataset(seqs, alphabet, args.max_len)
    # val_dataset = PromoterDataset(val_seq, val_query, alphabet, args.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.padded_collate_packed,
    )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=val_dataset.padded_collate_packed,
    # )

    checkpoint_callback = ModelCheckpoint(
        dirpath="model/",
        filename="sanity_0.1_human_small_random_0.1_dropout_small_{epoch}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        detect_anomaly=True,
        log_every_n_steps=10,
        precision="bf16",
        strategy="ddp",
        devices=2,
        # val_check_interval=0.25,
        callbacks=[checkpoint_callback],
    )
 

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        # val_dataloaders=val_loader,
    )
    # with open('logs/bert_train_distribution.pkl', 'wb') as file:
    #     pickle.dump(train_dataset.counter, file)

if __name__ == "__main__":
    main()
