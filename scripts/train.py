import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as lightning
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_optimizer import Adafactor

from poet.alphabets import Alphabet, Uniprot21
from poet.models.poet import PoET

IGNORE_INDEX = -100


class ATCG(Alphabet):
    """tokenize DNA sequence like this?"""

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
    """Still need to figure out how each family is organized"""

    def __init__(self, sequences: dict, alphabet, max_length):
        self.alphabet = alphabet
        self.sequences = sequences
        self.num_special_characters = 4
        self.ids = list(sequences.keys())
        self.max_len = max_length

    def __getitem__(self, idx):
        id = self.ids[idx]
        return self.pack_inputs(id)

    def sample_and_fit_sequences(
        self,
        query_id_or_sequence: Optional[
            str
        ] = None,  # Changed parameter name to be more explicit
        member_ids: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        fasta_dataset: Optional[Dict[str, str]] = None,
        max_seq_length: int = 1024,
        include_query: bool = True,
        sample: bool = True,
        truncate: bool = False,
        chunk_size: int = 10,
        max_skips: int = 2,
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

        # Retrieve and possibly truncate the query
        query_sequence = (
            self.get_sequence(query_id_or_sequence, fasta_dataset)
            if include_query
            else None
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

        if member_ids and len(member_ids) > 0:
            if sample:
                member_weights = np.array(
                    weights
                    if weights is not None
                    else [1.0 / len(member_ids)] * len(member_ids)
                )
                member_weights /= member_weights.sum()

                skip_rounds = 0
                while leftover > 0 and skip_rounds < max_skips:
                    sampled_ids = np.random.choice(
                        member_ids, size=chunk_size, replace=True, p=member_weights
                    )
                    added = False

                    for mid in sampled_ids:
                        seq = self.get_sequence(mid, fasta_dataset)
                        if not seq:
                            continue

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
            else:
                # Sequential processing with length limits
                for mid in member_ids:
                    seq = self.get_sequence(mid, fasta_dataset)
                    if not seq:
                        continue

                    seq_len = len(seq) + self.num_special_characters
                    if seq_len <= leftover:
                        passages.append(seq)
                        passage_lengths.append(seq_len)
                        leftover -= seq_len
                        total_tokens += seq_len
                    elif truncate and leftover > 0:
                        seq = seq[: leftover - self.num_special_characters]
                        passages.append(seq)
                        passage_lengths.append(leftover)
                        total_tokens += leftover
                        leftover = 0
                        break

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
        sample: str,
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
        """
    [previous docstring remains the same]
    """

        def tokenize_sequence(text: str, reverse_sequence: bool) -> np.ndarray:
            if reverse_sequence:
                text = text[::-1]
            if isinstance(text, str):
                text_bytes = text.encode("ascii")
            else:
                text_bytes = text  # Already bytes
            tokens = self.alphabet.encode(text_bytes)
            if reverse_sequence:
                return np.concatenate(
                    [[self.alpahbet.stop_token], tokens, [self.alphabet.start_token]]
                )
            else:
                return np.concatenate(
                    [[self.alphabet.start_token], tokens, [self.alphabet.stop_token]]
                )

        def truncate_sequence(seq: str, max_len: int) -> str:
            """Helper to truncate a sequence to max length while accounting for special tokens"""
            if len(seq) + self.num_special_characters > max_len:
                return seq[: max_len - self.num_special_characters]
            return seq

        # Initialize sequences and lengths
        all_sequences = []
        all_lengths = []

        for query_sequence in self.sequences[sample]:
            tokens = tokenize_sequence(query_sequence, reverse_sequence)
            all_sequences.append(tokens)
            all_lengths.append(len(tokens))

        # Concatenate all sequences
        if all_sequences:
            tokens = np.concatenate(all_sequences)
            tokens = torch.from_numpy(tokens).long()
        else:
            tokens = torch.empty(0, dtype=torch.long)

        # Add padding if needed
        current_length = sum(all_lengths)
        if not skip_padding and current_length < self.max_len:
            padding_length = self.max_len - current_length
            all_lengths.append(padding_length)
            padding = torch.full((padding_length,), padding_idx, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding])

        # # Create labels (same as tokens but with padding masked)
        # labels = tokens.clone()

        return {
            "tokens": tokens,
            "seq_lens": all_lengths,
            "label": sample,
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
        # padded_labels = []
        for x in batch:
            token = x["tokens"]
            label = x["label"]
            pad_len = target_length - token.size(0)
            if pad_len > 0:
                token = F.pad(token, (0, pad_len), value=padding_idx)
                label = F.pad(label, (0, pad_len), value=padding_idx)
            padded_tokens.append(token)
            # padded_labels.append(torch.Tensor([label]))
        tokens_tensor = torch.stack(padded_tokens, dim=0)
        # labels_tensor = torch.stack(padded_labels, dim=0)

        # Collate segment sizes (each sample may have several segments).
        max_segments = max(len(x["seq_lens"]) for x in batch)
        bsz = len(batch)
        segment_sizes = torch.zeros((bsz, max_segments), dtype=torch.int32)
        for i, x in enumerate(batch):
            seq_lens = torch.tensor(x["seq_lens"], dtype=torch.int32)
            segment_sizes[i, : len(seq_lens)] = seq_lens

        collated_batch = {
            "tokens": tokens_tensor,
            # "labels": labels_tensor,
            "segment_sizes": segment_sizes,
        }
        # collated_batch = self._to_cuda(collated_batch)
        return collated_batch

    def __len__(self):
        return len(self.sequences)

    def _to_cuda(self, tok_dict):
        return {k: v.to("cuda") for k, v in tok_dict.items()}


""" no idea what this does, but the poet people use it to warm up the model
"""


def jit_warmup(embedding_model: PoET, alphabet: Uniprot21):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().cuda()
    segment_sizes = torch.tensor(segment_sizes).long().cuda()
    _ = embedding_model(x.unsqueeze(0), segment_sizes.unsqueeze(0))


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
        self.model = PoET(**init).cuda().half()

    def forward(self, xs: torch.Tensor, segment_sizes: torch.Tensor) -> torch.Tensor:
        return self.model(xs, segment_sizes)

    def training_step(self, batch, batch_idx):
        xs = batch["tokens"]
        segment_sizes = batch["segment_sizes"]
        logits = self(xs, segment_sizes)

        # Calculate loss (assuming next token prediction)
        targets = xs[:, 1:].contiguous()  # Shift targets by 1
        logits = logits[:, :-1, :].contiguous()  # Remove last logit
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adafactor(
            self.model.parameters(),
            lr=1e-2,
        )


def main():
    parser = argparse.ArgumentParser(description="Train a PromoterModel")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_len", type=int, default=200, help="Maximum sequence length"
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

    example = {"BRCA": {"ATTC", "ATCG"}, "A19": {"ATTC", "ATGG"}}

    alphabet = ATCG(
        mask=True, include_gap=True, include_startstop=True, distinct_startstop=True
    )

    model = PromoterModel()

    dataset = PromoterDataset(example, alphabet, args.max_len)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.padded_collate_packed,
    )

    trainer = lightning.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )

    jit_warmup(model, alphabet)
    trainer.fit(
        model,
        train_loader,
    )
    trainer.save_checkpoint(args.ckpt_path)


if __name__ == "__main__":
    main()
