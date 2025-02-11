import copy
import logging
import math
import time
from functools import reduce
from typing import List, Optional, Dict, Any, Sequence

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from pathlib import Path
import random
import tqdm

from transformers import Cache, StaticCache, DynamicCache

from protriever.auxiliary_tasks import ContactPredictionHead
from protriever import dist_utils
from protriever.custom_softmax import create_group_tensor, grouped_softmax_scatter, grouped_log_softmax, group_logsumexp
from protriever.index import DistributedFAISSIndex
from protriever.training_sampler import UniformSampler
from protriever.packing_util import packed_block_causal_mask, packed_cross_block_causal_mask, packed_prefix_lm_mask
from enum import Enum, auto

class MaskType(Enum):
    """Types of masking for attention."""
    BLOCK_CAUSAL = auto()  # Original block causal masking
    CROSS_BLOCK_CAUSAL = auto()  # Cross-document block causal masking
    CAUSAL = auto()  # Simple causal masking
    PREFIX_LM = auto()



logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

class Protriever_poet(nn.Module):
    def __init__(self, opt, reader, retriever, reader_tokenizers, retriever_tokenizer, contact_prediction=False):
        super(Protriever_poet, self).__init__()
        self.fabric = None  # Initialize fabric attribute

        self.reader = reader
        self.retriever = retriever
        # if opt.compile:
        #     # compile_config = {
        #     #     "dynamic": True,  # Allow dynamic shapes
        #     #     "fullgraph": False,
        #     #     "mode": "reduce-overhead"
        #     # }
        #     compile_config = {}
        #     # self.reader.encoder = torch.compile(self.reader.encoder)
        #     # self.reader.decoder = torch.compile(self.reader.decoder)
        #     self.reader = torch.compile(self.reader, **compile_config)
        
        
        _, self.reader_decoder_tokenizer = reader_tokenizers
        self.retriever_tokenizer = retriever_tokenizer
        self.opt = opt
        self.ESM_MAX_SEQ_LENGTH = 1024  # For protein sequences
        self.max_seq_context = opt.max_seq_context
        self.num_special_characters = 2 #TODO: this is a hack to get the number of special characters in the tokenizer
        

        if opt.training_sampler == "uniform":
            self.training_sampler = UniformSampler(opt.n_context)
        else:
            self.training_sampler = None

        if contact_prediction:
            self.contact_head = self.build_contact_head()
        # logger.info(f"Contact head built with {self.contact_head.num_features} features")

    def build_contact_head(self) -> ContactPredictionHead:
        cfg = self.reader.decoder.config
        # print(cfg.num_hidden_layers)
        # print(cfg.num_attention_heads)
        logger.info(f"Building contact head with {cfg.num_hidden_layers * cfg.num_attention_heads} features")
        contact_head = ContactPredictionHead(
            cfg.num_hidden_layers * cfg.num_attention_heads,
            eos_idx=self.reader_decoder_tokenizer.eos_token_id,
        )
        contact_head.requires_grad_(False)
        return contact_head
    
    # def _get_fp16_retriever_copy(self):
    #     if hasattr(self.retriever, "module"):
    #         retriever_to_copy = self.retriever.module
    #     else:
    #         retriever_to_copy = self.retriever
    #     logger.info(f"Copying retriever to half precision for retriever: {retriever_to_copy}")
    #     return copy.deepcopy(retriever_to_copy).half().eval()
        
    def _get_fp16_retriever_copy(self):
        if hasattr(self.retriever, "module"):
            retriever_to_copy = self.retriever.module
        else:
            retriever_to_copy = self.retriever

        # Save the original precision and mode
        original_dtype = next(retriever_to_copy.parameters()).dtype
        original_mode = retriever_to_copy.training

        # Switch to half precision and eval mode
        # logger.info(f"Switching retriever to half precision and eval mode for retriever: {retriever_to_copy}")
        retriever_to_copy.to(torch.bfloat16).eval()

        return retriever_to_copy, original_dtype, original_mode



    @torch.no_grad()
    def build_index(self, index: DistributedFAISSIndex, passages: list[str], gpu_embedder_batch_size: int, logger=None, step: int = None, from_scratch: bool = False):
        # Will always override embeddings in memory (if present). Shape [n_passages, emb_dim]
        index.init_embeddings(passages)
        n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
        # retrieverfp16 = self._get_fp16_retriever_copy()
        retriever_to_copy, original_dtype, original_mode = self._get_fp16_retriever_copy()
        logger.info(f"Retriever now in half precision")
        total = 0
        for i in range(n_batch):
            batch = passages[
                i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size
            ]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, gpu_embedder_batch_size),
                truncation=True,
            )

            embeddings = retriever_to_copy(**_to_cuda(batch_enc, self.fabric), is_passages=True)
            index.embeddings[total : total + len(embeddings)] = embeddings
            total += len(embeddings)


            if i % 5 == 0 and i > 0: # should be 500
                mem_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"Number of passages encoded: {total} | Memory: {mem_allocated_gb:.3f} GB")

        
        logger.info(f"Reverting retriever back to original precision: {original_dtype} and mode: {'train' if original_mode else 'eval'}...")
        retriever_to_copy.to(original_dtype)
        if original_mode:
            retriever_to_copy.train()
        else:
            retriever_to_copy.eval()
        logger.info(f"Retriever is now back to {original_dtype} and {'train' if original_mode else 'eval'} mode.")


        # Barrier appears to cause crashes?
        # dist_utils.barrier()
        final_mem_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"{total} passages encoded on process: {dist_utils.get_rank()} | Memory: {final_mem_allocated_gb:.3f} GB")
        
        # Define embeddings_path
        if not from_scratch:
            initial_embeddings_path = Path(self.opt.embeddings_path)
            initial_index_path = Path(self.opt.save_index_path)
            embeddings_path = initial_embeddings_path / f"step_{step}"
            index_path = initial_index_path / f"step_{step}"
        else:
            embeddings_path = Path(self.opt.initial_embeddings_path)
            index_path = Path(self.opt.initial_embeddings_path)
        # Save embeddings immediately after generation in case of OOM issues.
        logger.info(f"Saving embeddings to {embeddings_path}, and index to {index_path}")
        index.train_index()
        index.save_index(
            index_path=index_path,
            total_saved_shards=self.opt.save_index_n_shards,
            save_faiss_index=index_path,
            save_embeddings=True,
            save_passages=False,
            embeddings_path=embeddings_path,
            passages_path=self.opt.passages_path,
        )
        index.clear_embeddings()
        
        # Newly generated embeddings are non-normalized and type float16.
        # Transform to float32 and normalize if necessary before training index.
        # index.embeddings = index._cast_to_torch32(index.embeddings)
        # if index.cosine_similarity:
        #     index.embeddings = F.normalize(embeddings)
        
        # logger.info(f"Index embeddings shape: {index.embeddings.shape} of type {index.embeddings.dtype}")
        # logger.info("Building FAISS index...")
        
        # index.save_faiss_index(path=index_path)
        # logger.info(f"FAISS index saved to {index_path}")


    def apply_distance_mask(self, passages, scores, distance_mask):
        filtered_passages = [p[m] for p, m in zip(passages, distance_mask)]
        filtered_scores = [s[m] for s, m in zip(scores, distance_mask)]
        return filtered_passages, filtered_scores

    def sample_uniformly(self, passages, scores, topk):
        # Check if there are enough passages to sample
        if len(passages) > topk:
            # Generate random indices for sampling without replacement
            sampled_indices = np.random.choice(len(passages), size=topk, replace=False)
            
            # Select passages and their corresponding scores using the indices
            sampled_passages = [passages[idx] for idx in sampled_indices]
            sampled_scores = [scores[idx] for idx in sampled_indices]
        else:
            # If there are not enough passages to sample from, take them all
            sampled_passages, sampled_scores = passages, scores
        
        return sampled_passages, sampled_scores

    def sample_by_distance(self, passages, scores, topk):
        # Ensure scores are a flat numpy array
        scores = np.array(scores)

        # Convert scores to negative to assume lower scores are better (closer)
        distances = -scores
        probabilities = torch.softmax(torch.tensor(distances), dim=0).numpy()

        # Sample indices based on calculated probabilities, with replacement
        sampled_indices = np.random.choice(len(passages), size=topk, replace=True, p=probabilities)

        # Select passages and scores based on sampled indices
        sampled_passages = [passages[idx] for idx in sampled_indices]
        sampled_scores = [scores[idx] for idx in sampled_indices]

        return sampled_passages, sampled_scores

    @torch.no_grad()
    def _retrieve(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
    ):
        self.retriever.eval()
        if len(query) > 0:
            query_emb = self.retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
        else:
            query_emb = torch.empty((0, self.retriever.config.hidden_size)).cuda()

        if self.training:
            self.retriever.train()

        search_start = time.time()


        # Determine the number of passages to retrieve
        # retrieve_ratio = self.opt.filtering_overretrieve_ratio # 1 by default, not sure what to do with this
        # Retrieve passages and scores
        
        
        #TODO: here is where we are going to add the spiking selected indices. These  are store
        topk_retrieval_search = topk + 1 if filtering_fun is not None else topk

        # Apply sampling if necessary
        if self.opt.sampling_distance_bound:
            distance_mask = scores <= self.opt.sampling_distance_bound
            passages, scores = self.apply_distance_mask(passages, scores, distance_mask)
        elif self.opt.retrieve_sampling_method == 'uniform':
            passages, scores = self.sample_uniformly(passages, scores, topk_retrieval_search)
        elif self.opt.retrieve_sampling_method == 'distance':
            passages, scores = self.sample_by_distance(passages, scores, topk_retrieval_search)
        else:
            passages, scores = index.search_knn(query_emb, topk_retrieval_search)
        
        
        # Apply filtering function
        if filtering_fun is not None:
            passages, scores = filtering_fun(batch_metadata, passages, scores, topk, training=self.training)

        # logger.info(f"passages in retrieve function are {passages}")
        iter_stats["runtime/search"] = (time.time() - search_start, 1)

        return passages, scores, query_emb


    # @torch.no_grad()
    # def retrieve_with_rerank(
    #     self,
    #     index,
    #     topk,
    #     query,
    #     query_ids_retriever,
    #     query_mask_retriever,
    #     batch_metadata=None,
    #     filtering_fun=None,
    #     iter_stats={},
    #     fasta_dataset=None  # Add fasta_dataset parameter
    # ):
    #     bsz = len(query)
    #     to_rerank = self.opt.n_to_rerank_with_retrieve_with_rerank

    #     # First, do the retrieval 
    #     passages, _, query_emb = self._retrieve(
    #         index,
    #         to_rerank,
    #         query,
    #         query_ids_retriever,
    #         query_mask_retriever,
    #         batch_metadata,
    #         filtering_fun,
    #         iter_stats,
    #     )

    #     # print("DEBUG: Retrieved passages:", passages[:2]) # Show first 2 passages

    #     retrieverfp16, original_dtype, original_mode = self._get_fp16_retriever_copy()
        
    #     # Convert passage IDs to sequences
    #     flat_passage_strings = []
    #     for passage_group in passages:
    #         sequences = []
    #         for p in passage_group:
    #             seq = self.get_sequence(p['id'], fasta_dataset)
    #             # print("recovered sequence is", seq)
    #             # print(f"DEBUG: Converting ID {p['id']} to sequence of length {len(seq) if seq else 'None'}, seq is {seq[:10] if seq else 'None'}")
    #             sequences.append(seq)
    #         flat_passage_strings.extend(sequences)

    #     # print("DEBUG: First few flat passage strings:", flat_passage_strings[:2])
        
    #     encoder_batch_size = min(len(flat_passage_strings), self.opt.per_gpu_embedder_batch_size)
    #     passage_emb, output_passages, output_scores = (
    #         query_emb.new_zeros(len(flat_passage_strings), query_emb.shape[-1]),
    #         [],
    #         [],
    #     )

    #     for b in range(0, len(flat_passage_strings), encoder_batch_size):
    #         batch = flat_passage_strings[b : b + encoder_batch_size]
    #         batch_enc = self.retriever_tokenizer(
    #             batch,
    #             padding="longest",
    #             return_tensors="pt",
    #             max_length=min(self.opt.text_maxlength, self.ESM_MAX_SEQ_LENGTH), 
    #             truncation=True,
    #         )
    #         batch_emb = retrieverfp16(**_to_cuda(batch_enc, self.fabric), is_passages=True).to(query_emb)
    #         passage_emb[b : b + encoder_batch_size] = batch_emb

    #     passage_emb = passage_emb.view(bsz, to_rerank, -1)
    #     retriever_scores = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
    #     top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, topk, dim=1)

    #     for i in range(bsz):
    #         output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
    #         output_scores.append(top_retriever_scores[i].tolist())
            
    #     # Revert retriever back to original state
    #     retrieverfp16.to(original_dtype)
    #     if original_mode:
    #         retrieverfp16.train()
    #     else:
    #         retrieverfp16.eval()
            
    #     return output_passages, output_scores
    
    @torch.no_grad()
    def retrieve_with_rerank(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
        fasta_dataset=None
    ):
        """
        Retrieves and reranks passages, handling multiple UniRef100 sequences per passage.
        For each initial passage, we may get multiple different UniRef100 sequences from its cluster.
        Each sequence is scored independently against its query to find the best matches.
        
        Args:
            index: FAISS index for initial retrieval
            topk: Number of top passages to return per query
            query: Batch of queries
            query_ids_retriever, query_mask_retriever: Query tokenization for retriever
            batch_metadata: Optional metadata for filtering
            filtering_fun: Optional filtering function
            iter_stats: Dictionary to track statistics
            fasta_dataset: Dataset instance for sequence lookup
        
        Returns:
            tuple: (output_passages, output_scores) where:
                - output_passages is a list of lists of passage dicts (bsz, topk)
                - output_scores is a list of lists of scores (bsz, topk)
        """
        bsz = len(query)
        to_rerank = self.opt.n_to_rerank_with_retrieve_with_rerank

        # Initial retrieval step gets candidates to rerank
        passages, _, query_emb = self._retrieve(
            index,
            to_rerank,
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata,
            filtering_fun,
            iter_stats,
        )

        # Number of UniRef100 sequences to sample per passage
        num_uni100_requested = getattr(self.opt, "rerank_multiple_uni100", 1)
        
        # We'll maintain flat lists for efficiency and use mapping arrays to track relationships
        all_sequences = []  # All UniRef100 sequences concatenated
        passage_to_query_idx = []  # Maps each sequence back to its query index
        passage_to_orig_passage = []  # Maps each sequence back to original passage dict
        
        # Process all passages to get their UniRef100 sequences
        for query_idx, passage_group in enumerate(passages):
            for passage in passage_group:
                # Get multiple sequences if requested, otherwise just one
                if num_uni100_requested > 1:
                    sequences = fasta_dataset.get_multiple(passage['id'], num=num_uni100_requested)
                else:
                    sequences = [self.get_sequence(passage['id'], fasta_dataset)]
                
                # Extend our tracking arrays
                num_seqs = len(sequences)
                all_sequences.extend(sequences)
                passage_to_query_idx.extend([query_idx] * num_seqs)
                passage_to_orig_passage.extend([passage] * num_seqs)
        
        # Handle case where no sequences were found
        if not all_sequences:
            return [[]] * bsz, [[]] * bsz
        
        # Set up retriever in eval and fp16 mode for efficiency
        retriever_fp16, original_dtype, original_mode = self._get_fp16_retriever_copy()
        
        # Embed all sequences in batches for memory efficiency
        encoder_batch_size = min(len(all_sequences), self.opt.per_gpu_embedder_batch_size)
        passage_emb = query_emb.new_zeros(len(all_sequences), query_emb.shape[-1])
        
        for b in range(0, len(all_sequences), encoder_batch_size):
            batch = all_sequences[b : b + encoder_batch_size]
            # Tokenize batch
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, self.ESM_MAX_SEQ_LENGTH),
                truncation=True,
            )
            # Get embeddings
            batch_emb = retriever_fp16(**_to_cuda(batch_enc, self.fabric), is_passages=True).to(query_emb)
            passage_emb[b : b + encoder_batch_size] = batch_emb
        
        # Convert mapping to tensor for efficient indexing
        passage_to_query_idx = torch.tensor(passage_to_query_idx, device=query_emb.device)
        
        # For each sequence, get its corresponding query embedding
        # This is more efficient than looping through queries
        query_emb_expanded = query_emb[passage_to_query_idx]  # Shape: [num_total_sequences, hidden_dim]
        
        # Compute similarity scores between queries and their passages using scaled dot product
        scores = torch.einsum("id,id->i", [query_emb_expanded, passage_emb])  # Shape: [num_total_sequences]
        scores = scores / math.sqrt(query_emb.size(-1))  # Scale by sqrt of embedding dimension
        
        # Process results for each query
        output_passages = []
        output_scores = []
        
        for i in range(bsz):
            # Get mask for all sequences belonging to this query
            query_mask = passage_to_query_idx == i
            if not query_mask.any():
                # No sequences for this query
                output_passages.append([])
                output_scores.append([])
                continue
                
            # Get scores and passages for this query
            query_scores = scores[query_mask]
            query_passages = [passage_to_orig_passage[j] for j, is_query in enumerate(query_mask) if is_query]
            
            # Take top-k if we have more than k sequences
            if len(query_scores) > topk:
                top_k_scores, top_k_indices = torch.topk(query_scores, topk)
                selected_passages = [query_passages[j] for j in top_k_indices.tolist()]
                selected_scores = top_k_scores.tolist()
            else:
                # If we have fewer than k sequences, take all of them
                selected_passages = query_passages
                selected_scores = query_scores.tolist()
                
            output_passages.append(selected_passages)
            output_scores.append(selected_scores)
        
        # Restore retriever to original state
        retriever_fp16.to(original_dtype)
        if original_mode:
            retriever_fp16.train()
        else:
            retriever_fp16.eval()
        
        return output_passages, output_scores

    @torch.no_grad()
    def retrieve(self, *args, **kwargs):
        retrieve_func = self.retrieve_with_rerank if self.opt.retrieve_with_rerank else self._retrieve
        passages, scores = retrieve_func(*args, **kwargs)[:2]
        return passages, scores


    def tokenize_query(self, query):
        """
        Tokenizes query sequences.

        Args:
            query (List[str]): List of query sequences (strings), length batch_size.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': Tensor of shape [batch_size, seq_length]
                - 'attention_mask': Tensor of shape [batch_size, seq_length]
        """
        query_tokens = self.retriever_tokenizer(
            query,
            max_length=min(self.opt.text_maxlength, self.ESM_MAX_SEQ_LENGTH),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return _to_cuda(query_tokens, self.fabric)
        
    def calculate_individual_losses(self, logits, labels, padding_idx=-100, query_positions=None):
        """
        Calculate loss for each sequence in the batch, exactly matching HuggingFace's implementation.
        Not calculating log of outputs
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        # Print shapes and content for debugging
        # print("logits shape:", logits.shape)
        # print("labels shape:", labels.shape)
        # print("unique labels:", torch.unique(labels))
        # print("padding_idx:", padding_idx)

        # Shift logits and labels: we predict everything except last token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten the tokens
        flat_shift_logits = shift_logits.view(-1, vocab_size)
        flat_shift_labels = shift_labels.view(-1)
        
        # Get per-token losses (with reduction='none' to get per-token values)
        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=padding_idx)
        token_losses = loss_fct(flat_shift_logits, flat_shift_labels)
        token_losses = token_losses.view(batch_size, -1)
        
        # Create mask for valid positions (non-padding)
        mask = (shift_labels != padding_idx).float()
        
        # If query positions provided, only calculate loss for query tokens
        if query_positions is not None:
            query_mask = torch.zeros_like(mask)
            for i, (start, end) in enumerate(query_positions):
                # Adjust positions to account for the shift
                query_mask[i, (start-1):(end-1)] = 1.0
            mask = mask * query_mask
        
        # Calculate average loss per sequence
        seq_lengths = mask.sum(dim=1).clamp(min=1)
        individual_losses = (token_losses * mask).sum(dim=1) / seq_lengths
        
        return individual_losses
    # def compress_indices(self, indices: list[int]) -> str:
    #     """
    #     Given a sorted list of integers, return a string that compresses contiguous indices.
    #     For example, [6,7,8,9,10] becomes "6-10".
    #     """
    #     if not indices:
    #         return ""
    #     segments = []
    #     start = indices[0]
    #     prev = indices[0]
    #     for i in indices[1:]:
    #         if i == prev + 1:
    #             prev = i
    #         else:
    #             segments.append(f"{start}-{prev}" if start != prev else f"{start}")
    #             start = i
    #             prev = i
    #     segments.append(f"{start}-{prev}" if start != prev else f"{start}")
    #     return ", ".join(segments)


    # def calculate_individual_losses(self, logits, labels, padding_idx=-100, query_positions=None):
    #     """
    #     Computes per-sample loss from logits and labels. If query_positions is provided,
    #     then for each sample the loss is computed only over the specified token regions.
    #     If a sample has a single query region (i.e. a tuple like (start, end)) it will be wrapped
    #     into a list so that the code works uniformly.
        
    #     The logits are expected to be of shape [batch_size, seq_len, vocab_size] and the labels
    #     [batch_size, seq_len]. Because language-modeling losses are typically computed with a shift,
    #     this function shifts logits (dropping the last time-step) and labels (dropping the first token).
        
    #     For debugging, the function prints the active token indices (in a compressed format) used
    #     for the loss calculation.
        
    #     Args:
    #         logits (Tensor): [B, L, vocab_size] model outputs.
    #         labels (Tensor): [B, L] ground-truth token IDs.
    #         padding_idx (int): Token ID to ignore.
    #         query_positions (List[List[Tuple[int,int]]] or List[Tuple[int,int]], optional): If provided, for
    #             each sample either a list of one or more (start, end) tuples (using the original, unshifted
    #             token indices) or a single (start, end) tuple. These define the region(s) over which the
    #             loss is computed.
                
    #     Returns:
    #         Tensor: [B] loss values (one per sample). If there are multiple regions per sample, the losses
    #         are averaged.
    #     """


    #     batch_size, seq_len, vocab_size = logits.size()

    #     # Shift logits and labels: token at position t predicts token at position t+1.
    #     shift_logits = logits[:, :-1, :].contiguous()  # shape: [B, L-1, vocab_size]
    #     shift_labels = labels[:, 1:].contiguous()        # shape: [B, L-1]
        
    #     # Compute per-token losses
    #     loss_fct = CrossEntropyLoss(reduction="none", ignore_index=padding_idx)
    #     flat_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    #     token_losses = flat_loss.view(batch_size, -1)  # shape: [B, L-1]
        
    #     # Base mask: ignore padding tokens.
    #     base_mask = (shift_labels != padding_idx).float()
        
    #     individual_losses = []
    #     for i in range(batch_size):
    #         if query_positions is not None and i < len(query_positions) and query_positions[i]:
    #             seg_losses = []
    #             seg_info = query_positions[i]
    #             # If a single tuple was provided instead of a list, wrap it in a list.
    #             if isinstance(seg_info, (tuple, list)) and len(seg_info) == 2 and isinstance(seg_info[0], int):
    #                 seg_info = [seg_info]
    #             # Process each segment in this sample.
    #             for seg_idx, (qstart, qend) in enumerate(seg_info):
    #                 # Adjust for the shift: original positions [qstart, qend) correspond to [qstart-1, qend-1) in shifted tokens.
    #                 start_idx = max(qstart - 1, 0)
    #                 end_idx = min(qend - 1, token_losses.size(1))  # end is exclusive
    #                 # Build a mask for tokens in this region.
    #                 seg_mask = torch.zeros_like(token_losses[i])
    #                 seg_mask[start_idx:end_idx] = 1.0
    #                 seg_mask = seg_mask * base_mask[i]
                    
    #                 # Debug print: print the active positions in compressed form.
    #                 active_indices = torch.nonzero(seg_mask).squeeze(1).tolist()
    #                 # print(f"Sample {i}, Segment {seg_idx} loss indices: {self.compress_indices(active_indices)}")
                    
    #                 denom = seg_mask.sum().item()
    #                 if denom > 0:
    #                     seg_loss = (token_losses[i] * seg_mask).sum() / denom
    #                     seg_losses.append(seg_loss)
    #                 else:
    #                     print(f"Warning: Sample {i}, Segment {seg_idx} has no active tokens!")
    #             # Average losses over segments for this sample.
    #             if seg_losses:
    #                 sample_loss = sum(seg_losses) / len(seg_losses)
    #             else:
    #                 sample_loss = torch.tensor(0.0, device=logits.device)
    #             individual_losses.append(sample_loss)
    #         else:
    #             # No query_positions provided: use full (non-padding) tokens.
    #             active_indices = torch.nonzero(base_mask[i]).squeeze(1).tolist()
    #             print(f"Sample {i} loss indices (full): {self.compress_indices(active_indices)}")
    #             denom = base_mask[i].sum().item()
    #             if denom > 0:
    #                 sample_loss = (token_losses[i] * base_mask[i]).sum() / denom
    #             else:
    #                 sample_loss = torch.tensor(0.0, device=logits.device)
    #             individual_losses.append(sample_loss)
        
    #     return torch.stack(individual_losses)
    def sample_and_fit_sequences(
        self,
        query_id_or_sequence: Optional[str] = None,  # Changed parameter name to be more explicit
        member_ids: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        fasta_dataset: Optional[Dict[str, str]] = None,
        max_seq_length: int = 1024,
        max_individual_seq_length: Optional[int] = None,
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
                return seq[:max_len - self.num_special_characters]
            return seq

        # Retrieve and possibly truncate the query
        query_sequence = self.get_sequence(query_id_or_sequence, fasta_dataset) if include_query else None
        
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
        # logger.info("printing inside of the sample and fit function")
        # logger.info("member_ids is {}".format(member_ids))
        if member_ids and len(member_ids) > 0:
            if sample:
                member_weights = np.array(weights if weights is not None else [1.0 / len(member_ids)] * len(member_ids))
                member_weights /= member_weights.sum()
                
                skip_rounds = 0
                while leftover > 0 and skip_rounds < max_skips:
                    sampled_ids = np.random.choice(member_ids, size=chunk_size, replace=True, p=member_weights)
                    added = False
                    
                    for mid in sampled_ids:
                        seq = self.get_sequence(mid, fasta_dataset)
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
            else:
                # Sequential processing with length limits
                for mid in member_ids:
                    seq = self.get_sequence(mid, fasta_dataset)
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
                    elif truncate and leftover > 0:
                        seq = seq[:leftover - self.num_special_characters]
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
            "total_tokens": total_tokens
        }
    
    def pack_inputs(
        self, 
        sample, 
        tokenizer,  # Uniprot21 tokenizer instance
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
        labels = tokens.clone()
        
        return {
            "tokens": tokens,
            "seq_lens": all_lengths,
            "labels": labels,
        }
      
    def padded_collate_packed(
        self, 
        batch: List[Dict[str, Any]], 
        fabric=None, 
        pad_mode: Optional[str] = None, 
        fixed_length: Optional[int] = None, 
        padding_idx: int = IGNORE_INDEX
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
                raise ValueError("fixed_length must be provided when pad_mode is 'fixed'")
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
            padded_tokens.append(token)
            padded_labels.append(label)
        tokens_tensor = torch.stack(padded_tokens, dim=0)
        labels_tensor = torch.stack(padded_labels, dim=0)
        
        # Collate segment sizes (each sample may have several segments).
        max_segments = max(len(x["seq_lens"]) for x in batch)
        bsz = len(batch)
        segment_sizes = torch.zeros((bsz, max_segments), dtype=torch.int32)
        for i, x in enumerate(batch):
            seq_lens = torch.tensor(x["seq_lens"], dtype=torch.int32)
            segment_sizes[i, :len(seq_lens)] = seq_lens
        
        collated_batch = {
            "tokens": tokens_tensor,
            "labels": labels_tensor,
            "segment_sizes": segment_sizes,
        }
        collated_batch = _to_cuda(collated_batch, fabric)
        return collated_batch
    
    def forward(
        self,
        index,
        query: Optional[List[str]] = None,  # Direct sequences
        query_ids: Optional[List[str]] = None,  # IDs to look up in fasta_dataset
        metadata: Optional[List[Dict[str, Any]]] = None,  # Metadata for each input
        passages: Optional[List[List[str]]] = None,  # Passages retrieved from index
        passages_weights: Optional[List[List[float]]] = None,  # Sampling weights for passages
        fasta_dataset=None,  # Optional dict to map identifiers to sequences
        filtering_fun=None,  # Optional function to filter retrieved passages
        train_retriever=False,  # Boolean indicating if retriever should be trained
        iter_stats: Optional[Dict[str, Any]] = None,
        query_only_loss: bool = False,
        reverse_sequence: bool = False,
        max_individual_seq_length: Optional[int] = None, 
    ):
        """
        Forward pass for training.

        Args:
            index: The FAISS index for retrieval.
            query (Optional[List[str]]): List of input sequences.
            query_ids (Optional[List[str]]): List of query IDs (e.g., UniRef50 IDs).
            metadata (List[Dict[str, Any]]): List of metadata dictionaries for each input.
            passages (List[List[str]]): List of lists containing passages for each input.
            passages_weights (List[List[float]]): List of lists containing weights for each passage.
            fasta_dataset: Optional dict to map identifiers to sequences.
            filtering_fun: Optional function to filter retrieved passages.
            train_retriever: Boolean indicating if retriever should be trained.
            iter_stats: Optional dict to store iteration statistics.
            query_only_loss: Boolean indicating whether to calculate loss only on query portion.
            max_individual_seq_length: Optional int to limit the length of individual sequences.

        Returns:
            Dict containing:
                - reader_loss: Mean loss across the batch
                - retriever_loss: Optional retriever training loss
                - individual_losses: Per-sequence losses
                - query_positions: Query positions in sequences if query_only_loss is True
        """

        forward_start = time.time()
        if query is None and query_ids is None:
            raise ValueError("Either query or query_ids must be provided.")

        # Setup query handling
        if query is not None:
            enum_queries = query
            use_fasta = False
            bsz = len(query)
        else:
            enum_queries = query_ids
            use_fasta = True
            bsz = len(query_ids)

        # Do initial retrieval if needed
        if self.opt.train_retriever:
            if not use_fasta:
                query_tokens = self.tokenize_query(enum_queries)
            else:
                query_sequences = [self.get_sequence(qid, fasta_dataset) for qid in enum_queries]
                query_tokens = self.tokenize_query(query_sequences)
            # logger.info(f"query_tokens are {query_tokens}")
            # logger.info(f"query_tokens shape are {query_tokens['input_ids'].shape}")

        if not self.opt.use_file_passages:
            retrieve_start = time.time()
            passages, _ = self.retrieve(
                index,
                self.opt.retriever_n_context,
                query_sequences if use_fasta else enum_queries,
                query_tokens["input_ids"],
                query_tokens["attention_mask"],
                batch_metadata=metadata,
                filtering_fun=filtering_fun,
                iter_stats=iter_stats,
            )
            if iter_stats is not None:
                iter_stats["runtime/retrieve"] = (time.time() - retrieve_start, 1)
        else:
            if passages is None:
                raise ValueError("Passages must be provided when use_file_passages is True")

        # Sample and fit sequences
        packed_batch = []
        query_positions = [] if query_only_loss else None
            
            # Sample and fit sequences for the entire batch
        sampled_sequences = []
        for i, query_item in enumerate(enum_queries):
            weights = passages_weights[i] if passages_weights else None
            sample = self.sample_and_fit_sequences(
                query_id_or_sequence=query_item,
                member_ids=passages[i],
                weights=weights,
                fasta_dataset=fasta_dataset if use_fasta else None,
                max_seq_length=self.max_seq_context,
                max_individual_seq_length=max_individual_seq_length,
                include_query=True,
                sample=True if passages_weights else False,
                truncate=False
            )
            sampled_sequences.append(sample)
        # logger.info(f"sampled_sequences are {sampled_sequences}")
        # Process retriever training
        retriever_loss = None
        if train_retriever:
            query_emb = self.retriever(
                query_tokens["input_ids"],
                query_tokens["attention_mask"],
                is_passages=False
            )
            # logger.info(f"Query embedding shape: {query_emb.shape}")

            # Get all passages in one batch
            all_passages = []
            for sample in sampled_sequences:
                all_passages.extend(sample["passages"])

            passage_tokens = self.retriever_tokenizer(
                all_passages,
                padding="max_length",
                return_tensors="pt",
                max_length=self.opt.text_maxlength,
                truncation=True,
            )
            passage_tokens = _to_cuda(passage_tokens, self.fabric)
            # logger.info(f"Combined passage tokens shape: {passage_tokens['input_ids'].shape}")
            
            all_passage_embs = self.retriever(**passage_tokens, is_passages=True)
            # logger.info(f"All passage embeddings shape: {all_passage_embs.shape}")

            # Repeat query embeddings to match passages
            passages_per_query = [len(s["passages"]) for s in sampled_sequences]
            # logger.info(f"Passages per query: {passages_per_query}")
            assert sum(passages_per_query) == all_passage_embs.size(0), "Mismatch in passage counts"
            
            query_emb_expanded = torch.repeat_interleave(query_emb, torch.tensor(passages_per_query).to(query_emb.device), dim=0)
            # logger.info(f"Query expanded shape: {query_emb_expanded.shape}, Passage embs shape: {all_passage_embs.shape}")
                    
            # Calculate retriever scores
            retriever_scores = torch.einsum("id,id->i", [query_emb_expanded, all_passage_embs])
            retriever_scores = retriever_scores / math.sqrt(query_emb.size(-1))
            # logger.info(f"Retriever scores shape: {retriever_scores.shape}, values: {retriever_scores}")

            if self.opt.gold_score_mode == "ppmean":
                gold_scores = self.perplexity_score(sampled_sequences)
            elif self.opt.gold_score_mode == "loop":
                gold_scores = self.loop_score(sampled_sequences)
            elif self.opt.gold_score_mode == "emdr":
                gold_scores = self.emdr_score(sampled_sequences)

            # logger.info(f"Gold scores shape: {gold_scores.shape}, values: {gold_scores}")
            if self.opt.gold_score_mode == "emdr":
                retriever_loss = self.logprob(retriever_scores, gold_scores, passages_per_query)
            else:
                retriever_loss = self.kldivloss(retriever_scores, gold_scores, passages_per_query)

            if self.training:
                self.reader.train()
            # logger.info(f"Retriever loss: {retriever_loss}")
                
        # Process reader training
        packed_batch = []
        for sample in sampled_sequences:
            packed = self.pack_inputs(
                sample=sample,
                tokenizer=self.reader_decoder_tokenizer,
                max_seq_length=self.max_seq_context,
                reverse_sequence=reverse_sequence,
                skip_padding=False,
                padding_idx=self.reader_decoder_tokenizer.mask_token
            )
            packed_batch.append(packed)

        collated_batch = self.padded_collate_packed(packed_batch, fabric=self.fabric)
        outputs = self.reader(
            xs=collated_batch["tokens"],
            segment_sizes=collated_batch["segment_sizes"]
        )

        individual_losses = self.calculate_individual_losses(
            outputs,
            collated_batch["labels"],
            padding_idx=self.reader_decoder_tokenizer.mask_token
        )
        reader_loss = individual_losses.mean()

        if iter_stats is not None:
            iter_stats["loss/reader_loss"] = (reader_loss.item(), bsz)
            if retriever_loss is not None:
                iter_stats["loss/retriever_loss"] = (retriever_loss.item(), bsz)
            iter_stats["runtime/forward"] = (time.time() - forward_start, 1)

        return {
            'reader_loss': reader_loss,
            'retriever_loss': retriever_loss,
            'individual_losses': individual_losses
        }
    
    def create_individual_samples(self, sampled_sequences: List[Dict[str, Any]]) -> (List[Dict[str, Any]], List[tuple]):
        """
        Given a list of samples (each with keys "sequence" and "passages"),
        create a list of individual samples where each sample contains the same query
        but only one passage. Also, record for each sample the query’s start and end
        positions in the packed sequence.
        """
        individual_samples = []
        query_positions = []  # For each sample, a tuple (query_start, query_end)
        for sample in sampled_sequences:
            query = sample["sequence"]
            for passage in sample["passages"]:
                indiv_sample = {
                    "sequence": query,
                    "passages": [passage],
                    "passage_lengths": [len(passage) + self.num_special_characters],
                    "query_length": len(query) + self.num_special_characters,
                }
                # In a packed sample, the query is appended after the passage.
                # So, the query starts right after the passage tokens.
                query_start = indiv_sample["passage_lengths"][0]
                query_end = query_start + indiv_sample["query_length"]
                individual_samples.append(indiv_sample)
                query_positions.append((query_start, query_end))
        return individual_samples, query_positions

    def create_leave_one_out_samples(self, sampled_sequences: List[Dict[str, Any]]) -> (List[Dict[str, Any]], List[tuple]):
        """
        For each sample (with a query and several passages), create leave-one-out
        samples in which for each passage one constructs a new sample that excludes
        that passage. Also record the query position in the new packed sample.
        """
        individual_samples = []
        query_positions = []
        for sample in sampled_sequences:
            passages = sample["passages"]
            query = sample["sequence"]
            for idx in range(len(passages)):
                # Exclude the passage at index idx.
                remaining = passages[:idx] + passages[idx+1:]
                leave_sample = {
                    "sequence": query,
                    "passages": remaining,
                    "passage_lengths": [len(p) + self.num_special_characters for p in remaining],
                    "query_length": len(query) + self.num_special_characters,
                }
                # In the packed sample, the query tokens come after all remaining passages.
                query_start = sum(leave_sample["passage_lengths"])
                query_end = query_start + leave_sample["query_length"]
                individual_samples.append(leave_sample)
                query_positions.append((query_start, query_end))
        return individual_samples, query_positions

    # -------------------------------------------------------------------------
    # New versions of the loss functions using the padded_collate_packed with pad_mode="largest"
    # and reusing the sample-and-fit helper logic.
    # -------------------------------------------------------------------------
    
    def perplexity_score(self, sampled_sequences: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Computes per-passage gold scores using perplexity loss over individual query+passage samples.
        Uses the new padded_collate_packed with pad_mode "largest".
        """
        with torch.no_grad():
            self.reader.eval()
            # Reuse sample_and_fit output: create one sample per passage.
            individual_samples, query_positions = self.create_individual_samples(sampled_sequences)
            
            # Pack each individual sample without per-sample padding.
            packed_list = [
                self.pack_inputs(
                    sample=sample,
                    tokenizer=self.reader_decoder_tokenizer,
                    max_seq_length=self.max_seq_context,
                    skip_padding=True,
                    padding_idx=self.reader_decoder_tokenizer.mask_token
                )
                for sample in individual_samples
            ]
            
            # Collate the batch, padding each sample up to the largest length in the batch.
            collated = self.padded_collate_packed(
                batch=packed_list,
                fabric=self.fabric,
                pad_mode="largest",
                padding_idx=self.reader_decoder_tokenizer.mask_token
            )
            outputs = self.reader(
                xs=collated["tokens"],
                segment_sizes=collated["segment_sizes"]
            )
            # print("outputs shape is {}".format(outputs.shape))
            losses = self.calculate_individual_losses(
                outputs,
                collated["labels"],
                padding_idx=self.reader_decoder_tokenizer.mask_token,
                query_positions=query_positions
            )
            # Return negative perplexity (so that higher gold score means better passage)
            return -losses

    
    def emdr_score(self, sampled_sequences: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Computes per-passage gold scores using emdr loss over individual query+passage samples.
        Uses the new padded_collate_packed with pad_mode "largest".
        As we are aggregating by passage instead of cacluating at the token level, 
        this is exact same as perplexity, except for the sign change at the end
        """
        with torch.no_grad():
            self.reader.eval()
            # Reuse sample_and_fit output: create one sample per passage.
            individual_samples, query_positions = self.create_individual_samples(sampled_sequences)
            
            # Pack each individual sample without per-sample padding.
            packed_list = [
                self.pack_inputs(
                    sample=sample,
                    tokenizer=self.reader_decoder_tokenizer,
                    max_seq_length=self.max_seq_context,
                    skip_padding=True,
                    padding_idx=self.reader_decoder_tokenizer.mask_token
                )
                for sample in individual_samples
            ]
            
            # Collate the batch, padding each sample up to the largest length in the batch.
            collated = self.padded_collate_packed(
                batch=packed_list,
                fabric=self.fabric,
                pad_mode="largest",
                padding_idx=self.reader_decoder_tokenizer.mask_token
            )
            outputs = self.reader(
                xs=collated["tokens"],
                segment_sizes=collated["segment_sizes"]
            )
            losses = self.calculate_individual_losses(
                outputs,
                collated["labels"],
                padding_idx=self.reader_decoder_tokenizer.mask_token,
                query_positions=query_positions
            )
            return losses

    def loop_score(self, sampled_sequences: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Computes leave-one-out (LOO) gold scores in a single forward pass.
        Uses the new padded_collate_packed (pad_mode "largest") and reuses a helper to
        create individual leave-one-out samples.
        """
        with torch.no_grad():
            self.reader.eval()
            individual_samples, query_positions = self.create_leave_one_out_samples(sampled_sequences)
            packed_list = [
                self.pack_inputs(
                    sample=sample,
                    tokenizer=self.reader_decoder_tokenizer,
                    max_seq_length=self.max_seq_context,
                    skip_padding=True,
                    padding_idx=self.reader_decoder_tokenizer.mask_token
                )
                for sample in individual_samples
            ]
            collated = self.padded_collate_packed(
                batch=packed_list,
                fabric=self.fabric,
                pad_mode="largest",
                padding_idx=self.reader_decoder_tokenizer.mask_token
            )
            outputs = self.reader(
                xs=collated["tokens"],
                segment_sizes=collated["segment_sizes"]
            )
            losses = self.calculate_individual_losses(
                outputs,
                collated["labels"],
                padding_idx=self.reader_decoder_tokenizer.mask_token,
                query_positions=query_positions
            )
            return losses

    def kldivloss(self, score: torch.Tensor, gold_score: torch.Tensor, passages_per_query: list) -> torch.Tensor:
        """
        Computes a KL divergence loss between the retriever’s scores and gold scores,
        but first normalizes (via softmax) within each query’s group of passages.
        
        Args:
            score (Tensor): 1D tensor of retriever scores (one per passage).
            gold_score (Tensor): 1D tensor of gold scores (one per passage).
            passages_per_query (list[int]): List giving the number of passages for each query.
            
        Returns:
            Tensor: The KLDiv loss computed over the grouped (normalized) scores.
        """
        # Create a group tensor (each entry indicates the query group for that passage)
        groups = create_group_tensor(passages_per_query, score.device)
        # Normalize gold scores into probabilities (per group) using temperature
        gold_probs = grouped_softmax_scatter(gold_score / self.opt.temperature_gold, groups)
        # Compute log-normalized retriever scores (per group) using temperature
        score_log_probs = grouped_log_softmax(score / self.opt.temperature_score, groups)
        # Note: torch.nn.KLDivLoss expects input as log-probabilities and target as probabilities.
        # print("score_log_probs shape is {}".format(score_log_probs.shape))
        # print("gold_probs shape is {}".format(gold_probs.shape))
        loss = torch.nn.KLDivLoss(reduction='batchmean')(score_log_probs, gold_probs)
        return loss

    def logprob(self, score: torch.Tensor, gold_score: torch.Tensor, passages_per_query: list) -> torch.Tensor:
        """
        Computes the negative log probability for EMDR training.
        
        The goal is to compute, for each query,
        
            log [ sum_{k in query} p_LM(q|d_k) * p_RETR(d_k|q) ]
        
        where p_LM and p_RETR are obtained by applying softmax within each query's
        group of passages. We assume that `score` and `gold_score` are 1D tensors of 
        raw logits (concatenated over all queries) and that the grouping is provided 
        by `passages_per_query` (a list of counts, one per query).
        
        Temperatures are applied from self.opt.temperature_score and 
        self.opt.temperature_gold.
        
        Returns:
            A scalar loss (the negative average over queries).
        """
        device = score.device
        # Create a group tensor indicating which query each passage belongs to.
        groups = create_group_tensor(passages_per_query, device)
        
        # Compute log-softmax for the LM/gold scores and retriever scores within each group.
        # (This gives you log p_LM and log p_RETR per passage.)
        log_gold = grouped_log_softmax(gold_score / self.opt.temperature_gold, groups)
        log_retr = grouped_log_softmax(score / self.opt.temperature_score, groups)
        
        # The joint log probability per passage is the sum.
        log_joint = log_gold + log_retr
        
        # Now, for each query, we want the log of the sum of exp(log_joint) over its passages.
        # We use our group_logsumexp helper to get one number per query.
        group_logprob = group_logsumexp(log_joint, groups)
        
        # The loss is the negative average over queries.
        loss = -group_logprob.mean()
        return loss

    @torch.no_grad()
    def generate(self, tokens, query):
        """
        Generate sequences using the reader.

        Args:
            tokens: Tokenized passages.
            query: Original query texts.
        """
        reader_input_ids = tokens['input_ids'].view(len(query), -1)
        reader_attention_mask = tokens['attention_mask'].view(len(query), -1)

        generated_ids = self.reader.generate(
            input_ids=reader_input_ids,
            attention_mask=reader_attention_mask,
            num_beams=self.opt.generation_num_beams,
            max_length=self.opt.generation_max_length,
            early_stopping=True,
        )
        if self.opt.warm_start:
            generated_texts = self.reader_decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            generated_texts = self.reader_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    def set_fabric(self, fabric):
        """Set the fabric instance"""
        self.fabric = fabric

    def get_sequence(self, protein_id_or_seq: str, fasta_dataset: Dict[str, str]) -> str:
        """
        Retrieve protein sequence by UniRef50/100 ID if fasta is given, otherwise return protein_id

        Args:
            protein_id (str): UniRef50 ID of the protein.

        Returns:
            str: Protein sequence.
        """
        if fasta_dataset is None:
            return protein_id_or_seq

        return fasta_dataset[protein_id_or_seq]
    
def _to_cuda(tok_dict, fabric):
    return {k: fabric.to_device(v) for k, v in tok_dict.items()}

def jit_warmup(embedding_model, alphabet):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().cuda()
    segment_sizes = torch.tensor(segment_sizes).long().cuda()
    print("inputs are", x.unsqueeze(0), segment_sizes.unsqueeze(0))
    _ = embedding_model.forward(x.unsqueeze(0), segment_sizes.unsqueeze(0))
