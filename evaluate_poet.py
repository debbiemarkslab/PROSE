import os
import time
from collections import defaultdict
from pathlib import Path
import logging
import numpy as np
import torch
from tqdm import tqdm
import json
import itertools
import string
import pandas as pd 

from torch.utils.data import DataLoader

from lightning.fabric import Fabric

from protriever import util
from protriever.options import get_options
from protriever.index_io import load_or_initialize_index, save_embeddings_and_index
from protriever.model_io import load_dataloader, create_checkpoint_directories, load_or_initialize_protriever_model, instantiate_fasta_dataset

from protriever.dataset import ProteinDMSDataset, TRRosettaContactDataset, ProteinClusterDatasetInverseWeighted, FastaDatasetByUniref, custom_collate_fn
from poet.msa.sampling import MSASampler, NeighborsSampler, RandomSampler, TopSampler
from poet.fasta import parse_stream
from poet.alphabets import Uniprot21
from poet.models.modules.packed_sequence import PackedTensorSequences

ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()

from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)

# if torch.cuda.is_available():
#     # For training, use high precision for better performance
#     torch.set_float32_matmul_precision('highest') #should be highest for evaluation
#     logger.info("Set float32 matmul precision to highest for evaluation")


def _get_eval_data_loader_dms(opt, data_path):
    """
    Creates a DataLoader for evaluation using the ProteinDMSDataset.
    """    
    # Create dataset
    transform = None
    dataset = ProteinDMSDataset(
        csv_path=data_path,
        transform=transform,
        use_file_passages=opt.use_file_passages
    )
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=opt.per_gpu_batch_size_eval,
        num_workers=opt.num_workers,
        shuffle=False,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False
    )
    
    
    return data_loader, dataset.largest_context_len()


def evaluate_likelihood(model, data_loader, opt, fabric, data_path, index, step, query_only_loss=False):
    pass

def get_seqs_from_fastalike(filepath: Path) -> list[bytes]:
    return [s for _, s in parse_stream(open(filepath, "rb"), upper=False)]


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> np.ndarray:
    return np.vstack(
        [
            alphabet.encode(s.translate(None, delete=ASCII_LOWERCASE_BYTES))
            for s in msa_sequences
        ]
    )


def initialize_msa_sampler(sampler_method, sim):
    """
    Initialize the appropriate MSA sampler based on method.
    """
    sampler_base = {
        "top": TopSampler,
        "random": RandomSampler,
        "neighbors": lambda: NeighborsSampler(theta=0.8, can_use_torch=True)
    }[sampler_method]()
    
    return MSASampler(
        method=sampler_base,
        force_include_first=False,
        max_similarity=sim
    )

def jit_warmup(embedding_model, alphabet):
    x = b"$WAAAGH*$WAAGW*"
    segment_sizes = [8, 7]
    x = alphabet.encode(x)  # encode x into the uniprot21 alphabet
    x = torch.from_numpy(x).long().cuda()
    segment_sizes = torch.tensor(segment_sizes).long().cuda()
    print("inputs are", x.unsqueeze(0), segment_sizes.unsqueeze(0))
    output = embedding_model.forward(x.unsqueeze(0), segment_sizes.unsqueeze(0))
    return output

def retrieve_passages(model, index, opt, query, query_enc, num_samples, batch_metadata=None):
    """
    Retrieves passages and returns their IDs for sample_and_fit_sequences.
    
    Args:
        model: Model instance
        index: FAISS index
        opt: Options
        query: Query text
        query_enc: Encoded query
        num_samples: Number of samples to retrieve
        batch_metadata: Optional batch metadata
        
    Returns:
        List[str]: List of passage IDs
    """
    if not opt.use_file_passages:
        query_ids_retriever = query_enc["input_ids"]
        query_mask_retriever = query_enc["attention_mask"]
        
        # Get raw passages from retrieval
        retrieved_passages, _ = model.retrieve(
            index,
            opt.n_context_eval[0],  # Number of passages to retrieve
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata=batch_metadata,
            filtering_fun=None,
            fasta_dataset=opt.fasta_dataset
        )
        print("DEBUG: retrieved_passages", retrieved_passages)
        if 'id' in retrieved_passages[0][0]:
            # Extract just the IDs from each passage dict
            passage_ids = [p['id'] for p in retrieved_passages[0]]
        else:
            pass
        return passage_ids
    else:
        return []
    
def get_context_from_msa(model, opt, msa_fasta_path):
    """
    Generates context from MSA sequences using different sampling parameters.
    
    Args:
        model: Model instance containing reader and tokenizer
        opt: Options containing:
            - debug: Boolean for debug mode
            - max_context_length_eval: List of context lengths to evaluate
            - similarity_values: List of similarity thresholds
            - sampler_method: Method for MSA sampling
            - seed: Random seed for sampling
        msa_fasta_path: Path to MSA fasta file
    
    Returns:
        dict: Context data with following structure:
            {
                "contexts": List[Dict], where each dict contains:
                    - "tokens": torch.Tensor of shape [1, seq_length] containing tokenized context
                    - "segment_sizes": torch.Tensor of shape [1, num_segments] containing the length 
                      of each sequence in the context before padding
                    - "metadata": dict with:
                        - "context_length": int, maximum length used for this context
                        - "similarity": float, similarity threshold used for MSA sampling
                        - "context_type": str, always "msa"

            The outer list length = len(max_context_length_eval) * len(similarity_values) if not debug,
            or 1 if in debug mode.
    """
    msa_sequences = get_seqs_from_fastalike(Path(msa_fasta_path))
    alphabet = model.reader_decoder_tokenizer
    msa = get_encoded_msa_from_a3m_seqs(msa_sequences=msa_sequences, alphabet=alphabet)

    contexts = []
    
    # Use consistent parameter names
    length_params = opt.max_context_length_eval if not opt.debug else [opt.max_context_length_eval[0]]
    similarity_params = opt.similarity_values if not opt.debug else [opt.similarity_values[0]]
    
    for max_tokens in tqdm(length_params, desc="Processing context lengths"):
        for similarity in similarity_params:
            sampler = initialize_msa_sampler(opt.sampler_method, similarity)
            sample_idxs = sampler.get_sample_idxs(
                msa=msa,
                gap_token=alphabet.gap_token,
                seed=opt.seed,
            )
            
            msa_prompt = model.sample_and_fit_sequences(
                member_ids=[msa_sequences[i].upper().translate(None, b"-") for i in sample_idxs],
                max_seq_length=max_tokens,
                truncate=False,
                include_query=False,
                sample=False
            )
            
            packed = model.pack_inputs(
                sample=msa_prompt,
                tokenizer=alphabet,
                max_seq_length=msa_prompt['total_tokens'],
                skip_padding=True
            )
            
            collated = model.padded_collate_packed([packed], fabric=model.fabric)
            
            contexts.append({
                "tokens": collated["tokens"],  # [1, seq_length]
                "segment_sizes": collated["segment_sizes"],  # [1, num_segments]
                "metadata": {
                    "context_length": max_tokens,
                    "similarity": similarity,
                    "context_type": "msa"
                }
            })
    
    return {"contexts": contexts}

def get_context_from_index(model, index, opt, eval_query):
    """
    Retrieves context sequences from the index using the query and processes them
    into the same format as MSA contexts.
    
    Args:
        model: Model instance containing retriever and tokenizer
        index: Index object for retrieval
        opt: Options containing:
            - debug: Boolean for debug mode
            - max_context_length_eval: List of context lengths to evaluate
            - n_context_eval: List of number of passages to retrieve
            - fasta_dataset: Dataset mapping IDs to sequences
        eval_query: Query string for retrieval

    Returns:
        dict: Context data with following structure:
            {
                "contexts": List[Dict], where each dict contains:
                    - "tokens": torch.Tensor of shape [1, seq_length] containing tokenized 
                      retrieved sequences concatenated together
                    - "segment_sizes": torch.Tensor of shape [1, num_segments] containing the length
                      of each retrieved sequence in the context before padding
                    - "metadata": dict with:
                        - "context_length": int, maximum length used for this context
                        - "n_context": int, number of passages retrieved
                        - "context_type": str, always "retrieval"

            The outer list length = len(max_context_length_eval) if not debug,
            or 1 if in debug mode.
            
    Note:
        - Retrieved sequences are packed together into a single context sequence
        - Each sequence in the context is wrapped with start/end tokens before concatenation
        - The segment_sizes tensor tracks the length of each individual sequence
        - The total context length (seq_length) includes all sequences plus special tokens
    """
    query = [eval_query]
    query_tokens = model.tokenize_query(query)
    contexts = []
    
    # Use consistent parameter names
    length_params = opt.max_context_length_eval if not opt.debug else [opt.max_context_length_eval[0]]
    n_context_params = opt.n_context_eval if not opt.debug else [opt.n_context_eval[0]]
    
    # Get initial passages
    passage_ids = retrieve_passages(
        model, index, opt, query, query_tokens, 1,
    )
    
    # Process for different context lengths
    for max_length in tqdm(length_params, desc="Processing context lengths"):

        # logger.info("fasta dataset is {}".format(opt.fasta_dataset))
        sequence_prompt = model.sample_and_fit_sequences(
            member_ids=passage_ids,
            max_seq_length=max_length,
            truncate=False,
            include_query=False,
            sample=False,
            fasta_dataset=opt.fasta_dataset
        )
        print("DEBUG: sequence_prompt", sequence_prompt)
        packed = model.pack_inputs(
            sample=sequence_prompt,
            tokenizer=model.reader_decoder_tokenizer,
            max_seq_length=sequence_prompt['total_tokens'],
            skip_padding=True
        )
        
        collated = model.padded_collate_packed([packed], fabric=model.fabric)
        
        contexts.append({
            "tokens": collated["tokens"],  # [1, seq_length]
            "segment_sizes": collated["segment_sizes"],  # [1, num_segments]
            "metadata": {
                "context_length": max_length,
                "n_context": n_context_params[0],
                "context_type": "retrieval"
            }
        })
    
    return {"contexts": contexts}

@torch.no_grad()
def evaluate_dms_poet(model, data_loader, opt, fabric, data_path, context_data, step, largest_context_len):
    """
    Evaluates model on DMS data using provided context, with optional bidirectional scoring.
    
    Args:
        model: Model instance
        data_loader: DataLoader containing variants to score
        opt: Options containing evaluation parameters
        fabric: Lightning Fabric instance
        data_path: Path to save results
        context_data: dict with structure:
            {
                "contexts": List[Dict], where each dict contains:
                    - "tokens": torch.Tensor of token IDs
                    - "segment_sizes": torch.Tensor of segment sizes
                    - "metadata": dict of context-specific metadata
            }
        step: Current training step
        largest_context_len: Maximum context length
    
    Returns:
        dict: Metrics including spearman correlations for different configurations
    """
    def score_dataloader_with_memory(memory, reverse=False):
        """Score all variants in dataloader with given memory"""
        all_scores = []
        all_batch_info = []
        
        for batch in tqdm(data_loader, desc="Scoring variants"):
            batch_variants = batch["sequence"]
            variant_samples = [
                model.sample_and_fit_sequences(
                    query_id_or_sequence=variant,
                    include_query=True,
                    max_seq_length=largest_context_len
                )
                for variant in batch_variants
            ]
            
            variant_packed = [
                model.pack_inputs(
                    sample=sample,
                    tokenizer=model.reader_decoder_tokenizer,
                    max_seq_length=largest_context_len,
                    skip_padding=False,
                    reverse_sequence=reverse
                )
                for sample in variant_samples
            ]
            
            collated_variants = model.padded_collate_packed(variant_packed, fabric=fabric)
            logits = model.reader.logits(
                collated_variants["tokens"][:, :-1],
                memory,
                preallocated_memory=True
            )
            targets = collated_variants["tokens"][:, 1:]
            
            losses = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                targets,
                ignore_index=model.reader_decoder_tokenizer.mask_token,
                reduction='none'
            ).float().sum(dim=1)
            
            all_scores.append(-losses)
            all_batch_info.append({
                "mutant": batch["mutant"],
                "DMS_score": batch["DMS_score"],
                "id": batch.get("id", None)
            })
            
        return torch.cat(all_scores), all_batch_info

    model.eval()
    dataset_wpred = []
    dataset_detailed = []
    metrics = defaultdict(list)
    model.reader = model.reader.half()
    all_ensemble_scores = []
    
    # Process each context
    for context in tqdm(context_data["contexts"], desc="Processing contexts"):
        # Generate memory from context
        memory = model.reader.embed(
            context["tokens"],
            context["segment_sizes"]
        )
        memory = model.reader.logits_allocate_memory(
            memory=memory,
            batch_size=opt.per_gpu_batch_size_eval,
            length=largest_context_len,
        )
        
        # Score variants
        forward_scores, batch_info = score_dataloader_with_memory(memory, reverse=False)
        forward_scores_list = forward_scores.tolist()
        
        if opt.reverse_sequence:
            reverse_scores, _ = score_dataloader_with_memory(memory, reverse=True)
            reverse_scores_list = reverse_scores.tolist()
            this_scores = (forward_scores + reverse_scores) / 2
        else:
            reverse_scores_list = None
            this_scores = forward_scores
            
        all_ensemble_scores.append(this_scores)
        
        # Store detailed results
        for idx in range(len(forward_scores)):
            detailed_entry = {
                "mutant": batch_info[idx // opt.per_gpu_batch_size_eval]["mutant"][idx % opt.per_gpu_batch_size_eval],
                "DMS_score": batch_info[idx // opt.per_gpu_batch_size_eval]["DMS_score"][idx % opt.per_gpu_batch_size_eval],
                **context["metadata"],  # Include context-specific metadata
                "forward_score": forward_scores_list[idx],
            }
            
            if batch_info[idx // opt.per_gpu_batch_size_eval]["id"] is not None:
                detailed_entry["id"] = batch_info[idx // opt.per_gpu_batch_size_eval]["id"][idx % opt.per_gpu_batch_size_eval]
                
            if reverse_scores_list is not None:
                detailed_entry["reverse_score"] = reverse_scores_list[idx]
                detailed_entry["average_score"] = (forward_scores_list[idx] + reverse_scores_list[idx]) / 2
            
            dataset_detailed.append(detailed_entry)
        
        # Store basic info for final results if this is first run
        if len(dataset_wpred) == 0:
            for idx, scores in enumerate(this_scores):
                ex = {
                    "mutant": batch_info[idx // opt.per_gpu_batch_size_eval]["mutant"][idx % opt.per_gpu_batch_size_eval],
                    "DMS_score": batch_info[idx // opt.per_gpu_batch_size_eval]["DMS_score"][idx % opt.per_gpu_batch_size_eval],
                }
                if batch_info[idx // opt.per_gpu_batch_size_eval]["id"] is not None:
                    ex["id"] = batch_info[idx // opt.per_gpu_batch_size_eval]["id"][idx % opt.per_gpu_batch_size_eval]
                dataset_wpred.append(ex)
                
        del memory
        torch.cuda.empty_cache()
    
    final_scores = torch.stack(all_ensemble_scores).mean(dim=0)
    
    # Add final scores to dataset
    for idx, score in enumerate(final_scores):
        dataset_wpred[idx]["likelihood"] = score.item()

    # Save results
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt, fabric)
        detailed_dataset_name = f"{dataset_name}-detailed"
        util.save_distributed_dataset(dataset_detailed, detailed_dataset_name, opt, fabric)
        
        metrics["spearman"], metrics["msd"] = util.calculate_metrics_from_saved_data(dataset_name, opt)
        print(metrics["spearman"], metrics["msd"])
        util.calculate_detailed_metrics_from_saved_data(detailed_dataset_name, opt, metrics)
        logger.info(f"Spearman: {metrics['spearman']}, MSD: {metrics['msd']}")

    return metrics

@torch.no_grad()
def run_retrieval_only(model, index, opt, data_loader, data_path, step=None):
    """
    Runs retrieval only (without generating answers) and saves the fasta dataset is .
    """
    model.eval()

    
    dataset_wpred = []
    model = util.get_model_if_wrapped(model)

    for batch in tqdm(data_loader, desc="Retrieving passages for batches"):
        query = batch.get("sequence", [""])
        batch_metadata = batch.get("id")
        query_enc = model.retriever_tokenize(query)

        # Move tensors to the appropriate device
        query_ids_retriever = query_enc["input_ids"]
        query_mask_retriever = query_enc["attention_mask"]

        retrieved_passages, _ = model.retrieve(
            index,
            opt.n_context_eval,
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata=batch_metadata,
            filtering_fun=None,  # No filtering function
            fasta_dataset = opt.fasta_dataset
        )

        # If example is a padding example then skip step
        if not query or not query[0]:
            continue

        for k in range(len(retrieved_passages)):
            if opt.write_results:
                ex = {"query": query[k], "passages": retrieved_passages[k]}
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)
    return {}


def run_evaluation(model, index, opt, fabric, step=None):
    """
    Run evaluation based on the specified evaluation modes.
    Returns a nested dictionary with metrics for each evaluation mode.
    """
    all_metrics = {
        "contact": {},
        "likelihood": {},
        "dms": {}
    }

    # Contact prediction evaluation
    if "contact" in opt.eval_modes:
        pass

    # Likelihood evaluation
    if "likelihood" in opt.eval_modes:
        # data_loader = _get_eval_data_loader_cluster(opt, n_context=opt.n_context, num_samples=100)
        data_loader = load_dataloader(opt, 
                            min_members=opt.n_context, 
                            n_context=opt.n_context,
                            batch_size=opt.per_gpu_batch_size_eval,
                            split="val",
                            max_batches=100,
                            sequential_val=False #TODO: add as a param for eval
                            )
        data_path = "likelihood_eval.txt"
        
        metrics = evaluate_likelihood(
            model=model,
            data_loader=data_loader,
            opt=opt,
            fabric=fabric,
            data_path=data_path,
            index=index,
            step=step,
            query_only_loss=opt.query_only_loss
        )
        all_metrics["likelihood"] = metrics

    # DMS evaluation
    if "dms" in opt.eval_modes:
        if len(opt.eval_dms_files) != len(opt.eval_query):
            raise ValueError("The number of evaluation data files does not match the number of queries")
        
        dms_metrics = {}
        for i, (data_path, eval_query) in enumerate(zip(opt.eval_dms_files, opt.eval_query)):
            dataset_name = os.path.basename(data_path)
            data_loader, largest_context_len = _get_eval_data_loader_dms(opt, data_path)
            fabric.setup_dataloaders(data_loader)

            if opt.retrieve_only:
                run_retrieval_only(model, index, opt, data_loader, data_path, step)
            else:
                # Get context data based on configuration
                if opt.eval_msa_files:
                    # Use MSA-based context if MSA files are provided
                    logger.info(f"Using MSA context for evaluation from {opt.eval_msa_files[i]}")
                    context_data = get_context_from_msa(
                        model=model,
                        opt=opt,
                        msa_fasta_path=opt.eval_msa_files[i]
                    )
                else:
                    # Use retrieval-based context if no MSA files
                    logger.info(f"Using retrieval context for evaluation with query: {eval_query}")
                    context_data = get_context_from_index(
                        model=model,
                        index=index,
                        opt=opt,
                        eval_query=eval_query
                    )

                # Evaluate using the provided context
                metrics = evaluate_dms_poet(
                    model=model,
                    data_loader=data_loader,
                    opt=opt,
                    fabric=fabric,
                    data_path=data_path,
                    context_data=context_data,
                    step=step,
                    largest_context_len=largest_context_len
                )
                dms_metrics[dataset_name] = metrics

            fabric.barrier()
        
        all_metrics["dms"] = dms_metrics
    return all_metrics

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()
    util.process_options_ref_file(opt)

    torch.manual_seed(opt.seed)

    # Initialize Fabric
    fabric = Fabric(accelerator="cuda", strategy="ddp", precision=opt.precision)

    # Replace SLURM initialization with Fabric setup
    fabric.launch()

    checkpoint_path, saved_index_path, save_initial_index_path = create_checkpoint_directories(opt)

    logger = util.init_basic_logger(fabric.is_global_zero, fabric.world_size > 1, os.path.join(checkpoint_path, "run.log"))
    if fabric.is_global_zero:
        options.print_options(opt)

    logger.info(f"world size: {fabric.world_size}")

    logger.info("opt.load_index_path: {}".format(opt.load_index_path))
    logger.info("opt.msa_dir_eval: {}".format(opt.msa_dir_eval))
    if not opt.msa_dir_eval:
        if opt.fasta_dataset:
            instantiate_fasta_dataset(opt)
        index, passages = load_or_initialize_index(opt)

    else:
        index, passages = None, None

    model, _, _, _, _, opt, step = load_or_initialize_protriever_model(opt, fabric, eval_only=True)
    model = fabric.setup(model)
    model.module.set_fabric(fabric)
    model.mark_forward_method("retrieve")
    model.mark_forward_method("sample_and_fit_sequences")
    model.mark_forward_method("build_index")
    fabric.barrier()



    if not opt.use_file_passages and not opt.msa_dir_eval and opt.load_index_path is None:
        indexing_start = time.time()
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger, step)

        # if opt.save_index_path is not None:
        #     save_embeddings_and_index(index, opt)

    # For case of retrieval only
    # if not opt.eval_query:
    #     opt.eval_query = [""] * len(opt.eval_data)

    logger.info("Start Evaluation for step {}".format(step))
    all_metrics = run_evaluation(model, index, opt, fabric, step)
    # Log validation metrics
    log_message = util.log_validation_metrics(step, all_metrics, fabric, opt)
    logger.info(log_message)

    logger.info("Finished evaluation")
