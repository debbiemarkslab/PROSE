# train.py

import logging
import os
import random
import time
from collections import defaultdict

# Import necessary modules
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

torch._dynamo.config.optimize_ddp = False

# when running compilation
# torch._logging.set_logs(dynamo=logging.DEBUG)

import contextlib
import inspect
import io
import subprocess
from pprint import pprint

import debugpy

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # For training, use high precision for better performance
    torch.set_float32_matmul_precision("medium")
    logger.info("Set float32 matmul precision to medium for training")


def log_memory_status(step, location):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    max_reserved = torch.cuda.max_memory_reserved() / 1e9

    logger.info(f"Step {step} - {location}:")
    logger.info(f"  Allocated: {allocated:.3f} GB")
    logger.info(f"  Reserved: {reserved:.3f} GB")
    logger.info(f"  Max Allocated: {max_allocated:.3f} GB")
    logger.info(f"  Max Reserved: {max_reserved:.3f} GB")

    # Capture the output of memory_summary
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        torch.cuda.memory_summary(device=None, abbreviated=False)
        memory_summary = buf.getvalue()

    # Log the memory summary
    logger.info(f"Detailed Memory Summary for Step {step} - {location}:")
    if memory_summary:
        logger.info(memory_summary)
    else:
        logger.info("No detailed memory summary available.")

    # Add a direct print of the memory summary to debug
    print(f"Debug - Memory Summary for Step {step} - {location}:")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    torch.cuda.reset_peak_memory_stats()


def train(
    model,
    index,
    passages,
    optimizer,
    scheduler,
    opt,
    checkpoint_path,
    data_loader,
    fabric,
):
    logger.info("Starting training loop")
    logger.info(f"opt.train_retriever is {opt.train_retriever}")
    fabric.seed_everything(opt.seed)
    alpha = 1.0  # Weighting between reader and retriever loss

    if opt.use_gradient_checkpoint_reader:
        model.reader.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info(
            f"Checking gradient checkpointing for reader: {model.reader.is_gradient_checkpointing}"
        )

    if opt.use_gradient_checkpoint_retriever:
        model.retriever.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info(
            f"Checking gradient checkpointing for retriever: {model.retriever.is_gradient_checkpointing}"
        )

    # Calculate total steps for initial training and fine-tuning
    total_steps = opt.total_steps + opt.final_finetune_steps

    # Initialize the DataLoader iterator
    data_iter = iter(data_loader)

    for step in range(total_steps + 1):
        # logger.info(f"step is {step}")
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reinitialize the DataLoader if we've exhausted it
            logger.info("Reinitializing DataLoader")
            data_iter = iter(data_loader)
            batch = next(data_iter)

        batch_size = len(batch["id"])  # TODO: make this better
        iter_stats = {}
        model.train()

        # logger.info(f"batch is {batch}")
        if not opt.use_file_passages and index_refresh_scheduler.is_time_to_refresh(
            step
        ):
            logger.info(f"index is trained: {index.is_index_trained()}")
            if step == 0 and index.is_index_trained():
                logger.info("index is already trained, skipping index building")
                continue
            elif step == 0 and not index.is_index_trained():
                # Build initial index if not prebuilt.
                indexing_start = time.time()
                logger.info(
                    "Building initial index from already calculated embeddings."
                )
                # Saves embeddings to opt.embeddings_path
                model.build_index(
                    index=index,
                    passages=passages,
                    gpu_embedder_batch_size=opt.per_gpu_embedder_batch_size,
                    logger=logger,
                    step=step,
                    from_scratch=True,
                )
                iter_stats["runtime/indexing"] = (time.time() - indexing_start, 1)
            else:
                # Refresh index.
                indexing_start = time.time()
                logger.info("Building index")
                # Saves embeddings to "opt.embeddings_path.{opt.name}/step_{step}"
                model.build_index(
                    index=index,
                    passages=passages,
                    gpu_embedder_batch_size=opt.per_gpu_embedder_batch_size,
                    logger=logger,
                    step=step,
                )
                iter_stats["runtime/indexing"] = (time.time() - indexing_start, 1)

        if opt.final_finetune_steps > 0 and step >= (
            opt.total_steps - opt.final_finetune_steps
        ):
            if step == (opt.total_steps - opt.final_finetune_steps):
                model.reader.n_context = opt.final_n_context
                # Update batch size (you'll need to handle this in your data loader)
                if hasattr(data_loader, "batch_sampler"):
                    data_loader.batch_sampler.batch_size = opt.final_batch_size
                else:
                    data_loader.batch_size = opt.final_batch_size

                # Optionally adjust learning rate for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.1  # Reduce LR by 10x

                # Force index refresh before fine-tuning
                if not opt.use_file_passages:
                    logger.info("Refreshing index before final fine-tuning")
                    model.module.build_index(
                        index=index,
                        passages=passages,
                        gpu_embedder_batch_size=opt.per_gpu_embedder_batch_size,
                        logger=logger,
                        step=step,
                    )

        train_step_start = time.time()

        # Determine if we're accumulating gradients
        is_accumulating = (step + 1) % opt.accumulation_steps != 0

        # Use no_backward_sync for efficient gradient accumulation
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # print(f'number of passages per query: {len(batch["passages"][0])}')
            # target = batch["sequence"]
            batch_metadata = batch.get("id", None)
            # logger.info(f"batch_metadata is {batch_metadata}")
            # print(f"we are in rank {fabric.global_rank} and id are {batch_metadata}")
            # print(f'step is {step} and is_accumulating is {is_accumulating}')
            # print(f"batch is {batch}")

            # Tokenize targets
            reverse = False
            if opt.reverse_sequence:
                if random.random() < 0.5:
                    reverse = True

            outputs = model(
                index=index,
                query_ids=batch["id"],
                passages=batch["passages"] if opt.use_file_passages else None,
                passages_weights=batch["weights"] if opt.use_file_passages else None,
                metadata=batch_metadata,
                filtering_fun=self_match_filter,
                train_retriever=opt.train_retriever
                and step > opt.freeze_retriever_steps,
                iter_stats=iter_stats,
                fasta_dataset=opt.fasta_dataset,
                reverse_sequence=reverse,
                max_individual_seq_length=opt.max_individual_seq_length,
                query_only_loss=opt.query_only_loss,  # Add this line
            )

            reader_loss = outputs["reader_loss"]
            retriever_loss = outputs["retriever_loss"]

            if retriever_loss is not None and opt.train_retriever:
                train_loss = alpha * reader_loss.float() + retriever_loss
            else:
                train_loss = reader_loss

            # Scale loss for gradient accumulation
            train_loss = train_loss / opt.accumulation_steps

            # Backward pass
            backward_start = time.time()
            # print(f'train_loss is {train_loss}')
            fabric.backward(train_loss)
            iter_stats["runtime/backward"] = (time.time() - backward_start, 1)

        # Update model if we're not accumulating or if it's the last step
        if not is_accumulating or step == opt.total_steps:
            # Clip gradients
            fabric.clip_gradients(model, optimizer, max_norm=opt.clip)
            if opt.train_retriever:
                fabric.clip_gradients(model, retr_optimizer, max_norm=opt.clip)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            if opt.train_retriever:
                retr_optimizer.step()
                retr_scheduler.step()
            # Zero gradients
            # print(f'step is {step} and zeroing gradients')
            optimizer.zero_grad()
            if opt.train_retriever:
                retr_optimizer.zero_grad()

        # print(f'train_loss is {train_loss}')
        if retriever_loss is not None and opt.train_retriever:
            iter_stats["loss/reader_loss"] = (
                reader_loss.item() * opt.accumulation_steps,
                batch_size,
            )
            iter_stats["loss/retriever_loss"] = (
                retriever_loss.item() * opt.accumulation_steps,
                batch_size,
            )
        iter_stats["loss/train_loss"] = (
            train_loss.item() * opt.accumulation_steps,
            batch_size,
        )
        iter_stats["runtime/train_step"] = (time.time() - train_step_start, 1)
        run_stats.update(iter_stats)

        # Logging
        if step % opt.log_freq == 0 and step > 0:
            log = util.log_training_metrics(
                step, run_stats.average_stats, fabric, scheduler=scheduler, opt=opt
            )
            logger.info(log)
            run_stats.reset()

        # Saving checkpoints
        if (
            step % opt.save_freq == 0
            and step > 0
            or step == opt.total_steps
            or step == total_steps
        ):
            checkpoint_name = (
                "final-pretrain"
                if step == opt.total_steps
                else "final-finetune"
                if step == total_steps
                else f"step-{step}"
            )
            logger.info(f"Saving checkpoint: {checkpoint_name}")
            # save_protriever_model(
            #     model,
            #     optimizer,
            #     scheduler,
            #     retr_optimizer,
            #     retr_scheduler,
            #     step,
            #     opt,
            #     checkpoint_path,
            #     f"step-{step}",
            #     fabric
            # )
            save_protriever_model_no_fabric(
                model,
                optimizer,
                scheduler,
                retr_optimizer,
                retr_scheduler,
                step,
                opt,
                checkpoint_path,
                checkpoint_name,
                fabric,
            )

        # Evaluation: add and step > 0 to avoid eval at step 0
        if (
            step % opt.eval_freq == 0
            and step > 0
            or step == opt.total_steps
            or step == total_steps
        ):
            eval_prefix = (
                "[Final Pretrain] "
                if step == opt.total_steps
                else "[Final Finetune] "
                if step == total_steps
                else ""
            )
            logger.info(f"{eval_prefix}Running evaluation at step {step}")
            all_metrics = run_evaluation(model, index, opt, fabric, step)
            log_message = util.log_validation_metrics(step, all_metrics, fabric, opt)
            logger.info(f"{eval_prefix}{log_message}")
            model.train()
            model.float()  # Ensure model is in float32 mode TODO: consistent

    logger.info("Training completed.")


if __name__ == "__main__":
    options = get_options()
    opt = options.parse()
    util.process_options_ref_file(opt)
    (
        checkpoint_path,
        saved_index_path,
        saved_initial_index_path,
    ) = create_checkpoint_directories(opt)

    # Initialize Fabric loggers
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(opt.checkpoint_dir, "tensorboard")
    )
    csv_logger = CSVLogger(save_dir=os.path.join(opt.checkpoint_dir, "csv"))

    # Create a list to hold our loggers
    loggers = [tb_logger, csv_logger]
    # Conditionally add WandbLogger
    if opt.wandb_key is not None:
        os.environ[
            "WANDB_API_KEY"
        ] = opt.wandb_key  # Ensure this environment variable is set
        wandb_logger = WandbLogger(
            project=opt.wandb_project, name=opt.name, config=vars(opt)
        )
        loggers.append(wandb_logger)
    else:
        logger.info("Wandb key not provided. Wandb logging will be disabled.")

    static_graph = False

    fabric = Fabric(
        accelerator="gpu",
        loggers=loggers,
        precision=opt.precision,
        # devices=4,
        # strategy="ddp",
        strategy=DDPStrategy(
            process_group_backend="gloo",
            find_unused_parameters=True,
            static_graph=static_graph,
        ),
    )

    # also tried gloo
    # seem to have to trigger static_graph to true when training retriever but if not should be put to false
    # Replace SLURM initialization with Fabric setup
    fabric.launch()
    torch.autograd.set_detect_anomaly(True)

    logger = util.init_basic_logger(
        fabric.is_global_zero,
        fabric.world_size > 1,
        os.path.join(checkpoint_path, "run.log"),
    )

    if fabric.is_global_zero:
        options.print_options(opt)
    fabric.barrier()

    logger.info(f"world size: {fabric.world_size}")

    data_loader = load_dataloader(
        opt,
        min_members=opt.cluster_min_members,
        n_context=opt.n_context,
        batch_size=opt.per_gpu_batch_size,
    )
    # Wrap the data loader with Fabric
    data_loader = fabric.setup_dataloaders(data_loader)

    index, passages = load_or_initialize_index(opt)
    (
        model,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        opt,
        step,
    ) = load_or_initialize_protriever_model(opt, fabric, eval_only=False)

    logger.info(f"going here, before compile")
    if opt.compile:
        # compile_config = {
        #     "dynamic": True,  # Allow dynamic shapes
        #     "fullgraph": False,
        #     "mode": "reduce-overhead"
        # }
        compile_config = {
            "dynamic": True,
        }
        # self.reader.encoder = torch.compile(self.reader.encoder)
        # self.reader.decoder = torch.compile(self.reader.decoder)
        model.reader = torch.compile(model.reader)

    # Setup model and optimizers with Fabric
    if retr_optimizer is not None:
        model, optimizer, retr_optimizer = fabric.setup(
            model, optimizer, retr_optimizer
        )
    else:
        model, optimizer = fabric.setup(model, optimizer)

    logger.info(f"fabric has set up model and optimizer")
    model.mark_forward_method(
        "build_index"
    )  # Changed from model.build_index to 'build_index'
    model.mark_forward_method("retrieve")

    # Set fabric instance in the model
    model.fabric = fabric  # Use model.module because of DDP wrapper

    logger.info("Start training")
    fabric.barrier()

    train(
        model,
        index,
        passages,
        optimizer,
        scheduler,
        opt,
        checkpoint_path,
        data_loader,
        fabric,
    )
