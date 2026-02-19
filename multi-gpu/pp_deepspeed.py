"""
Pipeline-parallel fine-tuning of Llama-3.1-8B with DeepSpeed PipelineModule.
Run: deepspeed --num_gpus=4 multi-gpu/pp_deepspeed.py
     deepspeed --num_gpus=4 multi-gpu/pp_deepspeed.py --ds_config path/to/ds.json
"""

import os
import gc
import json
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset


# === CONFIG ===
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B"
DATA_PATH   = "alpaca_gpt4_data.json"
MAX_SEQ_LEN = 1024
BATCH_SIZE  = 1     # per-device (per-GPU) micro-batch size
GRAD_ACCUM  = 8     # micro-batches accumulated before one optimizer step
NUM_STAGES  = 4     # pipeline stages — must equal --num_gpus
NUM_LAYERS  = 32    # Llama-3.1-8B has 32 decoder layers
LR          = 3e-6
EPOCHS      = 3
LOG_STEPS   = 5
OUTPUT_DIR  = "./outputs/pp_deepspeed"


# === DATA ===
def load_alpaca_dataset(path=DATA_PATH, eval_size=1000):
    with open(path) as f:
        data = json.load(f)
    random.shuffle(data)
    train, eval_ = data[:-eval_size], data[-eval_size:]
    return preprocess(train), preprocess(eval_)

def preprocess(dataset):
    texts = []
    for row in dataset:
        if row["input"] == "":
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}"
            ).format_map(row)
        else:
            text = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            ).format_map(row)
        texts.append(text)
    return Dataset.from_dict({"text": texts})

def pack(tokenizer, hf_dataset, max_seq_len=MAX_SEQ_LEN):
    """
    Flatten all examples into one long token stream, then slice into
    fixed-length chunks of max_seq_len tokens.
    input_ids = chunk[:-1], labels = chunk[1:] — already shifted by 1 for next-token prediction.
    No padding needed — every chunk is exactly max_seq_len tokens long.
    """
    all_ids = []
    for text in hf_dataset["text"]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)  # mark the boundary between examples
        all_ids.extend(ids)

    packed = []
    for i in range(0, len(all_ids) - max_seq_len, max_seq_len):
        chunk = all_ids[i : i + max_seq_len + 1]
        packed.append({"input_ids": chunk[:-1], "labels": chunk[1:]})
    return packed


class PackedDataset(torch.utils.data.Dataset):
    """
    Thin wrapper that presents packed chunks as a PyTorch Dataset.

    DeepSpeed PipelineModule expects each item to be a (inputs, labels) tuple:
      • inputs  — fed into the first pipeline stage (EmbeddingStage)
      • labels  — routed directly to the loss function on the last stage
    Intermediate stages never see the labels; DeepSpeed manages the routing.
    """
    def __init__(self, packed_data):
        self.data = packed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels    = torch.tensor(item["labels"],    dtype=torch.long)
        return input_ids, labels   # (inputs, labels) — the pipeline contract


# === MODEL CACHE ===
# When DeepSpeed builds a PipelineModule, it calls LayerSpec.build() — i.e. cls(*args) —
# ONLY on the rank(s) that host each layer. By passing model_name + layer_idx as
# constructor args (picklable), each rank loads the base model exactly ONCE via this
# shared cache, regardless of how many layers that rank hosts. After PipelineModule
# construction, call _free_model_cache() to reclaim the CPU RAM — the layer objects
# extracted by each wrapper instance remain alive because those wrappers hold
# Python references to them.
_MODEL_CACHE: dict = {}

def _get_or_load_model(model_name: str) -> AutoModelForCausalLM:
    """Return the cached base model, loading from disk if not yet cached on this rank."""
    if model_name not in _MODEL_CACHE:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[rank {rank}] Caching {model_name} (layers for this stage will be kept after build)...")
        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,   # stream weight tensors one-by-one to cap peak CPU RAM
        )
        # Untie embed_tokens ↔ lm_head before splitting across stages.
        # Llama ties these by default (tie_word_embeddings=True) to save memory.
        # After partitioning they live on different GPUs — independent copies are required.
        if m.config.tie_word_embeddings:
            m.lm_head.weight = nn.Parameter(m.lm_head.weight.detach().clone())
        _MODEL_CACHE[model_name] = m
    return _MODEL_CACHE[model_name]

def _free_model_cache():
    """
    Release the CPU model cache after all LayerSpecs on this rank have been built.
    Each wrapper already holds a reference to its extracted layer object, so those
    tensors remain alive and the garbage collector will not free them.
    After this call, each rank holds only its stage's parameters (~2–3 GB instead of ~16 GB).
    """
    _MODEL_CACHE.clear()
    gc.collect()
    print(f"[rank {dist.get_rank()}] Model cache freed — only stage layers remain in CPU memory.")


# === PIPELINE STAGE DEFINITIONS ===
# DeepSpeed's PipelineModule treats the model as a flat sequence of nn.Module layers.
# It partitions this sequence across NUM_STAGES GPUs and orchestrates a GPipe-style
# micro-batch schedule:
#
#   Micro-batch  │  Stage 0   Stage 1   Stage 2   Stage 3
#   ─────────────┼──────────────────────────────────────────
#   mb 0 forward │  F0  →     F0  →     F0  →     F0 → loss
#   mb 1 forward │  F1  →     F1  →     F1  →     F1 → loss
#   ...
#   mb 7 backward│  B7  ←     B7  ←     B7  ←     B7
#   optimizer    │ ──────────── step ─────────────────────
#
# Only hidden_states flow between stages; attention masks and position_ids are
# reconstructed locally inside each wrapper to avoid extra inter-stage communication.
#
# Layer layout for Llama-3.1-8B (32 decoder layers) across 4 stages:
#   Stage 0 : EmbeddingStage  + layers[ 0.. 7]   (9 modules)
#   Stage 1 : layers[ 8..15]                      (8 modules)
#   Stage 2 : layers[16..23]                      (8 modules)
#   Stage 3 : layers[24..31] + FinalNormAndLMHead (9 modules)

class EmbeddingStage(nn.Module):
    """
    First pipeline stage — token IDs → dense hidden states.
    Llama uses learned token embeddings; RoPE is applied inside each attention layer.
    Constructor arg is model_name (a string) so LayerSpec stays picklable.
    """
    def __init__(self, model_name):
        super().__init__()
        # _get_or_load_model hits the cache on all but the first call on this rank
        self.embed_tokens = _get_or_load_model(model_name).model.embed_tokens

    def forward(self, input_ids):
        # input_ids : (batch, seq_len) — integer token IDs from the DataLoader
        # returns   : (batch, seq_len, hidden_size) — float hidden states for stage 1
        return self.embed_tokens(input_ids)


class LlamaLayerWrapper(nn.Module):
    """
    Wraps a single LlamaDecoderLayer to be compatible with PipelineModule.

    PipelineModule requires each layer to accept one tensor and return one tensor.
    LlamaDecoderLayer normally receives (hidden_states, attention_mask, position_embeddings, ...)
    and returns a tuple. This wrapper:
        1. Accepts only hidden_states from the previous stage.
        2. Reconstructs position_ids locally (arange over seq_len) — no cross-stage data needed.
        3. Computes rotary position embeddings (cos, sin) via self.rotary_emb.
        4. Passes attention_mask=None (valid for packed sequences with no padding).
        5. Extracts and returns only hidden_states from the output tuple.
    Constructor args are picklable (model_name: str, layer_idx: int).

    Note on position_embeddings vs position_ids:
        Transformers ≥ 4.45 refactored the Llama attention API. LlamaAttention.forward() no
        longer computes RoPE internally from position_ids — it requires pre-computed
        position_embeddings=(cos, sin) from the model-level rotary_emb module. Passing only
        position_ids leaves position_embeddings=None, which causes a TypeError at runtime.
    """
    def __init__(self, model_name, layer_idx):
        super().__init__()
        m = _get_or_load_model(model_name)
        self.layer      = m.model.layers[layer_idx]
        self.rotary_emb = m.model.rotary_emb   # LlamaRotaryEmbedding; lives on this stage's GPU

    def forward(self, hidden_states):
        seq_len      = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # Compute (cos, sin) for this sequence length using the model's rotary embedding module.
        # Required by LlamaAttention.forward() as of transformers 4.45+ — passing only
        # position_ids no longer works because RoPE is now applied at the model level.
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # attention_mask=None → full (causal) attention across all tokens in the chunk.
        # This is correct because packing removes padding — every token is real data.
        outputs = self.layer(
            hidden_states,
            attention_mask=None,
            position_embeddings=(cos, sin),
        )
        return outputs[0]   # LlamaDecoderLayer returns (hidden_states, *extras); keep only [0]


class FinalNormAndLMHead(nn.Module):
    """
    Last pipeline stage — hidden states → vocabulary logits.
    Combines the final RMS norm and the linear vocab projection on the same GPU (stage 3),
    avoiding one extra pipeline communication hop.
    Constructor arg is model_name (a string) so LayerSpec stays picklable.
    """
    def __init__(self, model_name):
        super().__init__()
        m = _get_or_load_model(model_name)
        self.norm    = m.model.norm   # LlamaRMSNorm (final layer norm)
        self.lm_head = m.lm_head     # nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(self.norm(hidden_states))  # (batch, seq_len, vocab_size)


def loss_fn(logits, labels):
    """
    Cross-entropy loss for next-token prediction.
    DeepSpeed calls this automatically on the last stage with:
      logits  — output of FinalNormAndLMHead  (batch, seq_len, vocab_size)
      labels  — the labels tensor from the (inputs, labels) dataset tuple

    pack() already shifts labels by 1: labels[i] = next token after input_ids[i].
    No additional shifting is needed here — logits[i] and labels[i] are already aligned.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,   # skip any positions marked as padding
    )


def main():
    # === ARG PARSING ===
    # The deepspeed launcher injects --local_rank into sys.argv; always parse it.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Injected automatically by the deepspeed launcher.")
    parser.add_argument("--ds_config", type=str, default=None,
                        help="Optional path to an external DeepSpeed JSON config. "
                             "If omitted, the inline DS_CONFIG dict below is used.")
    args = parser.parse_args()

    if not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    # === DATA ===
    if rank == 0:
        print("Loading dataset...")
    train_hf, _ = load_alpaca_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_packed  = pack(tokenizer, train_hf)
    train_dataset = PackedDataset(train_packed)

    # Steps per epoch: each engine.train_batch() call consumes BATCH_SIZE × GRAD_ACCUM samples.
    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps     = EPOCHS * steps_per_epoch
    warmup_steps    = int(0.1 * total_steps)

    if rank == 0:
        print(f"Packed train chunks: {len(train_packed):,}  |  steps/epoch: {steps_per_epoch}")

    # === MODEL LOADING ===
    # Preload the base model into _MODEL_CACHE one rank at a time.
    # This staggers disk I/O so ranks don't compete for CPU bandwidth simultaneously.
    # Peak RAM note: earlier ranks keep their cache alive until _free_model_cache() below,
    # so the peak across all ranks is still world_size × model_size in the worst case —
    # but the staggering prevents simultaneous disk reads and reduces CPU saturation.
    if rank == 0:
        print("Loading model (staggered across ranks to reduce CPU I/O contention)...")
    for r in range(world_size):
        if rank == r:
            _get_or_load_model(MODEL_NAME)
        dist.barrier()   # rank r+1 begins loading only after rank r finishes

    # === PIPELINE MODULE ===
    # LayerSpec(cls, *args) stores only a class reference + picklable constructor args.
    # PipelineModule calls LayerSpec.build() = cls(*args) ONLY on the rank that hosts
    # each layer — so _get_or_load_model() is a cache hit (no re-download).
    pipeline_layers = (
        [LayerSpec(EmbeddingStage, MODEL_NAME)]
        + [LayerSpec(LlamaLayerWrapper, MODEL_NAME, i) for i in range(NUM_LAYERS)]
        + [LayerSpec(FinalNormAndLMHead, MODEL_NAME)]
    )

    if rank == 0:
        print(f"Pipeline: {len(pipeline_layers)} total modules → {NUM_STAGES} stages "
              f"(~{len(pipeline_layers) // NUM_STAGES} per stage)")

    pipe_model = PipelineModule(
        layers=pipeline_layers,
        loss_fn=loss_fn,
        num_stages=NUM_STAGES,
        partition_method="uniform",  # divide the layer list into equal-length chunks
    )

    # All LayerSpecs for this rank have been built — release the base model cache.
    # Each stage wrapper already holds its own reference to its extracted layer objects,
    # so those tensors remain alive. Steady-state RAM drops from ~16 GB to ~2–3 GB per rank.
    _free_model_cache()

    # === DEEPSPEED CONFIG ===
    # Inline dict is the default. Pass --ds_config to override with an external JSON file.
    #
    # train_batch_size = per_device_batch × dp_degree × grad_accum
    #   per_device_batch (train_micro_batch_size_per_gpu) : samples per GPU per forward pass
    #   dp_degree                                         : number of data-parallel replicas
    #   grad_accum       (gradient_accumulation_steps)    : micro-batches before one optimizer step
    #
    # For pure pipeline parallelism (all GPUs in one pipeline, dp_degree = 1):
    #   train_batch_size = 1 × 1 × 8 = 8
    # For hybrid PP + DP (e.g. 8 GPUs, 4-stage pipeline, 2 dp replicas):
    #   dp_degree = num_gpus / num_stages = 2  →  train_batch_size = 1 × 2 × 8 = 16
    #
    # DeepSpeed validates: train_batch_size == micro_batch × grad_accum × dp_world_size.
    # Values MUST be consistent or DeepSpeed raises an assertion error.
    if args.ds_config:
        with open(args.ds_config) as f:
            config = json.load(f)
        if rank == 0:
            print(f"Using external DeepSpeed config: {args.ds_config}")
    else:
        config = {
            "train_micro_batch_size_per_gpu": BATCH_SIZE,
            "gradient_accumulation_steps":    GRAD_ACCUM,
            "train_batch_size":               BATCH_SIZE * GRAD_ACCUM,   # dp_degree=1 → 1×8 = 8
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr":           LR,
                    "betas":        [0.9, 0.999],
                    "eps":          1e-8,
                    "weight_decay": 0.0,
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr":    0,
                    "warmup_max_lr":    LR,
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps":  total_steps,
                },
            },
            "bf16": {"enabled": True},
            "pipeline": {
                "activation_checkpoint_interval": 0, # optional: set to 1 to checkpoint activations every layer
            },
            "steps_per_print":      LOG_STEPS,
            "wall_clock_breakdown": False,
        }

    # === DEEPSPEED INITIALIZE ===
    # deepspeed.initialize() wraps PipelineModule in a PipelineEngine that:
    #   • assigns each LayerSpec to a GPU stage and moves its parameters there
    #   • implements the GPipe schedule (forward all micro-batches, then backward all)
    #   • manages gradient accumulation across GRAD_ACCUM micro-batches
    #   • calls the optimizer only after all micro-batches are accumulated
    #
    # Passing training_data lets DeepSpeed build the DataLoader with a DistributedSampler
    # matched to the data-parallel topology. engine.train_batch() takes no arguments.
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_model,
        config=config,
        training_data=train_dataset,
    )

    if rank == 0:
        print(f"\nEffective batch size: {BATCH_SIZE} (micro) × {GRAD_ACCUM} grad_accum = "
              f"{BATCH_SIZE * GRAD_ACCUM} samples/step")
        print(f"Total optimizer steps: {total_steps}  (warmup: {warmup_steps})\n")

    # === TRAINING LOOP ===
    # engine.train_batch() encapsulates one complete optimizer step:
    #   1. Fetches GRAD_ACCUM micro-batches from the internal DataLoader.
    #   2. GPipe forward  — each micro-batch flows stage 0 → 1 → 2 → 3; loss_fn on stage 3.
    #   3. GPipe backward — gradients flow stage 3 → 2 → 1 → 0 for each micro-batch.
    #   4. Gradients accumulated over all micro-batches → optimizer.step() → zero_grad().
    # Returns the average loss across the micro-batches (scalar tensor on the last rank).
    engine.train()
    global_step = 0

    for epoch in range(EPOCHS):
        if rank == 0:
            print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")

        for _ in range(steps_per_epoch):
            loss = engine.train_batch()   # one full forward → backward → optimizer step
            global_step += 1

            # Only rank 0 (first pipeline stage) prints — avoids duplicate lines.
            if rank == 0 and global_step % LOG_STEPS == 0:
                print(f"  step {global_step:>5} | loss {loss.item():.4f}")

        # Save a DeepSpeed checkpoint at the end of each epoch.
        # Each GPU writes its own stage's weights + optimizer state + scheduler state.
        engine.save_checkpoint(OUTPUT_DIR, tag=f"epoch_{epoch + 1}")
        if rank == 0:
            print(f"  Checkpoint saved → {OUTPUT_DIR}/epoch_{epoch + 1}/")

    # DeepSpeed pipeline checkpoints are sharded: each GPU saves its own stage weights.
    # To produce a standard HuggingFace model from these shards:
    #   1. Gather shards using zero_to_fp32.py or reassemble stage state_dicts manually.
    #   2. Load the merged state dict into AutoModelForCausalLM and call save_pretrained().
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nTraining complete. DeepSpeed checkpoints at: {OUTPUT_DIR}/")
        print("To convert to HuggingFace format: gather stage shards, load into "
              "AutoModelForCausalLM.from_pretrained(), call save_pretrained().")


if __name__ == "__main__":
    main()
