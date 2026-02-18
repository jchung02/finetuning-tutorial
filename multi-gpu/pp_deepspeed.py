"""
Description : Pipeline-parallel fine-tuning of Llama-3.1-8B with DeepSpeed PipelineModule.
Key concepts: PipelineModule, LayerSpec, LlamaLayerWrapper, GPipe microbatch schedule,
              inline DeepSpeed config dict, train_batch_size formula.
Run         : deepspeed --num_gpus=4 multi-gpu/pp_deepspeed.py
Requirements: pip install deepspeed transformers datasets torch
"""

# === IMPORTS ===
# 1. stdlib
import os
import json
import random
import argparse

# 2. torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# 3. DeepSpeed
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

# 4. HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset


# === CONFIG ===
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B"
DATA_PATH   = "alpaca_gpt4_data.json"
MAX_SEQ_LEN = 1024
BATCH_SIZE  = 1     # per-device (per-GPU) micro-batch size
GRAD_ACCUM  = 8     # micro-batches accumulated before one optimizer step
NUM_STAGES  = 4     # pipeline stages — must equal --num_gpus
LR          = 3e-6
EPOCHS      = 3
LOG_STEPS   = 5
OUTPUT_DIR  = "./outputs/pp_deepspeed"


# === SECTION 0: ARGUMENT PARSING ===
# deepspeed launcher injects --local_rank into sys.argv; always parse it.
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1,
                    help="Injected automatically by the deepspeed launcher.")
args = parser.parse_args()


# === SECTION 1: DATA LOADING & PACKING ===
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def pack(hf_dataset, max_seq_len=MAX_SEQ_LEN):
    """
    Flatten all examples into one long token stream, then slice into
    fixed-length chunks of (max_seq_len + 1) tokens.
    input_ids  = chunk[:-1]  (the model input)
    labels     = chunk[1:]   (next-token targets, shifted by 1)
    No padding is needed — every chunk is exactly max_seq_len tokens long.
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


train_hf, eval_hf = load_alpaca_dataset()
train_packed = pack(train_hf)

train_dataset = PackedDataset(train_packed)

# Steps per epoch: each train_batch() call consumes BATCH_SIZE × GRAD_ACCUM samples.
# (Pipeline stages are serial — all 4 GPUs process the same micro-batch, not different ones.)
steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
total_steps     = EPOCHS * steps_per_epoch
warmup_steps    = int(0.1 * total_steps)


# === SECTION 2: PIPELINE STAGE DEFINITIONS ===
#
# DeepSpeed's PipelineModule treats the model as a flat sequence of nn.Module layers.
# It partitions this sequence across NUM_STAGES GPUs and orchestrates a GPipe-style
# micro-batch schedule:
#
#   Micro-batch  │  Stage 0   Stage 1   Stage 2   Stage 3
#   ─────────────┼──────────────────────────────────────────
#   mb 0 forward │  F0        F0        F0        F0 → loss
#   mb 1 forward │  F1        F1        F1        F1 → loss
#   ...          │
#   mb 7 backward│  B7        B7        B7        B7
#   ...          │
#   optimizer    │ ──────────── step ──────────────────────
#
# Each layer must accept a single tensor (or tuple) and return a single tensor (or tuple).
# Only that tensor flows along the pipeline; attention masks and position_ids are
# reconstructed locally inside each wrapper to avoid inter-stage communication overhead.

class EmbeddingStage(nn.Module):
    """
    First pipeline stage — token IDs → dense hidden states.

    Llama uses learned token embeddings with no separate positional embedding table.
    Rotary position encodings (RoPE) are applied inside each attention layer.
    """
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens  # nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids):
        # input_ids : (batch, seq_len) — integer token IDs from the DataLoader
        # returns   : (batch, seq_len, hidden_size) — float hidden states for stage 1
        return self.embed_tokens(input_ids)


class LlamaLayerWrapper(nn.Module):
    """
    Wraps a single LlamaDecoderLayer to be compatible with PipelineModule.

    Why a wrapper?
      PipelineModule requires each layer to accept one tensor and return one tensor.
      LlamaDecoderLayer normally receives (hidden_states, attention_mask,
      position_ids, ...) and returns a tuple. This wrapper:
        1. Accepts only hidden_states from the previous stage.
        2. Reconstructs position_ids locally (arange over seq_len) — no cross-stage data needed.
        3. Passes attention_mask=None (valid for packed sequences with no padding).
        4. Extracts and returns only hidden_states from the output tuple.
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer  # one of model.model.layers[i]

    def forward(self, hidden_states):
        seq_len      = hidden_states.shape[1]
        # Reconstruct position_ids from sequence length.
        # Each token's position is simply its index in the sequence.
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # attention_mask=None → full (causal) attention across all tokens in the chunk.
        # This is correct because packing removes padding — every token is real data.
        outputs = self.layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
        )
        return outputs[0]   # LlamaDecoderLayer returns (hidden_states, *extras); keep only [0]


class FinalNormAndLMHead(nn.Module):
    """
    Last pipeline stage — hidden states → vocabulary logits.

    Combines the final RMS norm and the linear vocabulary projection so they
    always reside on the same GPU (stage 3), avoiding one extra pipeline hop.

    Note on weight tying:
      Llama ties embed_tokens.weight ↔ lm_head.weight by default.
      Because embed_tokens lives on stage 0 and lm_head on stage 3, they must
      be independent tensors on different GPUs. We untie them before building
      the pipeline (see Section 3 below).
    """
    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm    = norm     # LlamaRMSNorm (final layer norm)
        self.lm_head = lm_head  # nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)


def loss_fn(logits, labels):
    """
    Cross-entropy loss for autoregressive (next-token) prediction.
    DeepSpeed calls this automatically on the last stage with:
      logits  — output of FinalNormAndLMHead
      labels  — the labels tensor from the (inputs, labels) dataset tuple
    """
    # Shift so token[i] predicts token[i+1]:
    #   logits : positions 0 .. T-2  →  predict positions 1 .. T-1
    shift_logits = logits[..., :-1, :].contiguous()   # (batch, T-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous()        # (batch, T-1)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,  # skip any positions marked as padding
    )


# === SECTION 3: LOAD BASE MODEL & ASSEMBLE PIPELINE ===
# Load on CPU with bf16 (halves RAM: ~16 GB for 8B vs 32 GB in fp32).
# All ranks load the full model here; DeepSpeed moves each stage's parameters
# to the correct GPU during initialize(). For very large models, consider
# using low_cpu_mem_usage=True or deepspeed.zero.Init().
print("Loading base model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,    # stream one layer at a time to avoid CPU OOM
)

# Untie embed_tokens ↔ lm_head.
# Llama shares these weights by default (tie_word_embeddings=True) to save memory.
# After partitioning, they live on different GPUs — we need independent copies.
if base_model.config.tie_word_embeddings:
    base_model.lm_head.weight = nn.Parameter(
        base_model.lm_head.weight.detach().clone()
    )

# Build the flat layer list using LayerSpec.
# LayerSpec(cls, *constructor_args) — DeepSpeed calls cls(*constructor_args) only
# on the rank(s) assigned to that stage, avoiding unnecessary instantiation.
#
# Layout for Llama-3.1-8B (32 decoder layers) across 4 stages:
#   Stage 0 : EmbeddingStage + layers[ 0.. 7]   (9 modules)
#   Stage 1 : layers[ 8..15]                     (8 modules)
#   Stage 2 : layers[16..23]                     (8 modules)
#   Stage 3 : layers[24..31] + FinalNormAndLMHead (9 modules)
pipeline_layers = (
    [LayerSpec(EmbeddingStage, base_model.model.embed_tokens)]
    + [LayerSpec(LlamaLayerWrapper, layer) for layer in base_model.model.layers]
    + [LayerSpec(FinalNormAndLMHead, base_model.model.norm, base_model.lm_head)]
)

print(f"Pipeline: {len(pipeline_layers)} total modules → {NUM_STAGES} stages "
      f"(~{len(pipeline_layers) // NUM_STAGES} per stage)")

pipe_model = PipelineModule(
    layers=pipeline_layers,
    loss_fn=loss_fn,
    num_stages=NUM_STAGES,
    partition_method="uniform",  # divide the layer list into equal-length chunks
)


# === SECTION 4: DEEPSPEED CONFIG (inline dict) ===
#
# train_batch_size = per_device_batch × dp_degree × grad_accum
#
#   per_device_batch  (train_micro_batch_size_per_gpu) : samples per GPU per forward pass
#   dp_degree                                          : number of DATA-PARALLEL replicas
#   grad_accum        (gradient_accumulation_steps)    : micro-batches before one optimizer step
#
# For pure pipeline parallelism (all GPUs in one pipeline, no data parallelism):
#   dp_degree = 1   →   train_batch_size = 1 × 1 × 8 = 8
#
# For hybrid pipeline + data parallelism (e.g. 8 GPUs, 4-stage pipeline, 2 dp replicas):
#   dp_degree = num_gpus / num_stages = 8 / 4 = 2
#   →   train_batch_size = 1 × 2 × 8 = 16
#
# DeepSpeed validates: train_batch_size == micro_batch × grad_accum × dp_world_size
# Values MUST be consistent or DeepSpeed raises an assertion error.
#
DS_CONFIG = {
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "gradient_accumulation_steps":    GRAD_ACCUM,
    "train_batch_size":               BATCH_SIZE * GRAD_ACCUM,  # dp_degree=1 → 1 × 8 = 8
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
        # OPTIONAL: gradient checkpointing — recompute activations on backward
        # to trade compute for memory. Set to 1 to checkpoint every layer.
        "activation_checkpoint_interval": 0,
    },
    "steps_per_print":      LOG_STEPS,
    "wall_clock_breakdown": False,
}


# === SECTION 5: INITIALIZE DEEPSPEED ENGINE ===
# deepspeed.initialize() wraps PipelineModule in a PipelineEngine that:
#   • assigns each LayerSpec to a GPU stage and moves parameters there
#   • implements the GPipe schedule (forward all micro-batches, then backward all)
#   • manages gradient accumulation across GRAD_ACCUM micro-batches
#   • calls the optimizer only after all micro-batches are accumulated
#
# Passing training_data lets DeepSpeed build the DataLoader with a DistributedSampler
# matched to the data-parallel topology. Call engine.train_batch() with no arguments.
engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=pipe_model,
    config=DS_CONFIG,
    training_data=train_dataset,
)

if dist.get_rank() == 0:
    print(f"\nEffective batch size : {BATCH_SIZE} (micro) × {GRAD_ACCUM} grad_accum = "
          f"{BATCH_SIZE * GRAD_ACCUM} samples/step")
    print(f"Total optimizer steps: {total_steps}  (warmup: {warmup_steps})\n")


# === SECTION 6: TRAINING LOOP ===
# engine.train_batch() encapsulates one complete optimizer step:
#   1. Fetches GRAD_ACCUM micro-batches from the internal DataLoader.
#   2. GPipe forward  — each micro-batch flows through stage 0 → 1 → 2 → 3.
#                       loss_fn() runs on stage 3, producing a scalar.
#   3. GPipe backward — gradients flow stage 3 → 2 → 1 → 0 for each micro-batch.
#   4. Gradients accumulated over all micro-batches → optimizer.step() → zero_grad().
# Returns the average loss across the micro-batches (scalar tensor on last rank).

engine.train()
global_step = 0

for epoch in range(EPOCHS):
    if dist.get_rank() == 0:
        print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")

    for _ in range(steps_per_epoch):
        loss = engine.train_batch()   # one full forward → backward → optimizer step
        global_step += 1

        # Only rank 0 (first pipeline stage) prints — avoids duplicate output.
        if dist.get_rank() == 0 and global_step % LOG_STEPS == 0:
            print(f"  step {global_step:>5} | loss {loss.item():.4f}")

    # Save a DeepSpeed checkpoint at the end of each epoch.
    # This saves model weights + optimizer state + scheduler state for each stage.
    engine.save_checkpoint(OUTPUT_DIR, tag=f"epoch_{epoch + 1}")
    if dist.get_rank() == 0:
        print(f"  Checkpoint saved → {OUTPUT_DIR}/epoch_{epoch + 1}/")


# === SECTION 7: SAVE ===
# DeepSpeed pipeline checkpoints are sharded: each GPU saves its own stage weights.
# To produce a standard single-file HuggingFace model:
#   1. Load the checkpoint with deepspeed.utils.zero_to_fp32.py (for ZeRO) or
#      manually reassemble stage shards into a full state_dict.
#   2. Load into AutoModelForCausalLM and call save_pretrained().
# For tutorial purposes we save the DeepSpeed checkpoint only.
if dist.get_rank() == 0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nTraining complete.")
    print(f"DeepSpeed checkpoints saved to: {OUTPUT_DIR}/")
    print("To convert to HuggingFace format, reassemble stage shards from each "
          "GPU's checkpoint and load into AutoModelForCausalLM.from_pretrained().")


if __name__ == "__main__":
    pass
