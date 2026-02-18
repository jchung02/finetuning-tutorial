"""
Description : Pipeline-parallel training of Llama-3.1-8B with Megatron-Core's 1F1B schedule.
Key concepts: parallel_state, TransformerConfig, GPTModel pre_process/post_process flags,
              get_forward_backward_func → 1F1B, forward_step_func contract.
Run         : torchrun --nproc_per_node=4 multi-gpu/pp_megatron.py
Requirements: see # Requirements block below
"""

# Requirements:
#
# Option A — NVIDIA PyTorch container (cleanest, all CUDA deps pre-installed):
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#       nvcr.io/nvidia/pytorch:25.03-py3
#   pip install megatron-core
#
# Option B — manual install on an existing CUDA 11.8+ environment:
#   pip install megatron-core
#   # megatron-core pulls: torch, apex (optional), transformer-engine (optional)
#   # This script uses get_gpt_layer_local_spec() — no transformer-engine required.
#
# GPU memory: Llama-3.1-8B randomly initialised in bf16 ≈ 16 GB total.
#             With 4 pipeline stages each stage holds ~4 GB of parameters.
#
# Reference: https://github.com/NVIDIA/Megatron-LM/tree/core_v0.14.0/examples/llama
#            https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/examples/run_simple_mcore_train_loop.py


# === IMPORTS ===
# 1. stdlib
import os
from functools import partial

# 2. torch
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

# 3. Megatron-Core
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


# === CONFIG ===
# Llama-3.1-8B architecture constants — taken directly from the official
# Megatron-LM examples/llama/train_llama3_8b_h100_fp8.sh
NUM_LAYERS  = 32
HIDDEN_SIZE = 4096
FFN_HIDDEN  = 14336     # SwiGLU intermediate (each of gate/up projections is this wide)
NUM_HEADS   = 32        # attention heads
GQA_GROUPS  = 8         # key-value heads (Grouped Query Attention)
KV_CHANNELS = 128       # head dimension = HIDDEN_SIZE // NUM_HEADS = 4096 // 32
VOCAB_SIZE  = 128256    # Llama 3.x BPE vocabulary
ROTARY_BASE = 1_000_000 # Llama 3.1 extended-context RoPE base (Llama 1/2 uses 10 000)

# Pipeline parallelism
NUM_STAGES       = 4                          # must match --nproc_per_node
LAYERS_PER_STAGE = NUM_LAYERS // NUM_STAGES   # 32 / 4 = 8 layers per stage

# Training
MAX_SEQ_LEN      = 1024
MICRO_BATCH      = 1   # samples per GPU per micro-batch
NUM_MICROBATCHES = 8   # micro-batches per optimizer step; ≥ NUM_STAGES enables steady-state 1F1B
LR               = 3e-6
EPOCHS           = 3
LOG_STEPS        = 5
OUTPUT_DIR       = "./outputs/pp_megatron"
SEED             = 42


# === SECTION 0: DISTRIBUTED INIT ===
def initialize_distributed(num_stages=NUM_STAGES):
    """
    Set up PyTorch distributed and Megatron-Core's parallel process groups.

    Megatron maintains TWO independent sets of NCCL groups:
      Tensor Model Parallel  (TP) — ranks that SHARE one layer, split along heads/features
      Pipeline Model Parallel (PP) — ranks that hold CONSECUTIVE layer blocks

    Here: TP=1 (no intra-layer split), PP=4 (four consecutive pipeline stages).
    initialize_model_parallel() must be called before constructing any GPTModel.
    """
    rank       = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,     # no tensor split for this example
        pipeline_model_parallel_size=num_stages,
    )
    model_parallel_cuda_manual_seed(SEED)  # reproducible weight init across TP ranks


initialize_distributed()
rank   = dist.get_rank()
device = torch.device(f"cuda:{rank}")


# === SECTION 1: TRANSFORMER CONFIG ===
#
# TransformerConfig is the single config dataclass that ALL Megatron-Core
# sub-modules (TransformerBlock, SelfAttention, MLP, ...) read from.
# It inherits from ModelParallelConfig, which carries TP/PP world sizes at runtime
# via parallel_state — you do NOT hard-code them here.
#
# Pipeline-relevant fields:
#   num_layers     — TOTAL layers across ALL stages; Megatron divides automatically.
#   pipeline_dtype — dtype of tensors shipped between stages over P2P NCCL links.
transformer_config = TransformerConfig(
    # ── Llama-3.1-8B architecture ─────────────────────────────────────────────
    num_layers=NUM_LAYERS,              # 32 total; each stage gets 32/4 = 8
    hidden_size=HIDDEN_SIZE,            # 4096
    num_attention_heads=NUM_HEADS,      # 32 attention heads
    num_query_groups=GQA_GROUPS,        # 8 KV heads — GQA halves KV cache vs MHA
    ffn_hidden_size=FFN_HIDDEN,         # 14336 intermediate for each gate/up projection
    kv_channels=KV_CHANNELS,            # per-head dimension (hidden / heads = 128)

    # ── Normalisation & activation ────────────────────────────────────────────
    normalization="RMSNorm",            # Llama uses RMSNorm; LayerNorm is the default
    gated_linear_unit=True,             # SwiGLU: FFN = W_down(silu(W_gate x) ⊙ W_up x)
    activation_func=F.silu,             # SiLU (swish) for the gated branch of SwiGLU

    # ── Regularisation ────────────────────────────────────────────────────────
    hidden_dropout=0.0,                 # Llama is trained without dropout
    attention_dropout=0.0,
    add_bias_linear=False,              # no bias in Q/K/V, FFN projections
    layernorm_zero_centered_gamma=True, # --apply-layernorm-1p: γ = 0 → scale = 1+γ

    # ── Pipeline communication ────────────────────────────────────────────────
    pipeline_dtype=torch.bfloat16,      # inter-stage activation tensors use bf16

    # ── Misc ──────────────────────────────────────────────────────────────────
    use_cpu_initialization=True,        # initialise weights on CPU, then move to GPU
)


# === SECTION 2: MODEL CONSTRUCTION — PRE/POST PROCESS FLAGS ===
#
# GPTModel uses two booleans to decide which sub-components to materialise:
#
#   pre_process=True  → build LanguageModelEmbedding    (token + position lookup)
#                        Only the FIRST pipeline stage needs it.
#
#   post_process=True → build final RMSNorm + output linear (lm_head)
#                        Only the LAST pipeline stage needs it.
#
# TransformerBlock (the layer stack inside GPTModel) automatically determines
# WHICH 8 layers to build on this rank from parallel_state:
#   stage 0 → layers [0, 8)    stage 1 → layers [8, 16)
#   stage 2 → layers [16, 24)  stage 3 → layers [24, 32)
# You do NOT specify this split — Megatron calculates it for you.
#
# get_gpt_layer_local_spec(normalization="RMSNorm") returns a ModuleSpec that
# describes each transformer layer using only pure Megatron-Core modules
# (ColumnParallelLinear, DotProductAttention, RMSNorm).
# No Transformer Engine required — safe to run without the NVIDIA TE package.

def build_model():
    is_first = parallel_state.is_pipeline_first_stage()
    is_last  = parallel_state.is_pipeline_last_stage()

    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQ_LEN,
        pre_process=is_first,   # stage 0: include embedding lookup
        post_process=is_last,   # stage 3: include final norm + lm_head
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=ROTARY_BASE,
        share_embeddings_and_output_weights=False,  # untied embed ↔ lm_head (Llama 3.x)
    )
    return model


model = build_model()
model.to(device)

if rank == 0:
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Stage {pp_rank}] Parameters on this GPU: {n_params:,}  "
          f"(layers {pp_rank * LAYERS_PER_STAGE}–{(pp_rank+1) * LAYERS_PER_STAGE - 1}"
          f"{' + embed' if pp_rank == 0 else ''}"
          f"{' + lm_head' if pp_rank == NUM_STAGES - 1 else ''})")


# === SECTION 3: SYNTHETIC DATASET ===
#
# Using randomly generated token sequences to focus on pipeline schedule logic.
# The Megatron-Core dataset pipeline (BlendedMegatronDatasetBuilder, GPTDataset)
# is production-grade but requires a pre-tokenised binary corpus.
#
# For real training, replace SyntheticDataset with:
#   from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
#   from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
# See: examples/run_simple_mcore_train_loop.py in the Megatron-LM repo.

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Generates fixed random token sequences for schedule demonstration.
    Each item is a dict with keys: tokens, labels, position_ids, loss_mask.
    ALL pipeline stages use the same dataset; each stage only uses the
    fields relevant to it (stage 0: tokens; last stage: labels).
    """
    def __init__(self, vocab_size, seq_len, size=512):
        torch.manual_seed(SEED)
        self._tokens = torch.randint(0, vocab_size, (size, seq_len))
        self.seq_len = seq_len

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        tokens = self._tokens[idx]
        labels = torch.roll(tokens, -1)   # shift by 1: token[i] predicts token[i+1]
        labels[-1] = tokens[-1]
        return {
            "tokens":       tokens,
            "labels":       labels,
            "position_ids": torch.arange(self.seq_len),
            "loss_mask":    torch.ones(self.seq_len),  # 1 = compute loss, 0 = ignore (padding)
        }


dataset    = SyntheticDataset(VOCAB_SIZE, MAX_SEQ_LEN, size=512)
dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=True, drop_last=True)


# === SECTION 4: OPTIMIZER ===
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)


# === SECTION 5: FORWARD STEP FUNCTION — THE PIPELINE CONTRACT ===
#
# This function is the ONLY user-facing hook in Megatron's pipeline scheduler.
# forward_backward_func() calls it once per micro-batch per stage, interleaving
# calls across ranks according to the 1F1B schedule.
#
# CONTRACT:
#   Signature : forward_step_func(data_iterator, model) → (output_tensor, loss_func)
#   data_iter : called via next() on EVERY rank for EVERY micro-batch.
#               Keeps iterators in sync across the pipeline.
#               Intermediate stages (1, 2) call next() but the model ignores the tokens —
#               it uses the activation tensor received from the previous stage via P2P.
#   output_tensor : activation sent to the NEXT stage (or loss on the last stage)
#   loss_func     : None on all stages except the last.
#                   On the last stage, called as loss_func(output_tensor) → (scalar, dict)
#
# What each stage actually uses:
#   Stage 0     : tokens, position_ids  (embedding lookup → send hidden_states to stage 1)
#   Stages 1, 2 : model.input_tensor set via set_input_tensor() from P2P recv
#                 tokens/position_ids are loaded but ignored by the model
#   Stage 3     : labels (compute loss with per-token cross-entropy)

def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """
        Reduce per-token cross-entropy into a scalar batch loss.
        Called only on the last pipeline stage.
          output_tensor : (batch, seq_len) — per-token CE losses from GPTModel
          loss_mask     : (batch, seq_len) — 1.0 for real tokens, 0.0 for padding
        """
        losses    = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss      = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm_loss": loss.detach()}

    # next() called on ALL stages — keeps all iterators advancing in lockstep.
    data         = next(data_iterator)
    tokens       = data["tokens"].to(device)        # (batch, seq_len) int64
    position_ids = data["position_ids"].to(device)  # (batch, seq_len) int64
    labels       = data["labels"].to(device)         # used only on the last stage
    loss_mask    = data["loss_mask"].to(device)      # used only in loss_func

    # attention_mask=None: valid here because get_gpt_layer_local_spec sets
    # AttnMaskType.causal, so each attention kernel applies causal masking internally.
    output_tensor = model(tokens, position_ids, attention_mask=None, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


# === SECTION 6: THE 1F1B PIPELINE SCHEDULE ===
#
# get_forward_backward_func() inspects parallel_state and selects the right schedule:
#   PP == 1  →  forward_backward_no_pipelining         (no overlap, baseline)
#   PP >  1  →  forward_backward_pipelining_without_interleaving  ← 1F1B  ← our case
#   PP >  1 + virtual pipeline stages →  forward_backward_pipelining_with_interleaving
#
# ─────────────────────────────────────────────────────────────────────────────────────
# 1F1B (One-Forward One-Backward) schedule — P=4 stages, M=8 micro-batches
#
#         │◄─────── warm-up (P−1) ──────►│◄──── steady state ────►│◄─ drain ─►│
# Stage 0 │ F0   F1   F2   F3           │ B0  F4  B1  F5  B2  F6 │ B3 B4 B5 B6│ B7
# Stage 1 │      F0   F1   F2           │ F3  B0  F4  B1  F5  B2 │ F6 B3 B4 B5│ B6 B7
# Stage 2 │           F0   F1           │ F2  F3  B0  F4  B1  F5 │ B2 F6 B3 B4│ B5 B6 B7
# Stage 3 │                F0           │ F1  F2  F3  B0  F4  B1 │ F5 B2 F6 B3│ B4 B5 B6 B7
#
# Fₙ = forward pass of micro-batch n  (activation flows stage 0 → 1 → 2 → 3)
# Bₙ = backward pass of micro-batch n (gradient flows stage 3 → 2 → 1 → 0)
#
# Key properties vs GPipe (all-F then all-B):
#
#   Property          │  GPipe                     │  1F1B
#   ──────────────────┼────────────────────────────┼─────────────────────────────
#   Activation memory │ store ALL M micro-batches  │ store only P micro-batches
#   Pipeline bubble   │ (P−1)/M of compute wasted  │ same fraction, but lower memory
#   Steady state      │ none (all F, then all B)   │ each stage alternates F and B
#   num_microbatches  │ any M                      │ M ≥ P for steady state
#
# Bubble fraction = (P−1) / M × 100%
# → P=4, M=8:  bubble ≈ 37.5%   (3 idle slots out of 8)
# → P=4, M=32: bubble ≈ 9.4%    (increase M to amortise the warm-up/drain overhead)
#
# The entire schedule runs inside forward_backward_func — your code never touches P2P.
# ─────────────────────────────────────────────────────────────────────────────────────

forward_backward_func = get_forward_backward_func()

if rank == 0:
    bubble_pct = (NUM_STAGES - 1) / NUM_MICROBATCHES * 100
    print(f"\nPipeline : {NUM_STAGES} stages  ×  {LAYERS_PER_STAGE} layers/stage  =  {NUM_LAYERS} total layers")
    print(f"Schedule : 1F1B  |  {NUM_MICROBATCHES} micro-batches/step  |  bubble ≈ {bubble_pct:.1f}%")
    print(f"           (increase NUM_MICROBATCHES to reduce bubble)\n")


# === SECTION 7: TRAINING LOOP ===
model.train()
global_step      = 0
steps_per_epoch  = len(dataloader) // NUM_MICROBATCHES  # each step consumes NUM_MICROBATCHES items

for epoch in range(EPOCHS):
    if rank == 0:
        print(f"--- Epoch {epoch + 1} / {EPOCHS} ---")

    data_iter = iter(dataloader)   # fresh iterator each epoch

    for _ in range(steps_per_epoch):
        optimizer.zero_grad()

        # forward_backward_func drives the complete 1F1B schedule for one optimizer step:
        #   1. Warm-up  : stages fill with P−1 forward passes (P2P sends activations forward)
        #   2. Steady   : every stage alternates F and B (P2P sends activations + gradients)
        #   3. Drain    : remaining backward passes flush backward through all stages
        #   4. Aggregate: gradients from all NUM_MICROBATCHES micro-batches are summed
        #
        # Returns a list of loss dicts — non-empty ONLY on the last pipeline stage.
        # Other stages return an empty list [].
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=model,
            num_microbatches=NUM_MICROBATCHES,
            seq_length=MAX_SEQ_LEN,
            micro_batch_size=MICRO_BATCH,
            forward_only=False,   # False = run full forward + backward
        )

        # Gradient clipping + optimizer step on ALL stages (each holds its own params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1

        # Loss is computed (and returned) only on the LAST pipeline stage.
        # Intermediate stages log a placeholder to confirm they are progressing.
        if global_step % LOG_STEPS == 0:
            if parallel_state.is_pipeline_last_stage() and losses_reduced:
                avg = sum(d["lm_loss"].item() for d in losses_reduced) / len(losses_reduced)
                print(f"  step {global_step:>5} | stage {rank} (last) | loss {avg:.4f}")
            elif rank == 0:
                print(f"  step {global_step:>5} | stage 0 (first)  | loss computed on last stage")


# === SECTION 8: CHECKPOINT SAVE ===
# Each pipeline stage holds a different shard of the model.
# For production use, megatron.core.dist_checkpointing handles all shards:
#   from megatron.core import dist_checkpointing
#   dist_checkpointing.save(model.sharded_state_dict(prefix=''), ckpt_dir)
#
# For this tutorial we save only the last stage's weights as a simple example.
dist.barrier()
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, f"stage_{rank}_weights.pt")
)
if rank == 0:
    print(f"\nPer-stage weights saved to {OUTPUT_DIR}/stage_{{rank}}_weights.pt")
    print("For a full model checkpoint across all stages, use megatron.core.dist_checkpointing.")

# Always clean up Megatron process groups after training
parallel_state.destroy_model_parallel()
dist.destroy_process_group()


if __name__ == "__main__":
    pass
