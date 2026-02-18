"""
Description : Fine-tune Llama-3.1-8B with HuggingFace Accelerate — one script, any parallelism strategy.
Key concepts: Accelerator init, accelerator.state inspection, prepare(), accumulate(),
              accelerator.backward(), mixed precision, gradient accumulation, and model saving —
              all identical regardless of whether you run DDP / FSDP / DeepSpeed ZeRO.
Run         : accelerate launch --config_file configs/single_gpu.yaml       multi-gpu/parallelism_accelerate.py
              accelerate launch --config_file configs/ddp_2gpu.yaml          multi-gpu/parallelism_accelerate.py
              accelerate launch --config_file configs/fsdp_zero3.yaml        multi-gpu/parallelism_accelerate.py
              accelerate launch --config_file configs/deepspeed_zero2.yaml   multi-gpu/parallelism_accelerate.py
              accelerate launch --config_file configs/deepspeed_zero3.yaml   multi-gpu/parallelism_accelerate.py
Requirements: pip install accelerate transformers datasets torch tqdm
              For DeepSpeed configs: pip install deepspeed
"""

# === IMPORTS ===
# 1. stdlib
import os
import json
import random

# 2. torch
import torch
from torch.utils.data import DataLoader

# 3. HuggingFace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset

# 4. Accelerate — the only library that changes between parallelism strategies is the config YAML
from accelerate import Accelerator
from tqdm.auto import tqdm


# === CONFIG ===
MODEL_NAME    = "meta-llama/Meta-Llama-3.1-8B"
DATA_PATH     = "alpaca_gpt4_data.json"   # relative to repo root (where you launch from)
MAX_SEQ_LEN   = 1024
BATCH_SIZE    = 1                          # per-device batch size
GRAD_ACCUM    = 8                          # effective batch = BATCH_SIZE × num_gpus × GRAD_ACCUM
LR            = 3e-6
EPOCHS        = 3
LOG_STEPS     = 5
OUTPUT_DIR    = "./outputs/parallelism_accelerate"


# === SECTION 0: ACCELERATOR INIT & STATE INSPECTION ===
# Accelerator reads your --config_file automatically — no code change needed between strategies.
# mixed_precision, distributed_type, num_processes all come from the YAML.
accelerator = Accelerator(
    gradient_accumulation_steps=GRAD_ACCUM,  # tells accumulate() when to sync gradients
    log_with="tensorboard",                  # OPTIONAL: swap for "wandb" if preferred
    project_dir=OUTPUT_DIR,
)

# Print parallelism details so students can verify the active strategy.
# accelerator.print() is rank-aware: only rank 0 prints, so output is never duplicated.
accelerator.print("\n" + "=" * 60)
accelerator.print("ACCELERATE STATE")
accelerator.print("=" * 60)
accelerator.print(accelerator.state)          # shows: distributed_type, mixed_precision, num_processes, etc.
accelerator.print("=" * 60 + "\n")


# === SECTION 1: DATA LOADING ===
def load_alpaca_dataset(path=DATA_PATH, eval_size=1000):
    """Load and split the Alpaca dataset, then format into prompt strings."""
    with open(path) as f:
        data = json.load(f)
    random.shuffle(data)
    train_data = data[:-eval_size]
    eval_data  = data[-eval_size:]
    return preprocess(train_data), preprocess(eval_data)

def preprocess(dataset):
    """Convert each Alpaca row into a single instruction-response string."""
    texts = []
    for row in dataset:
        if row["input"] == "":
            # No additional input context — use the shorter template
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}"
            ).format_map(row)
        else:
            # Additional input provided — use the longer template
            text = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            ).format_map(row)
        texts.append(text)
    return Dataset.from_dict({"text": texts})

# Only load data on the main process first, then all processes access it.
# This avoids every GPU reading the file simultaneously.
with accelerator.main_process_first():
    train_hf_dataset, eval_hf_dataset = load_alpaca_dataset()

accelerator.print(f"Train examples: {len(train_hf_dataset):,}  |  Eval examples: {len(eval_hf_dataset):,}")


# === SECTION 2: TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default


# === SECTION 3: SEQUENCE PACKING ===
# Instead of padding every sequence to MAX_SEQ_LEN (wasteful), we concatenate all
# token IDs into one long stream and slice fixed-length chunks — no padding needed.
def pack(hf_dataset, max_seq_len=MAX_SEQ_LEN):
    """
    Tokenize all examples, concatenate into one token stream, then slice into
    fixed-length (max_seq_len) chunks. Each chunk becomes one training example.
    Labels are shifted by 1 (next-token prediction).
    """
    # Tokenize without padding — we want the raw token IDs
    all_token_ids = []
    for text in hf_dataset["text"]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)   # add EOS between examples
        all_token_ids.extend(ids)

    packed = []
    # Slice the flat token stream into (max_seq_len + 1) chunks:
    #   input_ids = tokens[0 : max_seq_len]
    #   labels    = tokens[1 : max_seq_len + 1]  (shifted by one position)
    for i in range(0, len(all_token_ids) - max_seq_len, max_seq_len):
        chunk = all_token_ids[i : i + max_seq_len + 1]
        packed.append({
            "input_ids": chunk[:-1],   # model input
            "labels":    chunk[1:],    # next-token targets
        })
    return packed

accelerator.print("Packing sequences...")
with accelerator.main_process_first():
    train_packed = pack(train_hf_dataset)
    eval_packed  = pack(eval_hf_dataset)

accelerator.print(f"Packed train batches: {len(train_packed):,}  |  Packed eval batches: {len(eval_packed):,}")


# === SECTION 4: DATALOADERS ===
# default_data_collator stacks list-of-dicts into batched tensors — no extra work needed
# because our pack() already produces fixed-length lists.
train_dataloader = DataLoader(
    train_packed,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=default_data_collator,
)
eval_dataloader = DataLoader(
    eval_packed,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=default_data_collator,
)


# === SECTION 5: MODEL ===
# Do NOT use device_map="auto" here — Accelerate (and DeepSpeed/FSDP) manages device placement.
# Load in bf16 to match our configs; flash_attention_2 requires bf16.
accelerator.print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# OPTIONAL: gradient checkpointing trades compute for memory (recomputes activations on backward)
model.gradient_checkpointing_enable()


# === SECTION 6: OPTIMIZER & SCHEDULER ===
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

# Total optimizer steps = epochs × (batches / grad_accum) — Accelerate handles the division
total_steps = EPOCHS * (len(train_dataloader) // GRAD_ACCUM)
warmup_steps = int(0.1 * total_steps)  # 0.1 treated as a ratio → 10 % warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


# === SECTION 7: PREPARE WITH ACCELERATE ===
# This is THE key Accelerate call — it wraps model, optimizer, and dataloaders for
# whatever parallelism strategy is specified in the YAML (DDP / FSDP / DeepSpeed).
# From here on, your code is the same for every strategy.
model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

accelerator.print(f"\nEffective batch size: {BATCH_SIZE} × {accelerator.num_processes} GPUs × {GRAD_ACCUM} grad_accum = "
                  f"{BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM}")
accelerator.print(f"Total optimizer steps: {total_steps}  |  Warmup steps: {warmup_steps}\n")


# === SECTION 8: TRAINING LOOP ===
def evaluate():
    """Run one pass over the eval set and return average loss (rank-0 only)."""
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_steps = torch.tensor(0, device=accelerator.device)

    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            total_loss  += outputs.loss.detach()
            total_steps += 1

    # All-reduce so every process has the global sum
    total_loss  = accelerator.reduce(total_loss,  reduction="sum")
    total_steps = accelerator.reduce(total_steps, reduction="sum")
    avg_loss = (total_loss / total_steps).item()
    model.train()
    return avg_loss


model.train()
global_step = 0

for epoch in range(EPOCHS):
    accelerator.print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")
    progress = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)

    for batch in progress:
        # accelerator.accumulate() is the clean way to handle gradient accumulation.
        # It automatically:
        #   • skips gradient sync across processes until GRAD_ACCUM steps are done (saves bandwidth)
        #   • makes optimizer.step() a no-op on accumulation steps (only steps when sync happens)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            # accelerator.backward() replaces loss.backward().
            # For DeepSpeed it calls engine.backward(); for FSDP/DDP it calls the standard backward.
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()      # no-op on accumulation steps (Accelerate handles this)
            optimizer.zero_grad()

        # accelerator.sync_gradients is True only when an actual optimizer step just happened
        if accelerator.sync_gradients:
            global_step += 1
            progress.set_postfix({"loss": f"{loss.detach().item():.4f}", "step": global_step})

            if global_step % LOG_STEPS == 0:
                accelerator.print(f"  step {global_step:>5} | loss {loss.detach().item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    # --- End of epoch: evaluate ---
    eval_loss = evaluate()
    accelerator.print(f"  >> Epoch {epoch + 1} eval loss: {eval_loss:.4f}")

    # Sync all processes before next epoch
    accelerator.wait_for_everyone()


# === SECTION 9: SAVE ===
# accelerator.unwrap_model() strips away DDP / FSDP / DeepSpeed wrappers and
# returns the raw HuggingFace model, which we can then save normally.
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        OUTPUT_DIR,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained(OUTPUT_DIR)
    accelerator.print(f"\nModel saved to {OUTPUT_DIR}")

accelerator.end_training()


if __name__ == "__main__":
    # `accelerate launch` calls this file as a module, so __main__ is reached normally.
    # Nothing extra needed here — all setup happens at module level above.
    pass
