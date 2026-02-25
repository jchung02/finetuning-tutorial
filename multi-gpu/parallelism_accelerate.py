"""
Fine-tune Llama-3.1-8B with HuggingFace Accelerate
Run : accelerate launch --config_file configs/single_gpu.yaml        multi-gpu/parallelism_accelerate.py
      accelerate launch --config_file configs/ddp_4gpu.yaml          multi-gpu/parallelism_accelerate.py
      accelerate launch --config_file configs/fsdp_zero3.yaml        multi-gpu/parallelism_accelerate.py
      accelerate launch --config_file configs/deepspeed_zero2.yaml   multi-gpu/parallelism_accelerate.py
      accelerate launch --config_file configs/deepspeed_zero3.yaml   multi-gpu/parallelism_accelerate.py
"""

import os
import json
import random

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset

from accelerate import Accelerator
from tqdm.auto import tqdm


# === CONFIG ===
MODEL_NAME    = "meta-llama/Meta-Llama-3.1-8B"
DATA_PATH     = "alpaca_gpt4_data.json"   
MAX_SEQ_LEN   = 1024
BATCH_SIZE    = 1                          # per-device batch size
GRAD_ACCUM    = 8                          # effective batch = BATCH_SIZE × num_gpus × GRAD_ACCUM
LR            = 3e-6
EPOCHS        = 3
LOG_STEPS        = 5
OUTPUT_DIR       = "./outputs/parallelism_accelerate"
CHECKPOINT_STEPS = 50   # save a full resumable checkpoint every N optimizer steps
CHECKPOINT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints2")

accelerator = Accelerator(
    gradient_accumulation_steps=GRAD_ACCUM,  
    log_with="tensorboard",                  
    project_dir=OUTPUT_DIR,
)

accelerator.print("\n" + "=" * 60)
accelerator.print("ACCELERATE STATE")
accelerator.print("=" * 60)
accelerator.print(accelerator.state)          
accelerator.print("=" * 60 + "\n")


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

# load data on the main process first, then all processes access it.
with accelerator.main_process_first():
    train_hf_dataset, eval_hf_dataset = load_alpaca_dataset()

accelerator.print(f"Train examples: {len(train_hf_dataset):,}  |  Eval examples: {len(eval_hf_dataset):,}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default

def pack(hf_dataset, max_seq_len=MAX_SEQ_LEN):
    """
    Tokenize all examples, concatenate into one token stream, then slice into
    fixed-length (max_seq_len) chunks. Each chunk becomes one training example.
    Labels are shifted by 1 (next-token prediction).
    """
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

accelerator.print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16, # flash_attention_2 requires bf16
    attn_implementation="flash_attention_2",
)

model.gradient_checkpointing_enable() # optional

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

# Total optimizer steps = epochs × (batches / grad_accum) — Accelerate handles the division
total_steps = EPOCHS * (len(train_dataloader) // GRAD_ACCUM)
warmup_steps = int(0.1 * total_steps) 

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

accelerator.print(f"\nEffective batch size: {BATCH_SIZE} × {accelerator.num_processes} GPUs × {GRAD_ACCUM} grad_accum = "
                  f"{BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM}")
accelerator.print(f"Total optimizer steps: {total_steps}  |  Warmup steps: {warmup_steps}\n")

accelerator.init_trackers("train")

def evaluate():
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_steps = torch.tensor(0, device=accelerator.device)

    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            total_loss  += outputs.loss.detach()
            total_steps += 1

    # All-reduce: every process has the global sum
    total_loss  = accelerator.reduce(total_loss,  reduction="sum")
    total_steps = accelerator.reduce(total_steps, reduction="sum")
    avg_loss = (total_loss / total_steps).item()
    model.train()
    return avg_loss

model.train()
global_step = 0

# if os.path.isdir(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
#     # Pick the latest checkpoint sub-folder 
#     ckpt_folders = sorted(
#         [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("step_")],
#         key=lambda x: int(x.split("_")[1]),
#     )
#     if ckpt_folders:
#         latest_ckpt = os.path.join(CHECKPOINT_DIR, ckpt_folders[-1])
#         accelerator.print(f"Resuming from checkpoint: {latest_ckpt}")
#         accelerator.load_state(latest_ckpt)
#         # Recover the global step from the folder name so logging stays accurate
#         global_step = int(ckpt_folders[-1].split("_")[1])

for epoch in range(EPOCHS):
    accelerator.print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")
    progress = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)

    for batch in progress:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss) # For DeepSpeed it calls engine.backward()

            optimizer.step()
            scheduler.step()      # no-op on accumulation steps (Accelerate handles this)
            optimizer.zero_grad()

        # accelerator.sync_gradients is True only when an actual optimizer step just happened
        if accelerator.sync_gradients:
            global_step += 1
            progress.set_postfix({"loss": f"{loss.detach().item():.4f}", "step": global_step})

            if global_step % LOG_STEPS == 0:
                accelerator.print(f"  step {global_step:>5} | loss {loss.detach().item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                accelerator.log({"train/loss": loss.detach().item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

            if global_step % CHECKPOINT_STEPS == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
                accelerator.save_state(ckpt_path)
                accelerator.print(f"  [checkpoint] saved to {ckpt_path}")

    eval_loss = evaluate()
    accelerator.print(f"  >> Epoch {epoch + 1} eval loss: {eval_loss:.4f}")

    epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
    accelerator.save_state(epoch_ckpt_path)
    accelerator.print(f"  [checkpoint] end-of-epoch {epoch + 1} saved to {epoch_ckpt_path}")

    # Sync all processes before next epoch
    accelerator.wait_for_everyone()


# accelerator.unwrap_model() strips away DDP / FSDP / DeepSpeed wrappers and returns the raw HuggingFace model, which we can then save normally.
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
    pass
