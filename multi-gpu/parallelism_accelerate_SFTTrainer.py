"""
Run : accelerate launch --config_file configs/single_gpu.yaml        multi-gpu/parallelism_accelerate_SFTTrainer.py
      accelerate launch --config_file configs/ddp_2gpu.yaml          multi-gpu/parallelism_accelerate_SFTTrainer.py
      accelerate launch --config_file configs/fsdp_zero3.yaml        multi-gpu/parallelism_accelerate_SFTTrainer.py
      accelerate launch --config_file configs/deepspeed_zero2.yaml   multi-gpu/parallelism_accelerate_SFTTrainer.py
      accelerate launch --config_file configs/deepspeed_zero3.yaml   multi-gpu/parallelism_accelerate_SFTTrainer.py

Compare with parallelism_accelerate.py to see what SFTTrainer abstracts away:
  - Manual pack() function        → packing=True in SFTConfig
  - Manual DataLoader setup       → handled by SFTTrainer
  - Manual optimizer + scheduler  → handled by TrainingArguments / SFTConfig
  - Manual accelerator.prepare()  → handled internally by Trainer
  - Manual training loop          → replaced by trainer.train()
  - Manual evaluate() + reduce    → handled internally by Trainer
  - Manual unwrap_model() + save  → replaced by trainer.save_model()
The same five config YAMLs work unchanged — only this script is different.
"""

import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from accelerate import Accelerator


# === CONFIG ===
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B"
DATA_PATH   = "alpaca_gpt4_data.json"
MAX_SEQ_LEN = 1024
BATCH_SIZE  = 1       
GRAD_ACCUM  = 8       # effective batch = BATCH_SIZE × num_gpus × GRAD_ACCUM
LR          = 3e-6
EPOCHS      = 3
LOG_STEPS   = 5
OUTPUT_DIR  = "./outputs/parallelism_accelerate_SFTTrainer"

accelerator = Accelerator() 

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

with accelerator.main_process_first():
    train_dataset, eval_dataset = load_alpaca_dataset()

accelerator.print(f"Train examples: {len(train_dataset):,}  |  Eval examples: {len(eval_dataset):,}")

accelerator.print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,           # flash_attention_2 requires bf16
    attn_implementation="flash_attention_2",
)

model.gradient_checkpointing_enable()     

# === TRAINING CONFIG ===
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM, 
    bf16=True,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,                        

    num_train_epochs=EPOCHS,
    logging_steps=LOG_STEPS,
    logging_first_step=True,
    eval_strategy="epoch",
    save_strategy="epoch",

    packing=True,
    max_length=MAX_SEQ_LEN,

    dataset_text_field="text",               
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,             # tokenizes on the fly
)

accelerator.print(
    f"\nEffective batch size: {BATCH_SIZE} × {accelerator.num_processes} GPUs × {GRAD_ACCUM} "
    f"grad_accum = {BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM}\n"
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
accelerator.print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    pass
