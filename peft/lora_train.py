"""
Description : Fine-tune Llama-3.1-8B with LoRA and QLoRA using HuggingFace PEFT.
Key concepts: LoraConfig, get_peft_model, BitsAndBytesConfig (4-bit), adapter save/load/merge.
Run         : python peft/lora_train.py
Requirements: pip install transformers datasets trl peft bitsandbytes torch
"""

# === IMPORTS ===
# 1. stdlib
import os
import json
import random

# 2. torch
import torch

# 3. HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# 4. PEFT
from peft import LoraConfig, get_peft_model, PeftModel


# === CONFIG ===
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B"
DATA_PATH   = "alpaca_gpt4_data.json"
MAX_SEQ_LEN = 1024
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
LR          = 3e-6
EPOCHS      = 3
LOG_STEPS   = 5


# === DATA LOADING ===
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

train_dataset, eval_dataset = load_alpaca_dataset()


# ==============================================================================
# SECTION 1 — LoRA: LOW-RANK ADAPTATION
# ==============================================================================
#
# Problem: Full fine-tuning updates every parameter in the model (7B+ weights).
#          This is slow, memory-heavy, and risks overwriting pretrained knowledge.
#
# LoRA solution: Freeze the original weights. For each chosen linear layer W,
# add two small trainable matrices A (d × r) and B (r × d) such that the
# effective weight update is ΔW = B × A. Because r ≪ d, the number of
# trainable parameters drops by 100-1000×.
#
#   W_new  =  W_frozen  +  B × A
#              (no grad)   (trained)
#
# At inference, B×A can be merged back into W so there is zero latency overhead.
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 1: LoRA")
print("=" * 60)

# --- 1a. Load the base model (full precision, single GPU) ---
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # Llama has no pad token by default

# --- 1b. LoRA hyperparameters ---
# TARGET MODULES — which linear layers receive LoRA adapters?
#
#   Attention-only (conservative, fewer params):
#     target_modules=["q_proj", "v_proj"]
#     Adapts only the query and value projections; smallest trainable footprint.
#     Good starting point when GPU memory is very tight.
#
#   Full attention (balanced):
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
#     Covers all four attention projections; recommended default.
#
#   Attention + MLP (aggressive, more expressive):
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                     "gate_proj", "up_proj", "down_proj"]
#     Also adapts the feed-forward sub-layers; useful when the task domain
#     differs strongly from pretraining (e.g. code, math, medical text).
#     Increases trainable params ~2-3× vs attention-only.

lora_config = LoraConfig(
    r=16,               # rank: size of the low-rank bottleneck; higher r = more capacity
                        # but more parameters. Typical range: 8–64.
    lora_alpha=32,      # scaling factor applied to ΔW = (alpha/r) × B×A.
                        # Rule of thumb: set alpha = 2×r so the effective scale stays ~1.
    lora_dropout=0.05,  # dropout applied to the LoRA path during training to reduce
                        # overfitting; 0.05–0.1 is common, use 0.0 for small datasets.
    bias="none",        # whether to train bias terms alongside LoRA; "none" is standard
    task_type="CAUSAL_LM",

    # Full attention projection matrices — good default for instruction fine-tuning
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# --- 1c. Wrap the base model with LoRA adapters ---
# get_peft_model() freezes all original weights and injects trainable A/B matrices
# into the specified target_modules. The rest of the model is unchanged.
lora_model = get_peft_model(base_model, lora_config)

# print_trainable_parameters() shows how many params are actually updated
lora_model.print_trainable_parameters()
# Example output: trainable params: 20,971,520 || all params: 8,051,232,768 || trainable%: 0.26

# --- 1d. Train with SFTTrainer (same API as full fine-tune — LoRA is transparent) ---
lora_output_dir = "./outputs/lora_adapter"

lora_trainer = SFTTrainer(
    model=lora_model,
    args=SFTConfig(
        output_dir=lora_output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,           # float → treated as ratio of total steps
        num_train_epochs=EPOCHS,
        bf16=True,
        logging_steps=LOG_STEPS,
        eval_strategy="epoch",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=True,               # pack multiple short examples into one sequence
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

lora_trainer.train()


# ==============================================================================
# SECTION 2 — QLoRA: QUANTIZED LOW-RANK ADAPTATION
# ==============================================================================
#
# Problem: Even LoRA requires the frozen base model in bf16/fp16 (~14 GB for 7B).
#          Consumer GPUs often have only 8–24 GB VRAM.
#
# QLoRA solution (Dettmers et al., 2023):
#   1. Load the base model in 4-bit NormalFloat (NF4) — ~4 GB for 7B params.
#      NF4 is information-theoretically optimal for normally-distributed weights.
#   2. Use double quantization: quantize the quantization constants themselves,
#      saving another ~0.4 bits/param.
#   3. Apply LoRA adapters on top in bf16 — adapters stay in full precision,
#      gradients are computed in bf16, base weights never leave 4-bit.
#
# Net result: fine-tune a 7B model on a single 16 GB GPU with minimal accuracy loss.
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 2: QLoRA (4-bit quantization + LoRA)")
print("=" * 60)

# --- 2a. 4-bit quantization config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # load weights in 4-bit NF4
    bnb_4bit_quant_type="nf4",                  # NormalFloat4 — best for LLM weight distributions
    bnb_4bit_compute_dtype=torch.bfloat16,      # upcast to bf16 for matrix multiplications
    bnb_4bit_use_double_quant=True,             # quantize the quantization constants (~0.4 bit saved)
)

# --- 2b. Load base model in 4-bit ---
# The model is loaded with quantized weights; bitsandbytes handles dequantization on-the-fly.
qlora_base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map="auto",                          # required when using bitsandbytes quantization
)

# prepare_model_for_kbit_training() is no longer needed in recent PEFT versions —
# gradient checkpointing and layer casting are handled automatically.
# OPTIONAL: uncomment if you encounter dtype issues with older PEFT (<0.7):
# from peft import prepare_model_for_kbit_training
# qlora_base_model = prepare_model_for_kbit_training(qlora_base_model)

# --- 2c. LoRA config for QLoRA ---
# r can often be raised compared to LoRA because the 4-bit base model gives back
# enough VRAM headroom. Including MLP layers is common in QLoRA papers.
qlora_config = LoraConfig(
    r=64,               # higher rank affordable because base model is 4-bit (saves ~10 GB)
    lora_alpha=128,     # keep alpha = 2×r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    # Attention + MLP: use when the task domain differs strongly from pretraining
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Wrap the 4-bit model with LoRA — adapters are still in bf16
qlora_model = get_peft_model(qlora_base_model, qlora_config)
qlora_model.print_trainable_parameters()

# --- 2d. Train with SFTTrainer (identical to LoRA — quantization is transparent) ---
qlora_output_dir = "./outputs/qlora_adapter"

qlora_trainer = SFTTrainer(
    model=qlora_model,
    args=SFTConfig(
        output_dir=qlora_output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        num_train_epochs=EPOCHS,
        bf16=True,
        logging_steps=LOG_STEPS,
        eval_strategy="epoch",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

qlora_trainer.train()


# ==============================================================================
# SECTION 3 — ADAPTER LIFECYCLE: SAVE / LOAD / MERGE
# ==============================================================================
#
# After training, you have two independent artifacts:
#   • The frozen base model   (unchanged; can be shared across many adapters)
#   • The LoRA adapter        (tiny; ~20-200 MB vs 14 GB for the full model)
#
# This separation is the key operational advantage of PEFT:
#   - Store / version / swap adapters without touching the base model.
#   - Merge for zero-latency inference when you no longer need to swap.
# ==============================================================================

print("\n" + "=" * 60)
print("SECTION 3: Adapter Lifecycle — Save / Load / Merge")
print("=" * 60)

# --- 3a. SAVE: only the adapter weights are saved (not the full model) ---
# save_pretrained() writes adapter_config.json + adapter_model.safetensors
# The directory is tiny: ~80 MB for r=16 on a 7B model (vs ~14 GB for full weights).
adapter_save_path = "./outputs/lora_adapter_saved"
lora_trainer.model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)
print(f"Adapter saved to: {adapter_save_path}")
print(f"Files: {os.listdir(adapter_save_path)}")

# --- 3b. LOAD: reload the base model and attach the saved adapter ---
# This is how you resume inference or continue training from a saved adapter.
print("\nLoading adapter onto a fresh base model...")

reload_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# PeftModel.from_pretrained() reads adapter_config.json, injects the LoRA matrices,
# and loads the saved adapter weights — base model weights remain frozen.
loaded_peft_model = PeftModel.from_pretrained(reload_base, adapter_save_path)
loaded_peft_model.eval()
print("Adapter loaded successfully.")

# OPTIONAL: swap to a different adapter at runtime without reloading the base model
# loaded_peft_model.load_adapter("./outputs/another_adapter", adapter_name="v2")
# loaded_peft_model.set_adapter("v2")

# --- 3c. MERGE: fuse adapter weights into the base model for zero-latency inference ---
# merge_and_unload() computes W_new = W_frozen + (alpha/r) × B×A for every adapted
# layer, writes the result back into the base model, and removes the adapter scaffolding.
# After merging:
#   ✓ No LoRA overhead at inference — standard transformer forward pass
#   ✓ Can be saved as a regular HuggingFace model and shared on the Hub
#   ✗ Adapter is no longer separable; you cannot swap or fine-tune the adapter further
print("\nMerging adapter weights into the base model...")
merged_model = loaded_peft_model.merge_and_unload()

merged_save_path = "./outputs/lora_merged_model"
merged_model.save_pretrained(merged_save_path)
tokenizer.save_pretrained(merged_save_path)
print(f"Merged model saved to: {merged_save_path}")

# --- 3d. Verify the merged model loads as a plain AutoModelForCausalLM ---
print("\nVerifying merged model loads without PEFT...")
plain_model = AutoModelForCausalLM.from_pretrained(
    merged_save_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Quick sanity-check generation
reload_tokenizer = AutoTokenizer.from_pretrained(merged_save_path)
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain what LoRA is in one sentence.\n\n### Response:\n"
inputs = reload_tokenizer(prompt, return_tensors="pt").to(plain_model.device)

with torch.no_grad():
    output_ids = plain_model.generate(**inputs, max_new_tokens=80, do_sample=False)

response = reload_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nSample generation from merged model:\n{response}")

print("\nDone. Adapter lifecycle complete.")


if __name__ == "__main__":
    pass
