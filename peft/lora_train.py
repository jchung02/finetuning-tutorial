# Run : python peft/lora_train.py

import os
import json
import random

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

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


##### 1: LoRA

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # Llama has no pad token by default

lora_config = LoraConfig(
    r=16,               
    lora_alpha=32,      
    lora_dropout=0.05,  
    bias="none",        
    task_type="CAUSAL_LM",

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

#  Wrap the base model with LoRA adapters 
lora_model = get_peft_model(base_model, lora_config) # freezes all original weights and injects trainable A/B matrices into the specified target_modules. 

lora_model.print_trainable_parameters() # trainable params: 13,631,488 || all params: 8,043,892,736 || trainable%: 0.1695

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

lora_trainer.train()


##### SECTION 2: QLoRA (4-bit quantization + LoRA)

# 4-bit quantization config 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          
    bnb_4bit_quant_type="nf4",                  
    bnb_4bit_compute_dtype=torch.bfloat16,      
    bnb_4bit_use_double_quant=True,            
)

qlora_base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config, # load weights in 4-bit with the above config
    attn_implementation="flash_attention_2",
    device_map="auto",                          
)

qlora_config = LoraConfig(
    r=64,               
    lora_alpha=128,     
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Wrap the 4-bit model with LoRA — adapters are still in bf16
qlora_model = get_peft_model(qlora_base_model, qlora_config)
qlora_model.print_trainable_parameters()

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

##### 3: Adapter Lifecycle — Save / Load / Merge

# SAVE: only the adapter weights are saved
adapter_save_path = "./outputs/lora_adapter_saved"
lora_trainer.model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)
print(f"Adapter saved to: {adapter_save_path}")
print(f"Files: {os.listdir(adapter_save_path)}")

# LOAD: reload the base model and attach the saved adapter
print("\nLoading adapter onto a fresh base model...")

reload_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

loaded_peft_model = PeftModel.from_pretrained(reload_base, adapter_save_path)
loaded_peft_model.eval()
print("Adapter loaded successfully.")

# MERGE: fuse adapter weights into the base model for zero-latency inference 
print("\nMerging adapter weights into the base model...")
merged_model = loaded_peft_model.merge_and_unload()

merged_save_path = "./outputs/lora_merged_model"
merged_model.save_pretrained(merged_save_path)
tokenizer.save_pretrained(merged_save_path)
print(f"Merged model saved to: {merged_save_path}")

print("\nVerifying merged model loads without PEFT...")
plain_model = AutoModelForCausalLM.from_pretrained(
    merged_save_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

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
