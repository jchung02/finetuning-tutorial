import json
import random
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")


with open("alpaca_gpt4_data.json", "r") as f:
    dataset = json.load(f)

random.shuffle(dataset) #shuffle inplace
train_dataset = dataset[:-1000]
eval_dataset = dataset[-1000:]

def preprocess(dataset):
    dataset_dict = {"text":[]}
    for row in dataset:
        if row["input"] == "":
            text = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response\n{output}").format_map(row)
        else:
            text = ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n ### Input:\n{input}\n\n### Response:\n{output}").format_map(row)
        dataset_dict["text"].append(text)
    return Dataset.from_dict(dataset_dict)

train_dataset = preprocess(train_dataset)
eval_dataset = preprocess(eval_dataset)

batch_size = 8
lr = 3e-6
epochs = 3
gradient_accumulation_steps = 8

training_args = SFTConfig(
    output_dir="./finetuning-tutorial",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=lr,
    logging_steps=5,
    num_train_epochs=epochs,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    eval_strategy="epoch",
    logging_first_step=True,
    max_length=1024,
    dataset_text_field="text",
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model("./finetuning-tutorial")