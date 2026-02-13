import json
import random
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    default_data_collator,
)
from torch.utils.data import DataLoader


# -------------------------------------------------
# 1. Load model & tokenizer
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")


# -------------------------------------------------
# 2. Load Alpaca dataset
# -------------------------------------------------
with open("alpaca_gpt4_data.json", "r") as f:
    alpaca = json.load(f)


# -------------------------------------------------
# 3. Prompt formatting
# -------------------------------------------------
def prompt_no_input(row):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ).format_map(row)


def prompt_input(row):
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n### Response:\n"
    ).format_map(row)

row = alpaca[232]
print(prompt_input(row))


def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)


# -------------------------------------------------
# 4. Build dataset
# -------------------------------------------------
prompts = [create_prompt(row) for row in alpaca] # all LLM inputs are here
EOS_TOKEN = tokenizer.eos_token
outputs = [row["output"] + EOS_TOKEN for row in alpaca]
dataset = [
    {"prompt": s, "output": t, "example": s + t}
    for s, t in zip(prompts, outputs)
]

tokenizer.pad_token = tokenizer.eos_token
tokenizer.encode("My experiments are going strong!", padding='max_length', max_length=10, return_tensors="pt")
random.shuffle(dataset) #shuffle inplace

train_dataset = dataset[:-1000]
eval_dataset = dataset[-1000:]

max_seq_len = 1024


# -------------------------------------------------
# 5. Packing function
# -------------------------------------------------
def pack(dataset, max_seq_len=1024):
    tkds_ids = tokenizer([s["example"] for s in dataset])["input_ids"]

    all_token_ids = []
    for tokenized_input in tkds_ids:
        all_token_ids.extend(tokenized_input)

    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len + 1):
        input_ids = all_token_ids[i : i + max_seq_len + 1]
        if len(input_ids) == (max_seq_len + 1):
            packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})
    return packed_ds


train_ds_packed = pack(train_dataset, max_seq_len)
eval_ds_packed = pack(eval_dataset, max_seq_len)


# -------------------------------------------------
# 6. Dataloader
# -------------------------------------------------
batch_size = 8
lr = 3e-6
epochs = 3
gradient_accumulation_steps = 32 // batch_size

train_dataloader = DataLoader(
    train_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator,
)

eval_dataloader = DataLoader(
    eval_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator,
    shuffle=False,
)


# -------------------------------------------------
# 7. Optimizer & Scheduler
# -------------------------------------------------
total_train_steps = epochs * len(train_dataloader) // gradient_accumulation_steps

optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5)
scheduler = get_cosine_schedule_with_warmup(
    optim,
    num_training_steps=total_train_steps,
    num_warmup_steps=total_train_steps // 10,
)


# -------------------------------------------------
# 8. Loss function
# -------------------------------------------------
def loss_fn(x, y):
    return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]),y.view(-1))


# -------------------------------------------------
# 9. Validation
# -------------------------------------------------
@torch.no_grad()
def validate():
    model.eval()

    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch["input_ids"] = batch["input_ids"].to("cuda")
        batch["labels"] = batch["labels"].to("cuda")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"])

    print("Eval Loss:", loss.item())
    model.train()


# -------------------------------------------------
# 10. Training Loop
# -------------------------------------------------
model.train()
train_step = 0

pbar = tqdm(total=total_train_steps)
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        batch["input_ids"] = batch["input_ids"].to("cuda")
        batch["labels"] = batch["labels"].to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = loss_fn(out.logits, batch["labels"]) / gradient_accumulation_steps
            loss.backward()
        if step % gradient_accumulation_steps == 0:
            print("Train loss:", loss.item())
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            train_step += 1
            pbar.update(1)
    validate()
pbar.close()