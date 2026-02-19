"""
Pipeline Parallel finetuning using torch.distributed.pipelining

Launch with torchrun:
  torchrun --nproc_per_node=<NUM_GPUS> multi-gpu/pp_torch2.py

Example (4 GPUs):
  torchrun --nproc_per_node=4 multi-gpu/pp_torch2.py
"""

import os
import json
import random
import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    default_data_collator,
)
from torch.utils.data import DataLoader
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe


# -------------------------------------------------
# 1. Distributed setup
# -------------------------------------------------
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ.get("LOCAL_RANK", rank))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl")

assert world_size > 1, "Pipeline parallelism requires at least 2 GPUs"


# -------------------------------------------------
# 2. Load model & tokenizer (CPU first, then split across GPUs)
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B", low_cpu_mem_usage=True
)
model.config.use_cache = False
model.config.return_dict = False 
# class CausalLMLogitsOnly(nn.Module):
#     def __init__(self, base):
#         super().__init__()
#         self.base = base
#         self.config = base.config 

#     def forward(self, input_ids):
#         out = self.base(input_ids=input_ids, use_cache=False)
#         return out.logits  # ✅ Tensor만 반환

# model = CausalLMLogitsOnly(model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")


# -------------------------------------------------
# 3. Load Alpaca dataset
# -------------------------------------------------
with open("alpaca_gpt4_data.json", "r") as f:
    alpaca = json.load(f)


# -------------------------------------------------
# 4. Prompt formatting
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


def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)


# -------------------------------------------------
# 5. Build dataset
# -------------------------------------------------
prompts = [create_prompt(row) for row in alpaca]
EOS_TOKEN = tokenizer.eos_token
outputs_text = [row["output"] + EOS_TOKEN for row in alpaca]
dataset = [
    {"prompt": s, "output": t, "example": s + t}
    for s, t in zip(prompts, outputs_text)
]

tokenizer.pad_token = tokenizer.eos_token

# Fixed seed so all ranks shuffle identically (pipeline parallelism = same data order)
random.seed(42)
random.shuffle(dataset)

train_dataset = dataset[:-1000]
eval_dataset = dataset[-1000:]

max_seq_len = 1024


# -------------------------------------------------
# 6. Packing function
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
# 7. Hyperparameters
# -------------------------------------------------
batch_size = 8
n_microbatches = 4  # Must evenly divide batch_size; keeps pipeline stages busy
lr = 3e-6
epochs = 3


# -------------------------------------------------
# 8. Dataloader (same data on all ranks — this is pipeline parallelism, not data parallelism)
# -------------------------------------------------
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
# 9. Pipeline parallelism setup
# -------------------------------------------------
# Split model into equal chunks of transformer layers across ranks
layers_per_rank = model.config.num_hidden_layers // world_size
if rank == 0:
    print(f"Pipeline: {layers_per_rank} layers per rank across {world_size} GPUs")

split_spec = {
    f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
    for i in range(1, world_size)
}
# split_spec = {
#     f"base.model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
#     for i in range(1, world_size)
# }

# Trace the model with an example input to build the pipeline graph
# example_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, max_seq_len))

# pipe = pipeline(
#     model,
#     mb_args=(example_input_ids,),
#     split_spec=split_spec,
# )
example_input_ids = torch.randint(
    0, tokenizer.vocab_size, (batch_size, max_seq_len), dtype=torch.long
)
pipe = pipeline(
    model,
    mb_args=(example_input_ids,),
    split_spec=split_spec,
)

# Build the pipeline stage for this rank (moves relevant weights to device)
stage = pipe.build_stage(rank, device=device)

# Free original full model to save memory
del model
torch.cuda.empty_cache()


# -------------------------------------------------
# 10. Loss function (applied at the last pipeline stage)
# -------------------------------------------------
# def loss_fn(output, target):
#     # Pipeline output may be CausalLMOutput, tuple, or raw tensor
#     if hasattr(output, "logits"):
#         logits = output.logits
#     elif isinstance(output, tuple):
#         logits = output[0]
#     else:
#         logits = output
#     return torch.nn.functional.cross_entropy(
#         logits.view(-1, logits.shape[-1]), target.view(-1)
#     )
def loss_fn(output, target):
    # return_dict=False 이면 output은 보통 (logits,) 또는 logits
    logits = output[0] if isinstance(output, (tuple, list)) else output
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target.view(-1),
    )

# -------------------------------------------------
# 11. Optimizer & Scheduler (only this stage's parameters)
# -------------------------------------------------
total_train_steps = epochs * len(train_dataloader)

optim = torch.optim.Adam(
    stage.submod.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5
)
scheduler = get_cosine_schedule_with_warmup(
    optim,
    num_training_steps=total_train_steps,
    num_warmup_steps=total_train_steps // 10,
)


# -------------------------------------------------
# 12. Pipeline schedule
# -------------------------------------------------
# Training schedule: forward + backward across microbatches (GPipe strategy)
train_schedule = ScheduleGPipe(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)


# -------------------------------------------------
# 13. Validation
# -------------------------------------------------
def validate():
    stage.submod.eval()
    eval_loss_sum = 0.0
    eval_steps = 0

    # Inference-only schedule (no loss_fn → forward only, no backward)
    eval_schedule = ScheduleGPipe(stage, n_microbatches=n_microbatches)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, disable=(rank != 0)):
            if rank == 0:
                eval_schedule.step(batch["input_ids"].to(device))
            elif rank == world_size - 1:
                output = eval_schedule.step()
                labels = batch["labels"].to(device)
                loss = loss_fn(output, labels)
                eval_loss_sum += loss.item()
                eval_steps += 1
            else:
                eval_schedule.step()

    if rank == world_size - 1:
        print(f"Eval Loss: {eval_loss_sum / max(eval_steps, 1):.4f}")

    stage.submod.train()


# -------------------------------------------------
# 14. Training Loop
# -------------------------------------------------
stage.submod.train()

pbar = tqdm(total=total_train_steps, disable=(rank != 0))
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        # Pipeline schedule: first rank feeds input, last rank computes loss
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if rank == 0:
                train_schedule.step(batch["input_ids"].to(device))
            elif rank == world_size - 1:
                losses = []
                train_schedule.step(
                    target=batch["labels"].to(device), losses=losses
                )
            else:
                train_schedule.step()

        optim.step()
        scheduler.step()
        optim.zero_grad(set_to_none=True)

        if rank == world_size - 1 and step % 10 == 0:
            avg_loss = sum(l.item() for l in losses) / len(losses)
            print(f"[Epoch {epoch}] Step {step}, Train loss: {avg_loss:.4f}")

        pbar.update(1)

    validate()

pbar.close()
dist.destroy_process_group()
