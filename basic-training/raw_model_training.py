# https://huggingface.co/docs/transformers/v4.18.0/en/training#train-in-native-pytorch

from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
print(dataset["train"][100])
# {'label': 0,
# 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so ...

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

 # Remove the text column because the model does not accept raw text as an input:
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
# Rename the label column to labels because the model expects the argument to be named labels:
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# Set the format of the dataset to return PyTorch tensors instead of lists:
tokenized_datasets.set_format("torch")

# Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning:
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# DataLoader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# Load model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# Optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Device
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop 
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
# Evaluation
from datasets import load_metric        
metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()