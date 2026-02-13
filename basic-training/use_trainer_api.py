# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=DVHs5aCA3l-_ 참고

import transformers
print(transformers.__version__)

## Load dataset

from datasets import load_dataset
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

# datasets["train"][10]
# {'text': 'The game\'s battle system , the BlitZ system , is carried over direct...'}

## Tokenizer
model_checkpoint = "distilgpt2"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Define a function that calls the tokenizer on our dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# print(tokenized_datasets["train"][1])
# {'input_ids': [796, 569, 18354, 7496, 17740, 6711, 796, 220, 198],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}


## Preprocessing
# block_size = tokenizer.model_max_length
block_size = 128

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# apply the function to our dataset
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))
# ' game and follows the " Nameless ", 

## Define model, Use Trainer
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

from transformers import Trainer, TrainingArguments
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-wikitext2",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

# initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# start training
trainer.train()

## Evaluation
import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# trainer.push_to_hub()
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("your-username/the-name-you-picked")
