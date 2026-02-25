# LLM Fine-tuning Tutorial

A collection of practical examples for fine-tuning large language models, covering single-GPU training, PEFT methods, and multi-GPU distributed training strategies.

## Contents

- `basic-training/` - Introductory training examples using the Trainer API
- `single-gpu/` - Basic single-GPU fine-tuning with vanilla training loop and SFTTrainer
- `multi-gpu/` - Distributed training examples including data parallelism, pipeline parallelism, and DeepSpeed/Accelerate integration
- `peft/` - Parameter-efficient fine-tuning with LoRA
- `configs/` - Configuration files for distributed training setups

## Requirements

```
pip install -r requirements.txt
```
