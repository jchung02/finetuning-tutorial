# ============================================================
# DeepSpeed Pipeline Parallelism을 이용한 LLaMA Fine-tuning
# ============================================================
#
# Pipeline Parallelism은 모델의 레이어들을 여러 GPU에 순차적으로 분배합니다.
#
#   [GPU 0]                    [GPU 1]
#   Embedding                  Transformer Layer 16~31
#   Transformer Layer 0~15     Norm + LM Head
#         ──── hidden states ────>
#
# DeepSpeed의 PipelineModule은 모델이 nn.Sequential처럼
# 레이어가 순차적으로 연결된 구조를 요구합니다.
# 따라서 HuggingFace 모델을 "Sequential한 레이어 리스트"로 분해해야 합니다.
#
# 참고: SFTTrainer는 Pipeline Parallelism을 지원하지 않으므로,
#       DeepSpeed의 PipelineEngine을 직접 사용합니다.
#
# 실행:
#   deepspeed --num_gpus 4 multi-gpu/pp_deepspeed2.py
# ============================================================

import os
import json
import random
import argparse

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM

import deepspeed
from deepspeed.pipe import PipelineModule


# ============================================================
# DeepSpeed Config (외부 JSON 파일 대신 코드 내 딕셔너리로 정의)
# ============================================================
# train_batch_size = micro_batch × gradient_accumulation × data_parallel_size
# 예: GPU 4개, PP=2 → DP=2, micro_batch=1 → gas = 8/(1×2) = 4
# ============================================================
DS_CONFIG = {
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    },
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
}


# ============================================================
# Step 1. Pipeline Stage용 Wrapper Module 정의
# ============================================================
# PipelineModule은 각 레이어가 "하나의 텐서 입력 → 하나의 텐서 출력"
# 형태인 Sequential 구조를 요구합니다.
# LLaMA의 각 구성요소를 이 인터페이스에 맞게 감싸줍니다.
#
# LLaMA 원래 구조:
#   input_ids → embed_tokens → [decoder_layer × 32] → norm → lm_head → logits
#
# Pipeline용 변환 (튜플 전달 방식):
#   input_ids → [EmbeddingLayer] → (hidden, cos, sin)
#            → [TransformerBlock × 32] → (hidden, cos, sin)
#            → [NormLMHeadLayer] → logits
#
# 최신 transformers에서 RoPE(cos, sin)는 LlamaModel 상위에서 한 번 계산되어
# 각 decoder layer로 전달됩니다. Pipeline에서는 이를 튜플에 담아 함께 넘깁니다.
# ============================================================

class EmbeddingLayer(nn.Module):
    """Token ID → Embedding + RoPE 계산 (파이프라인 첫 번째 스테이지)"""

    def __init__(self, embed_tokens, rotary_emb):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb  # LlamaModel.rotary_emb

    def forward(self, input_ids):
        # input_ids가 float로 캐스팅될 수 있으므로 long으로 명시적 변환
        hidden_states = self.embed_tokens(input_ids.long())

        # RoPE (cos, sin) 계산 — 모든 Transformer Block에서 공유
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_len, device=hidden_states.device
        ).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        return (hidden_states, cos, sin)


class TransformerBlockLayer(nn.Module):
    """하나의 Transformer Decoder Block (파이프라인 중간 스테이지)"""

    def __init__(self, decoder_layer):
        super().__init__()
        self.decoder_layer = decoder_layer

    def forward(self, inputs):
        # 이전 레이어에서 전달받은 튜플 언패킹
        hidden_states, cos, sin = inputs

        output = self.decoder_layer(
            hidden_states,
            position_embeddings=(cos, sin),
        )
        # hidden_states를 갱신하고, cos/sin은 그대로 다음 레이어로 전달
        return (output[0], cos, sin)


class NormLMHeadLayer(nn.Module):
    """Final LayerNorm + LM Head (파이프라인 마지막 스테이지)"""

    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, inputs):
        # cos, sin은 더 이상 필요 없으므로 버림
        hidden_states, cos, sin = inputs
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


# ============================================================
# Step 2. 모델을 Sequential Layer 리스트로 분해
# ============================================================
# LLaMA 내부 구조:
#   model.model.embed_tokens          (nn.Embedding)
#   model.model.layers[0]             (LlamaDecoderLayer)
#   model.model.layers[1]             (LlamaDecoderLayer)
#   ...
#   model.model.layers[31]            (LlamaDecoderLayer)
#   model.model.norm                  (RMSNorm)
#   model.lm_head                     (nn.Linear)
#
# 이를 평탄한(flat) 리스트로 변환:
#   [EmbeddingLayer, Block_0, Block_1, ..., Block_31, NormLMHeadLayer]
#   → 총 34개 레이어 → DeepSpeed가 num_stages개 GPU에 자동 분배
# ============================================================

def get_sequential_layers(model):
    layers = []
    # EmbeddingLayer에 rotary_emb도 함께 전달 (LlamaModel 상위에 위치)
    layers.append(EmbeddingLayer(model.model.embed_tokens, model.model.rotary_emb))
    for block in model.model.layers:
        layers.append(TransformerBlockLayer(block))
    layers.append(NormLMHeadLayer(model.model.norm, model.lm_head))
    return layers


# ============================================================
# Step 3. Causal LM Loss 정의
# ============================================================
# PipelineModule의 loss_fn으로 전달됩니다.
# 마지막 스테이지 GPU에서 logits과 labels를 받아 loss를 계산합니다.
#
#  DataLoader → (input_ids, labels)
#                    ↓           ↓
#             첫번째 스테이지   마지막 스테이지의 loss_fn
# ============================================================

def causal_lm_loss(logits, labels):
    """
    Causal LM: token[i]로 token[i+1]을 예측
    logits를 한 칸 앞으로, labels를 한 칸 뒤로 shift하여 비교합니다.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ============================================================
# Step 4. 데이터셋 준비
# ============================================================
# Pipeline에서 DataLoader는 (input, label) 튜플을 반환해야 합니다.
#   - input (input_ids) → 첫 번째 스테이지로 전달
#   - label (labels)    → 마지막 스테이지의 loss_fn으로 전달
# ============================================================

def prepare_dataset(tokenizer, max_seq_length=1024):
    with open("alpaca_gpt4_data.json", "r") as f:
        raw_data = json.load(f)
    random.shuffle(raw_data)

    # Alpaca 포맷으로 프롬프트 구성
    texts = []
    for row in raw_data:
        if row["input"] == "":
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}"
            ).format_map(row)
        else:
            text = (
                "Below is an instruction that describes a task, "
                "paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n### Response:\n{output}"
            ).format_map(row)
        texts.append(text)

    # 토크나이즈
    tokenizer.pad_token = tokenizer.eos_token
    encodings = tokenizer(
        texts,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]

    # Labels: padding 토큰 위치를 -100으로 마스킹 (loss 계산에서 제외)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return torch.utils.data.TensorDataset(input_ids, labels)


# ============================================================
# Step 5. 학습
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="LLaMA Pipeline Parallelism Fine-tuning"
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--pipeline_parallel_size", type=int, default=2,
        help="Number of pipeline stages (= number of GPUs for PP)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # 분산 환경 초기화
    deepspeed.init_distributed(dist_backend="nccl")
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(args.local_rank)
    torch.manual_seed(args.seed)

    # --- 토크나이저 로드 ---
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

    # --- 모델 로드 ---
    # 주의: 모든 GPU에서 전체 모델을 로드한 뒤 PipelineModule이 분할합니다.
    # 메모리가 부족하면 LayerSpec을 사용한 지연 생성(deferred construction)을 고려하세요.
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
    )

    # --- 모델을 Sequential Layer 리스트로 분해 ---
    layers = get_sequential_layers(model)
    del model  # 원본 모델 메모리 해제

    # --- PipelineModule 생성 ---
    # DeepSpeed가 layers를 num_stages개 GPU에 자동 분배
    pipe_model = PipelineModule(
        layers=layers,
        loss_fn=causal_lm_loss,
        num_stages=args.pipeline_parallel_size,
        activation_checkpoint_interval=0,  # >0 설정 시 activation checkpointing 활성화
    )

    # --- 데이터셋 준비 ---
    train_dataset = prepare_dataset(tokenizer)

    # --- DeepSpeed 엔진 초기화 ---
    # config를 딕셔너리로 직접 전달 (외부 JSON 파일 불필요)
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_model,
        model_parameters=[p for p in pipe_model.parameters() if p.requires_grad],
        training_data=train_dataset,
        config=DS_CONFIG,
    )

    # --- 학습 루프 ---
    # engine.train_batch()가 micro-batch 스케줄링을 자동 처리합니다.
    # Pipeline에서는 여러 micro-batch가 동시에 서로 다른 스테이지를 통과하며,
    # 이를 통해 GPU idle time(bubble)을 최소화합니다.
    for step in range(1, args.steps + 1):
        loss = engine.train_batch()

        # loss는 마지막 스테이지에서만 유의미한 값을 가짐
        if step % 10 == 0 and engine.is_last_stage():
            print(f"Step {step}/{args.steps} | Loss: {loss.item():.4f}")

    # --- 체크포인트 저장 ---
    # DeepSpeed 형식으로 저장 (각 스테이지별 파라미터)
    engine.save_checkpoint("./outputs/pp_deepspeed2")


if __name__ == "__main__":
    main()
