# -*- coding: utf-8 -*-
"""train_sft.py — 코치 QLoRA 파인튜닝 (Qwen3-8B, RTX 4070 12GB).

목표(제1원칙 순서): ①근거순종(참고데이터 있으면 그대로/없는 수치는 "몰라")
②솔 페르소나 ③한국어 격겜 유창성. 숫자 지식은 RAG(coach_kb)가 담당 —
가중치에 지식을 넣는 게 아니라 '행동'을 가르친다.

데이터: dataset/rag_train.jsonl (build_dataset.py 산출, messages 포맷 —
trl SFTTrainer가 토크나이저 챗템플릿을 자동 적용한다).

Windows 주의: Unsloth는 triton 의존이라 안 씀. 순정 peft+trl+bitsandbytes.

사용:
  python src/train_sft.py --dry     # 5스텝 스모크(OOM/API 확인)
  python src/train_sft.py           # 본 학습 (기본 2 epoch)
산출: lora_out/coach_ragA/  (LoRA 어댑터 + 토크나이저)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3-8B"
OUT_DIR = ROOT / "lora_out" / "coach_ragA"
TRAIN_JSONL = ROOT / "dataset" / "rag_train.jsonl"
EVAL_JSONL = ROOT / "dataset" / "rag_eval.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="5스텝 스모크 테스트")
    ap.add_argument("--epochs", type=float, default=3.0)   # 야간 학습 기준(과적합 경계는 3~4)
    ap.add_argument("--resume", action="store_true",
                    help="output_dir의 마지막 체크포인트부터 이어서")
    args = ap.parse_args()

    # 주의: Windows에서 datasets(pyarrow)는 torch/transformers보다 '먼저' import.
    # 반대 순서면 DLL 충돌로 세그폴트(실측).
    from datasets import load_dataset
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print(f"[load] {BASE_MODEL} (4-bit nf4)", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    model.config.use_cache = False           # gradient checkpointing과 충돌 방지
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    lora = LoraConfig(
        # v4: alpha 32->16 — LoRA 영향력을 절반으로. 행동(거절·근거순종)은 배우되
        # 베이스의 대화 유창성을 덜 덮어쓰게(v1~v3 공통 부작용 완화).
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    ds_train = load_dataset("json", data_files=str(TRAIN_JSONL), split="train")
    ds_eval = load_dataset("json", data_files=str(EVAL_JSONL), split="train")
    print(f"[data] train {len(ds_train)} / eval {len(ds_eval)}", flush=True)

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        max_length=1024,                      # 예시가 짧아 충분(잘림 거의 없음)
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,        # 유효 배치 16
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        max_steps=5 if args.dry else -1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        # 25스텝(~20분)마다 저장 -> 중단해도 그 지점부터 --resume 으로 이어함
        save_strategy="no" if args.dry else "steps",
        save_steps=25,
        # 6개 보관 -> 2에폭(~step 200) 시점 체크포인트가 살아남는다.
        # 3에폭이 과학습이면 재학습 없이 2에폭 지점으로 병합해 비교 가능(보험).
        save_total_limit=6,
        eval_strategy="no" if args.dry else "epoch",
        per_device_eval_batch_size=2,
        report_to=[],
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=cfg, peft_config=lora,
                         train_dataset=ds_train, eval_dataset=ds_eval,
                         processing_class=tok)

    print("[train] start", flush=True)
    resume = bool(args.resume and any(OUT_DIR.glob("checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume or None)
    if not args.dry:
        trainer.save_model(str(OUT_DIR))
        tok.save_pretrained(str(OUT_DIR))
        print(f"[done] adapter -> {OUT_DIR}", flush=True)
    else:
        print("[dry] smoke OK", flush=True)


if __name__ == "__main__":
    main()
