"""
train_lora.py  —  Qwen2.5-VL-3B QLoRA 파인튠 (자가학습 데이터로 무브식별 특화)

train_data/{train,val}.jsonl (convert_train.py 산출)로 Qwen2.5-VL-3B-Instruct를 4bit QLoRA 학습.
입력=두인물 시트(추론과 동일 포맷), 정답=무브 표기. 프롬프트 토큰은 마스킹(정답만 학습).
RTX 4070 12GB: 4bit + grad checkpoint + batch1(+grad accum) + 이미지 픽셀 제한으로 맞춤.

사용: python train_lora.py [--epochs 3] [--smoke]   (--smoke=2스텝만, 동작점검)
출력: qwen_lora/  (LoRA 어댑터)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
import punish_engine as pe

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
OUT = pe.app_dir() / "qwen_lora"
DATA = pe.app_dir() / "train_data"
MAXPX = 512 * 512          # 시트 픽셀 상한(인게임 프레임 디테일 보존 — 판별 정확도↑)


def load_rows(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def main():
    global MODEL, MAXPX, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt-dir", default=None, help="체크포인트 폴더(중단·재개용). 지정 시 주기 저장")
    ap.add_argument("--save-steps", type=int, default=40, help="몇 스텝마다 체크포인트 저장")
    ap.add_argument("--resume", action="store_true", help="ckpt-dir의 마지막 체크포인트에서 재개")
    ap.add_argument("--model", default=MODEL, help="베이스 모델(7B 실험 등)")
    ap.add_argument("--maxpx", type=int, default=MAXPX, help="시트 픽셀 상한(7B는 VRAM 위해 낮춤)")
    ap.add_argument("--out", default=str(OUT), help="어댑터 저장 폴더")
    a = ap.parse_args()
    MODEL, MAXPX, OUT = a.model, a.maxpx, Path(a.out)

    from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig, TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from qwen_vl_utils import process_vision_info

    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 256, max_pixels=MAXPX)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train = load_rows(DATA / "train.jsonl")
    if a.smoke:
        train = train[:2]
    print(f"train {len(train)}")

    def msgs_of(r):
        return [{"role": "user", "content": [{"type": "image", "image": r["image"]},
                                             {"type": "text", "text": r["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text",
                 "text": json.dumps({"move": r["answer"]}, ensure_ascii=False)}]}]

    def collate(batch):
        r = batch[0]                                  # batch1 (패딩/마스킹 단순화)
        m = msgs_of(r)
        full_text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        prompt_text = proc.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(m)
        enc = proc(text=[full_text], images=imgs, videos=vids, return_tensors="pt", padding=True)
        plen = proc(text=[prompt_text], images=imgs, videos=vids, return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100
        labels[labels == proc.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return dict(enc)                              # CPU 텐서 — Trainer가 device로 이동

    ckpt = a.ckpt_dir or (str(OUT) + "_ckpt")            # 중단·재개용 체크포인트 폴더
    save_kw = dict(save_strategy="steps", save_steps=a.save_steps, save_total_limit=2) \
        if not a.smoke else dict(save_strategy="no")
    args = TrainingArguments(
        output_dir=ckpt, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=(0.01 if a.smoke else a.epochs), learning_rate=1e-4,
        bf16=True, logging_steps=2, report_to=[],
        optim="paged_adamw_8bit",                        # QLoRA 표준: 옵티마이저 메모리 반감(7B fit)
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=(2 if a.smoke else -1),
        remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=False,
        **save_kw)
    tr = Trainer(model=model, args=args, train_dataset=train, data_collator=collate)
    # 재개: 명시적 --resume 이거나, ckpt 폴더에 체크포인트가 있으면 자동 이어하기
    resume = None
    if not a.smoke and Path(ckpt).exists() and any(Path(ckpt).glob("checkpoint-*")):
        if a.resume or True:                             # 체크포인트 있으면 항상 이어하기(중단내성)
            resume = True
            print(f"[재개] {ckpt}의 마지막 체크포인트에서 이어감", flush=True)
    tr.train(resume_from_checkpoint=resume)
    if not a.smoke:
        model.save_pretrained(str(OUT)); proc.save_pretrained(str(OUT))
        print(f"\n저장 -> {OUT}  (체크포인트: {ckpt})")
    else:
        print("\n스모크 OK — 학습 루프 동작")


if __name__ == "__main__":
    main()
