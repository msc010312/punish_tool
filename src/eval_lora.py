"""
eval_lora.py  —  파인튠 효과 검증: 같은 val셋에서 base-3B vs LoRA-3B 무브 정확도 비교.

train_data/val.jsonl(보류셋, 학습 미포함)으로, base Qwen2.5-VL-3B 와 LoRA 적용본이
각각 정답 무브를 맞히는 비율을 잰다. (val 정답은 자동라벨이므로 '라벨 재현율' 기준 —
LoRA가 base보다 높으면 학습이 패턴을 익힌 것.)
사용: python eval_lora.py
"""
from __future__ import annotations
import json, re
from pathlib import Path

import torch
import punish_engine as pe

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA = pe.app_dir() / "qwen_lora"
VAL = pe.app_dir() / "train_data" / "val.jsonl"


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    rows = [json.loads(l) for l in VAL.open(encoding="utf-8") if l.strip()]
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 256, max_pixels=512 * 512)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda")

    def predict(model, r):
        m = [{"role": "user", "content": [{"type": "image", "image": r["image"]},
                                          {"type": "text", "text": r["prompt"]}]}]
        text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(m)
        enc = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=48, do_sample=False)
        gen = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        try:
            mv = json.loads(gen).get("move", "")
        except Exception:
            mm = re.search(r'"move"\s*:\s*"([^"]+)"', gen)
            mv = mm.group(1) if mm else gen.strip().split()[0] if gen.strip() else ""
        return mv

    def score(model, tag):
        ok = 0
        for r in rows:
            mv = predict(model, r)
            hit = mv.lower().replace(".", "") == r["answer"].lower().replace(".", "")
            ok += hit
            print(f"  [{tag}] {r['char'][:8]:8} 정답={r['answer']:8} 예측={mv:10} {'O' if hit else 'X'}", flush=True)
        print(f"=== {tag} 정확도 {ok}/{len(rows)} = {ok/len(rows)*100:.0f}% ===\n", flush=True)
        return ok

    print(">>> BASE (파인튠 전)", flush=True)
    b = score(base, "base")
    print(">>> LoRA (파인튠 후)", flush=True)
    lora = PeftModel.from_pretrained(base, str(LORA))
    l = score(lora, "lora")
    print(f"\n결과: base {b}/{len(rows)} -> LoRA {l}/{len(rows)}  "
          f"({'개선 +' if l > b else '변화 '}{l-b})")


if __name__ == "__main__":
    main()
