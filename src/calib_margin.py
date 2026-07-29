"""
calib_margin.py — 채점 margin(1등-2등 로그우도 격차)이 실제 정답률과 어떻게 연동되는지 측정.

margin 구간별로 (그 구간에 속한 예측의 정확도, 커버리지)를 내서 게이팅 임계를 정한다.
목표: "이 margin 이상이면 정확도 X% 이상" 을 찾아 리포트가 그때만 단정하게.
사용: python calib_margin.py
"""
from __future__ import annotations
import json
import re

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
    model = PeftModel.from_pretrained(base, str(LORA)); model.eval()

    def score(image, prompt, cand):
        m = [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]},
             {"role": "assistant", "content": [{"type": "text",
              "text": json.dumps({"move": cand}, ensure_ascii=False)}]}]
        full = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        pr = proc.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(m)
        enc = proc(text=[full], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        plen = proc(text=[pr], images=imgs, videos=vids, return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone(); labels[:, :plen] = -100
        with torch.no_grad():
            return float(-model(**enc, labels=labels).loss.item())

    def norm(s):
        return re.sub(r"[^0-9a-z]", "", (s or "").lower())

    recs = []
    for k, r in enumerate(rows):
        cands = r.get("cands")
        if not cands or len(cands) < 2:
            continue
        sc = sorted(((score(r["image"], r["prompt"], c), c) for c in cands), reverse=True)
        margin = sc[0][0] - sc[1][0]
        recs.append((margin, norm(sc[0][1]) == norm(r["answer"])))
        if (k + 1) % 30 == 0:
            print(f"  ...{k+1}/{len(rows)}", flush=True)
    recs.sort(reverse=True)
    n = len(recs)
    print(f"표본 {n}\n임계   커버리지        정확도(그 이상)")
    for thr in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
        sel = [ok for mg, ok in recs if mg >= thr]
        if sel:
            print(f"{thr:4.2f}  {len(sel):3d}/{n} ({len(sel)/n*100:2.0f}%)   {sum(sel)}/{len(sel)} = {sum(sel)/len(sel)*100:.0f}%")


if __name__ == "__main__":
    main()
