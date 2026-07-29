"""
eval_score.py — 후보 '확률 채점'(closed-set) 방식 vs 자유생성 정확도 비교.

자유생성: 모델이 move 문자열을 생성 -> 후보로 스냅 (현행)
확률채점: 각 후보에 대해 target {"move":"<cand>"}의 로그우도를 계산 -> 최고 후보 선택
  = 쓰레기 출력 불가 + 전체 확률신호 활용. 고정 후보셋에선 보통 더 정확.
사용: python eval_score.py
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
        """target {"move":"cand"}의 평균 토큰 로그우도(높을수록 그 후보일 확률↑)."""
        m = [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]},
             {"role": "assistant", "content": [{"type": "text",
              "text": json.dumps({"move": cand}, ensure_ascii=False)}]}]
        full = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        pr = proc.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(m)
        enc = proc(text=[full], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        plen = proc(text=[pr], images=imgs, videos=vids, return_tensors="pt")["input_ids"].shape[1]
        with torch.no_grad():
            logits = model(**enc).logits
        ids = enc["input_ids"][0]
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tok = ids[1:]
        tgt = range(plen - 1, len(ids) - 1)                # assistant 토큰 구간
        vals = [lp[i, tok[i]].item() for i in tgt]
        return sum(vals) / max(len(vals), 1)

    def norm(s):
        return re.sub(r"[^0-9a-z]", "", (s or "").lower())

    hit_gen = hit_score = n = 0
    for r in rows:
        cands = r.get("cands")
        if not cands:
            continue
        n += 1
        # closed-set 채점
        best = max(cands, key=lambda c: score(r["image"], r["prompt"], c))
        hit_score += (norm(best) == norm(r["answer"]))
        # 자유생성(비교용)
        mm = [{"role": "user", "content": [{"type": "image", "image": r["image"]},
                                           {"type": "text", "text": r["prompt"]}]}]
        text = proc.apply_chat_template(mm, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(mm)
        enc = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=48, do_sample=False)
        gen = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        try:
            g = json.loads(gen).get("move", "")
        except Exception:
            g = (re.search(r'"move"\s*:\s*"([^"]+)"', gen) or [None, gen[:10]])[1]
        hit_gen += (norm(g) == norm(r["answer"]))
        if n % 30 == 0:
            print(f"  {n}: 채점 {hit_score}/{n} ({hit_score/n*100:.0f}%) | 생성 {hit_gen}/{n} ({hit_gen/n*100:.0f}%)", flush=True)

    print(f"\n=== 자유생성 {hit_gen}/{n} = {hit_gen/n*100:.0f}% ===", flush=True)
    print(f"=== 확률채점 {hit_score}/{n} = {hit_score/n*100:.0f}% ===", flush=True)


if __name__ == "__main__":
    main()
