"""
make_verify.py  —  사람검증용 소량 세트 생성 (7b-라벨 vs LoRA 불일치 케이스)

val 시트들에 LoRA를 돌려, 자동라벨(7b 투표)과 LoRA 예측이 '다른' 것만 골라낸다(=둘 중 하나가
틀림 → 검증가치 최고). 시트를 verify/ 로 복사하고 답안 양식(verify_set.json)을 만든다.
사용자는 각 시트를 보고 '진짜 무브'를 적어주면, 그걸로 라벨 교정 + 재학습.

사용: python make_verify.py [--cap 50]
출력: verify/<i>_<char>_7b=<라벨>_lora=<예측>.png  +  verify/verify_set.json
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path

import torch
import punish_engine as pe

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA = pe.app_dir() / "qwen_lora"
VAL = pe.app_dir() / "train_data" / "val.jsonl"
OUT = pe.app_dir() / "verify"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--split", default="val", choices=["val", "train"])  # train=라벨노이즈 검출(암기거부 문항)
    a = ap.parse_args()
    global VAL, OUT
    if a.split == "train":
        VAL = pe.app_dir() / "train_data" / "train.jsonl"
        OUT = pe.app_dir() / "verify_train"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
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

    def predict(r):
        m = [{"role": "user", "content": [{"type": "image", "image": r["image"]},
                                          {"type": "text", "text": r["prompt"]}]}]
        text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(m)
        enc = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=48, do_sample=False)
        gen = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        try:
            return json.loads(gen).get("move", "")
        except Exception:
            mm = re.search(r'"move"\s*:\s*"([^"]+)"', gen); return mm.group(1) if mm else gen.strip()[:10]

    disagree = []
    for r in rows:
        lora = predict(r)
        if lora.lower().replace(".", "") != r["answer"].lower().replace(".", ""):
            disagree.append({"image": r["image"], "char": r["char"],
                             "label_7b": r["answer"], "lora": lora})
    disagree = disagree[: a.cap]
    print(f"불일치 {len(disagree)}건 (검증 대상)", flush=True)

    vset = []
    for i, d in enumerate(disagree):
        name = f"{i:02d}_{d['char'].split()[0]}_7b={d['label_7b']}_lora={d['lora']}.png"
        name = re.sub(r'[\\/:*?"<>|]', "_", name)
        shutil.copy(d["image"], OUT / name)
        vset.append({"i": i, "sheet": name, "char": d["char"],
                     "label_7b": d["label_7b"], "lora": d["lora"], "correct": ""})
    (OUT / "verify_set.json").write_text(json.dumps(vset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT} ({len(vset)}개 시트 + verify_set.json). 각 시트 보고 correct에 진짜 무브 적어줘.", flush=True)


if __name__ == "__main__":
    main()
