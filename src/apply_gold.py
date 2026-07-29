"""apply_gold.py — 사용자 골드 판정(verify_train/gold.json)을 auto train.jsonl에 반영.
정답이 시트 후보에 있으면 교정, 없거나 bad면 행 제거(시트에 정답 후보가 없어 학습 불가)."""
import glob
import hashlib
import json
import os

import punish_engine as pe

ROOT = pe.app_dir()


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


gold = {int(k): v for k, v in json.load(open(ROOT / "verify_train/gold.json")).items()}
vt = {d["i"]: d for d in json.load(open(ROOT / "verify_train/verify_set.json", encoding="utf-8"))}
sheet_h = {md5(p): os.path.basename(p) for p in glob.glob(str(ROOT / "train_data/sheets/*.png"))}
v2base = {}
for f in glob.glob(str(ROOT / "verify_train/*.png")):
    i = int(os.path.basename(f)[:2])
    h = md5(f)
    if h in sheet_h:
        v2base[i] = sheet_h[h]

rows = [json.loads(l) for l in (ROOT / "train_data/train.jsonl").open(encoding="utf-8") if l.strip()]
by_base = {os.path.basename(r["image"]): r for r in rows}

# 시트에 구워진 후보 무브 = verify_set엔 없음 -> 골드가 7b라벨 또는 LoRA예측과 일치하면 후보에 있음이 보장.
# 그 외(둘 다 아님)는 후보에 없을 가능성 높아 제거(보수적).
fix = drop = 0
drop_bases = set()
for i, g in gold.items():
    base = v2base.get(i)
    if not base or base not in by_base:
        continue
    d = vt[i]
    norm = lambda s: s.lower().replace(".", "")
    if g == "bad":
        drop_bases.add(base); drop += 1
    elif norm(g) in (norm(d["label_7b"]), norm(d["lora"])):
        by_base[base]["answer"] = g if g != norm(d["label_7b"]) else d["label_7b"]
        # 표기는 시트 후보 표기 그대로 사용
        by_base[base]["answer"] = d["label_7b"] if norm(g) == norm(d["label_7b"]) else d["lora"]
        fix += 1
    else:
        drop_bases.add(base); drop += 1              # 정답이 후보에 없음 -> 제거

rows = [r for r in rows if os.path.basename(r["image"]) not in drop_bases]
(ROOT / "train_data/train.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
print(f"골드 반영: 교정/확정 {fix}, 제거 {drop} -> auto train {len(rows)}행")
