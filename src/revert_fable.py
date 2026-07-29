"""fable 판정(교정5·제거6) 전면 원복 — 사용자 검수에서 오판 확인됨(컬러스왑 혼동)."""
import json, glob, hashlib, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import vlm_id as v
import punish_engine as pe

ROOT = pe.app_dir()

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

vt = {d["i"]: d for d in json.load(open(ROOT / "verify_train/verify_set.json", encoding="utf-8"))}
sheet_h = {md5(p): p for p in glob.glob(str(ROOT / "train_data/sheets/*.png"))}
v2sheet = {}
for f in glob.glob(str(ROOT / "verify_train/*.png")):
    i = int(os.path.basename(f)[:2])
    h = md5(f)
    if h in sheet_h:
        v2sheet[i] = sheet_h[h]
print("시트 매핑:", len(v2sheet), "/33")

tr = ROOT / "train_data/train.jsonl"
rows = [json.loads(l) for l in tr.open(encoding="utf-8") if l.strip()]
corr = {15, 16, 29, 31, 32}
drop = {5, 7, 8, 17, 20, 22}

n = 0
for i in corr:                                   # 교정 원복
    base = os.path.basename(v2sheet[i])
    for r in rows:
        if os.path.basename(r["image"]) == base:
            r["answer"] = vt[i]["label_7b"]; n += 1
print("교정 원복:", n)

add = 0
img_dir = os.path.dirname(rows[0]["image"])
for i in drop:                                   # 제거 복원
    base = os.path.basename(v2sheet[i])
    if not any(os.path.basename(r["image"]) == base for r in rows):
        ch = vt[i]["char"]
        rows.append({"image": os.path.join(img_dir, base),
                     "prompt": v._PROMPT.format(ctx="상황: 학습", atk=ch),
                     "answer": vt[i]["label_7b"], "char": ch})
        add += 1
print("제거 복원:", add)
tr.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
print("train.jsonl ->", len(rows), "행")

vl = ROOT / "train_data/val.jsonl"
vrows = [json.loads(l) for l in vl.open(encoding="utf-8") if l.strip()]
back = {"0146.png": "2P", "0188.png": "623X"}
for r in vrows:
    b = os.path.basename(r["image"])
    if b in back:
        r["answer"] = back[b]
vl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in vrows), encoding="utf-8")
print("val 원복 완료")
