"""
bench_7b.py — 배포 경로(ollama qwen2.5vl:7b)를 사람라벨 val로 벤치마크.

val.jsonl의 시트를 7b에 그대로 보내(cands enum 제한, LoRA와 동일 시트·프롬프트) 정확도 측정.
주의: 배포 파이프라인의 3창 투표는 영상 타이밍이 필요해 여기선 단발(single-shot) — 투표는
정밀도(거르기)용이라 단발 비교가 LoRA와 공정.
사용: python bench_7b.py
"""
from __future__ import annotations
import json

import cv2
import punish_engine as pe
import vlm_id as v

VAL = pe.app_dir() / "train_data" / "val.jsonl"


def main():
    rows = [json.loads(l) for l in VAL.open(encoding="utf-8") if l.strip()]
    ok = n = 0
    norm = lambda s: (s or "").lower().replace(".", "")
    for r in rows:
        if "cands" not in r:
            continue
        sheet = cv2.imread(r["image"])
        if sheet is None:
            continue
        schema = json.loads(json.dumps(v._SCHEMA))
        schema["properties"]["move"]["enum"] = r["cands"] + ["unsure"]
        try:
            res = v._ask_ollama(v._b64(sheet), r["prompt"], "qwen2.5vl:7b", schema)
            mv = v._snap_move(res.get("move", ""), [{"char": r["char"], "move": m} for m in r["cands"]])
        except Exception as e:
            mv = f"ERR:{e}"
        n += 1
        hit = norm(mv) == norm(r["answer"])
        ok += hit
        print(f"  [{n:3d}] {r['char'][:8]:8} 정답={r['answer']:8} 7b={mv:10} {'O' if hit else 'X'}", flush=True)
    print(f"\n=== 7b 단발 정확도 {ok}/{n} = {ok/max(n,1)*100:.0f}% ===", flush=True)


if __name__ == "__main__":
    main()
