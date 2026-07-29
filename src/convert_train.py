"""
convert_train.py  —  dataset_auto/manifest.jsonl -> Qwen2.5-VL LoRA 학습용 데이터

각 자동라벨(프레임 4장 + char + move)을 **추론과 동일 포맷**(두인물 시트 + 후보 + 프롬프트)으로
변환한다. 후보셋은 manifest에 없으므로 [정답무브 + 같은캐릭 다른무브 distractor] 로 합성(셔플).
정답 = move. 학습/검증 분할(85/15, 같은 (char,move) 누수 방지 위해 단순 셔플 분할).

출력: train_data/sheets/*.png  +  train_data/{train,val}.jsonl
      각 줄: {"image": path, "prompt": str, "answer": move, "char": char}
사용: python convert_train.py
"""
from __future__ import annotations
import json, random
from pathlib import Path

import cv2
import punish_engine as pe
import vlm_id as v

random.seed(0)
SRC = pe.app_dir() / "dataset_auto" / "manifest.jsonl"
OUT = pe.app_dir() / "train_data"
SHEETS = OUT / "sheets"


def char_moves_with_ref(char: str) -> list[str]:
    """그 캐릭의 ref이미지 있는 무브 목록(distractor 풀). 삭제된 기술(236D 등) 제외."""
    d = v.IMG_DIR / char.replace(" ", "_")
    return [p.stem for p in d.glob("*.png") if not pe.is_removed_move(p.stem)] if d.is_dir() else []


def main():
    if not SRC.exists():
        raise SystemExit("dataset_auto/manifest.jsonl 없음 — collect_auto 먼저")
    SHEETS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    pools = {}
    out = []
    for i, r in enumerate(rows):
        ch, mv = r["char"], r["move"]
        frames = [cv2.imread(p) for p in r["frames"] if Path(p).exists()]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        pool = pools.setdefault(ch, char_moves_with_ref(ch))
        distract = [m for m in pool if m != mv]
        random.shuffle(distract)
        cands_mv = [mv] + distract[:3]
        random.shuffle(cands_mv)
        if mv not in cands_mv or len(cands_mv) < 2:
            continue
        cands = [{"char": ch, "move": m} for m in cands_mv]
        sheet = v.build_sheet(frames, cands)
        sp = SHEETS / f"{i:04d}.png"
        cv2.imwrite(str(sp), sheet)
        prompt = v._PROMPT.format(ctx="상황: 학습", atk=ch)
        out.append({"image": str(sp), "prompt": prompt, "answer": mv, "char": ch})

    random.shuffle(out)
    nv = max(1, int(len(out) * 0.15))
    val, train = out[:nv], out[nv:]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in train), encoding="utf-8")
    (OUT / "val.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in val), encoding="utf-8")
    print(f"변환 완료: train {len(train)} / val {len(val)} -> {OUT}")


if __name__ == "__main__":
    main()
