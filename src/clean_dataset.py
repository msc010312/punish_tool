"""
clean_dataset.py  —  기존 자동라벨(dataset_auto)에서 버스트·제외캐릭 제거

manifest의 각 라벨에 대해 저장된 크롭프레임으로 is_burst 검사 + EXCLUDE_CHARS(Bedman) 필터.
버스트/제외캐릭이면 manifest에서 빼고 클립 파일도 삭제. 청소된 manifest로 다시 쓴다.
사용: python clean_dataset.py
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import punish_engine as pe
import analyze_match as am
import vlm_id as v

MAN = pe.app_dir() / "dataset_auto" / "manifest.jsonl"


def main():
    rows = [json.loads(l) for l in MAN.open(encoding="utf-8") if l.strip()]
    keep, drop_burst, drop_char = [], 0, 0
    for r in rows:
        if r["char"] in am.EXCLUDE_CHARS:
            drop_char += 1; _rm(r); continue
        frames = [cv2.imread(p) for p in r["frames"] if Path(p).exists()]
        frames = [f for f in frames if f is not None]
        if frames and v.is_burst(frames):
            drop_burst += 1; _rm(r); continue
        keep.append(r)
    MAN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n", encoding="utf-8")
    print(f"청소: 버스트 -{drop_burst}, 제외캐릭 -{drop_char} | 남은 라벨 {len(keep)} (이전 {len(rows)})", flush=True)


def _rm(r):
    for p in r.get("frames", []):
        Path(p).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
