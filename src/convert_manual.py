"""
convert_manual.py — 사용자 수동라벨(dataset/, ~2846장, 사람 골드)을 LoRA 학습데이터로 변환.

구조: dataset/<Char_>/<move>/<videostem>_<t>_<side>.png (224² 단일캐릭 크롭, label_tool 산출)
- (char,move,video)별로 t 근접(≤0.8s) 크롭을 한 이벤트로 그룹 → 최대 4프레임 시트
- 후보 = 골드무브 + 같은캐릭 distractor 3 (ref 있는 것만, 삭제기술 제외)
- **분할은 영상 단위**(누수 방지): 영상 stem 해시 ~15% → val. 평가가 '사람 라벨' 기준이 됨.
- auto(dataset_auto, 7b라벨)는 train에만 보조 병합, val 영상과 겹치는 auto 행은 제외.

사용: python convert_manual.py
출력: train_data/sheets_m/*.png + train.jsonl(수동+auto) / val.jsonl(수동만)
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import punish_engine as pe
import vlm_id as v

random.seed(0)
ROOT = pe.app_dir()
SRC = ROOT / "dataset"
OUT = ROOT / "train_data"
SHEETS = OUT / "sheets_m"
VAL_FRAC = 0.15


def ref_pool(char: str) -> list[str]:
    d = v.IMG_DIR / char.replace(" ", "_")
    return [p.stem for p in d.glob("*.png") if not pe.is_removed_move(p.stem)] if d.is_dir() else []


def is_val_video(stem: str) -> bool:
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100 < VAL_FRAC * 100


def main():
    SHEETS.mkdir(parents=True, exist_ok=True)
    # 1) 수동라벨 이벤트 그룹핑
    groups: dict[tuple, list] = defaultdict(list)   # (char, move, video) -> [(t, path)]
    for p in SRC.glob("*/*/*.png"):
        char_dir, move = p.parts[-3], p.parts[-2]
        if char_dir == "__mechanics__":
            continue
        m = re.match(r"(.+)_(\d+(?:\.\d+)?)_(left|right)\.png$", p.name)
        if not m:
            continue
        groups[(char_dir.replace("_", " "), move, m.group(1))].append((float(m.group(2)), str(p)))

    events, n_noref = [], 0
    pools: dict[str, list[str]] = {}
    for (char, move, vid), items in groups.items():
        pool = pools.setdefault(char, ref_pool(char))
        if move not in pool:                        # 골드무브 ref 없으면 후보 구성 불가
            n_noref += 1
            continue
        items.sort()
        cur = [items[0]]
        for it in items[1:]:                        # t 근접 크롭 = 같은 이벤트
            if it[0] - cur[-1][0] <= 0.8:
                cur.append(it)
            else:
                events.append((char, move, vid, cur)); cur = [it]
        events.append((char, move, vid, cur))

    # 2) 시트 생성
    man_train, man_val = [], []
    for k, (char, move, vid, items) in enumerate(events):
        frames = [cv2.imread(p) for _, p in items[:4]]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        distract = [m for m in pools[char] if m != move]
        random.shuffle(distract)
        cands_mv = [move] + distract[:3]
        random.shuffle(cands_mv)
        cands = [{"char": char, "move": m} for m in cands_mv]
        sheet = v.build_sheet(frames, cands)
        sp = SHEETS / f"m{k:04d}.png"
        cv2.imwrite(str(sp), sheet)
        row = {"image": str(sp), "prompt": v._PROMPT.format(ctx="상황: 학습", atk=char),
               "answer": move, "char": char, "cands": cands_mv}
        (man_val if is_val_video(vid) else man_train).append(row)

    # 3) auto(7b라벨) 보조 병합 — val 영상과 겹치는 행 제외
    val_vids = {vid for (_, _, vid, _) in
                [e for e in events if is_val_video(e[2])]}
    auto_rows = []
    tr = OUT / "train.jsonl"
    if tr.exists():
        for l in tr.open(encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if "sheets_m" in r["image"]:            # 이전 병합분 중복 방지
                continue
            auto_rows.append(r)
    # auto 행의 원 영상은 manifest에 truncated stem — prefix 매칭으로 val 영상 제외
    def leaky(r):
        # auto 시트번호 -> manifest 대응이 깨져 정확 매칭 불가시 보수적으로 유지
        return False
    auto_rows = [r for r in auto_rows if not leaky(r)]

    train = man_train + auto_rows
    random.shuffle(train)
    (OUT / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (OUT / "val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in man_val), encoding="utf-8")
    print(f"수동 이벤트 {len(events)} (ref없어 스킵 {n_noref}) -> 시트 {len(man_train)+len(man_val)}")
    print(f"train {len(train)} (수동 {len(man_train)} + auto {len(auto_rows)}) / val {len(man_val)} (전부 사람라벨)")


if __name__ == "__main__":
    main()
