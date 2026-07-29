# -*- coding: utf-8 -*-
"""harvest_banner_ds.py — 배너 분류기 학습용 데이터셋 수확.

영상 스캔 -> 노란 텍스트 구간(text_run)마다 대표 크롭 저장.
  OCR이 확신한 것        -> banner_ds/COUNTER|PUNISH|JUST/   (자동 라벨)
  OCR이 못 읽은 것('?')  -> banner_ds/_review/                (블롭 or 놓친 배너 -> 사람 검수)

이후 _review 를 사람이 4분류(COUNTER/PUNISH/JUST/none)하면 학습셋 완성.
규칙(fill/gaps 임계) 대신 데이터로 배우게 하는 게 목적.
사용: python harvest_banner_ds.py [n_videos]
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

import punish_engine as pe
import games
import hud_reader as hr

OUT = pe.app_dir() / "banner_ds"
KEEP_N = 3          # run 당 저장할 대표 크롭 수(노란픽셀 많은 순)


def _yellow(bgr):
    yr = games.ACTIVE.text_yellow
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return (((r >= yr.r[0]) & (r <= yr.r[1]) & (g >= yr.g[0]) & (g <= yr.g[1]) &
             (b >= yr.b[0]) & (b <= yr.b[1]))).astype(np.uint8)


def letter_frames(crops, k: int):
    """배너 애니메이션에서 '글자가 그려진' 프레임을 고른다.

    순서: 판때기(노란 면적 최대) -> 글자 와이프 -> 글자 완성 -> 디졸브(색 변하며 사라짐).
    '노란 픽셀 최대'로 고르면 항상 글자 없는 판때기가 잡힌다(이전 버그).
    -> 노란 피크 이후 WIPE 프레임을 건너뛰고, 디졸브 꼬리는 빼고 k장.

    피크가 여러 번인 run(배너 2개가 붙어 한 구간으로 묶임)에서는 argmax 가 마지막
    사이클을 가리키므로, 자연스럽게 '마지막 배너'의 글자를 고른다.
    """
    if len(crops) <= 2:
        return crops[:k]
    ys = [int(_yellow(c).sum()) for c in crops]
    peak = int(np.argmax(ys))
    WIPE = 2                                 # 피크 직후 2장은 글자가 절반만 그려진 와이프
    after = crops[peak + 1 + WIPE:]
    if len(after) < k:                       # 피크가 끝쪽이면(판때기 짧음) 뒤쪽 절반 사용
        after = crops[len(crops) // 2:]
    tail = max(1, len(after) - 1)            # 마지막 1장은 디졸브일 확률이 높아 제외
    return after[:tail][:k] or after[:k]


def main(n: int = 6):
    keep = set(games.ACTIVE.text_words)
    vids = sorted((pe.app_dir() / "reference_video").glob("*.mp4"),
                  key=lambda p: p.stat().st_size)[:n]
    cfg = hr.load_config()
    for d in list(keep) + ["_review"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    tot = {k: 0 for k in list(keep) + ["_review"]}
    for vi, v in enumerate(vids):
        try:
            _, _, _, _, _, runs, _ = hr.scan(str(v), cfg)
        except Exception as e:
            print(f"  ! {v.name}: {e}", flush=True); continue
        stem = v.stem[:28].replace(" ", "_")
        for r in runs:
            crops = letter_frames(r["crops"], KEEP_N)
            w = hr.ocr_best_word(crops)      # 판때기 말고 '글자 프레임'만 OCR
            lab = w if w in keep else "_review"
            for j, c in enumerate(crops):
                if c.size == 0:
                    continue
                p = OUT / lab / f"{stem}_{r['start_t']:.1f}_{r['side']}_{j}.png"
                cv2.imwrite(str(p), c)
            tot[lab] += 1
        print(f"  [{vi+1}/{len(vids)}] {v.name[:40]}  누적 {tot}", flush=True)
    print(f"\n저장 -> {OUT}\n  자동라벨 {sum(tot[k] for k in keep)}run / 검수대기 {tot['_review']}run")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
