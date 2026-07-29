# -*- coding: utf-8 -*-
"""build_banner_templates.py — OCR로 배너 exemplar 수확 -> banner_templates.npz.

여러 영상 스캔 -> OCR이 분류한 COUNTER/PUNISH/JUST 배너의 마스크특징(aspect,vec) 수집.
런타임(banner_match)은 이 템플릿으로 OCR 미스를 폴백 분류. Tesseract가 부트스트랩 후 빠져도 됨.
사용: python build_banner_templates.py [n_videos]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import punish_engine as pe
import games
import hud_reader as hr
import banner_match as bm

OUT = pe.app_dir() / "banner_templates.npz"


def main(n: int = 6):
    keep = set(games.ACTIVE.text_words)
    vids = sorted((pe.app_dir() / "reference_video").glob("*.mp4"), key=lambda p: p.stat().st_size)[:n]
    cfg = hr.load_config()
    words, aspects, vecs = [], [], []
    for vi, v in enumerate(vids):
        try:
            _, _, _, _, _, runs, _ = hr.scan(str(v), cfg)
        except Exception as e:
            print(f"  ! {v.name}: {e}"); continue
        got = 0
        for r in runs:
            w = hr.ocr_best_word(r["crops"])
            if w not in keep:
                continue
            f = bm._best_frame_feat(r["crops"])
            if f is None:
                continue
            words.append(w); aspects.append(f[0]); vecs.append(f[1]); got += 1
        print(f"  [{vi+1}/{len(vids)}] {v.name[:45]}: +{got}  누적 {len(words)}", flush=True)
        if words:
            np.savez(OUT, words=np.array(words), aspects=np.array(aspects, np.float32),
                     vecs=np.array(vecs, np.float32))
    from collections import Counter
    print(f"\n저장 -> {OUT}  총 {len(words)}  분포 {dict(Counter(words))}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
