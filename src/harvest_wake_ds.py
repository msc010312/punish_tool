# -*- coding: utf-8 -*-
"""harvest_wake_ds.py — 흰색(기상) REVERSAL 배너 수확.

기상 리버설은 작은 사이드 배너가 아니라 '화면 좌/우 중앙에 크게' 흰 글자로 뜬다
(주황 가드경직 리버설과 위치·크기·색이 다름). hud_config.json 의
reversal_wake_left / reversal_wake_right 영역에서 '흰 픽셀 커버리지'가 튀는 구간을
런으로 잡아 대표 프레임을 저장한다.

배경 흰색(구름/폭포/흰 의상)이 오검을 낼 수 있어 자동 라벨은 하지 않는다.
전부 _review 로 보내 사람이 판정한다.

사용: python harvest_wake_ds.py [n_videos]
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

import punish_engine as pe
import hud_reader as hr

OUT = pe.app_dir() / "banner_ds_wake"
TPL_F = pe.app_dir() / "reversal_wake_tpl.npz"
STRIDE = 4
WHITE_COV = 0.15        # 흰 글자 커버리지 문턱(글자 0.28 vs 배경 0.03~0.08) — 1차 트리거
NCC_MIN = 0.30          # REVERSAL 글자모양 유사도 — 2차 필터(배경 흰색 걸러냄)
KEEP_N = 3

_TD = np.load(TPL_F)
_TPL, _TH, _TW = _TD["tpl"], int(_TD["TH"]), int(_TD["TW"])


def _ncc(bgr):
    if bgr is None or bgr.size == 0:
        return -1.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (_TW, _TH)); g -= g.mean(); s = g.std()
    if s <= 1e-6:
        return -1.0
    g /= s
    return float((g * _TPL).sum() / g.size)


def _regions():
    cfg = hr.load_config() or {}
    r = cfg.get("regions", {})
    def f(b):
        return (b["x0"], b["y0"], b["x1"], b["y1"])
    if "reversal_wake_left" not in r:
        sys.exit("hud_config.json 에 reversal_wake_left/right 없음")
    return f(r["reversal_wake_left"]), f(r["reversal_wake_right"])


def _crop(frame, reg):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = reg
    return frame[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]


def _white_cov(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 45)
    return float(white.mean())


def letter_frames(crops, k):
    """흰 글자는 판때기가 없어 '커버리지 피크 = 글자 완성'. 피크 주변 k장."""
    if len(crops) <= k:
        return crops
    covs = [_white_cov(c) for c in crops]
    peak = int(np.argmax(covs))
    lo = max(0, peak - k // 2)
    return crops[lo:lo + k] or crops[:k]


def main(n: int = 44):
    lreg, rreg = _regions()
    vids = sorted((pe.app_dir() / "reference_video").glob("*.mp4"),
                  key=lambda p: p.stat().st_size)[:n]
    (OUT / "_review").mkdir(parents=True, exist_ok=True)
    tot = 0
    for vi, v in enumerate(vids):
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"  ! 못 엶 {v.name}"); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        gr = hr.find_game_region(str(v))
        stem = v.stem[:28].replace(" ", "_")
        runs = {"L": dict(on=False), "R": dict(on=False)}
        fi = 0
        found = 0
        while True:
            if not cap.grab():
                break
            if fi % STRIDE == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                g = hr.crop_game(fr, gr)
                for side, reg in (("L", lreg), ("R", rreg)):
                    c = _crop(g, reg)
                    cov = _white_cov(c)
                    st = runs[side]
                    if cov >= WHITE_COV:
                        if not st["on"]:
                            st.update(on=True, t=fi/fps, crops=[c])
                        else:
                            st["crops"].append(c)
                    elif st["on"]:
                        st["on"] = False
                        sel = letter_frames(st["crops"], KEEP_N)
                        # 2차 필터: 대표 프레임이 REVERSAL 글자모양이어야 저장(배경 흰색 제거)
                        if not sel or max(_ncc(c) for c in sel) < NCC_MIN:
                            continue
                        for j, cc in enumerate(sel):
                            if cc.size:
                                p = OUT / "_review" / f"{stem}_{st['t']:.1f}_{side}_{j}.png"
                                cv2.imwrite(str(p), cc)
                        found += 1; tot += 1
            fi += 1
        cap.release()
        print(f"  [{vi+1}/{len(vids)}] {v.name[:40]}  런 {found}  (누적 {tot})", flush=True)
    print(f"\n저장 -> {OUT}/_review  {tot}런  (사람 검수 필요)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 44)
