# -*- coding: utf-8 -*-
"""banner_match.py — 배너(COUNTER/PUNISH/JUST) 마스크 템플릿매칭.

Tesseract가 스타일라이즈드 폰트 배너를 절반가량 놓치는 문제 보강. 노란색 마스크로 글자만
분리 -> 종횡비(aspect) 하드필터 + 마스크 최근접으로 3단어 분류. 이펙트(노란 블롭)는
텍스트 게이트(글자 사이 gap)로 배제. 템플릿(banner_templates.npz)은 OCR로 부트스트랩 수확.

용도: hud_reader.ocr_text_events 에서 OCR="?"(미스)일 때 폴백. 검증: LOVO 90%, 회수 정밀~90%.
"""
from __future__ import annotations
import functools
from pathlib import Path

import cv2
import numpy as np

import punish_engine as pe
import games

VW, VH = 160, 40
ASPECT_TOL = (0.72, 1.39)     # 종횡비 ±39% (JUST≠PUNISH≠COUNTER 분리)
THR = 0.82                    # 폴백 채택 임계(정밀 우선. 팬텀 카운터 방지)
# 노란 '이펙트 블롭'이 배너로 오검출되던 문제:
#   실측 - 진짜 글자 fill 0.19/gaps 154, 블롭 fill 0.42~0.50/gaps 69~123
FILL_MAX = 0.35               # 글자는 성기다. 이보다 꽉 차면 블롭
GAPS_MIN = 8                  # 글자 사이 빈 열이 충분해야 텍스트


def _yr():
    return games.ACTIVE.text_yellow


def yellow_mask(bgr: np.ndarray) -> np.ndarray:
    yr = _yr()
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return (((r >= yr.r[0]) & (r <= yr.r[1]) & (g >= yr.g[0]) & (g <= yr.g[1]) &
             (b >= yr.b[0]) & (b <= yr.b[1]))).astype(np.uint8)


def feat(bgr: np.ndarray):
    """노란마스크 -> 타이트bbox. 반환 (aspect, 정규화벡터, is_text). 마스크 부족시 None."""
    m = yellow_mask(bgr)
    ys, xs = np.where(m)
    if len(xs) < 40:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if w < 24 or h < 8:
        return None
    sub = m[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    aspect = w / h
    fill = float(sub.mean())
    gaps = int(np.sum(sub.mean(axis=0) < 0.12))          # 글자 사이 빈 열 = 텍스트 근거
    is_text = (0.08 <= fill <= FILL_MAX) and (aspect >= 1.4) and (gaps >= GAPS_MIN)
    v = cv2.resize(sub, (VW, VH), interpolation=cv2.INTER_AREA).flatten()
    n = np.linalg.norm(v)
    return (aspect, (v / n if n > 0 else v), is_text)


@functools.lru_cache(maxsize=1)
def _templates():
    p = pe.app_dir() / "banner_templates.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["words"], d["aspects"].astype(np.float32), d["vecs"].astype(np.float32)


def available() -> bool:
    return _templates() is not None


def _best_frame_feat(crops: list[np.ndarray]):
    """여러 프레임 중 노란픽셀 가장 많은(선명한) 것의 feat."""
    best, cov = None, -1
    for c in crops:
        v = int(yellow_mask(c).sum())
        if v > cov:
            f = feat(c)
            if f is not None:
                best, cov = f, v
    return best


def is_text_like(crops: list[np.ndarray]) -> bool:
    """이 배너 구간이 '글자'인가(노란 이펙트 블롭이 아닌가). OCR 결과 검증용 시각 게이트."""
    f = _best_frame_feat(crops)
    return bool(f and f[2])


def classify(crops: list[np.ndarray]) -> str:
    """배너 크롭들 -> COUNTER/PUNISH/JUST (템플릿 최근접). 텍스트 아니거나 미달이면 '?'."""
    tpl = _templates()
    if tpl is None or not crops:
        return "?"
    f = _best_frame_feat(crops)
    if f is None or not f[2]:                 # 텍스트 게이트(이펙트 배제)
        return "?"
    asp, v, _ = f
    words, aspects, vecs = tpl
    ok = (aspects / asp <= 1 / ASPECT_TOL[0]) & (aspects / asp >= 1 / ASPECT_TOL[1])
    ok &= (asp / aspects >= ASPECT_TOL[0]) & (asp / aspects <= ASPECT_TOL[1])
    if not ok.any():
        return "?"
    sims = vecs[ok] @ v
    j = int(sims.argmax())
    if float(sims[j]) < THR:
        return "?"
    return str(words[ok][j])
