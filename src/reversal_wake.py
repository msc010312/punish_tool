# -*- coding: utf-8 -*-
"""reversal_wake.py — 흰색(기상) REVERSAL 감지.

기상 리버설은 화면 좌/우 중앙에 '흰 글자'로 크게 뜬다(작은 사이드 배너와 위치·색이 다름).
reversal_wake_tpl.npz(143개 확정 표본 평균)과의 회색조 NCC 로 판정. torch 불필요.

문턱 0.50: 검출 88% / none오검 0% (검증셋 기준) -> 오정보 0 우선.
"""
from __future__ import annotations
import functools

import cv2
import numpy as np

import punish_engine as pe

TPL_F = pe.app_dir() / "reversal_wake_tpl.npz"
NCC_MIN = 0.50          # 배포 문턱(오검 0%)
WHITE_COV = 0.15        # 흰 글자 커버리지 1차 트리거


@functools.lru_cache(maxsize=1)
def _tpl():
    if not TPL_F.exists():
        return None
    d = np.load(TPL_F)
    return d["tpl"].astype(np.float32), int(d["TH"]), int(d["TW"])


def available() -> bool:
    return _tpl() is not None


def white_cover(bgr) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 45)).mean())


def score(bgr) -> float:
    t = _tpl()
    if t is None or bgr is None or bgr.size == 0:
        return -1.0
    T, TH, TW = t
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (TW, TH)); g -= g.mean(); s = g.std()
    if s < 8.0:                    # 빈/평평 프레임(글자 없음)
        return -1.0
    g /= s
    return float((g * T).sum() / g.size)


def is_wake(crops) -> bool:
    """런의 대표 프레임들 중 하나라도 문턱을 넘으면 흰색 리버설."""
    return bool(crops) and max((score(c) for c in crops), default=-1.0) >= NCC_MIN
