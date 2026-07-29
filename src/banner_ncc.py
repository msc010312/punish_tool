# -*- coding: utf-8 -*-
"""banner_ncc.py — 배너 4분류를 회색조 정규화상관(NCC) 템플릿 매칭으로.

배너는 폰트·크기·위치가 고정이라 색마스크 없이 원본 회색조로 매칭된다.
클래스별 평균 템플릿(여러 장 정렬 평균)을 만들고, 질의 크롭과의 NCC 최대값으로 분류.
torch·티서랙트 불필요, 템플릿은 npz 수십 KB.

  build_templates() : banner_ds/{CLASS}/*.png 로 클래스 평균 템플릿 저장
  classify(bgr)     : (label, score, margin)  — margin=1등-2등 NCC 갭(신뢰도)
  python banner_ncc.py           # 템플릿 빌드 + LOVO 자기평가
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import cv2
import numpy as np

import punish_engine as pe

TPL = pe.app_dir() / "banner_templates_ncc.npz"
CLASSES = ["COUNTER", "PUNISH", "JUST", "REVERSAL"]
TH, TW = 48, 160          # 템플릿 정규화 크기(회색조)


def _prep(bgr) -> np.ndarray | None:
    """회색조 -> 고정크기 -> 대비정규화(밝기/명암 불변). 매칭용 벡터."""
    if bgr is None or bgr.size == 0:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (TW, TH))
    g -= g.mean()
    s = g.std()
    return g / s if s > 1e-6 else None


def _video_of(fname: str) -> str:
    """LOVO용: 파일명 앞부분(=원본 영상 stem)."""
    m = re.match(r"(.+?)_[0-9.]+_P[12]_\d+\.png$", Path(fname).name)
    return m.group(1) if m else Path(fname).name


def _load(cls: str, per: int = 400):
    fs = sorted(glob.glob(str(pe.app_dir() / "banner_ds" / cls / "*.png")))[:per]
    out = []
    for f in fs:
        v = _prep(cv2.imread(f))
        if v is not None:
            out.append((f, _video_of(f), v))
    return out


def _load_labeled(cls: str):
    """사람이 검수한 banner_labels.json 기준으로 로드(자동라벨보다 정확)."""
    import json
    lp = pe.app_dir() / "banner_labels.json"
    if not lp.exists():
        return None
    lab = json.loads(lp.read_text(encoding="utf-8"))
    out = []
    for f, v in lab.items():
        if v != cls:
            continue
        im = cv2.imread(f)
        q = _prep(im) if im is not None else None
        if q is not None:
            out.append((f, _video_of(f), q))
    return out


def build_templates(per: int = 400):
    """수동 라벨(banner_labels.json) 우선, 없으면 폴더 자동라벨."""
    tpl = {}
    for c in CLASSES:
        labeled = _load_labeled(c)
        src = "수동라벨" if labeled else "자동라벨"
        vecs = [v for _, _, v in (labeled if labeled else _load(c, per))]
        if not vecs:
            print(f"  ! {c}: 샘플 없음"); continue
        m = np.mean(vecs, 0)
        m -= m.mean(); m /= (m.std() + 1e-6)
        tpl[c] = m.astype(np.float32)
        print(f"  {c:9s} 템플릿 <- {len(vecs)}장 ({src})")
    np.savez_compressed(TPL, **tpl)
    print(f"저장 -> {TPL}")
    return tpl


def _tpl():
    d = np.load(TPL)
    return {c: d[c] for c in d.files}


def classify(bgr, templates=None):
    q = _prep(bgr)
    if q is None:
        return "?", 0.0, 0.0
    T = templates or _tpl()
    n = q.size
    scores = sorted(((float((q * t).sum() / n), c) for c, t in T.items()), reverse=True)
    (s1, c1), (s2, _) = scores[0], scores[1]
    return c1, s1, s1 - s2


def evaluate(per: int = 400):
    """Leave-one-VIDEO-out: 질의 영상의 크롭을 뺀 나머지로 템플릿을 만들어 채점.
    같은 영상 프레임 유출 방지 -> 실배포에 가까운 정확도."""
    data = {c: _load(c, per) for c in CLASSES}
    vids = sorted({v for c in CLASSES for _, v, _ in data[c]})
    conf = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    tot = 0
    for held in vids:
        T = {}
        for c in CLASSES:
            vecs = [v for _, vid, v in data[c] if vid != held]
            if not vecs:
                continue
            m = np.mean(vecs, 0); m -= m.mean(); m /= (m.std() + 1e-6)
            T[c] = m.astype(np.float32)
        for c in CLASSES:
            for f, vid, v in data[c]:
                if vid != held:
                    continue
                n = v.size
                pred = max(T, key=lambda k: (v * T[k]).sum() / n)
                conf[c][pred] += 1; tot += 1
    correct = sum(conf[c][c] for c in CLASSES)
    print(f"\nLOVO 정확도: {correct}/{tot} = {correct/tot*100:.1f}%  (영상 {len(vids)}개)")
    print("혼동행렬 (행=정답, 열=예측):")
    print("           " + "".join(f"{c[:4]:>7}" for c in CLASSES))
    for c in CLASSES:
        row = "".join(f"{conf[c][c2]:>7}" for c2 in CLASSES)
        rc = sum(conf[c].values())
        acc = conf[c][c] / rc * 100 if rc else 0
        print(f"  {c:9s}{row}   ({acc:.0f}%)")


if __name__ == "__main__":
    print("템플릿 빌드:")
    build_templates()
    evaluate()
