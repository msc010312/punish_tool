"""
dustloop_aug.py  —  Dustloop 렌더(RGBA)를 인게임처럼 증강 (도메인 랜덤화)

캐릭터(알파)만 떼서 랜덤 배경(인게임 크롭 블러)에 합성 + 블러/색변형 → 인게임 크롭 도메인에 근접.
미등장 기술도 학습 커버하려는 목적. 인게임 크롭이 주력, 이건 보충.
"""
from __future__ import annotations
import glob
import random

import cv2
import numpy as np

import punish_engine as pe

ROOT = pe.app_dir()
_BG = None


def bg_pool():
    global _BG
    if _BG is None:
        _BG = glob.glob(str(ROOT / "dataset" / "*" / "*" / "*.png"))
    return _BG


def augment(path: str) -> np.ndarray | None:
    """Dustloop RGBA -> 224x224 BGR (인게임풍 증강). 실패시 None."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.shape[2] < 4:
        return None
    a = im[:, :, 3]
    ys, xs = np.where(a > 30)
    if len(xs) < 10:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    char = im[y0:y1 + 1, x0:x1 + 1]                     # 캐릭만 크롭(RGBA)
    ch, cw = char.shape[:2]
    scale = random.uniform(0.62, 0.96) * 224 / max(ch, cw)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    char = cv2.resize(char, (nw, nh), interpolation=cv2.INTER_AREA)

    # 배경 = 랜덤 인게임 크롭 강블러 (스테이지·이펙트 질감)
    bg = cv2.imread(random.choice(bg_pool()))
    bg = cv2.resize(bg, (224, 224))
    bg = cv2.GaussianBlur(bg, (0, 0), random.uniform(4, 10))

    ox = random.randint(0, 224 - nw); oy = random.randint(0, 224 - nh)
    rgb = char[:, :, :3].astype(float)
    al = (char[:, :, 3:4].astype(float) / 255.0)
    reg = bg[oy:oy + nh, ox:ox + nw].astype(float)
    bg[oy:oy + nh, ox:ox + nw] = (rgb * al + reg * (1 - al)).astype(np.uint8)

    # 전역 증강: 약블러 + 색변형 + 밝기
    if random.random() < 0.6:
        bg = cv2.GaussianBlur(bg, (0, 0), random.uniform(0.5, 1.6))
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(float)
    hsv[:, :, 1] *= random.uniform(0.8, 1.2)
    hsv[:, :, 2] *= random.uniform(0.75, 1.2)
    bg = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bg


if __name__ == "__main__":   # 시각화: 원본 | 증강x3 | 실제 인게임
    import sys
    from pathlib import Path
    random.seed(0)
    samples = [("Sol_Badguy", "623H"), ("Ky_Kiske", "236S"), ("May", "2_8S"),
               ("Potemkin", "214H"), ("Faust", "236H")]
    rows = []
    for ch, mv in samples:
        dl = sorted(glob.glob(str(ROOT / "dustloop_images" / ch / f"{mv}*.png")))
        ing = sorted(glob.glob(str(ROOT / "dataset" / ch / mv / "*.png")))
        if not dl:
            continue
        cells = []
        orig = cv2.imread(dl[0], cv2.IMREAD_UNCHANGED)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGRA2BGR) if orig.shape[2] == 4 else orig
        cells.append(cv2.resize(orig, (150, 150)))
        for _ in range(3):
            cells.append(cv2.resize(augment(dl[0]), (150, 150)))
        cells.append(cv2.resize(cv2.imread(ing[0]), (150, 150)) if ing
                     else np.zeros((150, 150, 3), np.uint8))
        strip = np.hstack(cells)
        cv2.putText(strip, f"{ch}/{mv}: 원본|증강x3|인게임", (4, 16), 0, 0.45, (0, 255, 255), 1)
        rows.append(strip)
    cv2.imwrite(str(ROOT / "dustloop_aug_preview.png"), np.vstack(rows))
    print("미리보기 -> dustloop_aug_preview.png")
