"""
gen_aug_cache.py  —  Dustloop 렌더를 미리 증강해 디스크에 저장 (학습 속도용)

각 렌더 K장 증강 -> dustloop_aug_cache/<char>/<base_move>/<stem>_aN.png
base_move = 멀티히트 접미사(_2 등) 제거해 in-game 클래스명과 정렬.
"""
from __future__ import annotations
import glob, re
from pathlib import Path

import cv2

import punish_engine as pe
import dustloop_aug as da

ROOT = pe.app_dir()
OUT = ROOT / "dustloop_aug_cache"
K = 6


def main():
    paths = glob.glob(str(ROOT / "dustloop_images" / "*" / "*.png"))
    n = 0
    for i, p in enumerate(paths):
        pp = Path(p)
        base = re.sub(r"_\d+$", "", pp.stem)        # 623H_2 -> 623H
        dst = OUT / pp.parent.name / base
        dst.mkdir(parents=True, exist_ok=True)
        for k in range(K):
            im = da.augment(p)
            if im is None:
                continue
            cv2.imwrite(str(dst / f"{pp.stem}_a{k}.png"), im)
            n += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)} 렌더 처리 ({n}장)", flush=True)
    print(f"완료: 증강 {n}장 -> {OUT}")


if __name__ == "__main__":
    main()
