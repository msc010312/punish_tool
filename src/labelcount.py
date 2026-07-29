# -*- coding: utf-8 -*-
"""labelcount.py — 라벨링 진행 확인. 캐릭별 무브당 크롭 수를 보여준다.
사용:  python labelcount.py            # 전체 캐릭 요약
       python labelcount.py Sol_Badguy # 그 캐릭 무브별 상세
"""
import os, glob, sys

BASE = os.path.join(os.path.dirname(__file__), "..", "dataset")


def counts(char):
    cd = os.path.join(BASE, char)
    out = {}
    for m in os.listdir(cd):
        md = os.path.join(cd, m)
        if os.path.isdir(md):
            out[m] = len(glob.glob(md + "/*.png")) + len(glob.glob(md + "/*.jpg"))
    return out


def main():
    if len(sys.argv) > 1:
        char = sys.argv[1]
        c = counts(char)
        tot = sum(c.values())
        print(f"{char}: 총 {tot}장 / 무브 {len(c)}개  (20장 목표 기준)")
        for m, n in sorted(c.items(), key=lambda x: -x[1]):
            bar = "#" * min(n, 20) + ("+" if n > 20 else "")
            mark = " OK" if n >= 20 else ""
            print(f"  {m:14s} {n:3d}  {bar}{mark}")
    else:
        chars = [d for d in sorted(os.listdir(BASE))
                 if os.path.isdir(os.path.join(BASE, d)) and not d.startswith("__")]
        for ch in chars:
            c = counts(ch)
            ready = sum(1 for n in c.values() if n >= 20)
            print(f"  {ch:24s} {sum(c.values()):4d}장 / {len(c)}무브 / 20장달성 {ready}개")


if __name__ == "__main__":
    main()
