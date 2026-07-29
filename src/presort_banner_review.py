# -*- coding: utf-8 -*-
"""presort_banner_review.py — banner_ds/_review 를 NCC 최고점수로 정렬(라벨링 순서용).

색 마스크도 torch 도 쓰지 않는다. 배너 4-클래스 NCC 템플릿(banner_ncc)과의 최고 유사도로
정렬 -> 진짜 배너(가려/흐려서 OCR이 놓친 것 포함)가 앞, 이펙트/배경 블롭이 뒤로 온다.
라벨러는 앞부터 찍고, 점수가 뚝 떨어지는 구간부터 [A]로 일괄 none.

자동 라벨은 하지 않는다 — _review 는 '진짜 배너 + 블롭'이 섞여 있어(NCC 점수 중첩)
자동 처리하면 둘 중 하나를 오염시킨다. 사람이 판정한다.

산출: banner_ds/_review_order.json  [{path, pred, sim}, ...]  (sim 내림차순)
사용: python presort_banner_review.py   (먼저 banner_ncc.py 로 템플릿 빌드)
"""
from __future__ import annotations
import glob
import json

import cv2

import punish_engine as pe
import banner_ncc as B


DS = pe.app_dir() / "banner_ds"
OUT = DS / "_review_order.json"


def main():
    if not B.TPL.exists():
        print("템플릿 없음 — 먼저: python banner_ncc.py"); return
    T = B._tpl()
    files = sorted(glob.glob(str(DS / "_review" / "*_0.png")))   # run 대표만
    print(f"검수 대상(run 대표) {len(files)}장")
    rows = []
    for i, f in enumerate(files):
        lab, s, _ = B.classify(cv2.imread(f), T)
        rows.append({"path": f, "pred": lab, "sim": round(s, 4)})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    rows.sort(key=lambda r: -r["sim"])
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 -> {OUT}  {len(rows)}건 (sim {rows[0]['sim']} ~ {rows[-1]['sim']})")
    print("자동 라벨 없음 — 라벨러에서 앞부터 찍고, 점수 급락 구간부터 [A]로 일괄 none")


if __name__ == "__main__":
    main()
