"""
hud_calibrate.py  —  HUD 영역을 '마우스로 직접' 지정하는 보정 툴

색 자동탐지가 길티기어 이펙트(불/플래시)에 약하므로, 사람이 풀해상도 프레임을 보고
HP바 위치를 직접 박스로 찍는다. 한 번 찍어두면 그 '고정 위치'로 안정적으로 읽는다.

저장 좌표는 해상도 비율(0~1)이라, 같은 HUD라면 다른 해상도 영상에도 대체로 재사용된다.

사용:
  python hud_calibrate.py "<영상.mp4>"            # 기본: 영상 중간 프레임으로 보정
  python hud_calibrate.py "<영상.mp4>" --t 90      # 90초 프레임으로 보정(깨끗한 장면 고르기)

조작:
  창이 뜨면 순서대로 박스를 드래그한다:
    1) P1(왼쪽) HP바  -> 바의 '꽉 찬 상태' 전체 길이에 맞춰 드래그
    2) P2(오른쪽) HP바
  각 박스 드래그 후 ENTER/SPACE 확정, c로 취소. 두 박스 다 찍으면 자동 저장.
  (선택) 버스트/텐션도 찍고 싶으면 --extra 옵션.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2

CONFIG = Path(__file__).with_name("hud_config.json")


def grab_frame(video: str, t: float | None):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"영상 못 엶: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = int((t * fps) if t is not None else n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(target, n - 1)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("프레임 읽기 실패")
    return frame


def roi_to_ratio(roi, W, H) -> dict:
    x, y, w, h = roi
    return {"x0": x / W, "y0": y / H, "x1": (x + w) / W, "y1": (y + h) / H}


def pick(frame, title) -> dict | None:
    print(f"  >> [{title}] 박스를 드래그하고 ENTER. (취소=c)")
    roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if roi[2] == 0 or roi[3] == 0:
        return None
    H, W = frame.shape[:2]
    return roi_to_ratio(roi, W, H)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=None, help="보정에 쓸 프레임 시각(초)")
    ap.add_argument("--extra", action="store_true", help="버스트/텐션 게이지도 지정")
    args = ap.parse_args()

    frame = grab_frame(args.video, args.t)
    H, W = frame.shape[:2]
    print(f"프레임 {W}x{H} 로 보정 시작. 깨끗한 장면이 아니면 --t 로 다른 시각 지정.")

    cfg = {"source": Path(args.video).name, "ref_size": [W, H], "regions": {}}

    p1 = pick(frame, "1) P1 (LEFT) HP bar")
    p2 = pick(frame, "2) P2 (RIGHT) HP bar")
    if not p1 or not p2:
        raise SystemExit("HP바 두 개를 모두 지정해야 저장됩니다.")
    cfg["regions"]["p1_hp"] = p1
    cfg["regions"]["p2_hp"] = p2

    if args.extra:
        for key, label in [("p1_burst", "P1 Burst"), ("p2_burst", "P2 Burst"),
                           ("p1_tension", "P1 Tension"), ("p2_tension", "P2 Tension")]:
            r = pick(frame, label)
            if r:
                cfg["regions"][key] = r

    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 완료 -> {CONFIG.name}")
    for k, v in cfg["regions"].items():
        print(f"  {k}: x {v['x0']:.3f}-{v['x1']:.3f}, y {v['y0']:.3f}-{v['y1']:.3f}")
    print("\n이제: python hud_reader.py \"<영상>\" --debug  로 검증하세요.")


if __name__ == "__main__":
    main()
