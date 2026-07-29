"""
localize.py  —  캐릭터 위치 검출 (사전학습 YOLO person)

GGST 캐릭터를 YOLOv8(person 클래스)로 검출. HUD 포트레이트(상단 모서리)는 위치로 제외.
용도:
  - 무브 ID 학습용 '공격자만' 깨끗하게 크롭 (반쪽화면 대신)
  - 두 캐릭 x좌표 -> 거리 (거리 2차 필터)

검증: YOLOv8m 으로 게임 프레임 ~80%에서 캐릭 검출, 박스 크롭 시 단일 캐릭 깔끔히 격리됨.
무겁다(torch+ultralytics) -> 데이터셋 빌드/학습 등 로컬 도구에서 사용. (경량 exe엔 미포함)
"""
from __future__ import annotations
import numpy as np

_model = None
# GGST 파인튠 검출기 우선, 없으면 범용 yolov8m 폴백
import punish_engine as _pe
_FINETUNED = _pe.app_dir() / "runs" / "ggst_det" / "weights" / "best.pt"


def _m():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(_FINETUNED) if _FINETUNED.exists() else "yolov8m.pt")
    return _model


def char_boxes(frame: np.ndarray, conf: float = 0.25) -> list[tuple[float, float, float, float, float]]:
    """캐릭터 박스 [(x1,y1,x2,y2,conf), ...] 좌->우 정렬. HUD 포트레이트 제외."""
    H, W = frame.shape[:2]
    r = _m()(frame, classes=[0], conf=conf, verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        # HUD 포트레이트: 상단 모서리(좌우 끝 + 위쪽)
        if cy < 0.20 * H and (cx < 0.15 * W or cx > 0.85 * W):
            continue
        # 배경/오검출 sanity: 너무 작거나(키<14%H) 가로로 납작한(w>1.9h) 박스 제외
        if bh < 0.14 * H or bw > 1.9 * bh:
            continue
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    out.sort(key=lambda b: b[0])
    return out


def attacker_box(frame: np.ndarray, attacker_side: str):
    """공격자 측(P1=좌/P2=우)에 해당하는 캐릭 박스. 없으면 None."""
    boxes = char_boxes(frame)
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0][:4]
    return (boxes[0] if attacker_side == "P1" else boxes[-1])[:4]


def crop_box(frame: np.ndarray, box, pad: int = 24, size: int = 224) -> np.ndarray:
    import cv2
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    return cv2.resize(frame[y1:y2, x1:x2], (size, size))
