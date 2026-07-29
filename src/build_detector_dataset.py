"""
build_detector_dataset.py  —  GGST 캐릭터 검출기 파인튜닝용 데이터셋 빌더 [부트스트랩]

범용 YOLO가 '확신하는'(2박스·분리·고conf) 프레임만 자동 라벨로 채택 -> 깨끗한 라벨.
이걸로 GGST 전용 검출기를 학습하면 이펙트·극단포즈 등 어려운 프레임까지 일반화 기대.
(라벨 누락 방지를 위해 '정확히 2박스' 프레임만 사용 — 1박스 프레임은 미검출 캐릭이 라벨 빠져 학습 오염)

출력(YOLO 포맷):
  detector_ds/images/{train,val}/*.jpg
  detector_ds/labels/{train,val}/*.txt   (각 줄: "0 cx cy w h" 정규화)
  detector_ds/data.yaml

사용:  python build_detector_dataset.py            # reference_video/ 전부
       python build_detector_dataset.py <영상...>  # 지정 영상
"""
from __future__ import annotations
import sys, random
from pathlib import Path

import cv2
from ultralytics import YOLO

import hud_reader as hr
import punish_engine as pe

ROOT = pe.app_dir()
OUT = ROOT / "detector_ds"
EVERY_SEC = 1.0          # 샘플 간격(초)
CONF = 0.25
MIN_BOTH_CONF = 0.40     # 두 박스 모두 이 conf 이상이어야 채택
VAL_RATIO = 0.15
_model = None


def _m():
    global _model
    if _model is None:
        _model = YOLO("yolov8x.pt")     # 라벨 품질 위해 가장 큰 모델로 부트스트랩
    return _model


def char_boxes(frame, conf=CONF):
    H, W = frame.shape[:2]
    r = _m()(frame, classes=[0], conf=conf, verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if cy < 0.20 * H and (cx < 0.15 * W or cx > 0.85 * W):   # HUD 포트레이트 제외
            continue
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    out.sort(key=lambda b: b[0])
    return out


def good_pair(boxes, W):
    """정확히 2박스 + 둘 다 고conf + 분리 -> 깨끗한 라벨로 채택."""
    if len(boxes) != 2:
        return None
    a, c = boxes
    if a[4] < MIN_BOTH_CONF or c[4] < MIN_BOTH_CONF:
        return None
    if abs((a[0] + a[2]) / 2 - (c[0] + c[2]) / 2) < 0.06 * W:
        return None
    return [a, c]


def yolo_line(box, W, H):
    x1, y1, x2, y2, _ = box
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process(video, idx):
    gr = hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, int(fps * EVERY_SEC))
    stem = "".join(ch if ch.isalnum() else "_" for ch in Path(video).stem)[:40]
    fi = kept = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if fi % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            fr = hr.crop_game(fr, gr)
            H, W = fr.shape[:2]
            pair = good_pair(char_boxes(fr), W)
            if pair:
                split = "val" if random.random() < VAL_RATIO else "train"
                name = f"{idx:02d}_{stem}_{fi}"
                cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (OUT / "labels" / split / f"{name}.txt").write_text(
                    "\n".join(yolo_line(b, W, H) for b in pair))
                kept += 1
        fi += 1
    cap.release()
    return kept


def main():
    random.seed(0)
    vids = sys.argv[1:] or [str(p) for p in (ROOT / "reference_video").glob("*.mp4")]
    if not vids:
        sys.exit("영상 없음 (reference_video/ 비었거나 인자 없음)")
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    total = 0
    for i, v in enumerate(vids):
        k = process(v, i)
        total += k
        print(f"[{i+1}/{len(vids)}] {Path(v).name[:50]} -> {k}장")
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames: [character]\n")
    print(f"\n완료: 총 {total}장 라벨 -> {OUT}\n학습: yolo detect train data={OUT/'data.yaml'} model=yolov8s.pt epochs=80 imgsz=960 device=0")


if __name__ == "__main__":
    main()
