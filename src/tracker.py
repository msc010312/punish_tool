"""
tracker.py  —  P1/P2 캐릭터 위치 추적 (색·스킨 무관, 위치 연속성 기반)

문제: YOLO 박스 2개 중 어느 게 P1/P2인지. 색은 스킨/컬러변경 때문에 못 씀.
해법: '위치 연속성'으로 추적.
  1) 라운드 시작(양쪽 HP 풀) = P1 좌/P2 우 (게임이 항상 리셋) -> 앵커
  2) 프레임마다 2박스를 직전 위치에 '최소 이동'으로 배정 (크로스업도 따라감)
  3) HP 풀 될 때마다 재앵커 -> 라운드 경계서 드리프트 리셋
  4) 박스 겹침/1개/0개 = '애매' 표시 -> 그 순간 이벤트는 데이터에서 스킵

반환: {frame_idx: (p1_box|None, p2_box|None, confident: bool)}  (every 프레임마다)
무겁다(YOLO 다수 프레임) -> 오프라인 데이터 구축용. 정확도 우선.
"""
from __future__ import annotations
import cv2
import numpy as np

import hud_reader as hr
import localize


def _cx(b):  # 박스 중심 x
    return (b[0] + b[2]) / 2


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _hp_pct(frame, rect) -> float:
    w = rect[2] - rect[0]
    return hr.cream_cols_in_rect(frame, rect) / max(w, 1)


def _separated(b1, b2, W) -> bool:
    """두 박스가 충분히 떨어져 있나(겹침/크로스업 순간 아님)."""
    return abs(_cx(b1) - _cx(b2)) > 0.06 * W


def track_players(video: str, cfg, every: int = 3, conf: float = 0.3,
                  max_seconds: float | None = None) -> dict:
    gr = hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    ok, f0 = cap.read()
    if not ok:
        return {}
    f0 = hr.crop_game(f0, gr)
    H, W = f0.shape[:2]
    l_rect, r_rect = hr.resolve_rects(cfg, W, H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- 1패스: 프레임별 박스 수집 (디코딩·YOLO 1회) ---
    raw = []                       # (fi, boxes)
    fi = 0
    limit = int(max_seconds * fps) if max_seconds else None
    while True:
        ok = cap.grab()
        if not ok or (limit and fi > limit):
            break
        if fi % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = hr.crop_game(frame, gr)
            raw.append((fi, localize.char_boxes(frame, conf)))
        fi += 1
    cap.release()

    # --- 2패스(메모리): 속도예측 트래킹 (크로스업=겹침구간을 속도로 통과) ---
    # 트랙 상태: pos(중심), vel(스텝당 변위), miss(연속 미검출 수)
    p1p = p2p = None               # 위치
    p1v = p2v = (0.0, 0.0)         # 속도
    p1m = p2m = 0
    gap = 0                        # 연속 0박스(라운드 전환 공백) 카운트
    REANCHOR_GAP = max(8, int(1.0 * fps / every))   # ~1초 공백 -> 라운드 전환 간주
    tracks: dict = {}

    def pred(pos, vel, miss):
        return (pos[0] + vel[0], pos[1] + vel[1]) if pos and miss < 6 else pos

    for fi, boxes in raw:
        p1box = p2box = None
        confident = False
        # 긴 검출공백(라운드 전환) 후 -> 트랙 리셋해 재앵커
        gap = gap + 1 if len(boxes) == 0 else 0
        if gap >= REANCHOR_GAP:
            p1p = p2p = None
            p1v = p2v = (0.0, 0.0)
        if p1p is None:
            # (재)앵커: 첫 2박스·분리 프레임 = 좌P1/우P2 (라운드시작은 캐릭이 시작위치로 리셋)
            if len(boxes) >= 2:
                bs = sorted(boxes, key=_cx)
                a, c = bs[0][:4], bs[-1][:4]
                if _separated(a, c, W):
                    p1box, p2box = a, c
                    p1p, p2p = _center(a), _center(c)
                    confident = True
        else:
            q1, q2 = pred(p1p, p1v, p1m), pred(p2p, p2v, p2m)   # 예측 위치
            if len(boxes) >= 2:
                b = sorted(boxes, key=lambda x: -x[4])[:2]
                a, c = b[0][:4], b[1][:4]
                ca, cc = _center(a), _center(c)
                d1 = _dist(ca, q1) + _dist(cc, q2)
                d2 = _dist(ca, q2) + _dist(cc, q1)
                p1box, p2box = (a, c) if d1 <= d2 else (c, a)
                confident = _separated(p1box, p2box, W)
            elif len(boxes) == 1:
                b = boxes[0][:4]; bc = _center(b)               # 1박스: 가까운 예측트랙에
                if _dist(bc, q1) <= _dist(bc, q2):
                    p1box = b
                else:
                    p2box = b
        # 상태 갱신 (검출=실측으로 vel 갱신, 미검출=예측으로 coast)
        if p1box:
            nc = _center(p1box)
            if p1p:
                p1v = (0.5 * p1v[0] + 0.5 * (nc[0] - p1p[0]),
                       0.5 * p1v[1] + 0.5 * (nc[1] - p1p[1]))
            p1p, p1m = nc, 0
        elif p1p:
            p1p, p1m = pred(p1p, p1v, p1m), p1m + 1
        if p2box:
            nc = _center(p2box)
            if p2p:
                p2v = (0.5 * p2v[0] + 0.5 * (nc[0] - p2p[0]),
                       0.5 * p2v[1] + 0.5 * (nc[1] - p2p[1]))
            p2p, p2m = nc, 0
        elif p2p:
            p2p, p2m = pred(p2p, p2v, p2m), p2m + 1
        tracks[fi] = (p1box, p2box, confident)
    tracks["_fps"] = fps
    return tracks


def boxes_at(tracks: dict, t: float):
    """시각 t에서 가장 가까운 추적 결과 (p1box, p2box, confident)."""
    fps = tracks.get("_fps", 60.0)
    target = int(t * fps)
    keys = [k for k in tracks if isinstance(k, int)]
    if not keys:
        return None, None, False
    k = min(keys, key=lambda x: abs(x - target))
    if abs(k - target) > fps * 0.3:    # 0.3s 이내 추적치 없으면 신뢰X
        return None, None, False
    return tracks[k]
