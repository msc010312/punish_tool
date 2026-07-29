# -*- coding: utf-8 -*-
"""input_reader.py — 리플레이 입력 표시(인풋 디스플레이) 판독기.

리플레이 극장의 입력 표시가 켜진 영상에서, 중앙 패드 패널(P1 위/P2 아래)의
스틱 볼 위치(9방향)와 버튼(P/K/S/HS/D) 점등을 프레임별로 읽는다.
이 (방향,버튼) 시퀀스가 커맨드 해석 -> 무브ID 확정의 입력이 된다.

기하(게임영역 비율, 2560x1440 리플레이 UI 실측 — 격자 오버레이로 확정):
  스틱 사각형  P1 (0.400,0.135)-(0.478,0.285) / P2 (0.400,0.315)-(0.478,0.465)
  버튼 4열 x간격 0.037 (P,K,S,HS 첫 행 / D,대시,RC,도발 둘째 행)
판독 원리:
  볼   = 진홍(H<8|H>172, S>120) 최대 연결성분의 중심 (꼬리·HS버블 오염 회피)
  버튼 = 원 내부 평균 V(밝기): 안 눌림=어두움(~60), 눌림=점등(>110)
"""
from __future__ import annotations
import numpy as np
import cv2

# 스틱 사각형 (x0,y0,x1,y1) — 게임영역 비율
STICK = {
    "P1": (0.400, 0.135, 0.478, 0.285),
    "P2": (0.400, 0.315, 0.478, 0.465),
}
# 버튼 중심 (첫 행 y / x 4개), 행 간격, 반지름(비율)
BTN_X = [0.530, 0.567, 0.604, 0.641]
BTN_Y = {"P1": 0.158, "P2": 0.338}
BTN_NAMES = ["P", "K", "S", "HS"]      # 첫 행. 둘째 행 첫 칸 = D
BTN_ROW2_DY = 0.045
BTN_R = 0.012                          # 원 반지름(게임영역 높이 비율 기준 근사)

DIRS = [6, 9, 8, 7, 4, 1, 2, 3]        # 각도(0=오른쪽, 반시계) -> 넘패드

# 버튼 원 자동 캘리브레이션 결과 캐시: {side: {name: (cx,cy)}} — 게임영역 비율
_BTN_CACHE: dict = {}


def calibrate_buttons(frame: np.ndarray) -> dict | None:
    """버튼 원(어두운 원 + 색 글자)을 프레임에서 직접 검출해 정확한 중심을 얻는다.
    하드코딩 좌표는 UI 버전·주변 밝기에 따라 배경을 찍어 가짜 눌림을 만든다(실측).
    반환 {P1:{P,K,S,HS,D:(cx,cy)}, P2:{...}} 또는 실패 시 None."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = {}
    for side, (y0, y1) in (("P1", (0.13, 0.28)), ("P2", (0.31, 0.46))):
        sub = (hsv[int(y0 * H):int(y1 * H), int(0.49 * W):int(0.68 * W), 2] < 110).astype(np.uint8)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(sub)
        r_exp = 0.0145 * H                      # 원 반지름 근사(실측 ~21px @1440p)
        area_lo, area_hi = 3.14 * (r_exp * 0.6) ** 2, 3.14 * (r_exp * 1.5) ** 2
        cands = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if area_lo <= a <= area_hi and 0.6 <= w_ / max(1, h_) <= 1.6:
                cands.append((cent[i][0], cent[i][1]))
        if len(cands) < 5:
            return None
        # 행 클러스터: y 근접(반지름 이내) 그룹 -> 첫 행 4개(P K S HS), 둘째 행 첫 칸(D)
        cands.sort(key=lambda c: c[1])
        rows = [[cands[0]]]
        for c in cands[1:]:
            if abs(c[1] - rows[-1][0][1]) < r_exp * 1.2:
                rows[-1].append(c)
            else:
                rows.append([c])
        rows = [sorted(r, key=lambda c: c[0]) for r in rows]
        top = next((r for r in rows if len(r) >= 4), None)
        if top is None:
            return None
        second = next((r for r in rows if r is not top and len(r) >= 1
                       and r[0][1] > top[0][1]), None)
        names = dict(zip(["P", "K", "S", "HS"], top[:4]))
        if second:
            names["D"] = second[0]
        # 게임영역 비율로 변환
        out[side] = {k: ((0.49 * W + x) / W, (y0 * H + y) / H) for k, (x, y) in names.items()}
    return out


def stick_dir(frame: np.ndarray, side: str) -> int | None:
    """스틱 방향(넘패드 1~9). 볼 미검출이면 None."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = STICK[side]
    sub = frame[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    ball = (((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 172)) &
            (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 80)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(ball)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 30:
        return None
    cx = cent[i][0] / sub.shape[1]
    cy = cent[i][1] / sub.shape[0]
    dx, dy = cx - 0.5, cy - 0.5
    if dx * dx + dy * dy < 0.03:                     # 데드존 = 중립
        return 5
    ang = np.degrees(np.arctan2(-dy, dx)) % 360
    return DIRS[int(((ang + 22.5) % 360) // 45)]


def button_vals(frame: np.ndarray, side: str, centers: dict | None = None) -> dict[str, float]:
    """버튼별 원 내부 평균 밝기(V). 점등 판정은 호출측에서 기준선 대비로.
    centers: calibrate_buttons() 결과(정확). 없으면 상수 폴백(부정확)."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r = max(3, int(BTN_R * H))
    out = {}
    for i, name in enumerate(BTN_NAMES + ["D"]):
        if centers and name in centers.get(side, {}):
            cx, cy = centers[side][name]
        elif name == "D":
            cx, cy = BTN_X[0], BTN_Y[side] + BTN_ROW2_DY
        else:
            cx, cy = BTN_X[i], BTN_Y[side]
        x, y = int(cx * W), int(cy * H)
        patch = hsv[max(0, y - r):y + r, max(0, x - r):x + r, 2]
        out[name] = float(patch.mean()) if patch.size else 0.0
    return out


def read_sequence(video: str, t0: float, t1: float, crop_game=None, game_region=None):
    """[t0,t1) 구간을 전 프레임 판독 -> [{t, P1:{dir,btn}, P2:{...}}, ...].
    버튼 점등 = V가 구간 중앙값보다 +35 이상."""
    import hud_reader as hr
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    gr = game_region if game_region is not None else hr.find_game_region(video)
    # 버튼 좌표 자동 캘리브레이션: 여러 프레임 성공분의 '중앙값' + 좌우패널 x 일치 검사.
    # 한 프레임 성공만 믿으면 가림 프레임에서 엉뚱한 원(둘째 행)을 잡을 수 있다(실측).
    centers = _BTN_CACHE.get(video)
    if not centers:
        good = []
        for dt in (0.0, 0.5, 1.0, -0.6, 1.5, -1.2, 2.0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t0 + dt) * fps)))
            ok, fr = cap.read()
            if not ok:
                continue
            c = calibrate_buttons(hr.crop_game(fr, gr))
            if not c or "P1" not in c or "P2" not in c:
                continue
            # 두 패널은 같은 x — 불일치면 오검(가림)으로 기각
            if any(abs(c["P1"][k][0] - c["P2"][k][0]) > 0.006
                   for k in c["P1"] if k in c["P2"]):
                continue
            good.append(c)
            if len(good) >= 3:
                break
        if good:
            centers = {}
            for side in ("P1", "P2"):
                keys = set.intersection(*(set(g[side]) for g in good))
                centers[side] = {
                    k: (sorted(g[side][k][0] for g in good)[len(good) // 2],
                        sorted(g[side][k][1] for g in good)[len(good) // 2])
                    for k in keys}
            _BTN_CACHE[video] = centers
    centers = centers or None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    rows = []
    raw = {("P1", n): [] for n in BTN_NAMES + ["D"]}
    raw.update({("P2", n): [] for n in BTN_NAMES + ["D"]})
    for i in range(int((t1 - t0) * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        g = hr.crop_game(fr, gr)
        row = {"t": t0 + i / fps}
        for side in ("P1", "P2"):
            bv = button_vals(g, side, centers)
            for n, v in bv.items():
                raw[(side, n)].append(v)
            row[side] = {"dir": stick_dir(g, side), "btnv": bv}
        rows.append(row)
    cap.release()
    # 점등 판정: 버튼별 기준선(중앙값) 대비 +35
    base = {k: float(np.median(v)) for k, v in raw.items() if v}
    for row in rows:
        for side in ("P1", "P2"):
            bv = row[side].pop("btnv")
            row[side]["btn"] = [n for n, v in bv.items() if v - base.get((side, n), 0) > 35]
    return rows
