# -*- coding: utf-8 -*-
"""input_moveid.py — 입력표시 기반 무브ID (정밀 계층 자동 활성).

영상에 리플레이 입력 표시(중앙 패드 패널)가 있으면, 카운터/펀ish 이벤트마다
공격자의 입력을 판독·해석해 시동기를 결정론적으로 확정한다(검증: 6/6).
없으면 아무것도 하지 않는다 — 기본 분석은 그대로.

이벤트 dict 에 붙이는 필드:
  input_move  : 확정 시동기(framedata 키). 예 "2K", "j.236S"
  input_moves : 동률 후보(근/원 S 등). 예 ["c.S","f.S"]
  input_atk   : 입력 타이밍으로 재판정한 공격자("P1"/"P2") — HP 귀속과 다를 수 있음
한계(정직): 좌우 위치 교대(사이드 스위치)는 아직 미추적 — P1=왼쪽 가정.
"""
from __future__ import annotations

import input_reader as ir
import command_parser as cp


def detect_input_display(video: str, game_region=None, probe_ts=(0.3, 0.5, 0.7)) -> bool:
    """중앙 패드 패널 존재 여부: 여러 지점에서 스틱 볼이 읽히고 버튼 캘리브레이션이 성립하면 True."""
    import cv2
    import hud_reader as hr
    gr = game_region if game_region is not None else hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    hits = 0
    for frac in probe_ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ok, fr = cap.read()
        if not ok:
            continue
        g = hr.crop_game(fr, gr)
        balls = sum(1 for s in ("P1", "P2") if ir.stick_dir(g, s) is not None)
        if balls == 2 and ir.calibrate_buttons(g):
            hits += 1
    cap.release()
    return hits >= 2          # 3지점 중 2곳 이상 성립 -> 입력표시 켜진 영상


def scan_presses(video: str, chars: dict, data: dict,
                 game_region=None, chunk: float = 30.0) -> dict | None:
    """경기 전체 입력을 한 번 스캔 -> {"P1": [(t, 무브키)], "P2": [...]}.
    입력표시 없으면 None. 같은 버튼 0.25s 내 재입력은 연타로 1회만."""
    if not detect_input_display(video, game_region):
        return None
    import cv2
    import hud_reader as hr
    gr = game_region if game_region is not None else hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 60.0)
    cap.release()
    mbs = {s: data.get(chars.get(s), {}) for s in ("P1", "P2")}
    if not mbs["P1"] or not mbs["P2"]:
        return None
    facing = {"P1": True, "P2": False}
    out = {"P1": [], "P2": []}
    last_press = {}
    t0 = 0.0
    while t0 < dur:
        t1 = min(dur, t0 + chunk)
        rows = ir.read_sequence(video, t0, t1, game_region=gr)
        for side in ("P1", "P2"):
            ml = set(mbs[side])
            for pt, btn in cp.presses_in(rows, side, t0, t1):
                if pt - last_press.get((side, btn), -9.0) < 0.25:
                    last_press[(side, btn)] = pt
                    continue
                last_press[(side, btn)] = pt
                res = cp.parse_press(rows, side, pt, btn, ml, facing[side])
                if res["moves"]:
                    out[side].append((pt, res["moves"][0]))
        t0 = t1
    return out


def move_stats_from(presses: dict, events: list[dict], chars: dict, data: dict) -> dict:
    """기술별 시도/적중. 적중 = 누름+발생 직후 상대 HP 하락(공중기는 늦은 히트 허용).
    정직한 한계: 입력 기준(경직 중 눌러 안 나간 입력 포함), 가드/헛침 구분 불가."""
    mbs = {s: data.get(chars.get(s), {}) for s in ("P1", "P2")}
    hits = {s: sorted(e["t"] for e in events if e.get("kind") == "hit" and e.get("side") == s)
            for s in ("P1", "P2")}
    stats = {"P1": {}, "P2": {}}
    for side in ("P1", "P2"):
        opp = "P2" if side == "P1" else "P1"
        for pt, key in presses[side]:
            st = getattr(mbs[side].get(key), "startup", None)
            exp = pt + (st / 60.0 if isinstance(st, int) else 0.15)
            late = 0.9 if key.startswith("j.") else 0.35   # 공중기는 하강하며 늦게 맞음
            hit = any(exp - 0.05 <= ht <= exp + late for ht in hits[opp])
            d = stats[side].setdefault(key, {"n": 0, "hit": 0})
            d["n"] += 1
            d["hit"] += int(hit)
    return stats


def link_stats_from(presses: dict, max_gap_f: int = 60, min_n: int = 3) -> dict:
    """연계 실행 통계: 같은 플레이어의 연속 입력 A->B(간격<=60F)를 전이별로 집계.
    '최적'은 지어내지 않는다 — 본인 최속 기록이 기준(외부 캔슬표 불필요, 사실만).
    반환 {"P1": [{"a","b","n","avg","min","max"}], ...} (n>=min_n, 평균간격 오름차순)."""
    out = {}
    for side in ("P1", "P2"):
        gaps: dict[tuple, list] = {}
        seq = presses[side]
        for (t1, a), (t2, b) in zip(seq, seq[1:]):
            g = (t2 - t1) * 60.0
            if 3.0 <= g <= max_gap_f:   # <3F = 동시입력(FD/RC 매크로) — 연계 아님
                gaps.setdefault((a, b), []).append(g)
        rows = []
        for (a, b), gs in gaps.items():
            if len(gs) >= min_n:
                rows.append({"a": a, "b": b, "n": len(gs),
                             "avg": sum(gs) / len(gs), "min": min(gs), "max": max(gs)})
        rows.sort(key=lambda r: -(r["avg"] - r["min"]))   # '본인 최속 대비 손해' 큰 순
        out[side] = rows
    return out


def collect_move_stats(video: str, events: list[dict], chars: dict, data: dict,
                       game_region=None, chunk: float = 30.0) -> dict | None:
    """(호환 유지) 기술별 시도/적중 통계만 필요할 때."""
    presses = scan_presses(video, chars, data, game_region, chunk)
    if presses is None:
        return None
    return move_stats_from(presses, events, chars, data)


def annotate(video: str, events: list[dict], chars: dict, data: dict,
             game_region=None) -> int:
    """카운터/펀ish 이벤트에 입력 확정 시동기 부착. 반환=확정 건수.
    입력표시 없으면 0건(무해). chars={'P1':캐릭명,'P2':...}, data=framedata."""
    if not detect_input_display(video, game_region):
        return 0
    import hud_reader as hr
    gr = game_region if game_region is not None else hr.find_game_region(video)
    mbs = {s: data.get(chars.get(s), {}) for s in ("P1", "P2")}
    if not mbs["P1"] or not mbs["P2"]:
        return 0
    facing = {"P1": True, "P2": False}       # P1=왼쪽 가정(사이드 스위치 미추적 — 한계)
    done = 0
    for e in events:
        if e.get("kind") not in ("counter", "punish"):
            continue
        atk0 = "P2" if e.get("victim") == "P1" else "P1"
        rows = ir.read_sequence(video, e["t"] - 1.4, e["t"] + 0.3, game_region=gr)
        atk, st = cp.identify_starter_both(rows, atk0, e["t"], mbs, facing)
        if st:
            e["input_move"] = st["move"]
            e["input_moves"] = st["moves"]
            e["input_atk"] = atk
            # 확정 무브의 counter 속성 = 진짜 카운터 레벨(옛 peak기반 level 대체)
            mv = mbs[atk].get(st["move"])
            lv = (getattr(mv, "counter", "") or "").strip()
            if lv:
                e["input_level"] = lv
            if atk != atk0:                   # 입력 타이밍이 귀속을 뒤집음(저데미지 시동 등)
                e["victim"] = "P2" if atk == "P1" else "P1"
            done += 1
    return done
