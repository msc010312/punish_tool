# -*- coding: utf-8 -*-
"""command_parser.py — 입력 시퀀스 -> 넘패드 커맨드(무브) 해석.

input_reader 가 읽은 (t, 방향, 버튼) 프레임 시퀀스에서, 버튼이 눌린 순간을 앵커로
직전 방향 히스토리를 모션 패턴과 대조해 커맨드를 결정한다.

  우선순위(격겜 입력 표준): 632146 > 236236/214214 > 41236/63214 > 623 > 236/214
                              > 22 > 차지([4]6/[2]8) > 커맨드노말(6S 등) > 노말
  좌우 반전: 넘패드 표기는 '상대 방향=6' 기준. 캐릭이 오른쪽이면 절대입력을 미러.
  버퍼: 모션 완성~버튼 사이 최대 BUF 프레임(GGST 선입력 관대).

정확도 원칙: 그 캐릭 무브리스트(framedata 키)에 실재하는 커맨드만 채택.
모션이 안 맞으면 한 단계씩 단순한 패턴으로 내려간다(틀린 상위 커맨드 단정 금지).
"""
from __future__ import annotations

MIRROR = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7, 2: 2, 5: 5, 8: 8}
BUF = 14          # 모션 마지막 방향 -> 버튼까지 허용 프레임(60fps)
HOLD_CHARGE = 28  # 차지 성립 프레임
BTN_KEY = {"HS": "H"}   # 입력표시 표기 -> 프레임데이터 표기

# 모션 패턴(방향 시퀀스) — 우선순위 순. max_span=패턴 전체가 이 프레임 안에 완성돼야
# 하는 제한(관대 변형용: 60fps 판독이 빠른 스틱 회전의 중간 방향을 놓칠 때 대비).
MOTIONS = [
    ("632146", [6, 3, 2, 1, 4, 6], None),
    ("236236", [2, 3, 6, 2, 3, 6], None),
    ("214214", [2, 1, 4, 2, 1, 4], None),
    ("41236",  [4, 1, 2, 3, 6], None),
    ("63214",  [6, 3, 2, 1, 4], None),
    ("623",    [6, 2, 3], None),
    ("421",    [4, 2, 1], None),
    ("236",    [2, 3, 6], None),
    ("214",    [2, 1, 4], None),
    ("236",    [2, 6], 9),      # 빠른 회전에서 3 판독 유실 -> 2->6 이 9F 안이면 236 인정
    ("214",    [2, 4], 9),
    ("22",     [2, 5, 2], None),
]


def _compress(dirs: list[tuple[float, int]]) -> list[tuple[float, int, int]]:
    """(t,dir) 프레임열 -> (t시작, dir, 지속프레임) 런들. None 은 직전 값 유지로 취급."""
    runs = []
    for t, d in dirs:
        if d is None:
            d = runs[-1][1] if runs else 5
        if runs and runs[-1][1] == d:
            runs[-1] = (runs[-1][0], d, runs[-1][2] + 1)
        else:
            runs.append((t, d, 1))
    return runs


def _match_motion(runs: list[tuple[float, int, int]], pattern: list[int],
                  press_i: int, fps: float, max_span: int | None = None) -> bool:
    """press 직전 런들에서 pattern 이 '순서대로, 사이 공백 없이(중립 1런 허용)' 등장하는가.
    마지막 패턴 방향의 끝~버튼 사이가 BUF 프레임 이내. max_span 이 있으면
    패턴 첫 방향 시작~버튼까지 총 프레임이 그 이내여야 한다(관대 변형의 오인 방지)."""
    pi = len(pattern) - 1
    first_ri = None
    started = False           # 패턴 꼬리를 찾기 전(=버튼 직전 BUF 안)은 아무 방향이나 스킵 허용
    skipped = 0               #   (모션 완성 후 스틱을 중립/아래로 되돌리고 누르는 게 흔함)
    for ri in range(press_i, -1, -1):
        d = runs[ri][1]
        if d == pattern[pi]:
            if not started:
                gap = sum(runs[k][2] for k in range(ri + 1, press_i + 1))
                if gap > BUF:
                    return False
                started = True
            first_ri = ri
            pi -= 1
            if pi < 0:
                if max_span is not None:
                    span = sum(runs[k][2] for k in range(first_ri, press_i + 1))
                    if span > max_span:
                        return False
                return True
        elif not started:
            skipped += runs[ri][2]
            if skipped > BUF:
                return False  # 버퍼 넘게 뒤져도 패턴 꼬리가 안 나오면 실패
        elif d == 5 or d == pattern[min(pi + 1, len(pattern) - 1)]:
            continue          # 중립/직전방향 잔류는 허용
        else:
            return False      # 다른 방향이 끼면 모션 끊김
    return False


def parse_press(rows: list[dict], side: str, press_t: float, btn: str,
                movelist: set[str], facing_right: bool, fps: float = 60.0) -> dict:
    """버튼이 눌린 순간의 커맨드 해석 -> {'cmd','move','alts'}.
    move = movelist 에 실재하는 최상위 매칭(없으면 None). alts = 반대 방향 가정 결과."""
    win = [(r["t"], r[side]["dir"]) for r in rows if press_t - 0.8 <= r["t"] <= press_t]
    if not win:
        return {"cmd": None, "moves": []}
    b = BTN_KEY.get(btn, btn)

    def resolve(mirror: bool) -> tuple[str, list[str]]:
        dirs = [(t, (MIRROR[d] if (d is not None and mirror) else d)) for t, d in win]
        runs = _compress(dirs)
        press_i = len(runs) - 1
        cur = runs[press_i][1]
        # 공중 판정: 최근 0.65s 안에 위 입력(7/8/9) -> 점프 중일 가능성. j. 키 우선 시도.
        air = any(d in (7, 8, 9) for t, d in dirs if press_t - 0.65 <= t)
        # 1) 모션들 (우선순위 순). 공중이면 j.모션 우선.
        for name, pat, span in MOTIONS:
            if _match_motion(runs, pat, press_i, fps, span):
                order = (f"j.{name}{b}", f"{name}{b}") if air else (f"{name}{b}", f"j.{name}{b}")
                hits = [k for k in order if k in movelist]
                if hits:
                    return name + b, hits[:1]
        # 2) 차지: 마지막 방향 6/8 이고 그 전 반대방향(4/2)을 HOLD_CHARGE+ 유지
        if cur in (6, 8) and press_i >= 1:
            back = {6: 4, 8: 2}[cur]
            held = sum(r[2] for r in runs[:press_i] if r[1] == back)
            if runs[press_i - 1][1] == back and held >= HOLD_CHARGE:
                key = f"[{back}]{cur}{b}"
                if key in movelist:
                    return key, [key]
        # 3) 노말/커맨드노말/공중노말 — 후보 수집 방식.
        # 버튼 점등 검출이 실제 누름보다 1~3F 늦어 스틱이 중립으로 돌아온 뒤 읽힐 수 있음
        # -> press 방향이 5인데 직전 방향 런이 방금(≤5F) 끝났으면 그 방향을 우선 후보로.
        dir_cands = [cur]
        if cur == 5 and press_i >= 1 and runs[press_i][2] <= 5 and runs[press_i - 1][1] != 5:
            dir_cands.insert(0, runs[press_i - 1][1])
        # 눌림 순간 볼 미검출(None-채움으로 온 방향)이면 실제 스틱은 모름 -> 중립도 후보
        if win and win[-1][1] is None and 5 not in dir_cands:
            dir_cands.append(5)
        air_press = cur in (7, 8, 9)
        moves: list[str] = []

        def add(key):
            if key and key in movelist and key not in moves:
                moves.append(key)

        for d in dir_cands:
            if air_press:                        # 누름 순간 위 방향 = 공중 확실
                add(f"j.{d}{b}"); add(f"j.{b}")
                continue
            if d in (1, 2, 3):                   # 앉은 방향 + 버튼 = 2X (게임 규칙)
                add(f"2{b}")
            elif d != 5:
                add(f"{d}{b}")
                if b == "D" and d in (4, 6):
                    add("6D or 4D")
            if d in (5, 4, 6):                   # 선 상태 노말. S는 근/원 둘 다 후보
                if b == "S":
                    add("c.S"); add("f.S")
                add(f"5{b}")
        if air and not air_press:                # 공중 '가능성'(직전 위 입력) -> 병기 후보
            add(f"j.{b}")
        if not moves and f"j.{b}" in movelist:   # 지상 키가 아예 없으면 공중 키
            add(f"j.{b}")
        d0 = dir_cands[0]
        return f"{d0}{b}", moves[:3]

    cmd, moves = resolve(mirror=not facing_right)
    alt_cmd, alt_moves = resolve(mirror=facing_right)
    return {"cmd": cmd, "moves": moves,
            "alt": {"cmd": alt_cmd, "moves": alt_moves} if alt_moves != moves else None}


def identify_starter(rows: list[dict], side: str, hit_t: float,
                     char_moves: dict, facing_right: bool) -> dict | None:
    """배너 시각과 '누른 시각+발생프레임'이 맞는 입력 = 시동기.

    지상기: 발동 즉시 히트권 -> (누름+발생)-배너 ∈ [-0.45, +0.10] (부호창).
      배너는 히트보다 0~0.4s 늦으므로 음수쪽이 정상. 피해자의 '끊긴' 기술은
      누름+발생 > 히트라 +쪽으로 늦어 자연히 배제된다.
    공중기(j.): 점프 궤적 따라 히트가 늦어질 수 있음 -> [-0.85, +0.10] 완화.
    반환 {'move','moves','cmd','press_t','err'}(err=|부호오차|, 지상 우선 동률시 작은 오차)."""
    ml = set(char_moves)
    best = None
    for pt, btn in presses_in(rows, side, hit_t - 1.1, hit_t + 0.05):
        res = parse_press(rows, side, pt, btn, ml, facing_right)
        for key in res["moves"]:
            st = getattr(char_moves.get(key), "startup", None)
            if not isinstance(st, int):
                continue
            signed = (pt + st / 60.0) - hit_t
            lo = -0.85 if key.startswith("j.") else -0.45
            if not (lo <= signed <= 0.10):
                continue
            err = abs(signed)
            if best is None or err < best["err"]:
                best = {"move": key, "moves": res["moves"], "cmd": res["cmd"],
                        "press_t": pt, "err": err}
    return best


def identify_starter_both(rows: list[dict], atk: str, hit_t: float,
                          moves_by_side: dict, facing: dict) -> tuple[str, dict | None]:
    """HP 귀속된 공격자(atk) 쪽에서 시동기를 찾고, 실패 시 반대쪽 폴백.
    잽 등 저데미지 시동은 HP로 귀속이 안 되는데(노이즈에 묻힘), 입력 타이밍은
    공격자 쪽만 정확히 맞으므로 그 자체가 귀속 신호가 된다.
    반환 (판정된 공격자, starter dict|None)."""
    st = identify_starter(rows, atk, hit_t, moves_by_side[atk], facing[atk])
    if st:
        return atk, st
    other = "P2" if atk == "P1" else "P1"
    st2 = identify_starter(rows, other, hit_t, moves_by_side[other], facing[other])
    if st2:
        return other, st2
    return atk, None


def presses_in(rows: list[dict], side: str, t0: float, t1: float) -> list[tuple[float, str]]:
    """[t0,t1] 안에서 '새로 눌린' 버튼들 (t, 버튼). 유지 중 재검출은 제외."""
    out = []
    prev: set[str] = set()
    for r in rows:
        cur = set(r[side]["btn"])
        if t0 <= r["t"] <= t1:
            for b in cur - prev:
                out.append((r["t"], b))
        prev = cur
    return out
