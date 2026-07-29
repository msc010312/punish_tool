"""
punish_engine.py  —  2D 격투게임 "확정 반격" 판정 엔진 (길티기어 스트라이브용 시작)

이 모듈이 하는 일 = 너의 도구의 "두뇌(②번 벽돌)".
영상 분석부(①번)가 "공격자 캐릭 A가 기술 X를 썼고, 그게 가드됐다"는 사실만 넘겨주면,
이 엔진이 프레임 데이터만으로 "방어자 캐릭 B는 무엇으로 확정 반격이 되는가"를 즉답한다.
=> 유저가 Dustloop 프레임표를 직접 안 뒤져도 되게 만드는 핵심.

데이터 형태(아래 SAMPLE_DATA)는 anitanotto/whens-my-turn 스크래퍼가 뱉는 JSON과 동일:
    { 캐릭터명: { 기술명: {"startup":int, "active":.., "recovery":.., "onBlock":int, "guard":str, ...} } }
풀 데이터는 그 스크래퍼를 꽂아 load_framedata()에 넘기면 그대로 작동한다.
"""

from __future__ import annotations
import functools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def app_dir() -> Path:
    """데이터 파일 위치. .exe(frozen)면 실행파일 폴더,
    개발 중이면 프로젝트 루트(소스는 src/ 안, 데이터는 루트)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent   # src/ -> 루트


def framedata_path() -> Path:
    """현재 활성 게임 프로필의 프레임데이터 파일 경로."""
    import games
    return app_dir() / games.ACTIVE.framedata_file


# ---------------------------------------------------------------------------
# 1) 데이터 로딩 / 정규화
# ---------------------------------------------------------------------------

def parse_frame_value(raw) -> Optional[int]:
    """
    Dustloop 값은 int일 수도, '-2~10' 같은 범위 문자열, '' 빈값일 수도 있다.
    '확정' 반격 판정에서는 항상 공격자에게 유리한(=방어자에게 빡센) 쪽을 가정해야 안전하다.
    예) onBlock 이 '-2~10' 이면, 보장하려면 최악(절댓값 가장 작은 -2)을 써야 한다.
    반환: 정규화된 정수, 판정 불가면 None.
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if s == "" or s == "-":
        return None
    # 범위 '-2~10' / '-5~-2'
    if "~" in s:
        nums = [int(n) for n in re.findall(r"[+-]?\d+", s)]
        if not nums:
            return None
        # 방어자 입장에서 가장 보수적인(절댓값 최소) 값 채택
        return min(nums, key=abs)
    m = re.match(r"^[+-]?\d+", s)
    return int(m.group()) if m else None


def parse_damage_total(raw) -> Optional[int]:
    """다단히트 총 데미지. Dustloop 표기: '20,25'->45(콤마=다단), '18*2'->36(n*횟수),
    '45'->45(단일). 다단기는 짧은 시간에 HP가 여러 번/한꺼번에 깎여 측정값이 총뎀일 수 있으므로,
    무브ID 매칭 때 첫히트(damage)와 총뎀 둘 다 후보로 쓴다. 판정불가면 None."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if s == "" or s == "-":
        return None
    mult = re.match(r"^\s*(\d+)\s*[*x×X]\s*(\d+)", s)   # '18*2' / '18×2'
    if mult:
        return int(mult.group(1)) * int(mult.group(2))
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if not nums:
        return None
    return sum(nums) if "," in s else nums[0]           # 콤마=다단 합, 아니면 첫 숫자


@dataclass
class Move:
    name: str
    startup: Optional[int]
    on_block: Optional[int]
    guard: str = ""          # "All" / "High" / "Low" / "" 등
    recovery: Optional[int] = None
    damage: Optional[int] = None   # 기본 데미지(첫 히트 숫자) — 무브 ID(데미지 매칭)용
    damage_total: Optional[int] = None  # 다단히트 총 데미지(뭉쳐 측정될 때 매칭용)
    counter: str = ""              # 카운터 레벨(Small/Mid/Large/Very Small) — 첫 토큰
    is_followup: bool = False
    is_air: bool = False      # j.~ 공중기/공중던지기 (지상 확정 반격 후보에서 제외)
    is_throw: bool = False    # 던지기 — 발생은 빨라도 가드백 때문에 거리 의존(조건부)
    is_special: bool = False  # 반격기/패리/가드캔슬/피니시블로 등 — 일반 콤보·확정기 아님
    reach: Optional[float] = None  # 리치(거리). 현재 Dustloop 스크랩엔 없음 -> 추후 별도 소스 연결용 훅

    @property
    def punishable_window(self) -> Optional[int]:
        """가드당했을 때 상대가 가진 확정 반격 프레임 수. 음수 onBlock일 때만 존재."""
        if self.on_block is None or self.on_block >= 0:
            return None
        return -self.on_block  # 예: onBlock -7 -> 7프레임 창


def load_framedata(raw: dict) -> dict[str, dict[str, Move]]:
    """스크래퍼 JSON -> {캐릭터: {기술명: Move}} 로 정규화."""
    out: dict[str, dict[str, Move]] = {}
    for char, moves in raw.items():
        out[char] = {}
        for mname, mdata in moves.items():
            if not isinstance(mdata, dict):
                continue
            disp_name = str(mdata.get("name", "")) if isinstance(mdata, dict) else ""
            guard = str(mdata.get("guard", ""))
            gl = guard.strip().lower()
            # 공중기 판정: 표기가 j.~ 로 시작하거나 이름이 'Air ...'(에어 던지기 등)
            is_air = mname.strip().startswith("j.") or disp_name.strip().startswith("Air ")
            # 던지기 판정: 이름에 'Throw' 또는 guard 칸이 던지기 표기
            is_throw = "throw" in disp_name.lower() or gl in ("throw", "ground throw")
            # 특수기 판정: 일반 타격기가 아닌 것 (콤보·확정 후속기로 잡으면 안 됨)
            #  - guard 칸이 빈 기술 = 반격기/패리/피니시블로 등 (예: Baiken 236P 'Hiiragi')
            #  - 'While Guarding:' = 가드캔슬(블록스턴에서만 나감)
            #  - 이름에 반격/패리/반사 키워드 (Potemkin 'Reflect Projectile' 등)
            #  - 슈퍼(각성필살기): 모션 입력이 길다(숫자 6개 이상, 예 236236/632146).
            #    표기 발생값이 '슈퍼플래시 이후' 1~2F라 콤보 연결창으로 오해되므로 제외.
            disp_lower = disp_name.lower()
            name_special = any(
                k in disp_lower for k in
                ("counter", "parry", "reflect", "deflect", "hiiragi",
                 "dodge", "reversal", "absorb", "autoguard", "armor")
            )
            n_motion_digits = sum(c.isdigit() for c in mname)
            is_super = n_motion_digits >= 6
            su_val = parse_frame_value(mdata.get("startup"))
            # 안전망: 발생 1~2F 비(非)던지기 = 슈퍼플래시 이후/차지-릴리스 등 표기 아티팩트.
            #   GGST 최속 일반기는 ~4~5F, 진짜 2F는 던지기(별도 처리)뿐이라 손실 없음.
            su_artifact = su_val is not None and su_val <= 2 and not is_throw
            is_special = (
                (gl == "" and not is_throw)
                or mname.strip().lower().startswith("while guarding:")
                or "guard cancel" in disp_lower
                or name_special
                or is_super
                or su_artifact
            )
            counter_lv = str(mdata.get("counter", "")).split(",")[0].strip()  # 첫 토큰(첫 히트 레벨)
            out[char][mname] = Move(
                name=mname,
                startup=parse_frame_value(mdata.get("startup")),
                on_block=parse_frame_value(mdata.get("onBlock")),
                guard=guard,
                recovery=parse_frame_value(mdata.get("recovery")),
                damage=parse_frame_value(mdata.get("damage")),
                damage_total=parse_damage_total(mdata.get("damage")),
                counter=counter_lv,
                is_followup=bool(mdata.get("followup", False)),
                is_air=is_air,
                is_throw=is_throw,
                is_special=is_special,
            )
    return out


HEALTH = 420   # GGST 표준 체력 (대부분 캐릭). HP% * HEALTH = 실제 데미지 환산.

# ---------------------------------------------------------------------------
# 캐릭터 방어 스탯(Defense·Guts) — 관측 데미지 -> base 데미지 역산
# 시동기(콤보 첫 히트)는 콤보보정(proration)이 안 걸리므로,
#   관측 = base × def_mult × guts배율(등급, 히트 직전 HP%)
# 를 역산하면 프레임데이터의 base 데미지와 직접 비교할 수 있다.
# (R.I.S.C./포지티브 보너스는 HUD에서 못 읽어 오차로 남음 — 중립 상황이면 대체로 0)
# ---------------------------------------------------------------------------
_GUTS_HP_STEPS = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)   # 표의 <=70% ... <=10% 문턱
_FALLBACK_CORR = 0.88   # 스탯 파일 없을 때의 옛 전역 보정(회귀 방지)


@functools.lru_cache(maxsize=1)
def load_char_stats() -> dict:
    """char_stats_ggst.json (scrape_char_stats.py 산출). 없으면 빈 dict."""
    p = app_dir() / "char_stats_ggst.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def guts_mult(rating: int, hp_pct: float) -> float:
    """Guts 배율. hp_pct = 히트 '직전' 방어측 HP(0~1). >70% 구간은 1.0."""
    table = load_char_stats().get("guts_table", {}).get(str(rating))
    if not table:
        return 1.0
    m = 1.0
    for step, mult in zip(_GUTS_HP_STEPS, table):   # 문턱을 내려가며 갱신 -> 해당 구간 배율
        if hp_pct <= step:
            m = mult
    return m


def unscale_damage(measured: float, defender_char: str | None,
                   defender_hp_pct: float | None, risc: float | None = None) -> float:
    """관측 데미지(HP%×HEALTH) -> base 데미지 추정.
    방어측 캐릭/HP를 모르면 옛 전역 보정(0.88)으로 폴백.
    risc: 히트 직전 방어측 R.I.S.C. 채움(0~1). 2%당 방어 1% 감소 = 배율 1/(1-0.5f),
    풀이면 데미지 2배(Dustloop) — 못 읽으면 0으로 취급(과대역산 방지)."""
    stats = load_char_stats().get("chars", {})
    st = stats.get(defender_char or "")
    if not st:
        return measured / _FALLBACK_CORR
    g = guts_mult(int(st.get("guts", 0)), defender_hp_pct if defender_hp_pct is not None else 1.0)
    r = 1.0 / (1.0 - 0.5 * min(1.0, max(0.0, risc))) if risc else 1.0
    return measured / (float(st.get("def_mult", 1.0)) * g * r)


# 삭제/구버전 기술 — Dustloop 스크랩에 남아있지만 현행 게임에 없음 (사용자 확인). 후보에서 전역 제외.
REMOVED_MOVES = {"236d"}


def is_removed_move(name: str) -> bool:
    return name.lower().replace(".", "").replace("_", "") in REMOVED_MOVES


def identify_move(char_moves: dict[str, Move], measured_damage: float,
                  counter_level: str | None = None,
                  tol: float = 0.25, max_results: int = 4,
                  include_air: bool = False, include_throw: bool = False) -> list[tuple[Move, float]]:
    """측정 데미지(+선택적 카운터레벨)로 '무슨 기술인지' 후보를 좁힌다 (속성 기반 무브 ID).
    measured_damage: 영상에서 잰 HP% * HEALTH.
    counter_level: 영상에서 본 레벨(Dustloop 표기 'Large'/'Mid' 등). 주면 그 레벨 기술을 우선.
    반환: [(Move, 상대오차), ...] 오차 작은 순. 후속/공중기 제외(지상 시동기 위주).
    카운터히트는 데미지 보정으로 base보다 크게 나오니 tol 넉넉히. 후보 압축용이지 확정 아님."""
    cands = []
    for mv in char_moves.values():
        if mv.damage is None or mv.damage <= 0 or mv.is_followup:
            continue
        if mv.is_air and not include_air:        # 공중기 — 점프인 카운터가 흔해 기본 포함 권장
            continue
        if mv.is_throw and not include_throw:
            continue
        if is_removed_move(mv.name):             # 삭제된 기술(236D 등) 제외
            continue
        err = abs(mv.damage - measured_damage) / max(measured_damage, 1.0)
        if mv.damage_total and mv.damage_total != mv.damage:   # 다단기: 첫히트/총뎀 중 가까운 쪽
            err = min(err, abs(mv.damage_total - measured_damage) / max(measured_damage, 1.0))
        if err <= tol:
            cands.append((mv, round(err, 3)))
    cands.sort(key=lambda x: x[1])
    # 레벨이 주어지면 그 레벨 기술만 남긴다(있을 때만) -> 종종 단일 후보로 특정됨
    if counter_level:
        lv = counter_level.strip().lower()
        matched = [(m, e) for m, e in cands if m.counter.lower() == lv]
        if matched:
            return matched[:max_results]
    return cands[:max_results]


def load_framedata_file(path: Path | None = None) -> Optional[dict[str, dict[str, Move]]]:
    """활성 게임의 프레임데이터 JSON 을 읽어 정규화. 없으면 None."""
    if path is None:
        path = framedata_path()
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return load_framedata(raw)


# ---------------------------------------------------------------------------
# 2) 핵심: 확정 반격 판정
# ---------------------------------------------------------------------------

@dataclass
class PunishResult:
    move: Move
    margin: int          # 창 - 발생. 클수록 여유. 0이면 칼타이밍 확정.

    def __str__(self) -> str:
        tag = "칼타이밍" if self.margin == 0 else f"여유 {self.margin}F"
        return f"{self.move.name} (발생 {self.move.startup}F, {tag})"


def find_punishers(
    defender_moves: dict[str, Move],
    window: int,
    *,
    exclude_followups: bool = True,
    exclude_air: bool = True,
    exclude_throws: bool = True,
    exclude_special: bool = True,
    distance: Optional[float] = None,   # 영상에서 추정한 두 캐릭 거리(있으면 리치로 2차 필터)
    max_results: int = 8,
) -> list[PunishResult]:
    """
    window = 공격자가 가드당해 노출된 프레임 수(punishable_window).
    방어자의 기술 중 startup <= window 인 '타격기'가 '확정' 반격 후보.
    빠른 순 정렬.

    exclude_air=True    : 공중기/공중던지기(j.~)는 지상 가드 직후 즉시 못 쓰므로 제외.
    exclude_throws=True : 던지기는 가드백(거리) 의존이라 별도 취급 -> find_throw_options() 로.
    exclude_special=True: 반격기/패리/가드캔슬(예: Baiken 236P Hiiragi)은 콤보·확정기 아님.
    """
    results: list[PunishResult] = []
    for mv in defender_moves.values():
        if mv.startup is None:
            continue
        if exclude_followups and mv.is_followup:
            continue
        if exclude_air and mv.is_air:
            continue
        if exclude_throws and mv.is_throw:
            continue
        if exclude_special and mv.is_special:
            continue
        # 거리 2차 필터(훅): 거리·리치가 둘 다 알려졌고 리치가 모자라면 제외.
        #   현재 mv.reach 는 항상 None(Dustloop 데이터에 리치 없음)이라 사실상 무동작.
        #   영상에서 거리를, 별도 소스에서 리치를 채우면 그 즉시 작동한다.
        if distance is not None and mv.reach is not None and mv.reach < distance:
            continue
        if mv.startup <= window:
            results.append(PunishResult(move=mv, margin=window - mv.startup))
    results.sort(key=lambda r: (r.move.startup, r.move.name))
    return results[:max_results]


def find_throw_options(
    defender_moves: dict[str, Move],
    window: int,
    *,
    max_results: int = 4,
) -> list[PunishResult]:
    """
    발생상 창에 들어오는 '지상 던지기'를 따로 추린다(공중던지기는 제외).
    주의: 발생이 창 안이어도 가드백(블록백) 거리 때문에 실제로 닿는지는 영상의
    거리 정보가 있어야 확정된다. 그래서 '조건부'로만 제시한다.
    """
    results: list[PunishResult] = []
    for mv in defender_moves.values():
        if mv.startup is None or not mv.is_throw or mv.is_air:
            continue
        if mv.startup <= window:
            results.append(PunishResult(move=mv, margin=window - mv.startup))
    results.sort(key=lambda r: (r.move.startup, r.move.name))
    return results[:max_results]


def analyze_blocked_move(
    data: dict[str, dict[str, Move]],
    attacker: str,
    move_name: str,
    defender: str,
) -> dict:
    """
    "공격자가 move_name 을 썼고 방어자가 가드함" 상황을 통째로 분석.
    영상 분석부가 (attacker, move_name, defender) 만 알아내면 이걸 호출하면 된다.
    """
    if attacker not in data or move_name not in data[attacker]:
        return {"error": f"프레임 데이터 없음: {attacker} / {move_name}"}
    if defender not in data:
        return {"error": f"프레임 데이터 없음: {defender}"}

    atk_move = data[attacker][move_name]
    window = atk_move.punishable_window

    if window is None:
        return {
            "attacker": attacker, "move": move_name,
            "on_block": atk_move.on_block,
            "punishable": False,
            "verdict": f"{move_name}는 가드 시 {atk_move.on_block:+}F → 확정 반격 없음(안전).",
            "punishers": [],
        }

    punishers = find_punishers(data[defender], window)
    throw_options = find_throw_options(data[defender], window)

    if punishers:
        best = punishers[0]
        verdict = (f"{attacker}의 {move_name}는 가드 시 {atk_move.on_block:+}F. "
                   f"{defender}는 최속 {best.move.name}({best.move.startup}F)로 확정 반격 가능.")
    elif throw_options:
        verdict = (f"{attacker}의 {move_name}는 -{window}F지만, "
                   f"{defender}에겐 확정 타격 반격이 없음. 근접이면 던지기 반격은 가능.")
    else:
        verdict = (f"{attacker}의 {move_name}는 -{window}F지만, "
                   f"{defender}에겐 {window}F 이내 반격기가 없어 확정 반격 불가.")

    return {
        "attacker": attacker, "move": move_name, "defender": defender,
        "on_block": atk_move.on_block, "punishable": True, "window": window,
        "verdict": verdict,
        "punishers": [str(p) for p in punishers],
        # 던지기는 가드백 거리에 따라 닿을 수도/헛잡일 수도 있어 '조건부'로 분리 제시
        "throw_options": [str(t) for t in throw_options],
        "throw_note": ("가드백 거리에 따라 닿으면 확정/안 닿으면 헛잡. "
                       "영상의 거리 정보로 확정 여부를 가려야 함."),
    }


# ---------------------------------------------------------------------------
# 3) 샘플 데이터 (검증용) — 풀 데이터는 스크래퍼 JSON으로 교체
#    아래 수치는 GGST 실제값 기반의 데모용 슬라이스다.
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "Sol": {
        "5P":  {"startup": 4,  "onBlock": -1, "guard": "All", "recovery": 9},
        "2K":  {"startup": 6,  "onBlock": -2, "guard": "Low", "recovery": 10},
        "c.S": {"startup": 7,  "onBlock": -2, "guard": "All", "recovery": 14},
        "5K":  {"startup": 8,  "onBlock": -3, "guard": "All", "recovery": 13},
        "6H":  {"startup": 9,  "onBlock": -27, "guard": "All", "recovery": 43},
        "f.S": {"startup": 9,  "onBlock": -7, "guard": "All", "recovery": 22},
    },
    "Faust": {
        "5P":  {"startup": 6,  "onBlock": -2, "guard": "All", "recovery": 11},
        "2K":  {"startup": 7,  "onBlock": -4, "guard": "Low", "recovery": 12},
        "c.S": {"startup": 8,  "onBlock": -1, "guard": "All", "recovery": 15},
        "6H":  {"startup": 25, "onBlock": -34, "guard": "High", "recovery": 44},
    },
    "May": {
        "5P":  {"startup": 5,  "onBlock": -1, "guard": "All", "recovery": 10},
        "2P":  {"startup": 6,  "onBlock": -2, "guard": "All", "recovery": 9},
        "c.S": {"startup": 7,  "onBlock": -3, "guard": "All", "recovery": 14},
        "6H":  {"startup": 28, "onBlock": -16, "guard": "All", "recovery": 36},
    },
}


def _demo():
    # 실데이터(framedata_ggst.json)가 있으면 그걸로, 없으면 샘플로 폴백
    data = load_framedata_file()
    if data is not None:
        print(f"[실데이터 로드: 캐릭터 {len(data)}명]")
        sol = "Sol Badguy"
    else:
        print("[framedata_ggst.json 없음 -> 샘플 데이터로 동작]")
        data = load_framedata(SAMPLE_DATA)
        sol = "Sol"
    print("=" * 64)
    print("DEMO: 가드 후 확정 반격 판정 엔진")
    print("=" * 64)

    cases = [
        (sol, "6H", "May"),       # -27, 확정 빵빵
        (sol, "f.S", "May"),      # -7, 일부만
        ("Faust", "6H", sol),     # 오버헤드, 확정
        (sol, "2K", "May"),       # -2, 확정 거의 없음
    ]
    for atk, mv, dfd in cases:
        r = analyze_blocked_move(data, atk, mv, dfd)
        print(f"\n[{atk} {mv} 가드 → {dfd} 반격]")
        print("  판정:", r["verdict"])
        if r.get("punishers"):
            print("  타격 확정:")
            for p in r["punishers"]:
                print("   -", p)
        if r.get("throw_options"):
            print(f"  던지기(조건부 — {r['throw_note']}):")
            for t in r["throw_options"]:
                print("   ~", t)

    print("\n" + "-" * 64)
    print("주의: 이 엔진은 '발생 프레임 ≤ 노출 프레임'만 본다. 실제 확정은")
    print("거리/밀려남(pushback)/리치/무적에 좌우되므로, 영상 분석부에서 거리를")
    print("같이 넘겨 2차 필터링하는 게 다음 정밀화 단계다.")


if __name__ == "__main__":
    _demo()
