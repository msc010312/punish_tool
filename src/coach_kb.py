# -*- coding: utf-8 -*-
"""coach_kb.py — 로컬 지식베이스 근거 검색(오프라인, 항상 가능).

대화형 코치가 프레임·기술·메카닉 질문을 받을 때, LLM이 숫자를 지어내지 않도록
번들 JSON(framedata / mechanics / combos / char_notes / move_names_ko)에서
관련 항목만 뽑아 '참고 데이터' 블록으로 돌려준다. 데이터에 없으면 빈 문자열 →
LLM은 '모른다'고 답하게 프롬프트가 강제한다(프로젝트 제1원칙: 잘못된 정보 금지).

  retrieve(question, chars) -> str   # 근거 블록(한국어). 없으면 ""
"""
from __future__ import annotations

import functools
import json
import re

# 한글 캐릭터 별칭 -> 로스터 영문 키. 참고 데이터(로직 아님) — move_names_ko.json 과 같은 성격.
KR_CHAR = {
    "에이비에이": "A.B.A", "아바": "A.B.A", "아방": "A.B.A",
    "앤지": "Anji Mito", "안지": "Anji Mito",
    "아스카": "Asuka R", "아스카R": "Asuka R",
    "액슬": "Axl Low", "액설": "Axl Low",
    "바이켄": "Baiken",
    "베드맨": "Bedman", "베드": "Bedman",
    "브리짓": "Bridget", "브리지트": "Bridget",
    "칩": "Chipp Zanuff", "칩프": "Chipp Zanuff",
    "엘페르트": "Elphelt Valentine", "엘펠트": "Elphelt Valentine", "엘프": "Elphelt Valentine",
    "파우스트": "Faust",
    "지오반나": "Giovanna", "지오": "Giovanna",
    "골드루이스": "Goldlewis Dickinson", "골드": "Goldlewis Dickinson",
    "해피카오스": "Happy Chaos", "해카": "Happy Chaos", "hc": "Happy Chaos",
    "이노": "I-No",
    "잭오": "Jack-O", "잭오프": "Jack-O",
    "잼": "Jam Kuradoberi",
    "쟈니": "Johnny", "자니": "Johnny", "조니": "Johnny",
    "카이": "Ky Kiske", "카이키스크": "Ky Kiske",
    "레오": "Leo Whitefang",
    "루시": "Lucy",
    "메이": "May",
    "밀리아": "Millia Rage", "밀리": "Millia Rage",
    "나고리유키": "Nagoriyuki", "나고리": "Nagoriyuki", "나고": "Nagoriyuki",
    "포템킨": "Potemkin", "포템": "Potemkin",
    "디지": "Queen Dizzy", "퀸디지": "Queen Dizzy", "디즈": "Queen Dizzy",
    "람레살": "Ramlethal Valentine", "람레": "Ramlethal Valentine", "람": "Ramlethal Valentine",
    "램리썰": "Ramlethal Valentine", "램리": "Ramlethal Valentine", "램": "Ramlethal Valentine",
    "신": "Sin Kiske", "신키스크": "Sin Kiske",
    "슬레이어": "Slayer",
    "솔": "Sol Badguy", "솔배드가이": "Sol Badguy",
    "테스타먼트": "Testament", "테스타": "Testament",
    "유니카": "Unika",
    "베놈": "Venom",
    "재토": "Zato-1", "자토": "Zato-1", "재토원": "Zato-1",
}

# 한글 메카닉 키워드 -> Dustloop Mechanics 섹션명(부분일치)
MECH_KW = {
    "로망캔슬": "Roman Cancel", "로망": "Roman Cancel", "rc": "Roman Cancel",
    "버스트": "Burst", "사이크버스트": "Burst", "psych": "Burst",
    "디플렉트": "Deflect Shield", "쉴드": "Deflect Shield", "실드": "Deflect Shield",
    "가드": "Guard", "블로킹": "Guard", "방어": "Guard",
    "페이스트": "Faultless Defense", "fd": "Faultless Defense", "폴트리스": "Faultless Defense",
    "위글": "Wild Assault", "와일드어썰트": "Wild Assault",
    "텐션": "Tension", "긴장": "Tension",
    "리스크": "R.I.S.C.", "risc": "R.I.S.C.",
    "카운터": "Counter Hit", "카운터히트": "Counter Hit",
    "대쉬": "Movement", "이동": "Movement", "잔상": "Movement",
    "커맨드": "Notation", "표기": "Notation",
}

_BTN = "PKSHD"

# GGST 유니버설 개틀링/캔슬 규칙 (출처: Dustloop GGST/Mechanics — Gatling Combination.
# 위키 원문을 한국어로 요약. 캐릭터별 예외 존재 가능성까지 원문 그대로 명시.)
GATLING_RULES_KO = (
    "지상: 5P·2P → 서로/자기자신/커맨드노말 | 5K·2K → D(더스트·2D)와 커맨드노말 | "
    "근접S(c.S) → S·H·D·커맨드노말 | 원거리S(f.S)·2S → H | "
    "공중: j.P→j.P, j.K→j.D, j.S→j.H·j.D, j.H→j.D. "
    "커맨드노말(6P/6S/6H 등)로의 개틀링은 위 규칙대로 가능하나 캐릭터별 예외가 있다. "
    "개틀링은 늦춰 눌러도 이어진다(딜레이 개틀링=프레임트랩). "
    "개틀링이 합법이어도 실전에서 안 이어지는 흔한 이유: ①히트 거리가 멀어 푸시백으로 "
    "다음 기술이 헛침 ②상대가 앉아 있어 타점 높은 기술이 헛침 ③히트스탑 중 너무 일찍 입력."
)

# cancel 필드 문자 해독(Dustloop MoveData_GGST)
_CANCEL_KO = {"S": "필살기", "J": "점프", "D": "대시", "R": "RRC", "P": "PRC",
              "N": "일반기", "B": "버스트"}

# 콤보/연결 질문 신호 -> 개틀링 규칙 블록 주입
COMBO_KW = ("이어", "연결", "개틀링", "게틀링", "콤보가 안", "콤보 안", "안 나가",
            "캔슬", "끊겨", "끊기", "헛치", "빗나", "안 맞", "connect", "gatling", "drop")

# GGST 랭크 시스템(2025.08 추가된 랭크매치 + 구 타워) — 출처: 공식/커뮤니티 위키
RANKS_KO = (
    "GGST 랭크매치(2025.8 도입, 캐릭터별 랭크): Iron→Bronze→Silver→Gold→Platinum→"
    "Diamond→Master(각 3단계) → 최고 랭크 Vanquisher. Vanquisher 안에는 "
    "Ignis·Virtus·Vindex(빈덱스) 세 등급이 있고 듀얼 레이트(DR)로 경쟁한다"
    "(DR 1600+면 최상위권). 즉 빈덱스=Vanquisher 내 등급으로 아주 높은 랭크. "
    "구 시스템은 타워 1~10층+천상계(Celestial). "
    "puddle.farm은 서드파티 GGST 전적·레이팅 추적 사이트(자체 레이팅 제공)다."
)
RANK_KW = ("랭크", "티어", "층", "천상", "셀레", "celestial", "뱅퀴셔", "vanquisher",
           "이그니스", "비르투스", "빈덱스", "vindex", "ignis", "virtus",
           "듀얼레이트", "듀얼 레이트", "dr ", "퍼들", "puddle", "전적", "레이팅", "승급")

# 한국 격겜 유저 용어 사전 — 질문에 등장하면 해당 정의를 근거로 주입
SLANG_KO = {
    "승룡": "승룡=623 커맨드 대공기 통칭(보통 무적 있는 대공 리버설. 솔은 볼카닉 바이퍼)",
    "무적기": "무적기=무적 프레임이 있는 기술(주로 리버설·각성기). 압박 탈출용, 가드되면 큰 확정반격 당함",
    "리버설": "리버설=기상/경직 풀리는 첫 프레임에 내는 기술. 주로 무적기",
    "f식": "F식=앉아가드 모션 중 점프공격을 겹쳐 가드 방향을 헷갈리게 하는 퍼지 셋업",
    "이지": "이지(선다)=두 가지를 동시에 못 막게 선택을 강요하는 심리",
    "딜캐": "딜캐=딜레이 캐치, 상대 기술 후딜에 확정 반격(=펀ish)",
    "짤짤이": "짤짤이=잽(P/K) 연타 견제",
    "히트확인": "히트확인=맞은 걸 보고 나서 콤보로 이어가는 것",
    "역가드": "역가드=상대 뒤쪽에서 공격이 나와 가드 방향이 반대가 되는 상황",
    "파해": "파해=상대 패턴/셋업을 깨는 대응법",
    "지르기": "지르기=리스크 감수하고 무적기/큰 기술을 내미는 것",
    "대공": "대공=공중 접근 요격(6P·승룡 등)",
    "바이트": "바이트=일부러 늦게 쳐서 상대 무적기/버스트를 헛치게 유도하는 것",
    "황로망": "황(옐로우)로망캔슬=가드 중 텐션 50%로 압박을 끊는 방어 옵션",
    "적로망": "적(레드)로망캔슬=히트 후 캔슬로 콤보 연장",
    "옵셀": "옵션셀렉트=한 입력으로 상황별 다른 기술이 나가게 하는 테크닉",
    "프레임트랩": "프레임트랩=일부러 틈을 만들어 상대 버튼을 유도해 카운터 내는 압박",
    "칼라": "칼라버스트=콤보 도중 정확한 타이밍의 사이크 버스트",
}

# 무적기 질문 신호 -> 캐릭터 invuln 컬럼 실조회(지어내지 않고 데이터로)
INVULN_KW = ("무적", "승룡", "리버설", "reversal", "dp")


def _paths():
    import punish_engine as pe
    d = pe.app_dir()
    return {
        "frame": d / "framedata_ggst.json",
        "mech": d / "mechanics_ggst.json",
        "combo": d / "combos_ggst.json",
        "notes": d / "char_notes_ggst.json",
        "kmoves": d / "move_names_ko.json",
    }


@functools.lru_cache(maxsize=1)
def _load() -> dict:
    out = {}
    for k, p in _paths().items():
        try:
            out[k] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out[k] = {}
    return out


def _looks_move(nm: str) -> bool:
    """'6S'처럼 기술표기 꼴인지(오탐 방지: 숫자+버튼 또는 j./c./f. 접두)."""
    import re as _re
    return bool(_re.fullmatch(r"(?:j\.)?(?:[cf]\.)?\d{0,6}[PKSHD]", nm))


def _norm_move(tok: str) -> str:
    """사용자 입력 기술표기 -> framedata 키 후보로 정규화. 예 '5hs'->'5H','j.s'->'j.S'."""
    t = tok.strip().replace(" ", "")
    t = re.sub(r"(?i)hs", "H", t)                 # HS -> H(길티 헤비 표기)
    # 근/원 접두 -> c./f.
    t = t.replace("근접", "c.").replace("근", "c.").replace("원거리", "f.").replace("원", "f.")
    t = re.sub(r"(?i)^j\.?", "j.", t)             # 점프 접두 통일
    # 버튼 문자만 대문자로
    t = re.sub(r"[pkshd]", lambda m: m.group().upper(), t)
    return t


def detect_chars(text: str, active=None) -> list[str]:
    """질문에 '명시적으로 언급된' 로스터 캐릭(영문 키). active는 무시 —
    매치 캐릭을 자동 포함하면 무관한 질문에도 개요가 딸려나오므로 언급된 것만."""
    fd = _load()["frame"]
    found = []
    low = text.lower()
    for kr, en in KR_CHAR.items():
        if kr.lower() in low and en not in found:
            found.append(en)
    for en in fd:
        toks = re.split(r"[\s.\-]+", en.lower())     # 이름/성 어느 토큰이든 등장하면
        if any(len(tk) >= 3 and tk in low for tk in toks) and en not in found:
            found.append(en)
    return found


def _move_line(mv: dict, kname: str = "") -> str:
    """framedata 한 기술 dict -> 한 줄 요약(값 있는 것만)."""
    parts = []
    for label, key in (("발생", "startup"), ("가드", "onBlock"),
                       ("가드타입", "guard"), ("데미지", "damage"), ("카운터", "counter")):
        v = mv.get(key)
        if v not in (None, "", "-"):
            parts.append(f"{label} {v}")
    c = (mv.get("cancel") or "").strip()
    if c:
        decoded = "/".join(_CANCEL_KO.get(ch, ch) for ch in c)
        parts.append(f"캔슬가능: {decoded}")
    tail = f" ({kname})" if kname else ""
    return ", ".join(parts) + tail


def retrieve(question: str, chars: dict | None = None, max_moves: int = 8) -> str:
    """질문에 대한 로컬 근거 블록. 없으면 ''. chars={'P1':..,'P2':..}(현재 매치)."""
    db = _load()
    fd, mech, combo, notes, km = (db["frame"], db["mech"], db["combo"],
                                  db["notes"], db["kmoves"])
    active = [c for c in (chars or {}).values() if isinstance(c, str)]
    who = detect_chars(question)          # 질문에 명시된 캐릭만
    L = []

    # 1) 기술 표기 -> 프레임데이터 (해당 캐릭 우선, 없으면 언급 캐릭 전체)
    raw_toks = re.findall(r"(?i)(?:j\.?)?(?:[cf]\.|근접|근|원거리|원)?\d{0,6}(?:hs|[pkshd])(?:~\S+)?", question)
    seen = set()
    move_hits = 0
    search_chars = who or active
    for tok in raw_toks:
        nm = _norm_move(tok)
        if len(nm) < 2:
            continue
        for ch in search_chars:
            moves = fd.get(ch, {})
            key = next((k for k in (nm, f"c.{nm}", f"f.{nm}") if k in moves), None)
            if key and (ch, key) not in seen:
                seen.add((ch, key))
                kname = km.get(ch, {}).get(key, "")
                L.append(f"- {ch} {key}: {_move_line(moves[key], kname)}")
                move_hits += 1
                if move_hits >= max_moves:
                    break
            elif key is None and ch in fd and (ch, nm) not in seen and _looks_move(nm):
                # 존재하지 않는 기술 = '없음'을 명시적 근거로(있다고 전제하고 떠드는 것 방지)
                seen.add((ch, nm))
                L.append(f"- {ch}에게 '{nm}' 기술은 데이터에 없음 — 존재하지 않는 기술일 "
                         f"가능성이 높으니 있다고 전제하지 말 것")
        if move_hits >= max_moves:
            break

    # 2) 캐릭터 개요(한글) + 대표 콤보 — 특정 기술을 안 물었을 때만(일반 질문)
    if who and move_hits == 0:
        for ch in who[:2]:
            ko = (notes.get(ch, {}) or {}).get("ko", "")
            if ko:
                L.append(f"- {ch} 개요: {ko}")
            cbs = [c for c in combo.get(ch, []) if c.get("recipe") and c.get("starters")]
            for c in cbs[:2]:
                dmg = f" ~{c['damage']}뎀" if c.get("damage") else ""
                L.append(f"- {ch} 콤보: {c['recipe']}{dmg}")

    # 2.5) 콤보/연결 질문 -> 유니버설 개틀링 규칙(진짜 시스템 지식) 주입
    low = question.lower()
    if any(k in low for k in COMBO_KW):
        L.append(f"- GGST 개틀링 규칙: {GATLING_RULES_KO}")

    # 2.6) 랭크/전적 얘기 -> 랭크 시스템 지식 주입(빈덱스 등 용어 이해)
    if any(k in low for k in RANK_KW):
        L.append(f"- GGST 랭크 지식: {RANKS_KO}")

    # 2.7) 유저 용어 -> 사전 정의 주입
    for term, desc in SLANG_KO.items():
        if term in low:
            L.append(f"- 용어: {desc}")

    # 2.8) 무적기 질문 -> 해당 캐릭 invuln 컬럼 실조회(추측 금지, 데이터로)
    if any(k in low for k in INVULN_KW) and who:
        for ch in who[:2]:
            inv = []
            for mkey, md in fd.get(ch, {}).items():
                v = str(md.get("invuln", "") or "").strip()
                v = re.sub(r"<[^>]+>", " ", v).strip()      # 위키 HTML 태그 제거
                if v and v not in ("-",):
                    inv.append(f"{mkey}(무적 {v})")
            if inv:
                L.append(f"- {ch} 무적 프레임 보유 기술: {', '.join(inv[:8])}")
            else:
                L.append(f"- {ch}: 데이터상 무적 프레임 기술 없음")

    # 3) 메카닉 섹션(키워드 매칭, 첫 매칭 1개, 길이 제한)
    sec_name = next((v for k, v in MECH_KW.items() if k.lower() in low), None)
    if sec_name:
        for s in mech.get("sections", []):
            if sec_name.lower() in s.get("name", "").lower():
                txt = re.sub(r"\s+", " ", s.get("text", "")).strip()[:700]
                L.append(f"- 메카닉 [{s['name']}]: {txt}")
                break

    return "\n".join(L)
