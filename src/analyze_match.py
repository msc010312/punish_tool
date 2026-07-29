"""
analyze_match.py  —  timeline.json + 프레임데이터 -> 사람이 읽는 코칭 리포트

①번(HUD 리더)이 뽑은 timeline.json 의 이벤트(피격 / COUNTER / PUNISH / JUST + 프레임이득)를
②번(punish_engine)의 프레임데이터와 묶어, "이 순간 뭘 할 수 있었나"를 자연어로 풀어준다.

핵심 연결:
  카운터/펀ish는 프레임 이득(+13/+18)을 준다. 그 이득 = 후속 연결창(window)이므로
  punish_engine.find_punishers(때린_캐릭_기술, window) 가 "이 캐릭이 그 이득으로 넣을 수
  있는 기술"을 그대로 계산해준다.

때린 쪽/맞은 쪽 판정:
  COUNTER/PUNISH 텍스트는 '맞은 쪽'에 가깝게 뜬다. 그래서 같은 시각의 HP 감소(피격) 쪽을
  교차검증해 victim(맞은 쪽)을 확정하고, attacker(때린 쪽)=반대편으로 잡는다. HP 단서가
  없으면 텍스트 측을 victim 으로 가정한다.

사용:
  python analyze_match.py timeline.json --p1 "Sol Badguy" --p2 "Baiken"
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import punish_engine as pe
import move_id
import move_names

OTHER = {"P1": "P2", "P2": "P1"}

# 크롭/검출 신뢰불가 캐릭 — 분석·수집에서 제외 (Bedman=딜라일라 컴패니언을 크롭함)
EXCLUDE_CHARS = {"Bedman"}

# 소환수/분신이 본체와 별개로 공격하는 캐릭 — 크롭·귀속이 본체를 놓칠 수 있어 저신뢰 취급.
# (Zato=에디, Jack-O=하인, Venom=볼. Bedman은 아예 제외) 무브ID는 수행하되 확신 게이트를 높인다.
COMPANION_CHARS = {"Zato-1", "Jack-O", "Venom"}

# 후보 생성 레시피 v2 (골드 74건 캘리브레이션: recall 15%->57%)
#  - 실측뎀은 기본뎀의 ~0.88배(방어계수 등) -> 보정
#  - 데미지 top-5(tol 0.4, 공중·던지기 포함) ∪ 캐릭 빈도 프라이어 top-7(수동라벨 분포)
DMG_CORR = 0.88
_PRIOR_CACHE: dict = {}


def candidates_for(data: dict, char: str, measured_dmg: float | None,
                   counter_level: str | None = None,
                   victim_char: str | None = None, victim_hp: float | None = None,
                   victim_risc: float | None = None) -> list[str]:
    """이벤트의 무브 후보 무브명 리스트 (데미지 ∪ 프라이어).
    victim 정보를 주면 Defense·Guts·RISC 정밀 역산, 없으면 옛 전역 보정(DMG_CORR)."""
    names: list[str] = []
    if measured_dmg is not None and char in data:
        if victim_char:
            base_est = pe.unscale_damage(measured_dmg * pe.HEALTH, victim_char, victim_hp, victim_risc)
        else:
            base_est = measured_dmg * pe.HEALTH / DMG_CORR
        names = [m.name for m, _ in pe.identify_move(
            data[char], base_est, counter_level=counter_level,
            tol=0.4, max_results=5, include_air=True, include_throw=True)]
    if not _PRIOR_CACHE:
        p = pe.app_dir() / "starter_prior_ggst.json"
        _PRIOR_CACHE.update(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"_": []})
    names += (_PRIOR_CACHE.get(char) or [])[:7]
    return list(dict.fromkeys(names))


def load_timeline(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _combo_ok(c: dict) -> bool:
    """스크랩 잔해 걸러내기 — 위키 마크업/산문이 recipe 로 들어온 항목은 표시 금지."""
    r = (c.get("recipe") or "").strip()
    if not r or "=" in r or len(r) > 90:
        return False
    return (">" in r) or ("~" in r)          # 콤보 표기가 아니면 제외


def load_combos() -> dict:
    """Dustloop 초보 콤보 (scrape_combos.py 산출). 시동기별 콤보 루트."""
    p = pe.app_dir() / "combos_ggst.json"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {ch: [c for c in cs if _combo_ok(c)] for ch, cs in raw.items()}
    except Exception:
        return {}


def needs_tension(recipe: str) -> bool:
    """RC/각성기 포함 콤보 = 텐션 50% 필요."""
    r = recipe or ""
    return (any(tok in r for tok in ("RRC", "YRC", "PRC", "BRC", "RC"))
            or any(sup in r for sup in ("236236", "214214", "632146", "641236")))


def corner_required(c: dict) -> bool:
    """'코너/벽 시작' 전용 콤보인가. 영상에서 위치를 모르므로 이런 콤보는 뒤로 빼고 경고.
    (Midscreen to Corner=미드에서 시작 -> 전용 아님. Corner/Cornered/Back to wall=전용)"""
    p = (c.get("position") or "").strip().lower()
    if not p:
        return False
    return (p.startswith(("corner", "cornered", "back", "near corner", "(near) corner",
                          "(close to) corner", "almost corner", "close to corner",
                          "outside corner", "point-blank back", "point blank back"))
            or "in corner" in p or "cornered" in p)


def combos_for(combos: dict, char: str, move_names: list[str]) -> list[dict]:
    """그 캐릭 콤보 중 시동기가 move_names 에 포함된 것.
    정렬: 위치무관/미드 콤보 먼저(데미지순), 코너전용은 뒤로 -> 위치 모를 때 오추천 방지."""
    out = [c for c in combos.get(char, []) if any(s in move_names for s in c.get("starters", []))]
    out.sort(key=lambda c: (corner_required(c), -c.get("damage", 0)))
    return out


def best_combo_damage(combos: dict, char: str, move_names: list[str]) -> int:
    """그 시동기에서 가능한 최고 데미지(전환 분석 기준). 없으면 0.
    위치를 모르므로 '코너 시작 전용'은 제외 — 코너 콤보를 최적이라 하면 '더 살릴 수
    있었다'가 오도가 된다(미드에선 불가능한 데미지)."""
    cs = [c.get("damage", 0) for c in combos.get(char, [])
          if any(s in move_names for s in c.get("starters", [])) and not corner_required(c)]
    return max(cs, default=0)


def converted_damage(events: list[dict], t: float, victim: str) -> float:
    """t의 카운터/펀ish 이후 victim이 '연속 콤보'로 잃은 총 HP% (전환 데미지)."""
    hits = sorted([h for h in events if h["kind"] == "hit" and h["side"] == victim
                   and 0 <= h["t"] - t <= 4.0], key=lambda h: h["t"])
    total, last = 0.0, t
    for h in hits:
        if h["t"] - last > 1.2:          # 1.2s 넘게 비면 콤보 종료
            break
        total += h.get("damage_pct", 0.0)
        last = h["t"]
    return total


def refine_by_combo(combo_db: dict, char: str, cand_moves: list[str], observed_total: float,
                    tol: float = 0.82) -> tuple[list[str], dict]:
    """관측 콤보 총데미지로 시동기 후보를 거른다.
    플레이어가 낸 데미지(observed_total)는 그 시동기의 '최대 루트 데미지'를 못 넘으므로,
    max_route < observed*tol 인 시동기는 물리적으로 불가능 -> 제거. (단발/짧은 콤보면 필터 안 됨)
    반환: (남은 후보, {move: max_route_dmg})."""
    scores = {m: best_combo_damage(combo_db, char, [m]) for m in cand_moves}
    if observed_total <= 1:                      # 데미지 신호 약함 -> 거르지 않음
        return cand_moves, scores
    kept = [m for m in cand_moves
            if scores[m] == 0 or scores[m] >= observed_total * tol]  # DB없으면 보류(유지)
    return (kept or cand_moves), scores


def observed_combo(events: list[dict], t: float, victim: str) -> tuple[float, int]:
    """t의 카운터/펀ish 이후 victim이 '연속 콤보'로 잃은 (총 절대데미지, 히트수)."""
    hits = sorted([h for h in events if h["kind"] == "hit" and h["side"] == victim
                   and 0 <= h["t"] - t <= 4.0], key=lambda h: h["t"])
    total, cnt, last = 0.0, 0, t
    for h in hits:
        if h["t"] - last > 1.2:
            break
        total += h.get("damage_pct", 0.0); cnt += 1; last = h["t"]
    return total * pe.HEALTH, cnt


def nearest_hit_victim(events: list[dict], t: float) -> str | None:
    """victim(맞은 쪽) = 카운터/펀ish 직후 '시작하는 콤보'의 대상 측.
    카운터/펀ish는 콤보 시동이므로, t 직후 시작하는 콤보(연속 피격 묶음)의 side 가 victim.
    텍스트가 뜬 쪽은 attacker/victim 신호로 신뢰 불가(검증 완료) — HP 감소가 ground truth.
    attacker = 그 반대편."""
    cand = [c for c in group_combos(events)
            if -0.3 <= c["start_t"] - t <= 1.2 and c["dmg"] > 0.02]
    if not cand:
        return None
    cand.sort(key=lambda c: abs(c["start_t"] - t))
    return cand[0]["side"]


def group_combos(events: list[dict], gap: float = 1.0) -> list[dict]:
    """연속 피격(같은 victim, 시간 간격<=gap)을 한 콤보로 묶는다."""
    hits = sorted([e for e in events if e["kind"] == "hit"], key=lambda e: e["t"])
    combos: list[dict] = []
    cur: dict | None = None
    for h in hits:
        if cur and h["side"] == cur["side"] and h["t"] - cur["last_t"] <= gap:
            cur["hits"] += 1
            cur["last_t"] = h["t"]
            cur["end_hp"] = h["hp_to"]
        else:
            if cur:
                combos.append(cur)
            cur = dict(side=h["side"], start_t=h["t"], last_t=h["t"], hits=1,
                       start_hp=h["hp_from"], end_hp=h["hp_to"])
    if cur:
        combos.append(cur)
    for c in combos:
        c["dmg"] = max(0.0, c["start_hp"] - c["end_hp"])
    return combos


def estimate_rounds(events: list[dict]) -> int:
    """라운드 리셋 = HP가 '저체력(<0.5)'에서 '거의 풀피(>0.85)'로 돌아오는 지점.
    (작은 측정 노이즈로 가짜 리셋이 잡히지 않게 엄격히 — hysteresis)."""
    resets = 0
    was_low = {"P1": False, "P2": False}
    for e in sorted([e for e in events if e["kind"] == "hit"], key=lambda e: e["t"]):
        s = e["side"]
        if e["hp_to"] < 0.12:      # 실제 KO 근처만 (0.5는 리플레이 UI 깜빡임에 가짜 리셋 발생)
            was_low[s] = True
        elif e["hp_from"] > 0.85 and was_low[s]:
            resets += 1
            was_low[s] = False
    return resets + 1


def followups(data: dict, char: str, window: int) -> tuple[list[str], list[str]]:
    """+window 이득으로 연결 가능한 기술을 (빠른 기본기, 특수기)로 나눠 반환.
    특수기 판정: 입력에 모션 숫자 3개 이상(236/214/623 등) = 커맨드 기술.
    (주의: '프레임상 연결'만 본 것 — 실제 콤보 루트/게이팅은 아직 모름)"""
    if char not in data:
        return [], []
    res = pe.find_punishers(data[char], window, max_results=25)
    normals, specials = [], []
    for p in res:
        if sum(c.isdigit() for c in p.move.name) >= 3:
            specials.append(str(p))
        else:
            normals.append(str(p))
    return normals[:4], specials[:3]


def fmt_time(t: float) -> str:
    return f"{int(t)//60}:{t%60:05.2f}"


def first_hit(events: list[dict], t: float, victim: str, ev: dict | None = None) -> dict | None:
    """t 직후 victim의 '첫 피격' 이벤트. 시동기 데미지(damage_pct)와
    히트 직전 HP(hp_from — Guts 역산에 필요)를 함께 준다.
    ev(카운터/펀ish 이벤트)에 60fps 정밀 재판독 값(fh_damage)이 있으면 그걸 우선
    — 15Hz 스캔의 다단히트 뭉개짐(측정 2~4배 과대)을 피한다."""
    if ev and ev.get("fh_damage") is not None:
        return {"damage_pct": ev["fh_damage"], "hp_from": ev.get("fh_hp_from", 1.0),
                "risc": ev.get("fh_risc")}
    cand = [h for h in events if h["kind"] == "hit" and h["side"] == victim and 0 <= h["t"] - t <= 0.4]
    if not cand:
        return None
    return min(cand, key=lambda h: h["t"])


def first_hit_damage(events: list[dict], t: float, victim: str) -> float | None:
    """t 직후 victim의 '첫 피격' 데미지(HP%). 카운터/펀ish 시동기의 데미지 ≈ 이 값."""
    h = first_hit(events, t, victim)
    return h.get("damage_pct", 0.0) if h else None


def missed_super_punishes(events: list[dict]) -> list[tuple[float, str]]:
    """슈퍼 플래시 후 결과 판정 (무브ID 불필요, HUD 타이밍 기반).
      - 직후 0.8s 내 큰 피격(>=15%) = 슈퍼 적중 -> 펀니쉬 상황 아님 (스킵)
      - 0.4~2.5s 내 피격 = 가드/헛친 슈퍼를 펀니쉬함 (punished)
      - 아무 피격 없음 = 가드/헛침인데 반격 못함 = MISSED (큰 기회 놓침)
    반환: [(t, 'missed'|'punished'), ...]"""
    supers = [e for e in events if e.get("kind") == "super"]
    hits = [e for e in events if e.get("kind") == "hit"]
    out = []
    for sp in supers:
        t = sp["t"]
        if any(0 <= h["t"] - t <= 0.8 and h.get("damage_pct", 0) >= 0.15 for h in hits):
            continue                                      # 슈퍼 적중
        punished = any(0.4 < h["t"] - t <= 2.5 for h in hits)
        out.append((t, "punished" if punished else "missed"))
    return out


def attacker_of(events: list[dict], e: dict) -> str:
    """카운터/펀ish를 '때린' 쪽(attacker).
    victim = t 근처에서 HP 깎인 쪽(미러전도 HP바는 P1/P2 구분됨) -> attacker = 반대.
    just_defense는 막은 쪽(side)."""
    if e["kind"] in ("counter", "punish"):
        near = [h for h in events if h["kind"] == "hit" and -0.3 <= h["t"] - e["t"] <= 0.6]
        if near:
            near.sort(key=lambda h: abs(h["t"] - e["t"]))
            return OTHER[near[0]["side"]]           # HP 깎인 쪽의 반대 = 때린 쪽
        if e.get("victim"):                          # hit 이벤트 없어도 HP델타로 본 피격측
            return OTHER[e["victim"]]
        return OTHER[e["side"]]                      # 폴백: 텍스트측=victim 가정
    return e["side"]


def attacker_confident(events: list[dict], e: dict) -> tuple[str | None, bool]:
    """attribution이 명확한지 판정. VLM은 누가 공격자인지 못 가리므로(순응 편향) HP로만 판단.
    한쪽만 명확히 피격 -> (attacker, True). 양쪽 피격(트레이드)·피격없음 -> 불확실 -> 라벨 스킵.
    이게 Unika 피해자 오라벨/캐릭 오attribution의 근본 방지책."""
    if e["kind"] not in ("counter", "punish"):
        return e["side"], True
    near = [h for h in events if h["kind"] == "hit" and -0.3 <= h["t"] - e["t"] <= 0.6]
    if not near:
        # hit 이벤트 없음 -> 배너-확인된 HP 델타로 판정(빈사·경타·플래시로 hit 누락분 회복).
        v, d, od = e.get("victim"), e.get("hp_drop", 0.0), e.get("other_drop", 0.0)
        if v and d >= 0.01:
            if od >= 0.02 and od >= 0.6 * d:        # 양쪽 유의미+비슷 -> 트레이드
                return None, False
            return OTHER[v], True                    # 피격측의 반대 = 공격자
        return OTHER[e["side"]], False              # HP 변화 없음(팬텀) -> 불확실
    near.sort(key=lambda h: abs(h["t"] - e["t"]))
    victim = near[0]["side"]
    vt = near[0]["t"]
    # 반대편도 근접(±0.25s)해서 맞았으면 트레이드/애매 -> 불확실
    if any(h["side"] == OTHER[victim] and abs(h["t"] - vt) <= 0.25 for h in near):
        return None, False
    return OTHER[victim], True


def enrich_vlm(events: list[dict], video: str, chars: dict, data: dict,
               max_cand: int = 5, progress_cb=None) -> None:
    """각 카운터/펀ish 이벤트에 로컬 VLM 식별 결과(e['vlm'])를 붙인다. 멱등(이미 있으면 스킵).
    데미지로 좁힌 후보가 2개+일 때만 호출(0~1개면 불필요). 서버/영상 없으면 조용히 패스 -> 데미지 폴백.
    e['vlm'] = {character, move, situation, confidence, note}.

    기본 OFF: 실배포 무브ID 정확도가 ~25%(5지선다 random 20%)로 검증돼, 단정하면 오정보가 됨.
    켜려면 환경변수 PUNISH_MOVEID=1. (motion 방식이 검증되면 기본 ON 복귀 예정)
    OFF면 리포트는 데미지 기반 후보 + 콤보만 제시 — 전부 실측 신뢰되는 정보."""
    import os
    if os.environ.get("PUNISH_MOVEID", "") != "1":
        return                                   # 무브ID 비활성(torch/DINOv2 로드도 안 함)
    if not video or not Path(video).exists():
        return
    try:
        import vlm_id
    except Exception:
        return
    if not vlm_id.available():                  # 서버/키 없으면 프레임추출 낭비 말고 종료
        return
    combo_db = load_combos()
    todo = [e for e in events if e["kind"] in ("counter", "punish") and "vlm" not in e]
    for i, e in enumerate(todo):
        attacker, conf = attacker_confident(events, e)
        if not conf:                             # attribution 애매(트레이드 등) -> 오라벨 방지 위해 스킵
            e["ambiguous"] = True
            continue
        ch = chars[attacker]
        if ch in EXCLUDE_CHARS:                  # 신뢰불가 캐릭(Bedman 등) 분석 제외
            e["excluded"] = True
            continue
        fh = first_hit(events, e["t"], OTHER[attacker], ev=e)
        md = fh.get("damage_pct", 0.0) if fh else None
        vic_hp = fh.get("hp_from") if fh else None
        vic_risc = fh.get("risc") if fh else None
        if md is None:                               # hit 이벤트 없음 -> 배너-확인 HP델타를 데미지로
            md = e.get("hp_drop") or None
        lv = {"large": "Large", "medium": "Mid"}.get(e.get("level", "")) if e["kind"] == "counter" else None
        # 후보 v2: 데미지(Defense·Guts·RISC 역산) ∪ 빈도 프라이어. max_cand로 캡.
        names = candidates_for(data, ch, md, lv, victim_char=chars[OTHER[attacker]], victim_hp=vic_hp,
                               victim_risc=vic_risc)
        cands = [{"char": ch, "move": m} for m in names if vlm_id.ref_path(ch, m)][:max_cand]
        if len(cands) < 2:                      # 0~1개면 이미 특정/비교불가 -> VLM 불필요
            continue
        frames = vlm_id.event_frames(video, e["t"])
        if vlm_id.is_burst(frames):             # 버스트(방어기)는 분석 대상 아님 -> 표시 후 제외
            e["burst"] = True
            continue
        kk = "카운터" if e["kind"] == "counter" else "펀니쉬"
        ctx = f"{kk} 시동기" + (f", 상대 HP {md*100:.0f}% 감소" if md is not None else "")
        # RAG 백엔드는 인덱스(dataset)와 동일한 공격자 크롭을 써야 정렬됨(신뢰도↑)
        if vlm_id.default_backend() == "rag":
            q = vlm_id.attacker_frames(video, e["t"], attacker) or frames
            r = vlm_id.identify(q, cands, ctx, attacker_char=ch, backend="rag")
        else:
            hud = vlm_id.hud_strip(video, e["t"])
            # 단일 후보채점 — confidence=margin(1등-2등 로그우도차, 보정됨: ≥0.2→67% ≥0.1→60%)
            r = vlm_id.identify(frames, cands, ctx, attacker_char=ch, hud=hud)
        if ch in COMPANION_CHARS and r.get("confidence", 0.0) < 0.2:  # 소환수는 고확신만 신뢰
            r["move"] = "unsure"
        e["vlm"] = r
        if progress_cb:
            progress_cb(i + 1, len(todo))


def build_report(tl: dict, p1: str, p2: str, data: dict, perspective: str = "all",
                 video: str | None = None, use_vlm: bool = True) -> str:
    """timeline + 캐릭터명 + 프레임DB -> 코칭 리포트 문자열. (GUI/CLI 공용)
    perspective: 'all'(둘 다) / 'P1' / 'P2' — 해당 플레이어가 '한' 것만 추려서 보여줌.
    video: VLM 식별용 영상경로(미지정시 tl['video']). use_vlm=False면 데미지 추정만."""
    events = tl["events"]
    chars = {"P1": p1, "P2": p2}
    if use_vlm:
        enrich_vlm(events, video or tl.get("video"), chars, data)
    combo_db = load_combos()
    persp_ok = lambda side: perspective == "all" or side == perspective
    lines: list[str] = []
    def out(s=""): lines.append(s)

    out("=" * 64)
    ptag = {"all": "P1+P2", "P1": f"P1 {p1}", "P2": f"P2 {p2}"}[perspective]
    out(f"코칭 리포트 [{ptag} 관점] — {tl.get('video','?')}")
    out(f"P1(좌)={p1}   P2(우)={p2}")
    out("=" * 64)

    # 캐릭터 이해 코너 — Dustloop 개요/고유시스템 (scrape_char_notes.py 산출물)
    notes_p = pe.app_dir() / "char_notes_ggst.json"
    if notes_p.exists():
        try:
            notes = json.loads(notes_p.read_text(encoding="utf-8"))
            out("\n[캐릭터 이해 — Dustloop]")
            for nm in dict.fromkeys([p1, p2]):
                info = notes.get(nm) or {}
                u = info.get("ko") or info.get("unique", "")   # 한국어 우선, 없으면 영어
                if u:
                    out(f"  ▷ {nm}: {u[:400]}")
                mech = info.get("ko_mech")                      # 고유 메커닉(있으면)
                if mech:
                    out(f"     └ 고유 메커닉: {mech}")
        except Exception:
            pass

    def cnt(kind, side=None):
        return sum(1 for e in events if e["kind"] == kind and not e.get("burst")
                   and (side is None or e["side"] == side))
    out("\n[요약]")
    out(f"  총 피격 이벤트: {cnt('hit')}  (P1 {cnt('hit','P1')} / P2 {cnt('hit','P2')})")
    out(f"  COUNTER: {cnt('counter')}   PUNISH: {cnt('punish')}   JUST(직전가드): {cnt('just_defense')}"
        f"   REVERSAL: {cnt('reversal_wake') + cnt('reversal_guard')}"
        f"(기상 {cnt('reversal_wake')}/가드경직 {cnt('reversal_guard')})")
    out(f"  (미상 텍스트 {cnt('text')}건은 OCR 불가로 제외)")

    combos = group_combos(events)
    real = [c for c in combos if c["hits"] >= 2]
    dealt = {"P1": 0.0, "P2": 0.0}
    for c in combos:
        dealt[OTHER[c["side"]]] += c["dmg"]
    out("\n[전투 흐름]")
    out(f"  추정 라운드 수: {estimate_rounds(events)}")
    out(f"  누적 가한 데미지(HP%합): {p1} {dealt['P1']*100:.0f}%  /  {p2} {dealt['P2']*100:.0f}%")
    top = [c for c in sorted(real, key=lambda c: c["dmg"], reverse=True) if persp_ok(OTHER[c["side"]])]
    out(f"  콤보(2히트+) {len(real)}개. 큰 콤보 Top 5:")
    for c in top[:5]:
        atk = chars[OTHER[c["side"]]]
        out(f"   {fmt_time(c['start_t'])}  {atk} -> {c['hits']}히트 / {c['dmg']*100:.0f}% 데미지")

    # 기술 사용 통계 (입력표시 영상 한정 — GUI 워커가 tl["move_stats"]로 전달)
    ms = tl.get("move_stats")
    if ms:
        out("\n[기술 사용 통계 — 입력 판독]")
        for side in ("P1", "P2"):
            rows_ = sorted((ms.get(side) or {}).items(), key=lambda kv: -kv[1]["n"])
            rows_ = [(k, v) for k, v in rows_ if v["n"] >= 2][:8]
            if not rows_ or not persp_ok(side):
                continue
            out(f"  {side}({chars[side]}):")
            for k, v in rows_:
                pct = v["hit"] / v["n"] * 100
                out(f"    {move_names.annotate(chars[side], k):26s} {v['n']:3d}회 시도 · 적중 {v['hit']}회 ({pct:.0f}%)")
        out("    * 입력 기준(경직 중 입력 포함) · 미적중은 가드/헛침 구분 없음")

    # 연계 실행 분석 — 기준은 외부 '최적표'가 아니라 본인 최속 기록(사실만 말한다)
    ls = tl.get("link_stats")
    if ls:
        out("\n[연계 실행 — 입력 간격 (기준: 본인 최속)]")
        for side in ("P1", "P2"):
            rows_ = (ls.get(side) or [])[:5]
            if not rows_ or not persp_ok(side):
                continue
            out(f"  {side}({chars[side]}):")
            for r_ in rows_:
                loss = r_["avg"] - r_["min"]
                note = "  ← 편차 큼, 연습 포인트" if loss >= 6 else ""
                out(f"    {r_['a']} → {r_['b']}: {r_['n']}회 · 평균 {r_['avg']:.0f}F"
                    f" (최속 {r_['min']:.0f}F ~ 최대 {r_['max']:.0f}F){note}")

    out("\n[카운터 · 확정반격 순간]")
    def _atk(e):
        return e.get("input_atk") or attacker_of(events, e)
    notable = [e for e in events if e["kind"] in ("counter", "punish")
               and not e.get("burst")            # 버스트(방어기) 제외
               and not e.get("ambiguous")        # attribution 애매(트레이드) 제외
               and (e.get("input_atk") or attacker_confident(events, e)[1])
               and chars.get(_atk(e)) not in EXCLUDE_CHARS
               and persp_ok(_atk(e))]
    if not notable:
        out("  (해당 관점의 카운터/펀ish 없음)")
    LV_ADV = {"very small": 0, "small": 6, "mid": 13, "large": 18}
    shown: list[tuple[str, str, float]] = []   # 동시(트레이드/양측배너) 중복 표시 방지
    for e in notable:
        attacker = _atk(e)
        atk_char = chars[attacker]
        if any(a == attacker and k == e["kind"] and abs(t0 - e["t"]) < 1.25 for a, k, t0 in shown):
            continue                           # 같은 순간·같은 공격자·같은 종류 -> 한 번만
        shown.append((attacker, e["kind"], e["t"]))
        # 무브 ID (데미지 기반 후보).
        # NOTE: 배너의 노란색 채움비율(peak_cov)은 카운터 '강도' 정보가 아니다(같은 배너).
        #       그걸 레벨로 쓰면 모든 카운터가 +13/+18F로 과대평가된다 -> 사용 금지.
        fh = first_hit(events, e["t"], OTHER[attacker], ev=e)
        md = fh.get("damage_pct", 0.0) if fh else None
        vic_hp = fh.get("hp_from") if fh else None       # 히트 직전 HP -> Guts 역산
        vic_risc = fh.get("risc") if fh else None        # 히트 직전 RISC -> 최대 2배 역산
        if md is None:                                   # hit 이벤트 없으면 배너-확인 HP델타 사용
            md = e.get("hp_drop") or None
        cands = []
        input_ok = False
        if e.get("input_move") and atk_char in data:
            # 입력표시(리플레이) 판독으로 확정된 시동기 — 데미지 추정보다 우선(검증 6/6).
            keys = e.get("input_moves") or [e["input_move"]]
            cands = [(data[atk_char][k], 0.0) for k in keys if k in data[atk_char]]
            input_ok = bool(cands)
        if not cands and md is not None and atk_char in data:
            # 방어측 Defense·Guts·RISC 역산으로 base 데미지 추정 -> 프레임데이터와 직접 비교.
            # (시동기=콤보 첫 히트라 콤보보정 없음)
            vic_char = chars.get(OTHER[attacker])
            base_est = pe.unscale_damage(md * pe.HEALTH, vic_char, vic_hp, vic_risc)
            cands = pe.identify_move(data[atk_char], base_est)
        # 콤보 구조로 후보 압축: 관측 콤보 데미지를 못 내는 시동기 제거(불가능)
        obs_dmg, obs_hits = observed_combo(events, e["t"], OTHER[attacker])
        combo_note = ""
        if cands and obs_dmg > 1 and not input_ok:
            names0 = [m.name for m, _ in cands]
            kept, _ = refine_by_combo(combo_db, atk_char, names0, obs_dmg)
            if 0 < len(kept) < len(names0):
                cands = [c for c in cands if c[0].name in kept]
                combo_note = (f"     └ 콤보 {obs_dmg:.0f}뎀/{obs_hits}히트 → 시동기 "
                              f"{'/'.join(kept)}만 가능 (약한 시동기 제외)")
        # 카운터 프레임이득: 후보 기술들의 counter 속성에서 도출(배너 아님).
        #   전원 일치 -> 단정 / 엇갈림 -> 범위표시 + 최소이득으로 보수적 추천 / 불명 -> 단정·추천 안 함
        adv = adv_max = None
        lvl_txt = ""
        if e["kind"] == "counter" and cands:
            lvls = [(m.counter or "").strip().lower() for m, _ in cands]
            advs = sorted({LV_ADV[l] for l in lvls if l in LV_ADV})
            if len(advs) == 1:
                adv = adv_max = advs[0]
                lvl_txt = next(l for l in lvls if l in LV_ADV).upper()
            elif advs:
                adv, adv_max = advs[0], advs[-1]         # 최소이득 기준(안 되는 콤보 추천 방지)
        dmg_names = [m.name for m, _ in cands[:4]]
        vlm = e.get("vlm") or {}
        # 확신 게이트(보정된 margin): ≥0.2=확정(정확도67%) / 0.1~0.2=유력(60%) / <0.1=단정 안 함.
        vconf = vlm.get("confidence", 0.0)
        vmove = vlm.get("move") if vconf >= 0.20 else None
        vpart = vlm.get("move") if 0.10 <= vconf < 0.20 else None
        vweak = vlm.get("move") if 0.05 <= vconf < 0.10 else None   # 낮은확신 참고 힌트
        if vmove in (None, "", "unsure"):
            vmove = None
        if vpart in ("", "unsure"):
            vpart = None
        if vweak in ("", "unsure"):
            vweak = None
        sit = vlm.get("situation", "")
        id_move = None
        if vmove:
            id_move = next((m for m in data.get(atk_char, {}).values() if m.name == vmove), None)
        elif len(cands) == 1:
            id_move = cands[0][0]
        if e["kind"] == "punish":
            label = "PUNISH(확정반격)"; advtxt = ""
        elif adv is None:
            label = "COUNTER"; advtxt = "  (프레임이득 불명 — 시동기 미확정)"
        elif adv == adv_max:
            label = f"COUNTER {lvl_txt}"; advtxt = f"  (+{adv}F)"
        else:
            label = "COUNTER"; advtxt = f"  (+{adv}~{adv_max}F, 시동기 미확정)"
        out(f"\n  {fmt_time(e['t'])}  {attacker}({atk_char})의 {label}{advtxt}")
        # 상황 경고 — 트레이드/가드/헛침이면 일방적 콤보 추천이 부적절
        combo_ok = sit not in ("trade", "blocked", "whiff")
        if sit == "trade":
            out("     ⚠ VLM: 트레이드(양쪽 동시 타격) — 일방적 콤보 상황 아님")
        elif sit == "blocked":
            out("     ⚠ VLM: 가드됨 — 히트 아님(카운터 텍스트 오인 가능)")
        elif sit == "whiff":
            out("     ⚠ VLM: 헛친 것으로 보임")
        # attribution 교차검증 — VLM이 본 캐릭이 분석값과 다르면 표시
        vc = vlm.get("character", "")
        if vc and atk_char and atk_char.split()[0].lower() not in vc.lower():
            out(f"     ⚠ VLM이 본 캐릭={vc} ≠ attribution={atk_char} — 사람 확인 필요")
        starter_names = []
        ann = lambda names: ' / '.join(move_names.annotate(atk_char, n) for n in names)
        if input_ok:
            # 리플레이 입력표시 판독으로 확정 — 추정 아님(검증 6/6). 근/원 S만 거리 미상.
            keys = [m.name for m, _ in cands]
            out(f"     · 시동기(입력 확인): {ann(keys)}")
            starter_names = keys
        elif vmove:                                    # margin≥0.2 -> 확정 표기(정확도~67%)
            dmg_ok = vmove in dmg_names
            tag = f"VLM 확신 {int(vconf*100)}" + ("·데미지일치" if dmg_ok else "")
            extra = f"  (데미지후보: {ann(dmg_names)})" if dmg_names and not dmg_ok else ""
            out(f"     · 시동기({tag}): {move_names.annotate(atk_char, vmove)}{extra}")
            starter_names = list(dict.fromkeys([vmove] + dmg_names))
        elif cands:                                  # 불확실 -> 단정하지 않고 후보 나열
            tag = "★특정" if len(cands) == 1 else "추정"
            mdtxt = f", 데미지 {md*pe.HEALTH:.0f}" if md is not None else ""
            if vpart:
                hint = f" · VLM 유력={move_names.annotate(atk_char, vpart)}(중간확신 {int(vconf*100)})"
            elif vweak:
                hint = f" · VLM 참고={move_names.annotate(atk_char, vweak)}(낮은확신 {int(vconf*100)})"
            else:
                hint = ""
            out(f"     · 시동기({tag}{mdtxt}): {ann(dmg_names)}{hint}")
            starter_names = list(dict.fromkeys(([vpart] if vpart else []) + dmg_names))
        if combo_note:
            out(combo_note)
        if starter_names and combo_ok:
            mc = combos_for(combo_db, atk_char, starter_names)
            # 텐션 필터: RC/각성기 콤보는 50% 필요 — 당시 텐션이 모자랐으면 뒤로(오추천 방지)
            ten = e.get("atk_tension")
            if ten is not None and ten < 0.48:
                mc = sorted(mc, key=lambda c: needs_tension(c.get("recipe", "")))
            uncertain = len(starter_names) > 1            # 시동기 미확정 -> 콤보는 조건부 참고
            for c in mc[:2]:                              # 위치무관 먼저, 코너전용은 뒤
                dtxt = f"{c['damage']}뎀 " if c.get("damage") else ""
                ptxt = f"({c['position']}) " if c.get("position") else ""
                dv = (c.get("difficulty") or "").strip()
                diff = f"[{dv}] " if dv and "|" not in dv and "=" not in dv else ""
                warn = "⚠코너 시작 전용 " if corner_required(c) else ""
                tn = ""
                if needs_tension(c.get("recipe", "")):
                    if ten is not None and ten < 0.48:
                        tn = f"⚠텐션 50% 필요(당시 {ten*100:.0f}%) "
                    else:
                        tn = "(텐션 50% 소모) "
                st = next((s for s in c.get("starters", []) if s in starter_names), "")
                cond = f"[{st}였다면] " if uncertain and st else ""
                out(f"     ▶ 콤보 {cond}{warn}{tn}{dtxt}{diff}{ptxt}{c['recipe']}")
            # 전환 분석: 실제로 콤보를 얼마나 살렸나 vs 최적
            opt = best_combo_damage(combo_db, atk_char, starter_names)
            if opt > 0:
                conv = converted_damage(events, e["t"], OTHER[attacker]) * pe.HEALTH
                if conv < opt * 0.55:
                    out(f"     ⚠ 전환 {conv:.0f}뎀 (최적 ~{opt}뎀) — 콤보를 더 살릴 수 있었어")
                else:
                    out(f"     ✓ 전환 {conv:.0f}뎀 (최적 ~{opt}뎀) — 잘 살림")
        # 프레임이득 후속기는 카운터에만(펀니쉬는 창 없음). 이득을 모르면 추천하지 않는다
        # — 안 되는 콤보를 "된다"고 하는 게 최악의 오정보.
        if e["kind"] == "counter":
            if adv is None:
                out("     └ (시동기를 특정 못 해 프레임이득 불명 — 후속기 추천 생략)")
            else:
                note = f" (최소이득 +{adv}F 기준)" if adv != adv_max else ""
                fn, fs = followups(data, atk_char, adv)
                if fn:
                    out(f"     └ 빠른 기본기{note}: " + ", ".join(fn))
                if fs:
                    out(f"     └ 콤보 특수기{note}: " + ", ".join(fs))
                # 확정 연결이 없으면 조용히 생략(반복되는 '없음' 줄은 노이즈)

    just = [e for e in events if e["kind"] == "just_defense" and persp_ok(e["side"])]
    if just:
        out("\n[직전가드(JUST) — 성공 시 텐션2배·퍼니쉬 쉬움]")
        for e in just:
            out(f"  {fmt_time(e['t'])}  {e['side']}({chars[e['side']]}) 직전가드 성공")

    # 버스트: 게이지 급락으로 감지. 버스트로 뜬 카운터는 위 분석에서 제외돼 있음.
    bursts = [e for e in events if e["kind"] == "burst" and persp_ok(e["side"])]
    if bursts:
        out("\n[사이크 버스트 사용]")
        for e in bursts:
            out(f"  {fmt_time(e['t'])}  {e['side']}({chars[e['side']]}) 버스트 사용"
                f"{' (카운터 성립)' if any(x.get('burst') and abs(x['t']-e['t'])<1.3 for x in events) else ''}")

    # 자원 코칭: 버스트 가득한데 치명 콤보를 그냥 맞은 순간 (버스트=콤보 탈출 자원)
    bm = [e for e in events if e["kind"] == "burst_missed" and persp_ok(e["side"])]
    if bm:
        out("\n[자원 코칭 — 버스트 아낀 대가]")
        for e in bm:
            out(f"  {fmt_time(e['t'])}  {e['side']}({chars[e['side']]}) 버스트 가득인데 "
                f"{e.get('hp_drop', 0)*100:.0f}% 콤보를 끝까지 맞음 — 치명 콤보엔 버스트 탈출 고려")

    # 리버설: side = 지른 쪽. 흰색=기상(reversal_wake), 주황=가드경직(reversal_guard).
    rev_evs = [e for e in events
               if e["kind"] in ("reversal_wake", "reversal_guard") and persp_ok(e["side"])]
    if rev_evs:
        out("\n[REVERSAL - 상대가 행동가능 첫 프레임에 필살기/무적기를 지름]")
        for e in sorted(rev_evs, key=lambda e: e["t"]):
            rev, opp = e["side"], OTHER[e["side"]]
            situ = "기상" if e["kind"] == "reversal_wake" else "가드경직 중"
            out(f"  {fmt_time(e['t'])}  {rev}({chars[rev]})가 {situ} 리버설  "
                f"-> {opp}({chars[opp]})는 압박 사이 방어/바이트(늦히트)로 대응")

    out("\n" + "-" * 64)
    out("표기: 숫자=방향키(236=파동승룡식)  >=이어서  ~=파생  66=대시  j.=점프  c.S/f.S=근/원S")
    out("      CH=카운터히트  CL=클린히트  dl=딜레이  WS=벽꽂기  RRC=레드 로만캔슬  en.=강화판")
    out("주의: 후속기 목록은 '발생<=프레임이득'만 본 1차 추정(근접 가정).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline", nargs="?", default="timeline.json")
    ap.add_argument("--p1", default="Sol Badguy")
    ap.add_argument("--p2", default="Baiken")
    ap.add_argument("--out", default="report.txt")
    ap.add_argument("--video", default=None, help="VLM 무브식별용 영상경로(있으면 사용)")
    args = ap.parse_args()

    tl = load_timeline(args.timeline)
    data = pe.load_framedata_file()
    if data is None:
        raise SystemExit("framedata_ggst.json 없음 — scrape_dustloop.py 먼저 실행")
    report = build_report(tl, args.p1, args.p2, data, video=args.video)
    print(report)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n리포트 저장 -> {args.out}")


if __name__ == "__main__":
    main()
