"""
calib_candidates.py — 후보 생성(데미지 매칭) 리콜 캘리브레이션.

발견: 사용자 골드 91건 중 후보에 정답이 있던 건 ~11%. 원인 규명·수정을 위해
골드 이벤트마다 (실측 HP%뎀) vs (프레임데이터 기본뎀) 비율 분포를 재고,
tol/후보수 N/비율보정 조합별 recall@N 을 측정한다. (CPU 전용 — 학습과 병행)
사용: python calib_candidates.py
"""
from __future__ import annotations
import json
import re
import statistics as st
from pathlib import Path

import punish_engine as pe
import analyze_match as am

ROOT = pe.app_dir()
TL = ROOT / "timelines"


def norm(s):
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def main():
    data = pe.load_framedata_file()
    rows = [json.loads(l) for l in (ROOT / "verify_pack/pack.jsonl").open(encoding="utf-8") if l.strip()]
    tls = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in TL.glob("*.json")}

    samples = []
    for r in rows:
        g = r.get("gold", "")
        if not g or g in ("bad", "MECH", "SKIP") or g.endswith("X") or "OR46D" in g:
            continue
        tl = tls.get(Path(r["video"]).stem)
        if not tl:
            continue
        events = tl["events"]
        victim = am.OTHER[r["attacker"]]
        md = am.first_hit_damage(events, r["t"], victim)
        if md is None:
            continue
        ch_moves = data.get(r["char"], {})
        mv = next((m for m in ch_moves.values() if norm(m.name) == norm(g)
                   or norm(m.name).startswith(norm(g) + "level")), None)
        if mv is None or not mv.damage:
            continue
        vic_ch = tl["p2"] if victim == "P2" else tl["p1"]
        samples.append({"char": r["char"], "gold_name": mv.name, "kind": r["kind"],
                        "measured": md * pe.HEALTH, "base": mv.damage,
                        "total": mv.damage_total or mv.damage, "victim": vic_ch,
                        "level": ""})
    print(f"캘리브레이션 샘플: {len(samples)}건")
    if not samples:
        return

    ratios = [s["measured"] / s["base"] for s in samples]
    rat_tot = [s["measured"] / s["total"] for s in samples]
    print(f"실측/기본뎀 비율: 중앙값 {st.median(ratios):.2f}  (사분위 "
          f"{sorted(ratios)[len(ratios)//4]:.2f}~{sorted(ratios)[3*len(ratios)//4]:.2f})")
    print(f"실측/총뎀 비율:   중앙값 {st.median(rat_tot):.2f}")

    # 레시피 격자: 보정계수 x tol x N x 공중/던지기 포함 -> recall@N
    print("\n보정  tol   N air | recall")
    med = st.median(ratios)
    for corr in (1.0, med):
        for tol in (0.25, 0.4, 0.6):
            for n in (4, 6, 8, 10):
                for air in (False, True):
                    hit = 0
                    for s in samples:
                        cands = pe.identify_move(data[s["char"]], s["measured"] / corr,
                                                 tol=tol, max_results=n,
                                                 include_air=air, include_throw=air)
                        if any(m.name == s["gold_name"] for m, _ in cands):
                            hit += 1
                    print(f"{corr:4.2f} {tol:4.2f} {n:2d}  {'O' if air else 'X'}  | "
                          f"{hit}/{len(samples)} = {hit/len(samples)*100:.0f}%")


def prior_recall():
    """레시피 v2: 데미지 top-k ∪ 캐릭별 빈도 프라이어 top-p (수동라벨 분포) -> recall."""
    import glob, os
    from collections import Counter
    data = pe.load_framedata_file()
    rows = [json.loads(l) for l in (ROOT / "verify_pack/pack.jsonl").open(encoding="utf-8") if l.strip()]
    tls = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in TL.glob("*.json")}
    # 캐릭별 시동기 프라이어(수동라벨 2846장 분포)
    prior: dict[str, Counter] = {}
    for p in glob.glob(str(ROOT / "dataset/*/*/*.png")):
        parts = Path(p).parts
        ch, mv = parts[-3].replace("_", " "), parts[-2]
        if ch == "  mechanics  ".strip():
            continue
        prior.setdefault(ch, Counter())[mv] += 1
    glob_prior = Counter()
    for c in prior.values():
        glob_prior.update(c)

    samples = []
    for r in rows:
        g = r.get("gold", "")
        if not g or g in ("bad", "MECH", "SKIP") or g.endswith("X") or "OR46D" in g:
            continue
        tl = tls.get(Path(r["video"]).stem)
        if not tl:
            continue
        md = am.first_hit_damage(tl["events"], r["t"], am.OTHER[r["attacker"]])
        if md is None:
            continue
        mv = next((m for m in data.get(r["char"], {}).values()
                   if norm(m.name) == norm(g) or norm(m.name).startswith(norm(g) + "level")), None)
        if mv:
            samples.append((r["char"], mv.name, md * pe.HEALTH))
    print(f"\n[v2] 샘플 {len(samples)} | 프라이어 캐릭 {len(prior)}")
    print("dmgN priN | recall  (보정0.88 tol0.4 공중포함)")
    for dn in (3, 4, 5):
        for pn in (3, 5, 7):
            hit = tot_n = 0
            for ch, gname, md in samples:
                dmg = [m.name for m, _ in pe.identify_move(data[ch], md / 0.88, tol=0.4,
                                                           max_results=dn, include_air=True, include_throw=True)]
                pri = [m for m, _ in (prior.get(ch) or glob_prior).most_common(pn)]
                cands = list(dict.fromkeys(dmg + pri))
                tot_n += len(cands)
                if gname in cands or any(norm(c) == norm(gname) for c in cands):
                    hit += 1
            print(f"  {dn}   {pn}  | {hit}/{len(samples)} = {hit/len(samples)*100:.0f}%  (평균후보 {tot_n/len(samples):.1f})")


if __name__ == "__main__":
    main()
    prior_recall()
