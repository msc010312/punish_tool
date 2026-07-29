"""
collect_verify.py — 사람 검수용 제안 수집 (골드 확장 파이프라인 v2)

기존 자동수집(만장일치 자동채택)과 달리, **모든 counter/punish 이벤트**를
HUD 포함 시트 + LoRA 제안(pre-fill)으로 뽑아 사람이 배치 검수하게 한다.
사람 답 = 골드. 적용된 수정: HUD명판·conf크롭·미러제외·236D차단·attribution필터·다단히트.

사용:  python collect_verify.py [--limit N(영상수)] [--per-video M]
출력:  verify_pack/<seq>_<char>_LoRA=<제안>.png + pack.jsonl (frames 경로·후보 포함)
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import cv2
import punish_engine as pe
import hud_reader as hr
import analyze_match as am
import vlm_id as v

OUT = pe.app_dir() / "verify_pack2"      # v2: 후보 레시피 v2(recall 57%) + 기검수 제외
OLD = pe.app_dir() / "verify_pack"
CACHE = pe.app_dir() / "timelines"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--per-video", type=int, default=40)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    data = pe.load_framedata_file()
    combo_db = am.load_combos()
    pack = OUT / "pack.jsonl"
    done = set()
    if pack.exists():
        done = {json.loads(l)["video"] for l in pack.open(encoding="utf-8") if l.strip()}
    man = pack.open("a", encoding="utf-8")
    seq = sum(1 for _ in pack.open(encoding="utf-8")) if pack.exists() else 0
    # v1에서 이미 검수(gold)된 이벤트 제외
    golded = set()
    old_pack = OLD / "pack.jsonl"
    if old_pack.exists():
        for l in old_pack.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("gold"):
                    golded.add((r["video"], round(r["t"], 1)))

    vids = sorted((pe.app_dir() / "reference_video").glob("*.mp4"))
    n_vid = 0
    for vp in vids:
        if n_vid >= a.limit:
            break
        tl = CACHE / (vp.stem + ".json")
        if not tl.exists() or vp.name in done:
            continue
        d = json.loads(tl.read_text(encoding="utf-8"))
        p1, p2, events = d["p1"], d["p2"], d["events"]
        if not p1 or not p2 or p1 == p2:                 # 미러전 제외
            continue
        chars = {"P1": p1, "P2": p2}
        n_ev = 0
        for e in events:
            if n_ev >= a.per_video or e["kind"] not in ("counter", "punish"):
                continue
            atk, conf = am.attacker_confident(events, e)
            if not conf:
                continue                                  # attribution 애매 제외
            ch = chars[atk]
            if ch in am.EXCLUDE_CHARS:
                continue
            md = am.first_hit_damage(events, e["t"], am.OTHER[atk])
            if md is None or ch not in data:
                continue
            if (vp.name, round(e["t"], 1)) in golded:     # 이미 검수됨
                continue
            lv = {"large": "Large", "medium": "Mid"}.get(e.get("level", "")) if e["kind"] == "counter" else None
            names = am.candidates_for(data, ch, md, lv)   # 후보 v2: 데미지∪프라이어 (recall 57%)
            cands = [{"char": ch, "move": m} for m in names if v.ref_path(ch, m)]
            if len(cands) < 2:
                continue
            frames = v.event_frames(str(vp), e["t"])
            if not frames or v.is_burst(frames):
                continue
            hud = v.hud_strip(str(vp), e["t"])
            guess = v.identify(frames, cands, f"{e['kind']} 시동기", attacker_char=ch, hud=hud)
            sheet = v.build_sheet(frames, cands, hud=hud)
            gm = re.sub(r'[\\/:*?"<>|]', "_", guess.get("move") or "unsure")
            name = f"{seq:04d}_{atk}_{ch.split()[0]}_LoRA={gm}.png"
            cv2.imwrite(str(OUT / name), sheet)
            man.write(json.dumps({"seq": seq, "sheet": name, "video": vp.name, "t": round(e["t"], 1),
                                  "attacker": atk, "char": ch, "kind": e["kind"],
                                  "cands": [c["move"] for c in cands],
                                  "lora": guess.get("move"), "gold": ""},
                                 ensure_ascii=False) + "\n")
            man.flush()
            seq += 1; n_ev += 1
            print(f"  {name}", flush=True)
        print(f"=== {vp.name[:45]} : {n_ev}건 ===", flush=True)
        n_vid += 1
    man.close()
    print(f"\n총 {seq}건 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
