"""
collect_auto.py  —  자가학습(self-training) 데이터 자동 수집 (수작업 라벨링·영상헌팅 없이)

이미 만든 VLM+투표 파이프라인을 매치영상에 돌려, **투표 3/3 일치 + unsure 아님** 인 고신뢰
이벤트만 (크롭, 무브, 상황) 라벨로 채택해 dataset_auto/ 에 적재한다. 투표 일치 = 품질 게이트.
이 자동라벨로 나중에 qwen2.5vl LoRA 파인튠 → 모델이 자기 확신답으로 자기를 날카롭게.

사용:
  python collect_auto.py <video> [--p1 NAME --p2 NAME] [--timeline path.json]
  # timeline 있으면 재스캔 생략(빠름). 없으면 자동 스캔.
출력: dataset_auto/<Char_folder>/<move>/<tag>_f{i}.png  +  dataset_auto/manifest.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

import cv2
import punish_engine as pe
import analyze_match as am
import vlm_id as v

OUT = pe.app_dir() / "dataset_auto"


def vote(video, t, ch, cands, windows=(-0.15, 0.0, 0.15)):
    """여러 프레임창으로 식별 → (최빈 move, 일치율, 대표 frames)."""
    moves, rep = [], None
    hud = v.hud_strip(video, t)                  # 캐릭 신원은 색이 아니라 HUD 명판으로
    for dt in windows:
        fr = v.event_frames(video, t + dt)
        if dt == 0.0:
            rep = fr
        moves.append(v.identify(fr, cands, "self-train", attacker_char=ch, hud=hud).get("move"))
    top, n = Counter(moves).most_common(1)[0]
    return top, n / len(windows), (rep or [])


def collect(video, p1, p2, events, man_path=None):
    OUT.mkdir(exist_ok=True)
    data = pe.load_framedata_file()
    combo_db = am.load_combos()
    chars = {"P1": p1, "P2": p2}
    man = (man_path or (OUT / "manifest.jsonl")).open("a", encoding="utf-8")
    kept = 0
    cp = [e for e in events if e["kind"] in ("counter", "punish")]
    for e in cp:
        atk, conf = am.attacker_confident(events, e)
        if not conf:                             # attribution 애매(트레이드) -> 오라벨 방지
            continue
        ch = chars[atk]
        if ch in am.EXCLUDE_CHARS:               # 신뢰불가 캐릭(Bedman) 라벨 안 함
            continue
        md = am.first_hit_damage(events, e["t"], am.OTHER[atk])
        if md is None or ch not in data:
            continue
        lv = {"large": "Large", "medium": "Mid"}.get(e.get("level", "")) if e["kind"] == "counter" else None
        names = [m.name for m, _ in pe.identify_move(data[ch], md * pe.HEALTH, counter_level=lv)[:5]]
        obs, _ = am.observed_combo(events, e["t"], am.OTHER[atk])
        names, _ = am.refine_by_combo(combo_db, ch, names, obs)
        cands = [{"char": ch, "move": m} for m in names if v.ref_path(ch, m)]
        if len(cands) < 2:
            continue
        if v.is_burst(v.event_frames(video, e["t"])):   # 버스트(방어기)는 라벨 안 함
            continue
        move, agree, frames = vote(video, e["t"], ch, cands)
        if agree < 1.0 or move in ("unsure", "", None) or not frames:
            continue                                    # 만장일치 + 확정 무브만 채택
        d = OUT / ch.replace(" ", "_") / move
        d.mkdir(parents=True, exist_ok=True)
        tag = f"{Path(video).stem[:20]}_{e['t']:.1f}"
        paths = []
        for i, fr in enumerate(frames):
            p = d / f"{tag}_f{i}.png"; cv2.imwrite(str(p), fr); paths.append(str(p))
        man.write(json.dumps({"char": ch, "move": move, "kind": e["kind"],
                              "t": round(e["t"], 1), "video": Path(video).name,
                              "frames": paths}, ensure_ascii=False) + "\n")
        kept += 1
        print(f"  ✓ {e['t']:6.1f} {ch[:10]:10} {move:8} (만장일치)", flush=True)
    man.close()
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--p1"); ap.add_argument("--p2")
    ap.add_argument("--timeline")
    a = ap.parse_args()
    data = pe.load_framedata_file()

    if a.timeline and Path(a.timeline).exists():
        events = json.loads(Path(a.timeline).read_text(encoding="utf-8"))["events"]
        p1, p2 = a.p1, a.p2
    else:
        import hud_reader as hr
        p1d, p2d = hr.detect_characters(a.video, list(data))
        p1, p2 = a.p1 or p1d, a.p2 or p2d
        cfg = hr.load_config()
        s, W, H, fps, rects, tr, wr = hr.scan(a.video, cfg)
        events = [e.as_dict() for e in hr.build_timeline(
            s, hr.ocr_text_events(tr) + hr.reversal_wake_events(wr))[0]]
    if not p1 or not p2:
        import hud_reader as hr
        d1, d2 = hr.detect_characters(a.video, list(data)); p1 = p1 or d1; p2 = p2 or d2
    print(f"P1={p1} P2={p2}  이벤트 {len(events)}", flush=True)
    n = collect(a.video, p1, p2, events)
    tot = sum(1 for _ in (OUT / "manifest.jsonl").open(encoding="utf-8")) if (OUT / "manifest.jsonl").exists() else 0
    print(f"\n채택 {n}개 적재 -> {OUT}  (누적 {tot}개)", flush=True)


if __name__ == "__main__":
    main()
