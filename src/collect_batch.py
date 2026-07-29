"""
collect_batch.py  —  여러 매치영상을 돌며 자가학습 데이터 자동 수집 (재개 가능).

reference_video/*.mp4 를 순회하며 각 영상에 collect_auto.collect 실행.
manifest.jsonl 에 이미 있는 영상은 건너뜀(재개). 영상마다 캐릭 자동검출.

사용: python collect_batch.py [--limit N] [--dir reference_video]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import punish_engine as pe
import hud_reader as hr
import analyze_match as am
import collect_auto as ca


def done_videos() -> set[str]:
    """모든 샤드 manifest(manifest*.jsonl) 합쳐서 완료영상 집합."""
    done = set()
    for mf in ca.OUT.glob("manifest*.jsonl"):
        for l in mf.open(encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["video"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--dir", default=str(pe.app_dir() / "reference_video"))
    ap.add_argument("--shard", type=int, default=0)      # 이 워커 번호
    ap.add_argument("--nshards", type=int, default=1)    # 총 워커 수 (병렬)
    a = ap.parse_args()
    data = pe.load_framedata_file()
    vids = sorted(Path(a.dir).glob("*.mp4"))
    done = done_videos()
    todo = [v for v in vids if v.name not in done][a.shard::a.nshards]   # 샤드 분담
    man_path = ca.OUT / (f"manifest_s{a.shard}.jsonl" if a.nshards > 1 else "manifest.jsonl")
    cache = pe.app_dir() / "timelines"; cache.mkdir(exist_ok=True)
    print(f"[shard {a.shard}/{a.nshards}] 전체 {len(vids)} / 완료 {len(done)} / 분담 {len(todo)} / 최대 {a.limit}", flush=True)

    n_proc = 0
    for vp in todo:
        if n_proc >= a.limit:
            break
        print(f"\n=== [{n_proc+1}] {vp.name[:50]} ===", flush=True)
        try:
            tl = cache / (vp.stem + ".json")
            if tl.exists():                        # 타임라인 캐시 -> 재스캔 생략
                d = json.loads(tl.read_text(encoding="utf-8"))
                p1, p2, events = d["p1"], d["p2"], d["events"]
                print(f"  (캐시) P1={p1} P2={p2} 이벤트={len(events)}", flush=True)
            else:
                p1, p2 = hr.detect_characters(str(vp), list(data))
                if not p1 or not p2:
                    print("  캐릭 검출 실패 — 건너뜀", flush=True); continue
                cfg = hr.load_config()
                s, W, H, fps, rects, tr, wr = hr.scan(str(vp), cfg)
                events = [e.as_dict() for e in hr.build_timeline(
                    s, hr.ocr_text_events(tr) + hr.reversal_wake_events(wr))[0]]
                tl.write_text(json.dumps({"p1": p1, "p2": p2, "events": events}, ensure_ascii=False),
                              encoding="utf-8")
                print(f"  P1={p1} P2={p2} 이벤트={len(events)} (스캔·캐시저장)", flush=True)
            if p1 == p2:                           # 미러전: 스틸컷 attribution 검증 불가 -> 제외
                print("  미러전 — 라벨 수집 제외", flush=True)
                n_proc += 1
                continue
            k = ca.collect(str(vp), p1, p2, events, man_path=man_path)
            print(f"  -> {k}개 채택", flush=True)
        except Exception as e:
            print(f"  ! 오류: {e}", flush=True)
        n_proc += 1

    from collections import Counter
    cnt = Counter(); tot = 0
    for mf in ca.OUT.glob("manifest*.jsonl"):
        for l in mf.open(encoding="utf-8"):
            if l.strip():
                d = json.loads(l); cnt[f"{d['char'][:8]}/{d['move']}"] += 1; tot += 1
    print(f"\n=== [shard {a.shard}] 처리 {n_proc}영상, 전체누적 라벨 {tot}개 ===", flush=True)
    print("상위 무브:", dict(cnt.most_common(12)), flush=True)


if __name__ == "__main__":
    main()
