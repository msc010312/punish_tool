"""
test_report.py  —  실제 분석 파이프라인 검증: 버스트/Bedman이 리포트에서 제외되는지 확인.

캐시 타임라인 + 영상으로 build_report를 돌리고, enrich_vlm이 표시한
burst/excluded 이벤트 수를 세서 리포트 청소가 실제 작동하는지 본다.
"""
from __future__ import annotations
import glob
from pathlib import Path

import punish_engine as pe
import analyze_match as am

TL = glob.glob(str(pe.app_dir() / "timelines" / "*MikenBancho*.json"))[0]
VID = glob.glob(str(pe.app_dir() / "reference_video" / "*MikenBancho*.mp4"))[0]


def main():
    tl = am.load_timeline(str(TL))
    p1, p2 = tl["p1"], tl["p2"]
    data = pe.load_framedata_file()
    events = tl["events"]
    cp = [e for e in events if e["kind"] in ("counter", "punish")]
    print(f"P1={p1} P2={p2} | counter/punish 이벤트 {len(cp)}개", flush=True)

    # enrich (버스트/Bedman 표시) — 진행률 출력
    am.enrich_vlm(events, VID, {"P1": p1, "P2": p2}, data,
                  progress_cb=lambda i, n: print(f"  VLM {i}/{n}", flush=True))
    burst = sum(1 for e in cp if e.get("burst"))
    excl = sum(1 for e in cp if e.get("excluded"))
    vlmd = sum(1 for e in cp if e.get("vlm"))
    print(f"→ 버스트 제외 {burst} | Bedman(제외캐릭) 제외 {excl} | VLM식별 {vlmd}", flush=True)

    report = am.build_report(tl, p1, p2, data, video=VID)
    Path(pe.app_dir() / "report_test.txt").write_text(report, encoding="utf-8")
    print("\n=== 리포트 (앞부분) ===\n" + report[:2500], flush=True)


if __name__ == "__main__":
    main()
