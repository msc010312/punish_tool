# -*- coding: utf-8 -*-
"""banner_calibrate.py — COUNTER/PUNISH/JUST 배너 영역을 드래그로 지정.

사람이 검증한 배너(banner_labels.json)의 실제 시각으로 자동 점프하고, '글자가 보이는'
프레임에서 박스를 드래그하면 hud_config.json 에 저장한다. 좌표는 '게임영역' 기준 비율.
한쪽만 그리면 반대쪽은 좌우 대칭으로 자동 생성.

  n / p : 다음/이전 배너 예시
  a / d : 시간 -0.03s / +0.03s  (판때기->글자 프레임 찾기)
  SPACE : 현재 프레임에서 박스 드래그 (드래그 후 ENTER 확정, C 취소)
  s     : 저장 후 종료      q : 저장 안 하고 종료

사용: python banner_calibrate.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import cv2

import punish_engine as pe
import hud_reader as hr

CONFIG = pe.app_dir() / "hud_config.json"
LABELS = pe.app_dir() / "banner_labels.json"
VDIR = pe.app_dir() / "reference_video"


def examples() -> list[tuple[str, float, str, str]]:
    """검증된 배너 run -> (video, t, side, word). 없으면 빈 리스트."""
    if not LABELS.exists():
        return []
    lab = json.loads(LABELS.read_text(encoding="utf-8"))
    vids = list(VDIR.glob("*.mp4"))
    out, seen = [], set()
    for p, w in lab.items():
        if w not in ("COUNTER", "PUNISH", "JUST") or not p.endswith("_0.png"):
            continue
        m = re.match(r"(.+)_([0-9.]+)_(P[12])_0\.png$", Path(p).name)
        if not m:
            continue
        stem, t, side = m.group(1), float(m.group(2)), m.group(3)
        v = next((x for x in vids if x.name.startswith(stem)), None)
        if not v or (v.name, t) in seen:
            continue
        seen.add((v.name, t))
        out.append((str(v), t, side, w))
    return out


def main():
    ex = examples()
    if not ex:
        print("검증된 배너가 없습니다. banner_label_tool.py 로 몇 개만 먼저 라벨하세요.")
        return
    print(f"배너 예시 {len(ex)}개.  n/p=이동  a/d=시간조절  SPACE=박스 드래그  s=저장  q=종료")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {"regions": {}}
    cfg.setdefault("regions", {})
    i, dt = 0, 0.25          # 기본 +0.25s (판때기 지나 글자 프레임)
    box = None               # (x0,y0,x1,y1) 게임영역 비율
    cap = None; cur_v = None

    while True:
        v, t, side, w = ex[i % len(ex)]
        if cur_v != v:
            if cap: cap.release()
            cap = cv2.VideoCapture(v); cur_v = v
            gr = hr.find_game_region(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + dt) * fps))
        ok, fr = cap.read()
        if not ok:
            i += 1; continue
        g = hr.crop_game(fr, gr)
        H, W = g.shape[:2]
        vis = g.copy()
        if box:
            # 그린 박스(초록) + 좌우 대칭 박스(하늘) 동시 표시 -> 양쪽 배너 다 검증 가능
            mir = (1 - box[2], box[1], 1 - box[0], box[3])
            for b, col, tag in ((box, (0, 255, 0), "drawn"), (mir, (255, 220, 0), "mirror")):
                p0 = (int(b[0]*W), int(b[1]*H)); p1 = (int(b[2]*W), int(b[3]*H))
                cv2.rectangle(vis, p0, p1, col, 2)
                cv2.putText(vis, tag, (p0[0], max(14, p0[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(vis, f"[{i%len(ex)+1}/{len(ex)}] {w} side={side} t={t:.1f}+{dt:+.2f}s",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, "SPACE=drag box(either side)  a/d=time  n/p=next  r=reset  s=save  q=quit",
                    (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
        cv2.imshow("banner calibrate", vis)
        k = cv2.waitKey(0) & 0xFF

        if k == ord('q'):
            break
        elif k == ord('n'):
            i += 1
        elif k == ord('p'):
            i -= 1
        elif k == ord('a'):
            dt -= 0.03
        elif k == ord('d'):
            dt += 0.03
        elif k == ord(' '):
            r = cv2.selectROI("banner calibrate", g, showCrosshair=True, fromCenter=False)
            if r[2] > 4 and r[3] > 4:
                box = (r[0]/W, r[1]/H, (r[0]+r[2])/W, (r[1]+r[3])/H)
                print(f"박스: {tuple(round(x,4) for x in box)}")
        elif k == ord('s'):
            if not box:
                print("먼저 SPACE 로 박스를 그리세요."); continue
            # 그린 쪽 + 좌우 대칭으로 반대쪽 생성
            drawn_left = (box[0] + box[2]) / 2 < 0.5
            mirror = (1 - box[2], box[1], 1 - box[0], box[3])
            left, right = (box, mirror) if drawn_left else (mirror, box)
            cfg["regions"]["banner_left"] = dict(zip(("x0", "y0", "x1", "y1"), [round(x, 5) for x in left]))
            cfg["regions"]["banner_right"] = dict(zip(("x0", "y0", "x1", "y1"), [round(x, 5) for x in right]))
            CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"저장 -> {CONFIG}\n  banner_left  {cfg['regions']['banner_left']}"
                  f"\n  banner_right {cfg['regions']['banner_right']}")
            break
    if cap: cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
