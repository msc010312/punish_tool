"""
extract_move_dataset.py  —  무브 ID(CNN) 학습용 데이터셋 빌더 [1단계]

매치 영상에서 카운터/펀ish 순간의 '공격자 영역'을 크롭해, 데미지 기반 무브 ID로 잠정 라벨링.
 - 데미지로 '단일 특정(★)'된 건 dataset/<캐릭>/<기술>/ 에 자동 분류 (신뢰 라벨)
 - 후보 여럿이면 dataset/<캐릭>/_review/ 에 모음 (사용자가 검수·분류)
이렇게 여러 영상을 돌려 라벨 데이터를 쌓는다. (CNN 학습은 그 다음 단계)

사용:  python extract_move_dataset.py "<영상.mp4>"   (캐릭터 자동 인식)
주의:  공격자 위치 검출이 없어 '공격자 측+중앙'을 넉넉히 크롭 -> CNN이 포즈에 집중하게 학습.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2

import hud_reader as hr
import punish_engine as pe
import analyze_match as am
import localize

CROP = 224                      # 학습 입력 크기
OUTROOT = pe.app_dir() / "dataset"


def safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s).strip("_") or "x"


def attacker_crop(frame, attacker_side: str):
    """공격자 측+중앙 위주로 크롭 (HUD 아래 플레이필드)."""
    H, W = frame.shape[:2]
    x0, x1 = (0.04, 0.66) if attacker_side == "P1" else (0.34, 0.96)
    y0, y1 = 0.18, 0.92
    crop = frame[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    return cv2.resize(crop, (CROP, CROP))


def main():
    if len(sys.argv) < 2:
        sys.exit("사용: python extract_move_dataset.py <영상.mp4>")
    video = sys.argv[1]
    data = pe.load_framedata_file()
    if data is None:
        sys.exit("framedata 없음")
    chars = list(data)

    print("캐릭터 인식 중…")
    p1, p2 = hr.detect_characters(video, chars)
    charmap = {"P1": p1, "P2": p2}
    print(f"  P1={p1}  P2={p2}")

    print("이벤트 스캔 중…(영상 길이에 따라 시간)")
    cfg = hr.load_config()
    samples, W, H, fps, rects, truns, _ = hr.scan(video, cfg)
    text_events = hr.ocr_text_events(truns)
    events, _, _ = hr.build_timeline(samples, text_events)
    ev = [e.as_dict() for e in events]

    gr = hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    stem = safe(Path(video).stem)
    n_auto = n_review = n_yolo = 0
    for e in ev:
        if e["kind"] not in ("counter", "punish"):
            continue
        atk = am.attacker_of(ev, e)
        ch = charmap.get(atk)
        if not ch or ch not in data:
            continue
        md = am.first_hit_damage(ev, e["t"], am.OTHER[atk])
        if md is None:
            continue
        cands = pe.identify_move(data[ch], md * pe.HEALTH)
        if not cands:
            continue
        # 공격자 박스를 YOLO로 찾아 깨끗하게 크롭 (히트 직전 active 프레임들 우선)
        crop = None
        for dt in (-0.10, -0.06, -0.03, 0.0, 0.04):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((e["t"] + dt) * vfps))
            ok, fr = cap.read()
            if not ok:
                continue
            fr = hr.crop_game(fr, gr)
            box = localize.attacker_box(fr, atk)
            if box is not None:
                crop = localize.crop_box(fr, box); n_yolo += 1
                break
        if crop is None:                           # YOLO 미검출 -> 반쪽화면 폴백
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(e["t"] * vfps))
            ok, fr = cap.read()
            if not ok:
                continue
            crop = attacker_crop(hr.crop_game(fr, gr), atk)
        if len(cands) == 1:                        # 데미지로 단일 특정 = 신뢰 라벨
            dest = OUTROOT / safe(ch) / safe(cands[0][0].name)
            n_auto += 1
        else:                                      # 애매 -> 검수 폴더(후보를 파일명에)
            cand_tag = "-".join(safe(m.name) for m, _ in cands[:3])
            dest = OUTROOT / safe(ch) / "_review"
            n_review += 1
        dest.mkdir(parents=True, exist_ok=True)
        tag = "" if len(cands) == 1 else f"_{cand_tag}"
        cv2.imwrite(str(dest / f"{stem}_{e['t']:.1f}{tag}.png"), crop)
    cap.release()
    print(f"완료: 자동라벨 {n_auto}장 / 검수대기 {n_review}장 (YOLO크롭 {n_yolo}장) -> {OUTROOT}")
    print("여러 영상을 돌려 데이터를 쌓고, _review 폴더를 기술별로 분류하면 학습 준비 끝.")


if __name__ == "__main__":
    main()
