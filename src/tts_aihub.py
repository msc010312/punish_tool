# -*- coding: utf-8 -*-
"""tts_aihub.py — AI Hub '감성 및 발화 스타일 동시 고려 음성합성 데이터' → SBV2 기반모델 코퍼스.

데이터 구조(실측)
  루트/01-1.정식개방데이터/Training/
    01.원천데이터/TS_<발화체>_NNN.zip   : wav들(pcm_s16le 44.1kHz mono, 파일명 F-..-NNN-XXXX.wav)
    02.라벨링데이터/TL_<발화체>_NNN.zip  : json들. json.sentences[].voice_piece.{filename,tr,duration},
                                          style.{emotion,style,intensity}, reciter.{id,gender}
  TS와 TL은 뒤 번호(NNN)로 짝. 오디오는 이미 SBV2 포맷이라 재샘플 불필요.

전략: 195GB 전부는 불필요(웜스타트가 91% 물려받음). 발화체별로 몇 zip씩만 뽑아
~수십 시간의 다양한 화자·감정 코퍼스를 만든다. 발표의 교훈(길이/노이즈 필터, 기호 확보)은
tts_corpus.py의 필터를 그대로 재사용.

디스크 안전: 오디오 zip을 임시폴더에 풀고 → 채택분만 목적지로 복사 → 임시 삭제.

사용:
  py -3.11 src/tts_aihub.py --hours 40 --per-style 4
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import wave
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AIHUB_ROOT = Path("d:/punishTool/133.감성 및 발화 스타일 동시 고려 음성합성 데이터"
                  "/01-1.정식개방데이터/Training")
AUDIO_DIR = AIHUB_ROOT / "01.원천데이터"
LABEL_DIR = AIHUB_ROOT / "02.라벨링데이터"
OUT_ROOT = ROOT / "tts" / "dataset" / "base_ko"

MIN_SEC = 1.0
MAX_SEC = 12.0          # 발표: 긴 클립이 수렴을 막았다
MIN_CHARS = 4
MAX_CHARS = 120

# 발화체 7종. 다양성을 위해 각각에서 고르게 뽑는다.
STYLES = ["구연체", "낭독체", "대화체", "독백체", "애니체", "중계체", "친절체"]


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def load_label_map(label_zip: Path) -> dict[str, dict]:
    """라벨 zip 하나에서 {wav파일명: {text, speaker, emotion, style, intensity}} 추출.

    reciter.id는 각 파일 내부 로컬 번호라 zip이 달라도 충돌한다(전부 1). 그래서
    화자키에 zip 식별자(발화체_번호)를 넣어 세션(=실화자) 단위로 유일하게 만든다.
    """
    out: dict[str, dict] = {}
    session = label_zip.stem.replace("TL_", "")  # 예: 구연체_001
    try:
        with zipfile.ZipFile(label_zip) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".json"):
                    continue
                try:
                    doc = json.loads(zf.read(name).decode("utf-8-sig"))
                except Exception:
                    continue
                # 최상위가 1개짜리 리스트로 감싸인 경우가 있어 언랩한다.
                records = doc if isinstance(doc, list) else [doc]
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    reciter = rec.get("reciter", {}) or {}
                    spk = f"aihub_{session}_{reciter.get('gender', 'x')}"
                    for s in rec.get("sentences", []):
                        if not isinstance(s, dict):
                            continue
                        vp = s.get("voice_piece", {}) or {}
                        fn = vp.get("filename")
                        text = vp.get("tr") or s.get("origin_text")
                        if not fn or not text:
                            continue
                        st = s.get("style", {}) or {}
                        out[Path(fn).name] = {
                            "text": " ".join(str(text).split()),
                            "speaker": spk,
                            "emotion": st.get("emotion", ""),
                            "style": st.get("style", ""),
                            "intensity": st.get("intensity", 0),
                        }
    except Exception as e:
        print(f"  [경고] 라벨 zip 실패 {label_zip.name}: {e}")
    return out


def normalize_check(text: str):
    sys.path.insert(0, str(ROOT / "tts" / "sbv2"))
    from style_bert_vits2.nlp.korean.normalizer import normalize_text
    return normalize_text(text)


def zip_number(p: Path) -> str:
    # TS_구연체_001.zip → 001
    return p.stem.split("_")[-1]


def audio_zips_for_style(style: str) -> list[Path]:
    return sorted(AUDIO_DIR.glob(f"TS_{style}_*.zip"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=40.0, help="목표 총 시간")
    ap.add_argument("--per-style", type=int, default=4, help="발화체당 최대 zip 수")
    ap.add_argument("--per-zip-sec", type=float, default=3600.0,
                    help="zip(화자)당 최대 초. 화자 다양성 확보용(0=무제한)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    wav_dir = out_root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    budget = args.hours * 3600
    total_sec = 0.0
    idx = 0
    lines: list[str] = []
    rep = {"채택": 0, "전사없음": 0, "길이초": 0, "길이문자": 0, "정규화탈락": 0}
    spk_counter: Counter = Counter()
    style_counter: Counter = Counter()
    punct_counter: Counter = Counter()

    print(f"목표 {args.hours}시간 / 발화체당 최대 {args.per_style} zip")

    # 발화체를 라운드로빈으로 인터리브해서 예산이 일찍 차도 다양성이 유지되게 한다.
    # [구연체_1, 낭독체_1, ..., 구연체_2, 낭독체_2, ...] 순으로 방문.
    per_style_zips = {s: audio_zips_for_style(s)[: args.per_style] for s in STYLES}
    schedule: list[tuple[str, Path]] = []
    for rank in range(args.per_style):
        for s in STYLES:
            if rank < len(per_style_zips[s]):
                schedule.append((s, per_style_zips[s][rank]))

    for style, az in schedule:
        if True:
            if total_sec >= budget:
                break
            num = zip_number(az)
            label_zip = LABEL_DIR / f"TL_{style}_{num}.zip"
            if not label_zip.exists():
                print(f"  {az.name}: 짝 라벨 없음, 건너뜀")
                continue
            labels = load_label_map(label_zip)
            if not labels:
                continue

            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                with zipfile.ZipFile(az) as zf:
                    for m in zf.namelist():
                        if m.lower().endswith(".wav"):
                            data = zf.read(m)
                            (tmp / Path(m).name).write_bytes(data)

                kept_here = 0
                zip_sec = 0.0
                for wav in sorted(tmp.glob("*.wav")):
                    # zip(=화자)당 상한. 한 화자에 데이터가 쏠리지 않게 해 화자 수를 늘린다.
                    if args.per_zip_sec and zip_sec >= args.per_zip_sec:
                        break
                    meta = labels.get(wav.name)
                    if not meta:
                        rep["전사없음"] += 1
                        continue
                    text = normalize_check(meta["text"])
                    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                        rep["길이문자"] += 1
                        continue
                    sec = wav_seconds(wav)
                    if not (MIN_SEC <= sec <= MAX_SEC):
                        rep["길이초"] += 1
                        continue
                    if total_sec >= budget:
                        break

                    name = f"base_{idx:06d}.wav"
                    shutil.copy2(wav, wav_dir / name)
                    lines.append(f"{name}|{meta['speaker']}|KO|{text}")
                    idx += 1
                    kept_here += 1
                    total_sec += sec
                    zip_sec += sec
                    rep["채택"] += 1
                    spk_counter[meta["speaker"]] += 1
                    style_counter[meta["style"] or style] += 1
                    for p in ("…", "?", "!", ","):
                        if p in text:
                            punct_counter[p] += 1

            print(f"  {az.name}: +{kept_here}  (누적 {rep['채택']}개 / {total_sec / 3600:.2f}h)")

    esd = out_root / "esd.list"
    io.open(esd, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print(f"\n{'=' * 56}")
    print(f"채택 {rep['채택']}개 / {total_sec / 3600:.2f}시간 → {esd}")
    print(f"제외: {rep}")
    print(f"화자 {len(spk_counter)}명, 발화체 {dict(style_counter)}")
    print("기호 포함:", {k: f"{v}({v / max(rep['채택'], 1) * 100:.1f}%)"
                       for k, v in punct_counter.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
