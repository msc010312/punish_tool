"""
prep_voice.py  —  긴 음성 wav -> XTTS 파인튜닝 데이터셋 (무음 기준 분할 + 한국어 전사)

ffmpeg silencedetect로 발화 구간을 잘라 클립(mono 22050)으로 저장하고, whisper로 한국어
전사해 Coqui XTTS 학습용 metadata 를 만든다.

출력: voice/dataset/wavs/*.wav  +  voice/dataset/metadata.csv  (audio_file|text|speaker)
       (+ metadata_train.csv / metadata_eval.csv 분할)
사용: python prep_voice.py "<input.wav>" [--max 120]
"""
from __future__ import annotations
import argparse, re, subprocess, sys
from pathlib import Path

OUTROOT = Path(r"D:/punishTool/voice/dataset")
WAVS = OUTROOT / "wavs"
SPK = "choihan"


def silence_intervals(path: str, noise="-32dB", dur=0.25):
    cmd = ["ffmpeg", "-i", path, "-af", f"silencedetect=noise={noise}:d={dur}", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log = r.stderr
    starts = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([0-9.]+)", log)]
    return starts, ends


def speech_segments(starts, ends, total, lo=1.2, hi=11.0):
    """무음 구간 사이 = 발화 구간. lo~hi초만 채택."""
    segs = []
    cur = 0.0
    # ends[i] = 무음 끝(발화 시작), starts[i] = 다음 무음 시작(발화 끝)
    pts = sorted([(s, "s") for s in starts] + [(e, "e") for e in ends])
    speech_start = 0.0
    in_speech = True
    for t, k in pts:
        if k == "s" and in_speech:           # 발화->무음
            if lo <= t - speech_start <= hi:
                segs.append((speech_start, t))
            in_speech = False
        elif k == "e" and not in_speech:      # 무음->발화
            speech_start = t
            in_speech = True
    if in_speech and lo <= total - speech_start <= hi:
        segs.append((speech_start, total))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max", type=int, default=120)
    ap.add_argument("--prefix", default="clip")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    WAVS.mkdir(parents=True, exist_ok=True)

    # 총 길이
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=nw=1:nk=1", a.input], capture_output=True, text=True).stdout.strip())
    starts, ends = silence_intervals(a.input)
    segs = speech_segments(starts, ends, dur)[: a.max]
    print(f"발화 구간 {len(segs)}개 추출 (총 {dur:.0f}s)", flush=True)

    # 클립 컷 (mono 22050)
    clips = []
    for i, (s, e) in enumerate(segs):
        out = WAVS / f"{a.prefix}_{i:04d}.wav"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", a.input,
                        "-ac", "1", "-ar", "22050", str(out)],
                       capture_output=True)
        clips.append(out)
    print(f"클립 {len(clips)}개 저장 -> {WAVS}", flush=True)

    # whisper 전사 (한국어)
    import whisper, warnings
    warnings.filterwarnings("ignore")
    m = whisper.load_model("small")
    rows = []
    for c in clips:
        r = m.transcribe(str(c), language="ko", fp16=True)
        txt = r["text"].strip().replace("|", " ")
        if len(txt) >= 2:
            rows.append((c.name, txt))
            print(f"  {c.name}: {txt}", flush=True)

    # metadata (audio_file|text|speaker). append면 기존 것 유지 후 합쳐 재분할
    new_lines = [f"wavs/{n}|{t}|{SPK}" for n, t in rows]
    existing = []
    mf = OUTROOT / "metadata.csv"
    if a.append and mf.exists():
        existing = [l for l in mf.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.startswith("audio_file|")]
    alll = existing + new_lines
    mf.write_text("\n".join(alll), encoding="utf-8")
    nv = max(1, len(alll) // 10)
    hdr = "audio_file|text|speaker_name\n"
    (OUTROOT / "metadata_eval.csv").write_text(hdr + "\n".join(alll[:nv]), encoding="utf-8")
    (OUTROOT / "metadata_train.csv").write_text(hdr + "\n".join(alll[nv:]), encoding="utf-8")
    print(f"\n전사 {len(rows)}개 추가 (총 {len(alll)}) -> train {len(alll)-nv}/eval {nv}", flush=True)


if __name__ == "__main__":
    main()
