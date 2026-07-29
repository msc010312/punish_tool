"""
prep_clips.py  —  이미 라인별로 잘린 클립 폴더 -> XTTS 학습 데이터셋 (분할 불필요, 변환+전사만)

폴더의 각 오디오(ogg/wav/mp3)를 22050 모노 wav로 변환하고 whisper로 한국어 전사 -> metadata.
무음 분할이 필요한 긴 파일은 prep_voice.py를, 라인별 클립은 이걸 쓴다.
사용: python prep_clips.py <폴더> [--min 1.2 --max 12 --prefix sol]
"""
from __future__ import annotations
import argparse, glob, subprocess, re
from pathlib import Path

OUTROOT = Path(r"D:/punishTool/voice/dataset")
WAVS = OUTROOT / "wavs"
SPK = "sol"


def dur(f):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=nw=1:nk=1", f], capture_output=True, text=True).stdout.strip() or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--min", type=float, default=1.2)
    ap.add_argument("--max", type=float, default=12.0)
    ap.add_argument("--prefix", default="sol")
    a = ap.parse_args()
    WAVS.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{a.indir}/*.ogg") + glob.glob(f"{a.indir}/*.wav")
                   + glob.glob(f"{a.indir}/*.mp3") + glob.glob(f"{a.indir}/*.m4a"))
    files = [f for f in files if a.min <= dur(f) <= a.max]
    print(f"적정 길이 클립 {len(files)}개", flush=True)

    clips = []
    for i, f in enumerate(files):
        out = WAVS / f"{a.prefix}_{i:04d}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", f, "-ac", "1", "-ar", "22050", str(out)], capture_output=True)
        clips.append(out)

    import whisper, warnings
    warnings.filterwarnings("ignore")
    m = whisper.load_model("small")
    rows = []
    for c in clips:
        t = m.transcribe(str(c), language="ko", fp16=True)["text"].strip().replace("|", " ")
        kr = len(re.findall(r"[가-힣]", t)) / (len(re.sub(r"\s", "", t)) or 1)
        if len(re.sub(r"\s", "", t)) >= 2 and kr >= 0.5:
            rows.append((c.name, t))
            print(f"  {c.name}: {t}", flush=True)

    lines = [f"wavs/{n}|{t}|{SPK}" for n, t in rows]
    h = "audio_file|text|speaker_name\n"
    nv = max(1, len(lines) // 10)
    (OUTROOT / "metadata.csv").write_text("\n".join(lines), encoding="utf-8")
    (OUTROOT / "metadata_eval.csv").write_text(h + "\n".join(lines[:nv]), encoding="utf-8")
    (OUTROOT / "metadata_train.csv").write_text(h + "\n".join(lines[nv:]), encoding="utf-8")
    print(f"\n전사 {len(rows)}개 -> train {len(lines)-nv}/eval {nv}", flush=True)


if __name__ == "__main__":
    main()
