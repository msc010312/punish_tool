# -*- coding: utf-8 -*-
"""tts_prep.py — 게임 추출 보이스(ogg) → Style-Bert-VITS2 학습 데이터셋.

단계(서브커맨드)
  audio      : 원본 → 44.1kHz 모노 wav, 앞뒤 무음 트림, 라우드니스 정규화
  transcribe : whisper 전사 → review.tsv (사람이 직접 교정하는 파일)
  build      : 교정된 review.tsv → esd.list (SBV2 포맷)
  stats      : 현재 데이터셋 통계

캐릭터/언어 추가는 PROFILES에 항목 하나 넣으면 끝. 경로/언어/소스는 프로파일에 모음.
모든 서브커맨드는 --profile 로 대상을 고른다(기본 sol_ko).

사용:
  py -3.11 src/tts_prep.py audio --profile sol_jp
  py -3.11 src/tts_prep.py transcribe --profile sol_jp --model large-v3
  py -3.11 src/tts_prep.py build --profile sol_jp
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "tts" / "dataset"

# SBV2 학습 샘플레이트(config의 sampling_rate와 반드시 일치)
TARGET_SR = 44100
# 라우드니스 목표. 외침/평상시의 음색 차이는 남고 음량만 균일해짐 → 스타일 벡터에 유리
LOUDNORM = "I=-23:TP=-1.5:LRA=11"
# 무음 판정 임계. 게임 보이스는 BGM이 없어 -45dB로 충분
SILENCE_DB = "-45dB"
# 이보다 짧으면 기합/숨소리로 보고 제외
MIN_SEC = 0.45
# 이보다 길면 SBV2 학습이 불안정(발표에서도 긴 클립이 문제였음)
MAX_SEC = 14.0

JPN_ROOT = ROOT / "sol/Exports/RED/Content/Chara/SOL/Common/Audio/Voice/JPN"


@dataclass(frozen=True)
class Source:
    """원본 오디오 묶음 하나. style은 SBV2 수동 스타일 지정에 그대로 쓰임."""

    style: str
    root: Path
    pattern: str
    note: str = ""
    # 파일명(확장자 제외)이 겹치면 첫 번째만 채택. JPN처럼 같은 대사가 여러
    # 보이스 뱅크(0/1/2/…)에 중복될 때 켠다.
    dedup_stem: bool = False


@dataclass(frozen=True)
class Profile:
    """한 (화자, 언어) 데이터셋의 정의."""

    speaker: str
    lang: str            # esd.list의 언어 코드 (KO / JP)
    whisper_lang: str    # whisper 전사 언어
    out_name: str        # tts/dataset/ 아래 폴더명
    sources: list[Source] = field(default_factory=list)


PROFILES: dict[str, Profile] = {
    # 한국어(더빙) — 커스텀 KO 모듈 경로. 기반모델 학습 필요.
    "sol_ko": Profile(
        speaker="SOL", lang="KO", whisper_lang="ko", out_name="SOL",
        sources=[
            Source("Battle",
                   ROOT / "sol/Exports/RED/Content/Chara/SOL/Common/Audio/Voice/KOR/0",
                   "**/*.ogg", "VS 조우대사 — 짧은 도발/외침"),
            Source("Normal", ROOT / "voice/sol_story_kr", "*.ogg",
                   "스토리 음성 — 차분한 장문"),
        ],
    ),
    # 일본어(원본 성우) — SBV2 기본 지원. JP-Extra 기반모델이 이미 있어 바로 파인튜닝.
    "sol_jp": Profile(
        speaker="SOL", lang="JP", whisper_lang="ja", out_name="SOL_JP",
        sources=[
            Source("Battle", JPN_ROOT, "*/VS_*/*.ogg",
                   "VS 조우대사 (전 보이스뱅크)", dedup_stem=True),
        ],
    ),
    # 한국어 기반모델용 다화자 코퍼스. esd.list는 tts_aihub.py가 만들며(화자는 줄마다
    # 다름), 오디오 audio/transcribe/build 단계는 거치지 않는다. tts_train의
    # stage→preprocess→train만 이 out_name을 대상으로 돈다.
    "base_ko": Profile(
        speaker="MULTI", lang="KO", whisper_lang="ko", out_name="base_ko",
        sources=[],
    ),
}

# 실행 중 선택된 프로파일. main()에서 세팅.
PROFILE: Profile = PROFILES["sol_ko"]


def out_dir() -> Path:
    return OUT_ROOT / PROFILE.out_name


def wav_dir() -> Path:
    return out_dir() / "wavs"


def review_path() -> Path:
    return out_dir() / "review.tsv"


def esd_path() -> Path:
    return out_dir() / "esd.list"


def duration(path: Path) -> float:
    """ffprobe로 초 단위 길이. 실패하면 0."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def collect() -> list[tuple[Source, Path]]:
    """선택된 프로파일의 모든 소스에서 원본 파일 수집."""
    items: list[tuple[Source, Path]] = []
    for src in PROFILE.sources:
        if not src.root.exists():
            print(f"[경고] 소스 없음: {src.root}")
            continue
        found = sorted(src.root.glob(src.pattern))
        if src.dedup_stem:
            seen: set[str] = set()
            deduped = []
            for p in found:
                if p.stem not in seen:
                    seen.add(p.stem)
                    deduped.append(p)
            print(f"  {src.style:8s} {len(found):4d}개 → 중복제거 {len(deduped):4d}개  {src.note}")
            found = deduped
        else:
            print(f"  {src.style:8s} {len(found):4d}개  {src.note}")
        items += [(src, p) for p in found]
    return items


def convert(src_file: Path, dst: Path) -> None:
    """무음 트림 → 라우드니스 정규화 → 44.1kHz 모노 wav.

    silenceremove를 앞/뒤 각각 적용하려고 areverse로 두 번 돌린다.
    """
    trim = (f"silenceremove=start_periods=1:start_silence=0.05"
            f":start_threshold={SILENCE_DB}:detection=peak")
    af = f"{trim},areverse,{trim},areverse,loudnorm={LOUDNORM}"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src_file),
         "-af", af, "-ar", str(TARGET_SR), "-ac", "1",
         "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def cmd_audio(_args) -> None:
    wav_dir().mkdir(parents=True, exist_ok=True)
    print("소스 스캔:")
    items = collect()

    rows: list[dict] = []
    skipped = {"짧음": 0, "김": 0}
    for i, (src, path) in enumerate(items):
        name = f"{PROFILE.speaker.lower()}_{src.style.lower()}_{i:04d}.wav"
        dst = wav_dir() / name
        convert(path, dst)
        sec = duration(dst)
        if sec < MIN_SEC:
            dst.unlink(missing_ok=True)
            skipped["짧음"] += 1
            continue
        if sec > MAX_SEC:
            dst.unlink(missing_ok=True)
            skipped["김"] += 1
            continue
        rows.append({"wav": name, "style": src.style,
                     "sec": f"{sec:.2f}", "src": path.name, "text": ""})
        if (i + 1) % 40 == 0:
            print(f"  변환 {i + 1}/{len(items)}")

    with io.open(review_path(), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["wav", "style", "sec", "src", "text"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    total = sum(float(r["sec"]) for r in rows)
    print(f"\n채택 {len(rows)}개 / 제외 {skipped}")
    print(f"총 {total / 60:.2f}분 → {wav_dir()}")
    print(f"전사 대기 파일: {review_path()}")


def cmd_transcribe(args) -> None:
    """whisper 전사. 결과는 review.tsv의 text 칸에 채워짐(사람이 교정할 초안)."""
    import whisper

    rows = list(csv.DictReader(
        io.open(review_path(), encoding="utf-8-sig"), delimiter="\t"))
    print(f"모델 로드: {args.model}")
    model = whisper.load_model(args.model, device=args.device)

    for i, r in enumerate(rows):
        if r["text"].strip() and not args.overwrite:
            continue
        res = model.transcribe(str(wav_dir() / r["wav"]), language=PROFILE.whisper_lang,
                               fp16=(args.device == "cuda"))
        r["text"] = " ".join(res["text"].split())
        if (i + 1) % 20 == 0:
            print(f"  전사 {i + 1}/{len(rows)}")

    with io.open(review_path(), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["wav", "style", "sec", "src", "text"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\n전사 완료 → {review_path()}")
    print("※ 이 파일의 text 칸을 직접 듣고 교정해야 함. 전사 오류가 곧 학습 오류.")


def cmd_build(_args) -> None:
    """교정된 review.tsv → SBV2 esd.list.

    포맷: wavs/파일명.wav|화자|언어|텍스트
    """
    rows = list(csv.DictReader(
        io.open(review_path(), encoding="utf-8-sig"), delimiter="\t"))
    lines, dropped = [], 0
    for r in rows:
        text = r["text"].strip()
        if not text:
            dropped += 1
            continue
        lines.append(f"{r['wav']}|{PROFILE.speaker}|{PROFILE.lang}|{text}")

    io.open(esd_path(), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"esd.list {len(lines)}줄 작성 (빈 텍스트 {dropped}개 제외) → {esd_path()}")


def cmd_stats(_args) -> None:
    if not review_path().exists():
        print("review.tsv 없음. 먼저 audio 실행.")
        return
    rows = list(csv.DictReader(
        io.open(review_path(), encoding="utf-8-sig"), delimiter="\t"))
    by_style: dict[str, list[float]] = {}
    filled = 0
    for r in rows:
        by_style.setdefault(r["style"], []).append(float(r["sec"]))
        filled += bool(r["text"].strip())
    print(f"{'스타일':10s} {'개수':>5s} {'총(분)':>8s} {'평균(초)':>9s}")
    for s, ds in sorted(by_style.items()):
        print(f"{s:10s} {len(ds):5d} {sum(ds) / 60:8.2f} {sum(ds) / len(ds):9.2f}")
    tot = [d for ds in by_style.values() for d in ds]
    print(f"{'합계':10s} {len(tot):5d} {sum(tot) / 60:8.2f} {sum(tot) / len(tot):9.2f}")
    print(f"전사 채워짐: {filled}/{len(rows)}")


def add_profile_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default="sol_ko", choices=list(PROFILES),
                   help="대상 (화자,언어) 프로파일")


def main() -> int:
    global PROFILE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audio"); a.set_defaults(fn=cmd_audio); add_profile_arg(a)
    t = sub.add_parser("transcribe")
    t.add_argument("--model", default="large-v3")
    t.add_argument("--device", default="cuda")
    t.add_argument("--overwrite", action="store_true")
    add_profile_arg(t)
    t.set_defaults(fn=cmd_transcribe)
    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build); add_profile_arg(b)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats); add_profile_arg(s)
    args = ap.parse_args()
    PROFILE = PROFILES[args.profile]
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
