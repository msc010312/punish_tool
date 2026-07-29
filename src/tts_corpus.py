# -*- coding: utf-8 -*-
"""tts_corpus.py — AI Hub 등 외부 한국어 음성 코퍼스 → SBV2 기반모델 학습 데이터.

NDC26 발표(넥슨게임즈)에서 얻은 교훈을 그대로 필터로 구현한다.

  1. 데이터셋 조합 실패
     평균 길이가 길고 절반 가까이 노이즈가 섞인 셋을 넣었더니 loss가 아예 수렴하지
     않았다(300GB를 통째로 버림). → 길이 상한과 SNR 하한으로 미리 걸러낸다.

  2. 문장기호 부족
     기반 모델 대본에 '…'가 없으면 파인튜닝에서 아무리 넣어도 표현되지 않았다.
     → 기호를 포함한 문장을 우선 채택하고, 최종 분포를 리포트한다.

  3. 치찰음(ㅅㅆㅈㅊ) 단어 부족
     대본에 없던 치찰음 단어는 발음 자체를 못 했다. → 치찰음 비율을 리포트한다.

AI Hub 배포본은 구조가 제각각이라(전사가 json/csv/txt, 오디오가 wav/pcm) 스캔은
어댑터로 분리했다. 새 배포본이 오면 LOADERS에 함수 하나 추가하면 된다.

사용:
  py -3.11 src/tts_corpus.py scan  --src <압축푼경로>
  py -3.11 src/tts_corpus.py build --src <압축푼경로> --hours 40
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import subprocess
import sys
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "tts" / "dataset" / "base_ko"

TARGET_SR = 44100          # SBV2 config의 sampling_rate와 일치해야 함
MIN_SEC = 1.0
MAX_SEC = 12.0             # 발표: 평균 길이가 긴 셋이 수렴을 막았다
MIN_CHARS = 4
MAX_CHARS = 120
LOUDNORM = "I=-23:TP=-1.5:LRA=11"

# 발표에서 "기반 모델에 없으면 끝까지 표현 안 된다"고 지목한 기호
WANTED_PUNCT = ("…", "?", "!", ",")
SIBILANTS = re.compile(r"[ㅅㅆㅈㅊ]|[사샤서셔소쇼수슈스시싸써쓰씨자져저조주즈지짜쩌쪼쭈찌차처초추츠치]")


@dataclass
class Utterance:
    """오디오 한 개와 그 대본."""

    audio: Path
    text: str
    speaker: str
    seconds: float = 0.0
    score: float = 0.0
    reject: str = ""


@dataclass
class ScanReport:
    total: int = 0
    kept: int = 0
    rejects: Counter = field(default_factory=Counter)
    speakers: Counter = field(default_factory=Counter)
    punct: Counter = field(default_factory=Counter)
    seconds: float = 0.0
    sibilant_hits: int = 0


# ── 전사 로더 ────────────────────────────────────────────────────────────────
# AI Hub 배포본마다 전사 형식이 달라 어댑터로 분리한다.
# 각 로더는 (오디오 stem -> 텍스트) 매핑을 돌려준다.

def load_json_transcripts(root: Path) -> dict[str, str]:
    """*.json — AI Hub에서 가장 흔한 형태. 파일당 발화 1개 또는 목록."""
    out: dict[str, str] = {}
    text_keys = ("transcription", "text", "script", "sentence", "orgtext",
                 "발화문", "전사문", "제시문")
    id_keys = ("fileName", "file_name", "filename", "id", "audio_id", "wav")

    def pick(d: dict, keys: tuple) -> str | None:
        for k in keys:
            for actual in d:
                if actual.lower() == k.lower() and isinstance(d[actual], str):
                    return d[actual]
        return None

    def walk(node, stem_hint: str) -> None:
        if isinstance(node, dict):
            text = pick(node, text_keys)
            ident = pick(node, id_keys)
            if text:
                stem = Path(ident).stem if ident else stem_hint
                out.setdefault(stem, text)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v, stem_hint)
        elif isinstance(node, list):
            for v in node:
                walk(v, stem_hint)

    for p in root.rglob("*.json"):
        try:
            walk(json.loads(io.open(p, encoding="utf-8-sig").read()), p.stem)
        except Exception:
            continue
    return out


def load_delimited_transcripts(root: Path) -> dict[str, str]:
    """*.csv / *.tsv / *.txt — '파일명<구분자>텍스트' 형태."""
    out: dict[str, str] = {}
    for p in list(root.rglob("*.csv")) + list(root.rglob("*.tsv")) + list(root.rglob("*.txt")):
        try:
            lines = io.open(p, encoding="utf-8-sig").read().splitlines()
        except Exception:
            continue
        for line in lines:
            for sep in ("|", "\t", ","):
                if sep in line:
                    head, _, tail = line.partition(sep)
                    stem, text = Path(head.strip()).stem, tail.strip().strip('"')
                    if stem and text and re.search(r"[가-힣]", text):
                        out.setdefault(stem, text)
                    break
    return out


LOADERS = (load_json_transcripts, load_delimited_transcripts)
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".pcm")


def probe_seconds(path: Path) -> float:
    """길이(초). wav는 헤더만 읽어 빠르게, 나머지는 ffprobe."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            pass
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def estimate_snr(path: Path) -> float:
    """대략적인 SNR(dB). 상위 10% 프레임 에너지 대비 하위 10%.

    발표에서 노이즈가 절반인 셋이 학습을 망쳤기에 싼 값으로라도 걸러낸다.
    정밀한 측정이 목적이 아니라 '명백히 지저분한 것'을 쳐내는 용도.
    """
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        win = max(1, int(sr * 0.02))
        frames = data[: len(data) // win * win].reshape(-1, win)
        energy = np.sort((frames**2).mean(axis=1))
        if len(energy) < 20:
            return 0.0
        noise = energy[: max(1, len(energy) // 10)].mean()
        signal = energy[-max(1, len(energy) // 10):].mean()
        if noise <= 0 or signal <= 0:
            return 99.0
        return 10.0 * math.log10(signal / noise)
    except Exception:
        return 99.0  # 측정 실패는 통과시킨다(과잉 배제 방지)


def normalize_for_check(text: str) -> str:
    sys.path.insert(0, str(ROOT / "tts" / "sbv2"))
    from style_bert_vits2.nlp.korean.normalizer import normalize_text

    return normalize_text(text)


def scan(src: Path, min_snr: float, verbose: bool) -> tuple[list[Utterance], ScanReport]:
    """코퍼스를 훑어 채택 가능한 발화 목록과 리포트를 만든다."""
    rep = ScanReport()

    transcripts: dict[str, str] = {}
    for loader in LOADERS:
        found = loader(src)
        if verbose:
            print(f"  {loader.__name__}: 전사 {len(found)}건")
        for k, v in found.items():
            transcripts.setdefault(k, v)
    print(f"전사 총 {len(transcripts)}건")

    audios = [p for ext in AUDIO_EXTS for p in src.rglob(f"*{ext}")]
    print(f"오디오 총 {len(audios)}개 — 검사 시작")

    kept: list[Utterance] = []
    for i, path in enumerate(audios):
        rep.total += 1
        if verbose and (i + 1) % 2000 == 0:
            print(f"  검사 {i + 1}/{len(audios)} (채택 {len(kept)})")

        raw = transcripts.get(path.stem)
        if not raw:
            rep.rejects["전사 없음"] += 1
            continue

        text = normalize_for_check(raw)
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            rep.rejects["길이(문자)"] += 1
            continue
        if not re.search(r"[가-힣]", text):
            rep.rejects["한글 없음"] += 1
            continue

        sec = probe_seconds(path)
        if not (MIN_SEC <= sec <= MAX_SEC):
            rep.rejects["길이(초)"] += 1
            continue

        if estimate_snr(path) < min_snr:
            rep.rejects["노이즈"] += 1
            continue

        # 발표의 교훈을 점수로: 원하는 기호와 치찰음이 든 문장을 우선 채택
        score = sum(3.0 for p in WANTED_PUNCT if p in text)
        score += min(len(SIBILANTS.findall(text)), 5) * 0.5
        score += random.random()  # 동점일 때 화자 편중을 막는 흔들기

        speaker = path.parent.name
        u = Utterance(path, text, speaker, sec, score)
        kept.append(u)
        rep.kept += 1
        rep.seconds += sec
        rep.speakers[speaker] += 1
        for p in WANTED_PUNCT:
            if p in text:
                rep.punct[p] += 1
        if SIBILANTS.search(text):
            rep.sibilant_hits += 1

    return kept, rep


def print_report(rep: ScanReport) -> None:
    print(f"\n{'=' * 56}")
    print(f"검사 {rep.total}개 → 채택 {rep.kept}개 ({rep.seconds / 3600:.2f}시간)")
    if rep.rejects:
        print("\n제외 사유:")
        for k, v in rep.rejects.most_common():
            print(f"  {k:12s} {v:7d}")
    print(f"\n화자 {len(rep.speakers)}명 (상위 5: "
          f"{[f'{k}:{v}' for k, v in rep.speakers.most_common(5)]})")
    if rep.kept:
        print("\n문장기호 포함률 — 발표가 지목한 핵심 지표:")
        for p in WANTED_PUNCT:
            n = rep.punct[p]
            mark = "  ← 부족" if p == "…" and n < rep.kept * 0.02 else ""
            print(f"  {p!r:6s} {n:6d}개 ({n / rep.kept * 100:5.1f}%){mark}")
        print(f"  치찰음 포함 {rep.sibilant_hits}개 "
              f"({rep.sibilant_hits / rep.kept * 100:.1f}%)")


def convert(src_file: Path, dst: Path) -> bool:
    af = f"loudnorm={LOUDNORM}"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src_file), "-af", af,
         "-ar", str(TARGET_SR), "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True,
    )
    return r.returncode == 0


def cmd_scan(args) -> None:
    _, rep = scan(Path(args.src), args.min_snr, args.verbose)
    print_report(rep)
    print("\n※ 이 리포트를 보고 --hours를 정한 뒤 build를 실행하세요.")


def cmd_build(args) -> None:
    kept, rep = scan(Path(args.src), args.min_snr, args.verbose)
    print_report(rep)

    # 목표 시간까지 점수 높은 것부터. 기호·치찰음이 든 문장이 먼저 들어간다.
    kept.sort(key=lambda u: -u.score)
    budget = args.hours * 3600
    chosen: list[Utterance] = []
    total = 0.0
    for u in kept:
        if total >= budget:
            break
        chosen.append(u)
        total += u.seconds

    wav_dir = OUT_ROOT / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{len(chosen)}개 ({total / 3600:.2f}시간) 변환 중...")

    lines: list[str] = []
    failed = 0
    for i, u in enumerate(chosen):
        name = f"base_{i:06d}.wav"
        if not convert(u.audio, wav_dir / name):
            failed += 1
            continue
        lines.append(f"{name}|{u.speaker}|KO|{u.text}")
        if (i + 1) % 1000 == 0:
            print(f"  변환 {i + 1}/{len(chosen)}")

    esd = OUT_ROOT / "esd.list"
    io.open(esd, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nesd.list {len(lines)}줄 (변환 실패 {failed}개) → {esd}")

    # 최종 채택본의 기호 분포. 발표의 실패를 반복하지 않으려면 여기서 확인해야 한다.
    final = Counter()
    for line in lines:
        text = line.split("|", 3)[3]
        for p in WANTED_PUNCT:
            if p in text:
                final[p] += 1
    print("최종 채택본 기호 분포:")
    for p in WANTED_PUNCT:
        print(f"  {p!r:6s} {final[p]:6d}개 ({final[p] / max(len(lines), 1) * 100:5.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("scan", cmd_scan), ("build", cmd_build)):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True, help="압축 푼 코퍼스 루트")
        p.add_argument("--min-snr", type=float, default=15.0)
        p.add_argument("--verbose", action="store_true")
        if name == "build":
            p.add_argument("--hours", type=float, default=40.0)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
