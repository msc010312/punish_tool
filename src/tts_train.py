# -*- coding: utf-8 -*-
"""tts_train.py — 준비된 데이터셋을 SBV2로 파인튜닝하는 러너.

tts_prep.py가 만든 tts/dataset/{out}/{wavs,esd.list}를 SBV2가 기대하는
Data/{model}/ 레이아웃으로 스테이징하고, 전처리→학습을 한 번에 돈다.

중요: 이 저장소는 symbols.py를 한국어 포함 125심볼로 늘렸다. 그래서 stock
JP-Extra 사전학습(112심볼)과 형상이 안 맞는다. tts_warmstart.py가 만든
pretrained_ko/G_0(125심볼, JP 심볼 전부 보존)로 갈아끼워야 로드된다.
JP·KO 파인튜닝 모두 이 웜스타트본에서 출발한다.

단계(서브커맨드)
  stage      : dataset → Data/{model}/raw + esd.list
  preprocess : resample→text→bert→style (SBV2 preprocess_all)
  train      : train_ms_jp_extra 실행 (웜스타트 G_0 주입)
  all        : 위 3단계 연속

사용:
  py -3.11 src/tts_train.py all --profile sol_jp --epochs 100 --batch 4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBV2 = ROOT / "tts" / "sbv2"
VENV_PY = ROOT / "tts" / "sbv2_env" / "Scripts" / "python.exe"
DATASET_ROOT = ROOT / "tts" / "dataset"
WARMSTART = SBV2 / "pretrained_ko"

# tts_prep.py의 프로파일과 대응(중복 정의를 피하려고 여기서 직접 임포트).
sys.path.insert(0, str(ROOT / "src"))
from tts_prep import PROFILES  # noqa: E402


def model_dir(model: str) -> Path:
    return SBV2 / "Data" / model


def run(cmd: list[str], cwd: Path) -> None:
    """서브프로세스 실행. venv 파이썬과 UTF-8 콘솔을 강제한다."""
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    import os
    full_env = {**os.environ, **env}
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=full_env, check=True)


def run_sbv2(script: str, script_args: list[str]) -> None:
    """SBV2 스크립트를 _run_noworker 래퍼로 실행한다.

    래퍼가 torchaudio/pyopenjtalk 워커 shim을 대상 import 전에 적용한다.
    """
    run([str(VENV_PY), "_run_noworker.py", script, *script_args], cwd=SBV2)


def cmd_stage(args) -> None:
    prof = PROFILES[args.profile]
    src = DATASET_ROOT / prof.out_name
    esd = src / "esd.list"
    if not esd.exists():
        sys.exit(f"esd.list 없음: {esd}\n먼저 tts_prep.py build --profile {args.profile}")

    md = model_dir(args.model)
    raw = md / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # 이미 44.1kHz 정규화된 wav를 raw/로 복사. preprocess의 resample이 raw→wavs로 옮긴다.
    n = 0
    for wav in (src / "wavs").glob("*.wav"):
        shutil.copy2(wav, raw / wav.name)
        n += 1

    # SBV2는 esd.list 첫 필드를 CWD(sbv2/) 기준 경로로 그대로 Path().is_file() 검사한다.
    # 그래서 bare 파일명을 resample 출력 위치(Data/{model}/wavs/…)로 다시 쓴다.
    wav_rel = f"Data/{args.model}/wavs"
    out_lines = []
    for line in esd.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 1)
        out_lines.append(f"{wav_rel}/{parts[0]}|{parts[1]}")
    (md / "esd.list").write_text("\n".join(out_lines) + "\n", "utf-8")
    print(f"스테이징 완료: wav {n}개 + esd.list(경로 재작성) → {md}")


def initialize(args) -> Path:
    """config.json 생성 + JP-Extra 사전학습을 models/로 스테이징.

    gradio 래퍼(preprocess_all.py)는 무거운 UI 의존이라 우회하고, initialize가 하던 일을
    직접 한다: 템플릿 config를 이 모델용 경로/화자수로 채우고 사전학습을 깐다.
    """
    import json

    md = model_dir(args.model)
    template = json.loads((SBV2 / "configs" / "config_jp_extra.json").read_text("utf-8"))
    template["data"]["training_files"] = str(md / "train.list")
    template["data"]["validation_files"] = str(md / "val.list")
    template["data"]["use_jp_extra"] = True
    template["train"]["batch_size"] = args.batch
    template["train"]["epochs"] = args.epochs
    template["model_name"] = args.model
    cfg_path = md / "config.json"
    cfg_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), "utf-8")

    models = md / "models"
    models.mkdir(parents=True, exist_ok=True)
    for f in ("G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"):
        shutil.copy2(SBV2 / "pretrained_jp_extra" / f, models / f)
    print(f"initialize: config.json + 사전학습 3개 → {md}")
    return cfg_path


def cmd_preprocess(args) -> None:
    md = model_dir(args.model)
    if not (md / "esd.list").exists():
        sys.exit(f"{md}/esd.list 없음. 먼저 stage.")

    cfg = initialize(args)

    # 1) resample: raw → wavs. 우리 wav는 이미 44.1k 정규화라 옵션 없이 통과만.
    run_sbv2("resample.py", ["--sr", "44100",
             "--input_dir", str(md / "raw"), "--output_dir", str(md / "wavs"),
             "--num_processes", "2"])

    # 2) esd.list → train.list/val.list (음소/톤/word2ph 부착)
    run_sbv2("preprocess_text.py", [
        "--transcription-path", str(md / "esd.list"),
        "--cleaned-path", str(md / "esd.list.cleaned"),
        "--train-path", str(md / "train.list"),
        "--val-path", str(md / "val.list"),
        "--config-path", str(cfg),
        "--val-per-lang", "4",
        "--use_jp_extra", "--yomi_error", "skip"])

    # 3) BERT 특징, 4) 스타일 벡터
    run_sbv2("bert_gen.py", ["-c", str(cfg)])
    run_sbv2("style_gen.py", ["-c", str(cfg), "--num_processes", "2"])
    print("preprocess 완료: train.list/val.list + BERT + style 생성됨")


def cmd_warmstart(args) -> None:
    """initialize가 복사한 stock G_0를 웜스타트본으로 교체."""
    md = model_dir(args.model)
    models = md / "models"
    if not models.exists():
        sys.exit(f"{models} 없음. preprocess를 먼저 실행(initialize가 여기 사전학습을 깖).")
    src = WARMSTART / "G_0.safetensors"
    if not src.exists():
        sys.exit(f"웜스타트본 없음: {src}\n먼저 tts_warmstart.py 실행.")
    shutil.copy2(src, models / "G_0.safetensors")
    print(f"웜스타트 G_0 주입: {src} → {models}/G_0.safetensors")
    print("  (D_0/WD_0는 언어 무관이라 stock 그대로 사용)")


def cmd_train(args) -> None:
    cmd_warmstart(args)  # 학습 직전 항상 웜스타트본 보장
    md = model_dir(args.model)
    run_sbv2("train_ms_jp_extra.py", [
        "--config", str(md / "config.json"),
        "--model", str(md),
        "--assets_root", str(SBV2 / "model_assets"),
        "--not_use_custom_batch_sampler"])


def cmd_all(args) -> None:
    cmd_stage(args)
    cmd_preprocess(args)
    cmd_train(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("stage", cmd_stage), ("preprocess", cmd_preprocess),
                     ("warmstart", cmd_warmstart), ("train", cmd_train),
                     ("all", cmd_all)):
        p = sub.add_parser(name)
        p.add_argument("--profile", default="sol_jp", choices=list(PROFILES))
        p.add_argument("--model", default=None, help="Data/ 아래 모델 폴더명(기본=프로파일 out_name)")
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--batch", type=int, default=4)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    if args.model is None:
        args.model = PROFILES[args.profile].out_name
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
