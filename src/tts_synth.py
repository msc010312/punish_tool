# -*- coding: utf-8 -*-
"""tts_synth.py — 학습한 SBV2 모델로 음성을 합성한다(JP·KO 공용).

학습은 Data/{model}/models/에 G_*.pth 체크포인트를 남긴다. 추론(TTSModel)은
.safetensors + style_vectors.npy를 요구하므로, 이 스크립트가 부족한 자산을 만든다:
  1) style_vectors.npy 없으면 per-file .npy를 평균내 Neutral 스타일 생성(default_style)
  2) G_*.pth의 'model' state_dict를 .safetensors로 변환
그다음 TTSModel로 문장들을 합성해 out_dir에 wav로 저장한다.

사용:
  py -3.11 src/tts_synth.py --model-dir tts/sbv2/Data/SOL_JP --lang JP \
      --texts "くたばりやがれ" "そいつは俺の獲物だ" --out-dir tts/samples/sol_jp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBV2 = ROOT / "tts" / "sbv2"
sys.path.insert(0, str(SBV2))


def _shims() -> None:
    """torch.load weights_only 등 _run_noworker와 동일한 호환 shim."""
    import torch

    if not hasattr(__import__("torchaudio"), "set_audio_backend"):
        import torchaudio
        torchaudio.set_audio_backend = lambda *a, **k: None  # type: ignore
    _orig = torch.load

    def _load(*a, **k):
        k["weights_only"] = False
        return _orig(*a, **k)

    torch.load = _load  # type: ignore


def latest_checkpoint(models_dir: Path) -> Path:
    cands = sorted(models_dir.glob("G_*.pth"),
                   key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
    if not cands:
        raise SystemExit(f"체크포인트 없음: {models_dir}/G_*.pth")
    return cands[-1]


def ensure_style_vectors(model_dir: Path, config_path: Path) -> Path:
    """style_vectors.npy 보장. 없으면 학습 wav들의 per-file .npy를 평균내 생성."""
    sv = model_dir / "style_vectors.npy"
    if sv.exists():
        return sv
    from default_style import save_neutral_vector

    save_neutral_vector(str(model_dir / "wavs"), str(model_dir),
                        str(config_path), str(config_path))
    print(f"style_vectors.npy 생성 → {sv}")
    return sv


def pth_to_safetensors(pth: Path) -> Path:
    """G_*.pth의 model state_dict를 .safetensors로 변환(추론용)."""
    import torch
    from safetensors.torch import save_file

    out = pth.with_suffix(".safetensors")
    if out.exists():
        return out
    ckpt = torch.load(str(pth), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # safetensors는 연속 텐서만 저장 가능 → contiguous화
    state = {k: v.contiguous() for k, v in state.items()
             if hasattr(v, "contiguous")}
    save_file(state, str(out))
    print(f"safetensors 변환 → {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="Data/{model} 경로")
    ap.add_argument("--lang", default="JP", choices=["JP", "KO", "EN", "ZH"])
    ap.add_argument("--checkpoint", default=None, help="G_*.pth (기본=최신)")
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--speaker-id", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--noise-w", type=float, default=0.5,
                    help="SDP 노이즈(음소 길이 흔들림). 낮추면 마찰음/자음이 덜 번짐. "
                         "솔 KO 확정값 0.5(ㅆ 번짐 완화)")
    ap.add_argument("--sdp-ratio", type=float, default=0.2,
                    help="DP↔SDP 혼합비. 낮추면 템포가 균일(덜 흔들림)")
    ap.add_argument("--intonation-scale", type=float, default=1.0,
                    help="억양 폭. >1 억양 강조(덜 밋밋), <1 평탄")
    ap.add_argument("--pitch-scale", type=float, default=1.0,
                    help="피치 배율(음높이). 1.0 유지 권장")
    ap.add_argument("--style", default="Neutral",
                    help="감정 스타일(Neutral/Battle/Story 등 config style2id)")
    ap.add_argument("--style-weight", type=float, default=1.0,
                    help="감정 강도. 클수록 스타일 평균에서 더 벌어짐(1~5)")
    ap.add_argument("--length", type=float, default=1.0)
    args = ap.parse_args()

    _shims()
    import soundfile as sf
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel

    model_dir = (ROOT / args.model_dir) if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    if not model_dir.exists():
        model_dir = SBV2 / args.model_dir  # Data/... 상대 지정 허용
    config_path = model_dir / "config.json"
    models_dir = model_dir / "models"

    pth = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(models_dir)
    if not pth.is_absolute():
        pth = models_dir / pth.name

    sv = ensure_style_vectors(model_dir, config_path)
    safet = pth_to_safetensors(pth)

    print(f"모델 로드: {safet.name} (lang={args.lang}, device={args.device})")
    model = TTSModel(model_path=safet, config_path=config_path,
                     style_vec_path=sv, device=args.device)
    model.load()

    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang = Languages[args.lang]
    for i, text in enumerate(args.texts):
        sr, audio = model.infer(text=text, language=lang, speaker_id=args.speaker_id,
                                noise=args.noise, noise_w=args.noise_w,
                                sdp_ratio=args.sdp_ratio, length=args.length,
                                intonation_scale=args.intonation_scale,
                                pitch_scale=args.pitch_scale,
                                style=args.style, style_weight=args.style_weight)
        out = out_dir / f"sample_{i:02d}.wav"
        sf.write(str(out), audio, sr)
        print(f"  [{i}] {out.name}  ({len(audio) / sr:.2f}s)  <- {text}")

    print(f"\n완료: {len(args.texts)}개 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
