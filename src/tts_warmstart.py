# -*- coding: utf-8 -*-
"""tts_warmstart.py — JP-Extra 사전학습 가중치를 한국어 학습의 출발점으로 변환.

배경
  NDC26 발표(넥슨게임즈)는 AI Hub 600GB로 한국어 기반 모델을 처음부터 학습했다.
  하지만 SBV2에서 언어에 의존하는 것은 텍스트 인코더(enc_p)뿐이고, 전체 73.2M 중
  66.6M(91%)를 차지하는 보코더/flow/duration/style은 언어와 무관하게 동작한다.
  그래서 그 91%를 JP-Extra 사전학습에서 그대로 물려받고, 나머지만 한국어로
  적응시킨다. 발표의 "기반 모델부터 만들기"를 소규모 자원에 맞게 축소한 경로.

가중치 형상이 바뀌는 곳(한국어 추가로 커진 테이블)은 기존 행을 그대로 복사하고
새 행만 초기화한다. 특히 음소 임베딩은 JP와 공유하는 심볼(a/i/u/e/o, g/n/m/s/h/N …)이
같은 자리에 남도록 '심볼 이름'으로 대응시킨다 — 인덱스로 자르면 전부 어긋난다.

사용:
  py -3.11 src/tts_warmstart.py --out tts/sbv2/pretrained_ko
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SBV2 = ROOT / "tts" / "sbv2"
sys.path.insert(0, str(SBV2))

# 새로 추가된 행을 채울 난수의 표준편차.
# SBV2가 임베딩을 nn.init.normal_(0, hidden**-0.5)로 만들기 때문에 같은 스케일을 쓴다.
HIDDEN_CHANNELS = 192

# JP-Extra 사전학습 당시의 심볼 순서(=가중치 행 순서). 지금의 SYMBOLS는 한국어가
# 섞여 들어가 순서가 달라졌으므로, 원본 순서를 재현해야 행을 옳게 옮길 수 있다.
def original_symbols() -> list[str]:
    from style_bert_vits2.nlp.symbols import (
        EN_SYMBOLS, JP_SYMBOLS, PAD, PUNCTUATION_SYMBOLS, ZH_SYMBOLS,
    )

    normal = sorted(set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS))
    return [PAD] + normal + PUNCTUATION_SYMBOLS


def convert(src: Path, dst: Path, seed: int) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    from style_bert_vits2.nlp.symbols import (
        LANGUAGE_ID_MAP, NUM_TONES, SYMBOLS,
    )

    torch.manual_seed(seed)
    sd = load_file(str(src))

    old_syms = original_symbols()
    old_index = {s: i for i, s in enumerate(old_syms)}
    std = HIDDEN_CHANNELS**-0.5

    def grow_rows(weight: "torch.Tensor", new_rows: int,
                  row_map: dict[int, int] | None = None) -> "torch.Tensor":
        """행 수를 new_rows로 늘린다. row_map이 있으면 new→old 대응대로 복사."""
        out = torch.randn(new_rows, weight.shape[1]) * std
        if row_map is None:
            out[: weight.shape[0]] = weight
        else:
            for new_i, old_i in row_map.items():
                out[new_i] = weight[old_i]
        return out

    report: list[str] = []

    # 1) 음소 임베딩 — 심볼 '이름'으로 대응. 한국어가 JP와 공유하는 심볼은
    #    학습된 표현을 그대로 물려받는다.
    key = "enc_p.emb.weight"
    if key in sd and sd[key].shape[0] != len(SYMBOLS):
        old = sd[key]
        row_map = {i: old_index[s] for i, s in enumerate(SYMBOLS) if s in old_index}
        sd[key] = grow_rows(old, len(SYMBOLS), row_map)
        report.append(f"{key}: {tuple(old.shape)} -> {tuple(sd[key].shape)}  "
                      f"({len(row_map)}개 심볼 전이, {len(SYMBOLS) - len(row_map)}개 신규)")

    # 2) 톤 임베딩 — 한국어 톤은 뒤에 붙으므로 앞부분을 그대로 둔다.
    key = "enc_p.tone_emb.weight"
    if key in sd and sd[key].shape[0] != NUM_TONES:
        old = sd[key]
        sd[key] = grow_rows(old, NUM_TONES)
        report.append(f"{key}: {tuple(old.shape)} -> {tuple(sd[key].shape)}")

    # 3) 언어 임베딩 — KO 행만 새로 생긴다.
    key = "enc_p.language_emb.weight"
    if key in sd and sd[key].shape[0] != len(LANGUAGE_ID_MAP):
        old = sd[key]
        sd[key] = grow_rows(old, len(LANGUAGE_ID_MAP))
        report.append(f"{key}: {tuple(old.shape)} -> {tuple(sd[key].shape)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(dst))

    if report:
        print(f"  {src.name}")
        for line in report:
            print(f"    {line}")
    else:
        print(f"  {src.name}: 형상 변경 없음(그대로 복사)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SBV2 / "pretrained_jp_extra"))
    ap.add_argument("--out", default=str(SBV2 / "pretrained_ko"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_dir, out_dir = Path(args.src), Path(args.out)
    print(f"웜스타트 변환: {src_dir} -> {out_dir}")
    for name in ("G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"):
        src = src_dir / name
        if not src.exists():
            print(f"  [건너뜀] {name} 없음")
            continue
        convert(src, out_dir / name, args.seed)

    print("\n판별자(D)와 WavLM(WD)은 언어 무관 — 형상 그대로 재사용됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
