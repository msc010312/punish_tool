# -*- coding: utf-8 -*-
"""한국어 BERT 특징 추출.

일본어/중국어 모듈은 문자 단위 토크나이저를 쓰기 때문에 토큰 i가 곧 문자 i였다.
한국어 BERT는 서브워드 토크나이저라 '한 토큰 = 여러 음절'이므로, offset mapping으로
토큰 특징을 문자 단위로 펼쳐서 word2ph(문자 기준)와 정렬시킨다.

BERT는 히든 사이즈가 1024인 모델이어야 한다. SBV2의 TextEncoder가
`nn.Conv1d(1024, hidden_channels, 1)`로 고정되어 있기 때문.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models

if TYPE_CHECKING:
    import torch


def __char_level_features(
    text: str, tokenizer, res: "torch.Tensor"
) -> "torch.Tensor":
    """토큰 단위 특징 [T, H] → 문자 단위 특징 [len(text)+2, H].

    앞뒤의 CLS/SEP 자리는 그대로 두고, 가운데를 원문 문자 수만큼 채운다.
    """
    import torch

    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    # 문자 위치 → 그 문자를 포함하는 토큰 인덱스
    owner = [-1] * len(text)
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end:  # CLS/SEP 등 특수 토큰
            continue
        for c in range(start, min(end, len(text))):
            owner[c] = tok_idx

    # 어떤 토큰에도 안 잡힌 문자(주로 공백)는 왼쪽 이웃을 따라간다.
    last = 0
    for c in range(len(text)):
        if owner[c] == -1:
            owner[c] = last
        else:
            last = owner[c]

    middle = torch.stack([res[owner[c]] for c in range(len(text))])
    return torch.cat([res[0:1], middle, res[-1:]], dim=0)


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> "torch.Tensor":
    """한국어 텍스트에서 음소 단위 BERT 특징을 뽑는다.

    Args:
        text: 정규화된 한국어 텍스트
        word2ph: 원문 각 문자에 음소가 몇 개 배정되는지 (len == len(text) + 2)
        device: 추론 디바이스
        assist_text: 보조 텍스트(스타일 힌트)
        assist_text_weight: 보조 텍스트 가중치

    Returns:
        torch.Tensor: [1024, 음소 수]
    """
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = bert_models.load_model(Languages.KO, device_map=device)
    bert_models.transfer_model(Languages.KO, device)
    tokenizer = bert_models.load_tokenizer(Languages.KO)

    style_res_mean = None
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True)
        # 일본어/중국어 구현과 동일하게 뒤에서 세 번째 은닉층을 쓴다.
        res = torch.cat(out["hidden_states"][-3:-2], -1)[0].cpu()
        res = __char_level_features(text, tokenizer, res)

        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            style_inputs = {k: v.to(device) for k, v in style_inputs.items()}
            style_out = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_out["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == len(text) + 2, (
        f"word2ph {len(word2ph)} != len(text)+2 {len(text) + 2}: {text!r}")

    phone_level_feature = []
    for i, repeat in enumerate(word2ph):
        if assist_text:
            assert style_res_mean is not None
            feat = (res[i].repeat(repeat, 1) * (1 - assist_text_weight)
                    + style_res_mean.repeat(repeat, 1) * assist_text_weight)
        else:
            feat = res[i].repeat(repeat, 1)
        phone_level_feature.append(feat)

    return torch.cat(phone_level_feature, dim=0).T
