# -*- coding: utf-8 -*-
"""한국어 G2P — 정규화된 한글 텍스트를 음소열로 변환한다.

파이프라인 (일본어에서 형태소분석/액센트추출/가나변환을 걷어낸 형태)
    정규화된 한글  ──g2pkk──▶  발음형 한글  ──자모 분해──▶  음소열

핵심 성질: 한국어 음운 변화는 음절 수를 보존한다(삼일전 → 사밀전, 3음절 → 3음절).
덕분에 원문 문자 ↔ 발음 문자가 1:1로 정렬되고, word2ph를 문자 단위로 정확히 만들 수 있다.

음소 심볼은 일본어와 음가가 가까운 것을 최대한 공유한다(a/i/u/e/o, k/g/n/m/s/h ...).
공유된 심볼은 JP-Extra 사전학습 가중치에서 웜스타트할 때 임베딩 행이 그대로 전이된다.
JP에 없는 것만 새로 추가: 모음 ae/eo/eu, 된소리 kk/tt/pp/ss/jj, 받침 kf/nf/tf/lf/mf/pf.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Optional

from style_bert_vits2.nlp.symbols import KO_SYMBOLS as REGISTERED_KO_SYMBOLS
from style_bert_vits2.nlp.symbols import PUNCTUATIONS

# ── 한글 자모 → 음소 ────────────────────────────────────────────────────────
# 초성 19개. 'ㅇ'은 음가가 없으므로 빈 리스트.
ONSET_PHONEMES: list[list[str]] = [
    ["g"], ["kk"], ["n"], ["d"], ["tt"], ["r"], ["m"], ["b"], ["pp"],
    ["s"], ["ss"], [], ["j"], ["jj"], ["ch"], ["k"], ["t"], ["p"], ["h"],
]

# 중성 21개. 반모음은 y/w로 앞에 붙인다.
NUCLEUS_PHONEMES: list[list[str]] = [
    ["a"], ["ae"], ["y", "a"], ["y", "ae"], ["eo"], ["e"], ["y", "eo"], ["y", "e"],
    ["o"], ["w", "a"], ["w", "ae"], ["w", "e"], ["y", "o"], ["u"], ["w", "eo"],
    ["w", "e"], ["w", "i"], ["y", "u"], ["eu"], ["eu", "i"], ["i"],
]

# 종성 28개(0 = 받침 없음). g2pkk가 이미 음절의 끝소리 규칙을 적용하므로
# 실제로는 7종성(ㄱㄴㄷㄹㅁㅂㅇ)만 들어오지만, 안전하게 전부 대표음으로 사상한다.
CODA_PHONEMES: list[list[str]] = [
    [],                                     # 받침 없음
    ["kf"], ["kf"], ["kf"],                 # ㄱ ㄲ ㄳ
    ["nf"], ["nf"], ["nf"],                 # ㄴ ㄵ ㄶ
    ["tf"],                                 # ㄷ
    ["lf"], ["kf"], ["mf"], ["pf"], ["lf"], ["tf"], ["pf"], ["lf"],  # ㄹ~ㅀ
    ["mf"], ["pf"], ["pf"],                 # ㅁ ㅂ ㅄ
    ["tf"], ["tf"],                         # ㅅ ㅆ
    ["N"],                                  # ㅇ  (JP의 발음(撥音) N과 음가가 가까워 공유)
    ["tf"], ["kf"], ["tf"], ["pf"], ["tf"],  # ㅈ ㅊ ㅋ ㅌ ㅍ
]

# JP_SYMBOLS에 없어서 새로 추가해야 하는 한국어 전용 음소
KO_EXTRA_SYMBOLS: list[str] = [
    "ae", "eo", "eu",                      # 모음
    "kk", "tt", "pp", "ss", "jj",          # 된소리
    "kf", "nf", "tf", "lf", "mf", "pf",    # 받침(불파음/유음/비음)
]

KO_SYMBOLS: list[str] = sorted(
    {p for table in (ONSET_PHONEMES, NUCLEUS_PHONEMES, CODA_PHONEMES)
     for phs in table for p in phs}
)

# 위 표가 만들어내는 음소와 symbols.py에 등록된 목록이 어긋나면 임베딩 인덱스가
# 조용히 밀린다. 학습 후에야 발음이 깨진 걸 발견하는 사고를 막으려고 import 시점에 막는다.
if set(KO_SYMBOLS) != set(REGISTERED_KO_SYMBOLS):
    raise RuntimeError(
        "한국어 음소표와 symbols.KO_SYMBOLS가 불일치합니다. "
        f"표에만 있음={sorted(set(KO_SYMBOLS) - set(REGISTERED_KO_SYMBOLS))}, "
        f"symbols에만 있음={sorted(set(REGISTERED_KO_SYMBOLS) - set(KO_SYMBOLS))}"
    )

# 한국어는 성조 언어가 아니므로 톤은 1종(전부 0)만 쓴다.
NUM_KO_TONES = 1

__HANGUL_BASE = 0xAC00
__HANGUL_LAST = 0xD7A3

__g2p_engine: Optional[object] = None


def __install_eunjeon_shim() -> None:
    """g2pkk는 형태소 분석기로 eunjeon을 import하는데 윈도우에서 빌드가 안 된다.

    동일한 MeCab-ko를 프리빌트 휠로 제공하는 python-mecab-ko를 eunjeon 이름으로
    끼워 넣는다. 두 패키지 모두 pos(text) -> [(형태소, 품사)] 시그니처로 동일하다.
    """
    if "eunjeon" in sys.modules:
        return
    import mecab  # python-mecab-ko

    shim = types.ModuleType("eunjeon")
    # find_spec()가 __spec__를 요구하므로 비어 있는 스펙을 달아 준다.
    shim.__spec__ = importlib.machinery.ModuleSpec("eunjeon", None)

    class Mecab:
        def __init__(self, *args, **kwargs) -> None:
            self.__mecab = mecab.MeCab()

        def pos(self, text: str):
            return self.__mecab.pos(text)

    shim.Mecab = Mecab  # type: ignore[attr-defined]
    sys.modules["eunjeon"] = shim


def __get_engine():
    """g2pkk 인스턴스는 사전 로딩이 무거우므로 한 번만 만든다."""
    global __g2p_engine
    if __g2p_engine is None:
        __install_eunjeon_shim()
        from g2pkk import G2p

        __g2p_engine = G2p()
    return __g2p_engine


def __syllable_to_phonemes(char: str) -> list[str]:
    """완성형 한글 한 글자를 음소 리스트로."""
    code = ord(char) - __HANGUL_BASE
    onset, rest = divmod(code, 588)
    nucleus, coda = divmod(rest, 28)
    return (ONSET_PHONEMES[onset] + NUCLEUS_PHONEMES[nucleus]
            + CODA_PHONEMES[coda])


def __is_hangul_syllable(char: str) -> bool:
    return __HANGUL_BASE <= ord(char) <= __HANGUL_LAST


def hangul_to_pronunciation(norm_text: str) -> str:
    """표기형 한글 → 발음형 한글 (연음·경음화·비음화·구개음화 등 적용).

    음절 수가 보존되지 않으면 문자 정렬이 깨지므로, 그럴 때는 원문을 그대로 쓴다.
    (드물지만 g2pkk가 기호 주변에서 길이를 바꾸는 경우가 있다.)
    """
    try:
        spoken = __get_engine()(norm_text)
    except Exception:
        return norm_text
    return spoken if len(spoken) == len(norm_text) else norm_text


def g2p(norm_text: str) -> tuple[list[str], list[int], list[int]]:
    """정규화된 한국어 텍스트를 (음소, 톤, word2ph)로 변환한다.

    japanese/g2p.py와 동일한 계약을 따른다:
      - phones/tones 양끝에 '_'가 붙는다
      - len(word2ph) == len(norm_text) + 2

    Returns:
        (phones, tones, word2ph)
    """
    spoken = hangul_to_pronunciation(norm_text)

    phones: list[str] = ["_"]
    word2ph: list[int] = [1]

    for src_char, spk_char in zip(norm_text, spoken):
        if __is_hangul_syllable(spk_char):
            got = __syllable_to_phonemes(spk_char)
        elif spk_char in PUNCTUATIONS:
            got = [spk_char]
        else:
            # 공백 등 발음이 없는 문자. BERT 정렬을 위해 자리는 유지하되 음소는 0개.
            got = []
        phones += got
        # word2ph는 '원문' 문자 기준이어야 BERT 특징과 맞는다.
        word2ph.append(len(got))

    phones.append("_")
    word2ph.append(1)

    # 음소가 0개인 문자가 있으면 BERT 특징이 소실되므로, 인접 문자로 몰아준다.
    word2ph = __redistribute_empty(word2ph)

    tones = [0] * len(phones)
    assert len(word2ph) == len(norm_text) + 2, (
        f"word2ph 길이 불일치: {len(word2ph)} != {len(norm_text) + 2}")
    assert sum(word2ph) == len(phones), (
        f"음소 수 불일치: {sum(word2ph)} != {len(phones)}")
    return phones, tones, word2ph


def __redistribute_empty(word2ph: list[int]) -> list[int]:
    """0인 칸을 없앤다.

    Bert-VITS2 계열은 word2ph에 0이 있어도 동작하지만, 그 문자의 BERT 특징이
    통째로 버려진다. 공백이 문맥을 가르는 위치라 정보를 잃지 않도록 오른쪽
    (없으면 왼쪽) 이웃에게 1을 빌려 준다. 총합은 보존된다.
    """
    out = list(word2ph)
    for i, v in enumerate(out):
        if v != 0:
            continue
        for j in list(range(i + 1, len(out))) + list(range(i - 1, -1, -1)):
            if out[j] > 1:
                out[j] -= 1
                out[i] = 1
                break
    return out
