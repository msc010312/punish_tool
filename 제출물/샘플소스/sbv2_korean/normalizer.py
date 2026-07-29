# -*- coding: utf-8 -*-
"""한국어 노멀라이저 — 숫자/기호를 읽을 수 있는 한글로 바꾼다.

G2P의 첫 단계. 발음 변환(g2p.py)은 입력이 한글이라고 가정하므로,
숫자·라틴문자·특수기호는 여기서 전부 한글이나 SBV2 punctuation으로 정리된다.

도메인 전용 읽기(예: 격투게임의 'F' → '프레임')는 여기 넣지 않는다.
그건 애플리케이션 쪽 전처리 책임이고, 여기는 범용 한국어 규칙만 담는다.
"""

from __future__ import annotations

import re
import unicodedata

from style_bert_vits2.nlp.symbols import PUNCTUATIONS

__SINO_DIGITS = "영일이삼사오육칠팔구"
__SINO_UNITS = ["", "십", "백", "천"]
__SINO_BIG = ["", "만", "억", "조", "경"]

# 발음이 없는(=읽지 않는) 기호를 SBV2 punctuation으로 사상
__PUNCT_MAP = {
    "：": ",", ":": ",", "；": ",", ";": ",",
    "，": ",", "、": ",", "。": ".", "．": ".",
    "！": "!", "？": "?", "‥": "…", "···": "…", "...": "…", "···": "…",
    "〜": "…", "~": "…", "―": "-", "—": "-", "–": "-",
    "“": "'", "”": "'", "‘": "'", "’": "'", '"': "'",
    "(": ",", ")": ",", "[": ",", "]": ",", "{": ",", "}": ",",
    "「": ",", "」": ",", "『": ",", "』": ",", "《": ",", "》": ",",
    "/": ",", "\\": ",", "·": ",", "…": "…",
}

# 라틴 알파벳 낱자 읽기 (약어가 그대로 남았을 때의 최후 수단)
__ALPHABET_KO = {
    "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프",
    "g": "지", "h": "에이치", "i": "아이", "j": "제이", "k": "케이",
    "l": "엘", "m": "엠", "n": "엔", "o": "오", "p": "피", "q": "큐",
    "r": "알", "s": "에스", "t": "티", "u": "유", "v": "브이",
    "w": "더블유", "x": "엑스", "y": "와이", "z": "지",
}

__CURRENCY_KO = {"₩": "원", "$": "달러", "€": "유로", "£": "파운드", "¥": "엔"}


def __sino_under_10000(n: int) -> str:
    """1~9999를 한자어 수사로. 십/백/천 앞의 '일'은 생략한다(십사 O, 일십사 X)."""
    out = ""
    for pos in range(3, -1, -1):
        d = (n // (10**pos)) % 10
        if d == 0:
            continue
        # 십/백/천 자리의 1은 읽지 않음
        if d == 1 and pos > 0:
            out += __SINO_UNITS[pos]
        else:
            out += __SINO_DIGITS[d] + __SINO_UNITS[pos]
    return out


def sino_number(n: int) -> str:
    """정수를 한자어 수사로 읽는다. 프레임/초/퍼센트 등 대부분의 단위가 이 계열."""
    if n == 0:
        return "영"
    sign = "마이너스 " if n < 0 else ""
    n = abs(n)
    chunks: list[str] = []
    idx = 0
    while n > 0:
        part = n % 10000
        if part:
            chunks.append(__sino_under_10000(part) + __SINO_BIG[idx])
        n //= 10000
        idx += 1
    return sign + "".join(reversed(chunks))


def __read_number(match: re.Match[str]) -> str:
    """정수/소수/천단위 콤마를 한글 읽기로."""
    raw = match.group(0)
    neg = raw.startswith("-")
    body = raw.lstrip("-").replace(",", "")
    if "." in body:
        head, tail = body.split(".", 1)
        head_ko = sino_number(int(head)) if head else "영"
        tail_ko = "".join(__SINO_DIGITS[int(c)] for c in tail if c.isdigit())
        text = f"{head_ko} 점 {tail_ko}"
    else:
        text = sino_number(int(body)) if body else ""
    return ("마이너스 " if neg else "") + text


def __read_alphabet(match: re.Match[str]) -> str:
    # 낱자 읽기도 앞뒤를 띄운다. '육에프'가 아니라 '육 에프'로 끊어 읽어야 자연스럽다.
    return " " + "".join(__ALPHABET_KO.get(c, "") for c in match.group(0).lower()) + " "


def normalize_text(text: str) -> str:
    """SBV2가 요구하는 형태(한글 + 허용 punctuation)로 정규화한다."""
    text = unicodedata.normalize("NFKC", text)

    # 단위/기호를 읽을 때는 앞뒤에 공백을 둔다. 붙여 두면 뒤에서 숫자를 읽은 뒤
    # '오퍼센트'처럼 한 어절이 되어 끊어 읽기가 사라진다.
    for src, dst in __CURRENCY_KO.items():
        text = text.replace(src, f" {dst} ")
    text = text.replace("%", " 퍼센트 ").replace("&", " 그리고 ")
    text = text.replace("+", " 플러스 ").replace("=", " 는 ")

    # 숫자보다 먼저 기호를 정리하면 '-'가 마이너스인지 대시인지 구분이 사라진다.
    # 따라서 숫자(부호/소수점/콤마 포함)를 먼저 읽어 치운다.
    text = re.sub(r"-?\d[\d,]*(?:\.\d+)?", __read_number, text)

    for src, dst in __PUNCT_MAP.items():
        text = text.replace(src, dst)

    text = re.sub(r"[A-Za-z]+", __read_alphabet, text)

    # 한글·허용 punctuation·공백만 남긴다
    allowed = "".join(re.escape(p) for p in PUNCTUATIONS)
    text = re.sub(rf"[^가-힣{allowed}\s]", "", text)

    # 중복 punctuation 축약, 공백 정리
    text = re.sub(r"([!?,.…'-])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
