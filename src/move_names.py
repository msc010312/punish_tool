# -*- coding: utf-8 -*-
"""공식 한국어 기술명 룩업 — move_names_ko.json (guiltygear.com/ggst/kr 커맨드리스트 기반).

프레임데이터의 move.name(numpad 표기)을 공식 한국어 기술명으로 매핑한다.
'236[H]'(차지)·'236P~6P'(파생)·'236S/H'(멀티버튼)·'j.236HH'(연타) 등 변형은
점진적으로 단순화해 base 입력으로 조회한다. 매핑 없으면 None(=통상기/미등록).
"""
import functools
import json
import re

import punish_engine as pe


@functools.lru_cache(maxsize=1)
def _load() -> dict:
    p = pe.app_dir() / "move_names_ko.json"
    try:
        d = json.load(p.open(encoding="utf-8"))
    except (OSError, ValueError):
        return {}                       # 파일 없거나 깨져도 주석만 생략(크래시 X)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _cands(name: str):
    """move.name 에서 조회 후보를 점진적으로 단순화하며 생성."""
    n = name.strip()
    yield n
    n = n.split(" or ")[0].strip()          # '236S or 214S' -> 첫 대안
    yield n
    base = n.split("~")[0].strip()          # '236P~6P' -> '236P'(파생 제거)
    yield base
    nb = base.replace("[", "").replace("]", "")   # '236[H]' -> '236H'(차지 제거)
    yield nb
    m = re.match(r"^(j\.)?(\d+)([PKSHD])(?:/[PKSHD])+$", nb)  # '22K/S/H' -> '22K'
    if m:
        yield f"{m.group(1) or ''}{m.group(2)}{m.group(3)}"
    m2 = re.match(r"^(j\.)?(\d+)([PKSHD])\3+$", nb)          # 'j.236HH' -> 'j.236H'
    if m2:
        yield f"{m2.group(1) or ''}{m2.group(2)}{m2.group(3)}"


def ko(char: str, move_name: str) -> str | None:
    tbl = _load().get(char)
    if not tbl or not move_name:
        return None
    for c in _cands(move_name):
        if c in tbl:
            return tbl[c]
    return None


def annotate(char: str, move_name: str) -> str:
    """'236K' -> '236K(용인)'. 매핑 없으면 원본 그대로."""
    k = ko(char, move_name)
    return f"{move_name}({k})" if k else move_name
