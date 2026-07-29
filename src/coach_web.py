# -*- coding: utf-8 -*-
"""coach_web.py — 온라인 근거(선택적). 인터넷 있을 때만 최신 정보 보강.

로컬 모델은 최신 패치·티어를 신뢰성 있게 모른다(학습 시점 이후 = 환각).
그래서 '최신/패치/너프' 류 질문이면 Dustloop 위키(무료 MediaWiki API, 키 불필요,
GGST 표준 소스)에서 관련 문서 발췌를 가져와 근거로 물린다. 오프라인이면 None →
코치는 '인터넷 없음/오래된 정보'라고 정직하게 답한다(제1원칙 유지).

배포 앱이 스스로 HTTP 요청 → 로컬 LLM에 컨텍스트로 주입(RAG). 과금 API 안 씀.

  web_context(question, chars) -> str | None
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

API = "https://www.dustloop.com/w/api.php"
UA = {"User-Agent": "PunishTool/1.0 (GGST coaching; local)"}

# 최신/외부 정보가 필요한 질문 신호. 하나라도 있으면 온라인 시도.
FRESH_KW = ("최신", "패치", "패치노트", "업데이트", "너프", "버프", "상향", "하향",
            "티어", "밸런스", "시즌", "신캐", "신규", "이번주", "요즘", "근황",
            "patch", "nerf", "buff", "tier", "update", "balance", "season", "meta",
            "찾아", "검색", "알아봐")


def online(timeout: float = 1.5) -> bool:
    try:
        urllib.request.urlopen(
            urllib.request.Request(API + "?action=query&format=json&meta=siteinfo",
                                   headers=UA), timeout=timeout).read(64)
        return True
    except Exception:
        return False


def _get(params: dict, timeout: float = 6.0):
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def search(query: str, n: int = 3) -> list[str]:
    """제목 목록 반환(Dustloop 검색)."""
    try:
        d = _get({"action": "query", "list": "search",
                  "srsearch": query, "srlimit": n})
        return [it["title"] for it in d.get("query", {}).get("search", [])]
    except Exception:
        return []


def extract(title: str, maxc: int = 1200) -> str:
    """문서 평문 발췌."""
    try:
        d = _get({"action": "query", "prop": "extracts", "explaintext": 1,
                  "exintro": 1, "titles": title, "redirects": 1})
        pages = d.get("query", {}).get("pages", {})
        for _, pg in pages.items():
            txt = re.sub(r"\s+", " ", pg.get("extract", "")).strip()
            if txt:
                return txt[:maxc]
    except Exception:
        pass
    return ""


def web_context(question: str, chars: dict | None = None) -> str | None:
    """최신정보 질문이면 Dustloop 발췌를 근거로. 아니면/오프라인이면 None."""
    low = question.lower()
    if not any(k.lower() in low for k in FRESH_KW):
        return None
    if not online():
        return None

    import coach_kb
    who = coach_kb.detect_chars(question, [c for c in (chars or {}).values()
                                           if isinstance(c, str)])
    titles: list[str] = []
    # 패치 질문이면 패치노트 문서 우선
    if any(k in low for k in ("패치", "patch", "업데이트", "update", "밸런스", "balance")):
        titles += search("GGST patch notes", 2)
    # 캐릭터 언급 시 해당 캐릭 위키
    for ch in who[:2]:
        titles += search(f"GGST {ch}", 1)
    if not titles:
        titles = search("GGST " + question, 2)

    blocks, seen = [], set()
    for t in titles[:3]:
        if t in seen:
            continue
        seen.add(t)
        ex = extract(t)
        if ex:
            blocks.append(f"- [{t}] {ex}")
    if not blocks:
        return None
    return "\n".join(blocks)
