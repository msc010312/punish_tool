"""
scrape_mechanics.py  —  Dustloop GGST 시스템 메커닉 스크랩

GGST/Mechanics 위키 페이지(섹션)를 긁어 구조화 -> mechanics_ggst.json.
용도:
  - 지식 참조(코칭·AI 멘트·문서)
  - 그중 '시각 이펙트'(버스트/로망캔슬/카운터히트 등)는 CNN 클래스 후보 -> 라벨러에서 라벨,
    카운터/펀ish 오검출(노란 이펙트 혼동) 해결.

시각 클래스(VISUAL)는 게임 지식으로 직접 지정(섹션 텍스트는 설명으로 보관).
"""
from __future__ import annotations
import json
import re
import urllib.request
import urllib.parse
from pathlib import Path

import punish_engine as pe

API = "https://www.dustloop.com/wiki/api.php"
UA = "Mozilla/5.0 (frame-data study tool; personal use)"
OUT = pe.app_dir() / "mechanics_ggst.json"

# 순간 시각 이펙트 = CNN으로 구분 가능(라벨러 클래스). name -> 화면에서 볼 단서
VISUAL = {
    "Red RC":        "빨간 충격파 (히트 후 로망캔슬)",
    "Yellow RC":     "노란 충격파, 상대 느려짐 (자기 기술 중 발동)",
    "Purple RC":     "보라 충격파 (기술 헛쳤을 때 로망캔슬)",
    "Blue RC":       "파란 충격파 (중립 로망캔슬)",
    "Gold Burst":    "금색 폭발, 중립에서 발동(공격형 버스트)",
    "Blue Burst":    "파란 폭발, 피격/가드 중 탈출(방어형 버스트)",
    "Counter Blitz": "패링/블리츠 섬광",
    "Deflect Shield": "가드 중 발동(P/K+D), 공격 쳐내는 방어 섬광/리플렉트",
    "Counter Hit":   "COUNTER 노란 글자 + 슬로모 (카운터히트 성립)",
    "Overdrive":     "초필살기 컷인/암전 연출",
}


def fetch_wikitext(page: str) -> str:
    q = {"action": "parse", "page": page, "prop": "wikitext", "format": "json"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=30))["parse"]["wikitext"]["*"]


def clean(wt: str) -> str:
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)                  # 템플릿 제거(1패스)
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)                  # 중첩 일부
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)  # [[a|b]]->b
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    wt = re.sub(r"<[^>]+>", "", wt)                         # 태그
    wt = re.sub(r"'''?", "", wt)                            # 볼드/이탤릭
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def split_sections(wt: str) -> list[dict]:
    parts = re.split(r"^(=+)\s*(.+?)\s*=+\s*$", wt, flags=re.M)
    out = []
    # parts: [pre, lvl, name, body, lvl, name, body, ...]
    for i in range(1, len(parts), 3):
        lvl = len(parts[i]); name = parts[i + 1].strip()
        body = clean(parts[i + 2]) if i + 2 < len(parts) else ""
        out.append({"name": name, "level": lvl, "text": body[:1200]})
    return out


def main():
    print("Mechanics 페이지 받는 중…")
    wt = fetch_wikitext("GGST/Mechanics")
    secs = split_sections(wt)
    data = {
        "source": "https://www.dustloop.com/w/GGST/Mechanics",
        "sections": secs,
        "visual_classes": VISUAL,     # 라벨러/CNN 클래스 + 시각 단서
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: 섹션 {len(secs)}개, 시각클래스 {len(VISUAL)}개 -> {OUT}")
    print("시각 클래스(라벨 가능):", ", ".join(VISUAL))


if __name__ == "__main__":
    main()
