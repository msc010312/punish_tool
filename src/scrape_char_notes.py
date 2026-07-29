"""
scrape_char_notes.py — Dustloop 캐릭터 페이지에서 개요/고유시스템 텍스트 수집.

각 캐릭 페이지(GGST/<char>)의 렌더링 HTML에서 리드 문단 + 'Unique Mechanic' 섹션을
평문으로 추출해 char_notes_ggst.json 저장. 코칭 리포트의 '상대 캐릭 이해' 코너에 주입용.
사용: python scrape_char_notes.py
"""
from __future__ import annotations
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import punish_engine as pe

API = "https://www.dustloop.com/wiki/api.php"
UA = "Mozilla/5.0 (frame-data study tool; personal use)"
OUT = Path(__file__).resolve().parent.parent / "char_notes_ggst.json"
DELAY = 0.8


def fetch_html(page: str, section: str | None = None) -> str | None:
    params = {"action": "parse", "page": page, "prop": "text", "format": "json",
              "redirects": "1"}
    if section is not None:
        params["section"] = section
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d.get("parse", {}).get("text", {}).get("*")
    except Exception:
        return None


def strip_tags(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = re.sub(r"&#\d+;|&\w+;", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def clean_notes(t: str) -> str:
    """무브 링크 호버카드의 프레임데이터 표가 텍스트로 섞여든 것 제거."""
    kw = r'(?:Guard|Startup|Active|Recovery|Total|Advantage|Level|Invuln\w*|Prorat\w*|RISC\w*|Attribute|Blockstun|Hitstun)'
    unit = kw + r'\s*(?:All|High|Low|Mid|[-+]?\d+~?\d*[.%]?|Throw|Foot|Head)?\s*'
    t = re.sub(r'(?:' + unit + r'){2,}', ' ', t)             # 표조각(유닛 2개+ 연속) 제거
    t = re.sub(r'\bGuard\s+(?:All|High|Low|Mid)\b(?!\s*Crush)', '', t)  # 잔재 단독 Guard레벨
    t = re.sub(r'\(\s*call\s+\w+\s*,?\s*', '(', t)
    t = re.sub(r'\s*,\s*\)', ')', t)
    t = re.sub(r'\(\s*[,)]*\s*\)', '', t)
    t = re.sub(r'\s{2,}', ' ', t).strip()
    t = re.sub(r'\bBedman\?', 'Bedman', t)
    t = re.sub(r'\bedit\b', ' ', t)                   # 위키 'edit' 링크 잔재
    # 전작 역사 언급 문장 제거 (Strive 코칭 무관: Déjà Vu 등 Xrd 얘기)
    hist = re.compile(r'previous (?:iteration|game|version)|in Xrd|last game|used to (?:have|be)', re.I)
    sents = [s for s in re.split(r'(?<=[.!])\s+', t) if not hist.search(s)]
    return re.sub(r'\s{2,}', ' ', " ".join(sents)).strip()   # 노이즈만 제거(문장 자르기는 호출측)


def first_sents(t: str, n: int) -> str:
    return " ".join(re.split(r'(?<=[.!])\s+', t)[:n]).strip()


def extract(html: str) -> dict:
    """Overview 섹션에서 아키타입 요약 + 'Unique Mechanic:' 라벨 메커닉을 분리 추출."""
    txt = strip_tags(html)
    txt = re.sub(r"^.{0,200}?Frame Data Resources(?: Overview)?\s*", "", txt)  # 상단 탭 제거
    low = txt.lower()
    mi = low.find("unique mechanic")
    # 아키타입 = Unique Mechanic 앞부분(없으면 전체) 정제 -> 앞 2문장
    overview = first_sents(clean_notes(txt[:mi] if mi >= 0 else txt), 2)
    # 메커닉 = 'Unique Mechanic(s)' 라벨 이후 ~ 다음 큰 섹션(Normal Moves 등) 전까지
    mech = ""
    if mi >= 0:
        tail = txt[mi:]
        tail = re.split(r"\bNormal Moves\b|\bMove List\b|\bCombos\b", tail)[0]
        tail = re.sub(r"^Unique Mechanics?\s*(?:edit)?\s*", "", tail, flags=re.I)
        mech = clean_notes(tail)
    return {"lead": overview[:600], "unique": overview[:600], "mechanic": mech[:900]}


def main():
    data = pe.load_framedata_file()
    # 기존 한국어(ko) 보존
    old = {}
    if OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))
    notes = {}
    for ch in data:
        html = fetch_html(f"GGST/{ch}", section="1")     # Overview 섹션
        if not html:
            print(f"  ! {ch}: 페이지 없음", flush=True)
            continue
        notes[ch] = extract(html)
        for k in ("ko", "ko_mech"):                       # 사람 번역·검수분 유지
            if ch in old and k in old[ch]:
                notes[ch][k] = old[ch][k]
        print(f"  ✓ {ch}: 개요 {len(notes[ch]['unique'])}자, 메커닉 {len(notes[ch]['mechanic'])}자", flush=True)
        time.sleep(DELAY)
    OUT.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 -> {OUT} ({len(notes)}캐릭)", flush=True)


if __name__ == "__main__":
    main()
