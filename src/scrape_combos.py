"""
scrape_combos.py  —  Dustloop 콤보 페이지에서 '초보자 콤보' 수집 (스타터별)

각 캐릭터의 .../Combos 페이지 위키텍스트에서 Beginner Combos 의 Recipe(콤보 표기)·난이도·설명을
긁어, 시동기(starter)별로 색인한다. 무브 ID가 시동기를 알려주면 그에 맞는 콤보를 추천하는 데이터.

예) Sol 'P 스타터' -> "5P or 2P > (5P or 2P xN) > (6P) > 6H/236KK"

저장: combos_ggst.json  { 캐릭: [ {title,difficulty,recipe,starters:[...],desc}, ... ] }
사용: python scrape_combos.py
"""
from __future__ import annotations
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.dustloop.com/wiki/api.php"
UA = "Mozilla/5.0 (frame-data study tool; personal use)"
OUT = Path(__file__).resolve().parent.parent / "combos_ggst.json"


def clean(s: str) -> str:
    s = re.sub(r"\{\{clr\|[^|]*\|([^}]*)\}\}", r"\1", s)   # {{clr|P|5P}} -> 5P
    s = re.sub(r"\{\{[^}]*\}\}", "", s)                    # 나머지 템플릿 제거
    s = re.sub(r"\[\[[^]|]*\|([^]]*)\]\]", r"\1", s)       # [[a|b]] -> b
    s = re.sub(r"\[\[([^]]*)\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)                         # html
    s = re.sub(r"'''?", "", s)
    return re.sub(r"\s+", " ", s).strip()


def starters_of(recipe: str) -> list[str]:
    """레시피 첫 구간에서 시동기 토큰 추출. '5P or 2P > ...' -> ['5P','2P']."""
    head = recipe.split(">")[0]
    head = re.sub(r"\b(AA|CH|dl\.?|delay|jc|adc|WS|CL|x?N)\b", " ", head, flags=re.I)
    head = head.replace("(", " ").replace(")", " ")
    toks = re.findall(r"j?\.?[1-9cf]?\.?[1-9]*[PKSHD]+", head)  # 5P,2K,c.S,6H,j.S,236K 등
    out, seen = [], set()
    for t in toks:
        t = t.strip(".")
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out[:4]


def fetch_combos(chara: str) -> list[dict]:
    page = "GGST/" + chara.replace(" ", "_") + "/Combos"
    q = {"action": "parse", "page": page, "prop": "wikitext", "format": "json"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))
    if "error" in d:
        return []
    wt = d["parse"]["wikitext"]["*"]
    combos, seen = [], set()

    def fld(block, key):
        m = re.search(r"\|\s*" + key + r"\s*=\s*(.+)", block)
        return clean(m.group(1)) if m else ""

    # 1) 상세 콤보표 {{GGST-ComboTableRow}} — combo·damage·position·difficulty·notes (주력)
    for block in re.split(r"\{\{GGST-ComboTableRow", wt)[1:]:
        cm = re.search(r"\|\s*combo\s*=\s*(.+)", block)
        if not cm:
            continue
        rc = clean(cm.group(1))
        if not rc or rc in seen:
            continue
        seen.add(rc)
        dm = re.search(r"\|\s*damage\s*=\s*(\d+)", block)
        combos.append({
            "title": "",
            "difficulty": fld(block, "difficulty"),
            "recipe": rc,
            "starters": starters_of(rc),
            "damage": int(dm.group(1)) if dm else 0,
            "position": fld(block, "position")[:16],
            "desc": fld(block, "notes")[:140],
        })

    # 2) 초보 콤보(Recipe=) — 데미지 없음, 보충용
    for m in re.finditer(r"\|\s*Recipe\s*=\s*(.+)", wt):
        rc = clean(m.group(1))
        if not rc or rc in seen:
            continue
        seen.add(rc)
        block = wt[max(0, m.start() - 900):m.start()]
        tm = re.findall(r"\|\s*Title\s*=\s*(.+)", block)
        dm = re.findall(r"\|\s*Difficulty\s*=\s*(.+)", block)
        om = re.findall(r"\|\s*Oneliner\s*=\s*(.+)", block)
        combos.append({
            "title": clean(tm[-1]) if tm else "",
            "difficulty": clean(dm[-1]) if dm else "",
            "recipe": rc,
            "starters": starters_of(rc),
            "damage": 0,
            "position": "",
            "desc": clean(om[-1])[:140] if om else "",
        })
    return combos


def main():
    chars = list(json.loads(
        (Path(__file__).resolve().parent.parent / "framedata_ggst.json").read_text(encoding="utf-8")))
    data = {}
    for c in chars:
        try:
            cb = fetch_combos(c)
        except Exception as e:
            print(f"  ! {c}: {e}"); cb = []
        data[c] = cb
        print(f"  {c:22} 초보콤보 {len(cb)}개")
        time.sleep(0.5)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for v in data.values())
    print(f"완료: {len(data)}캐릭 / 콤보 {total}개 -> {OUT.name}")


if __name__ == "__main__":
    main()
