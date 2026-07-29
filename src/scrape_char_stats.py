# -*- coding: utf-8 -*-
"""scrape_char_stats.py — 캐릭터 방어 스탯(Defense·Guts) 수집기.

Dustloop Cargo 테이블 `ggstCharacters` 에서 캐릭별 defense/guts 를 받아
char_stats_ggst.json 으로 저장한다. 시동기 데미지 역산(무브 후보 좁히기)에 쓰인다.

  데미지 공식(Dustloop GGST/Damage 문서):
    받는 데미지 = base × (defense+256)/256 × guts배율(등급, 현재HP%)
  Guts 배율표(GGST/Frame Data#Guts): >70% 구간은 100%.

변형 상태(나고 혈량, 지오 텐션 등)는 name 에 괄호로 붙어 오며 그대로 보존한다 —
기본 이름만 조회하면 기본 상태 값을 얻는다.

사용: python scrape_char_stats.py
"""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.dustloop.com/wiki/api.php"
UA = "Mozilla/5.0 (frame-data study tool; personal use)"
OUT = Path(__file__).resolve().parent.parent / "char_stats_ggst.json"

# GGST/Frame Data#Guts 표. 키=등급, 값=[<=70,<=60,<=50,<=40,<=30,<=20,<=10]% 배율
GUTS_TABLE = {
    0: [0.97, 0.92, 0.89, 0.84, 0.75, 0.66, 0.56],
    1: [0.96, 0.91, 0.87, 0.82, 0.73, 0.63, 0.53],
    2: [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50],
    3: [0.94, 0.89, 0.83, 0.78, 0.67, 0.57, 0.47],
    4: [0.93, 0.88, 0.81, 0.76, 0.64, 0.53, 0.44],
    5: [0.92, 0.87, 0.79, 0.74, 0.60, 0.50, 0.41],
}


def fetch() -> list[dict]:
    url = API + "?" + urllib.parse.urlencode({
        "action": "cargoquery", "tables": "ggstCharacters",
        "fields": "name,defense,guts", "limit": "200", "format": "json",
    })
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [row["title"] for row in payload.get("cargoquery", [])]


def main():
    rows = fetch()
    chars = {}
    for r in rows:
        name = r.get("name", "").strip()
        if not name or r.get("defense") in (None, ""):
            continue
        d = int(r["defense"]); g = int(r.get("guts") or 0)
        chars[name] = {
            "defense": d,
            "def_mult": round((d + 256) / 256, 4),   # 받는 데미지 배수
            "guts": g,
        }
    out = {
        "_source": "Dustloop cargo ggstCharacters + GGST/Frame Data#Guts",
        "_formula": "damage = base * def_mult * guts_mult(guts, hp_pct); hp>70% -> guts 1.0",
        "guts_table": {str(k): v for k, v in GUTS_TABLE.items()},
        "chars": chars,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 -> {OUT}  캐릭터 {len(chars)}명(상태변형 포함)")
    for n, v in list(chars.items())[:5]:
        print(f"  {n:20s} def_mult={v['def_mult']}  guts={v['guts']}")


if __name__ == "__main__":
    main()
