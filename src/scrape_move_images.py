"""
scrape_move_images.py  —  Dustloop 기술 이미지(레퍼런스 라이브러리) 수집

각 기술의 'images' 필드(예: 'GGST Sol Badguy 6H.png')를 받아 실제 파일 URL로 변환 후 다운로드.
무브 ID(애니메이션 인식)의 레퍼런스로 쓸 캐릭터별 동작 이미지 셋을 만든다.

사용:  python scrape_move_images.py "Sol Badguy"
저장:  dustloop_images/<char>/<input>.png   (멀티히트는 _1,_2)
주의: 레퍼런스는 클린 렌더라 게임 화면과 도메인 갭이 큼 — 매칭 방식은 별도 연구.
"""
from __future__ import annotations
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.dustloop.com/wiki/api.php"
UA = "Mozilla/5.0 (frame-data study tool; personal use)"
OUTROOT = Path(__file__).resolve().parent.parent / "dustloop_images"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def move_images(chara: str) -> list[tuple[str, str]]:
    """[(input, image_filename), ...] — 멀티이미지는 분해."""
    q = {"action": "cargoquery", "tables": "MoveData_GGST",
         "fields": "input,images", "where": f"chara='{chara}'",
         "limit": "500", "format": "json"}
    rows = _get(API + "?" + urllib.parse.urlencode(q)).get("cargoquery", [])
    out = []
    for r in rows:
        t = r["title"]; inp = (t.get("input") or "").strip()
        imgs = (t.get("images") or "").strip()
        if not inp or not imgs:
            continue
        for fn in imgs.split(";"):
            fn = fn.strip()
            if fn:
                out.append((inp, fn))
    return out


def file_url(filename: str) -> str | None:
    title = "File:" + filename.replace(" ", "_")
    q = {"action": "query", "titles": title, "prop": "imageinfo",
         "iiprop": "url", "format": "json"}
    pages = _get(API + "?" + urllib.parse.urlencode(q)).get("query", {}).get("pages", {})
    for p in pages.values():
        ii = p.get("imageinfo")
        if ii:
            return ii[0]["url"]
    return None


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def main():
    chara = sys.argv[1] if len(sys.argv) > 1 else "Sol Badguy"
    outdir = OUTROOT / safe(chara)
    outdir.mkdir(parents=True, exist_ok=True)
    items = move_images(chara)
    print(f"{chara}: 기술 이미지 {len(items)}개 수집 -> {outdir}")
    seen = {}
    for inp, fn in items:
        url = file_url(fn)
        if not url:
            print(f"  ! URL 없음: {fn}"); continue
        key = safe(inp)
        seen[key] = seen.get(key, 0) + 1
        suffix = f"_{seen[key]}" if seen[key] > 1 else ""
        dest = outdir / f"{key}{suffix}.png"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                dest.write_bytes(r.read())
            print(f"  {inp:16} -> {dest.name}")
        except Exception as e:
            print(f"  ! 실패 {inp}: {e}")
        time.sleep(0.4)
    print("완료.")


if __name__ == "__main__":
    main()
