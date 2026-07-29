"""
scrape_dustloop.py  —  길티기어 스트라이브 프레임 데이터 수집기

Dustloop 위키는 MediaWiki + Cargo 확장으로 돌아가고, Cargo는 데이터를 구조화된
JSON으로 뱉어주는 공식 쿼리 API를 연다. HTML 파싱이 아니라 'DB 쿼리'에 가깝다.

  엔드포인트 : https://www.dustloop.com/wiki/api.php
  action     : cargoquery
  tables     : MoveData_GGST   (GGST 기술 데이터 테이블)
  limit      : 한 번에 최대 500개  ->  offset 을 500씩 늘려 전체 수집

결과는 punish_engine.load_framedata() 가 그대로 먹는 형태로 저장한다:
    { 캐릭터명: { 기술표기(input): {startup, onBlock, guard, recovery, ...} } }

표준 라이브러리만 사용(설치 불필요). 서버 매너용으로 페이지마다 짧은 딜레이를 둔다.
"""

from __future__ import annotations
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.dustloop.com/wiki/api.php"
TABLE = "MoveData_GGST"
FIELDS = [
    "chara", "input", "name",
    "damage", "startup", "active", "recovery",
    "onBlock", "onHit", "guard",
    "level", "invuln", "prorate", "riscGain",
    "counter", "riscLoss", "wallDamage",   # counter = 카운터 레벨(Small/Mid/Large 등)
    "cancel",   # 캔슬 대상 문자열(S=필살기 J=점프 D=대시 R=RRC P=PRC 등) — 콤보 코칭용
]
PAGE = 500          # Cargo API 한 페이지 최대치
DELAY = 0.7         # 요청 사이 딜레이(초) — 서버 부담 방지
RETRIES = 4
OUT = Path(__file__).resolve().parent.parent / "framedata_ggst.json"  # 프로젝트 루트(데이터 위치)

UA = "Mozilla/5.0 (frame-data study tool; personal use)"


def fetch_page(offset: int) -> list[dict]:
    """offset 위치에서 최대 PAGE개 행을 받아 [{필드:값}, ...] 로 반환."""
    params = {
        "action": "cargoquery",
        "tables": TABLE,
        "fields": ",".join(FIELDS),
        "limit": str(PAGE),
        "offset": str(offset),
        "format": "json",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # Cargo 응답: {"cargoquery": [{"title": {필드:값}}, ...]}
            return [row["title"] for row in payload.get("cargoquery", [])]
        except Exception as e:  # 520 등 일시적 오류는 백오프 후 재시도
            last_err = e
            wait = DELAY * attempt * 2
            print(f"  ! offset {offset} 실패({attempt}/{RETRIES}): {e} -> {wait:.1f}s 후 재시도")
            time.sleep(wait)
    raise RuntimeError(f"offset {offset} 수집 실패: {last_err}")


def scrape() -> dict[str, dict[str, dict]]:
    """전 캐릭터/전 기술을 받아 {캐릭터: {기술표기: {필드}}} 로 그룹화."""
    data: dict[str, dict[str, dict]] = {}
    offset = 0
    total = 0
    while True:
        rows = fetch_page(offset)
        if not rows:
            break
        for r in rows:
            chara = (r.get("chara") or "").strip()
            key = (r.get("input") or "").strip() or (r.get("name") or "").strip()
            if not chara or not key:
                continue
            # 유니버설 메커닉은 per-캐릭 무브에서 제외 (2026.06 패치로 Wild Assault 삭제됨).
            # Wild Assault는 캐릭 개별기가 아니라 시스템 기술 -> mechanics 쪽에서 다룸.
            if (r.get("name") or "").strip() in ("Wild Assault", "Charged Wild Assault"):
                continue
            data.setdefault(chara, {})
            # 같은 표기가 또 나오면(파생기 등) 덮어쓰지 않게 _2, _3 으로 보존
            uniq = key
            n = 2
            while uniq in data[chara]:
                uniq = f"{key}_{n}"
                n += 1
            data[chara][uniq] = {
                "name": (r.get("name") or "").strip(),
                "damage": r.get("damage", ""),
                "startup": r.get("startup", ""),
                "active": r.get("active", ""),
                "recovery": r.get("recovery", ""),
                "onBlock": r.get("onBlock", ""),
                "onHit": r.get("onHit", ""),
                "guard": r.get("guard", ""),
                "level": r.get("level", ""),
                "invuln": r.get("invuln", ""),
                "prorate": r.get("prorate", ""),
                "riscGain": r.get("riscGain", ""),
                "counter": r.get("counter", ""),      # 카운터 레벨
                "riscLoss": r.get("riscLoss", ""),
                "wallDamage": r.get("wallDamage", ""),
                "cancel": r.get("cancel", ""),        # 캔슬 대상(콤보 코칭용)
            }
        total += len(rows)
        print(f"  offset {offset:>5}: {len(rows)}행 수집 (누적 {total})")
        offset += PAGE
        if len(rows) < PAGE:   # 마지막 페이지
            break
        time.sleep(DELAY)
    return data


def main() -> None:
    print("=" * 60)
    print("Dustloop GGST 프레임 데이터 수집 시작")
    print("=" * 60)
    data = scrape()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    n_char = len(data)
    n_move = sum(len(v) for v in data.values())
    print("-" * 60)
    print(f"완료: 캐릭터 {n_char}명 / 기술 {n_move}개 -> {OUT.name}")


if __name__ == "__main__":
    main()
