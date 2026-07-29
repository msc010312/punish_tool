"""
apply_pack2.py — verify_pack2의 GUI 검수 골드(gold 필드)를 학습데이터로 반영.

- gold가 후보에 있으면 기존 시트 재사용, 없으면(전환행 포함) 프레임 재추출해 새 시트
- bad/unsure 제외. 표기는 resolve()로 레퍼런스 파일명에 매칭(점·대소문자·대괄호 무시,
  레벨 변형 prefix 매칭, 언더스코어 별칭).
사용: python apply_pack2.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import cv2
import punish_engine as pe
import analyze_match as am
import vlm_id as v

ROOT = pe.app_dir()
PACK = ROOT / "verify_pack2"
SHEETS = ROOT / "train_data" / "sheets_p2"


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def resolve(gold: str, cands: list[str], char: str) -> str | None:
    g = norm(gold)
    if not g or g in ("bad", "unsure", "b", "u", "un"):  # b/u/un = 입력창 타이핑 축약
        return None
    for c in cands:
        if norm(c) == g or norm(c).startswith(g + "level"):
            return c
    d = v.IMG_DIR / char.replace(" ", "_")
    if d.is_dir():
        for p in d.glob("*.png"):
            if norm(p.stem) == g or norm(p.stem).startswith(g + "level"):
                return p.stem
    return None


def main():
    SHEETS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (PACK / "pack.jsonl").open(encoding="utf-8") if l.strip()]
    out, stats = [], {"reuse": 0, "rebuilt": 0, "skip": 0, "fail": 0}
    for r in rows:
        g = r.get("gold", "")
        if not g or norm(g) in ("bad", "unsure"):
            stats["skip"] += 1
            continue
        mv = resolve(g, r.get("cands") or [], r["char"])
        if not mv:
            stats["fail"] += 1
            print(f"  ! seq{r['seq']}: '{g}' 해석 실패 ({r['char']})", flush=True)
            continue
        if mv in (r.get("cands") or []):
            img = str(PACK / r["sheet"])
            stats["reuse"] += 1
        else:                                          # 골드 후보 포함 시트 재생성(전환행 포함)
            vid = str(ROOT / "reference_video" / r["video"])
            frames = v.event_frames(vid, r["t"])
            if not frames:
                stats["fail"] += 1
                continue
            hud = v.hud_strip(vid, r["t"])
            cands_mv = list(dict.fromkeys([mv] + (r.get("cands") or [])))[:4]
            if len(cands_mv) < 3:                      # distractor 보충(프라이어)
                extra = [m for m in am.candidates_for({}, r["char"], None)
                         if m != mv and v.ref_path(r["char"], m)]
                cands_mv = list(dict.fromkeys(cands_mv + extra))[:4]
            sheet = v.build_sheet(frames, [{"char": r["char"], "move": m} for m in cands_mv], hud=hud)
            img = str(SHEETS / f"q{r['seq']:04d}.png")
            cv2.imwrite(img, sheet)
            stats["rebuilt"] += 1
        out.append({"image": img, "prompt": v._PROMPT.format(ctx="상황: 학습", atk=r["char"]),
                    "answer": mv, "char": r["char"]})

    tr = ROOT / "train_data" / "train.jsonl"
    old = [json.loads(l) for l in tr.open(encoding="utf-8") if l.strip()]
    old = [x for x in old if "sheets_p2" not in x["image"] and str(PACK) not in x["image"]]
    merged = old + out
    tr.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in merged), encoding="utf-8")
    print(f"\n골드 반영 {len(out)} (재사용 {stats['reuse']} / 재생성 {stats['rebuilt']}) | "
          f"제외 {stats['skip']} | 실패 {stats['fail']}", flush=True)
    print(f"train.jsonl: {len(old)} -> {len(merged)}행", flush=True)


if __name__ == "__main__":
    main()
