"""
apply_pack.py — verify_pack에 대한 사용자 골드 판정을 학습데이터로 반영.

- 정답이 시트 후보에 있으면 기존 시트 그대로 학습행 생성
- 후보에 없으면 영상에서 프레임 재추출 -> [골드+데미지후보] 새 시트 제작
- bad/메커닉/멀티무브 -> 제외. 와일드카드(22X 등)는 후보 중 유일 매칭이면 해석.
- 수동 val 영상과 겹치는 팩 영상은 train에서 제외(누수 방지).
출력: pack 골드 행을 train.jsonl에 병합(+ pack.jsonl gold 필드 기록)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import cv2
import punish_engine as pe
import vlm_id as v

ROOT = pe.app_dir()
PACK = ROOT / "verify_pack"
SHEETS = ROOT / "train_data" / "sheets_p"

# 사용자 판정 (2026-07-04). 특수: bad/mech=제외, 멀티무브=제외(주석), 와일드카드=후보로 해석
G = {0:'bad',1:'236P',2:'2P',3:'6P',4:'c.S',5:'2P',6:'f.S',7:'bad',8:'c.S',9:'bad',10:'f.S',
     11:'bad',12:'f.S',13:'f.S',14:'6S',15:'f.S',16:'2K',17:'bad',18:'6P',19:'2P',20:'bad',
     21:'c.S',22:'bad',23:'f.S',24:'623S',25:'j.K',26:'bad',27:'MECH',28:'bad',29:'623S',
     30:'2K',31:'bad',32:'2K',33:'c.S',34:'2K',35:'MECH',36:'MECH',37:'bad',38:'2K',39:'bad',
     40:'bad',41:'2D',42:'236S',43:'22X',44:'5K',45:'5P',46:'2D',47:'bad',48:'632146H',
     49:'c.S',50:'bad',51:'2D',52:'j.K',53:'bad',54:'bad',55:'SKIP',56:'j.D',57:'[4]6S',
     58:'bad',59:'2P',60:'236P',61:'6P',62:'MECH',63:'bad',64:'6P',65:'5K',66:'f.S',67:'bad',
     68:'f.S',69:'bad',70:'bad',71:'en.236K',72:'SKIP',73:'6K',74:'bad',75:'f.S',76:'j.S',
     77:'5P',78:'bad',79:'bad',80:'bad',81:'f.S',82:'f.S',83:'bad',84:'236X',85:'2K',
     86:'bad',87:'bad',88:'2K',89:'2K',90:'bad',91:'236X',92:'bad',93:'SKIP',94:'bad',
     95:'bad',96:'bad',97:'22S',98:'f.S',99:'j.S',100:'bad',101:'bad',102:'bad',103:'214K',
     104:'j.K',105:'j.H',106:'c.S',107:'bad',108:'j.H',109:'f.S',110:'6P',111:'bad',
     112:'bad',113:'bad',114:'2D',115:'MECH',116:'236P',117:'2S',118:'5P',119:'214K',
     120:'bad',121:'bad',122:'OR46D',123:'c.S',124:'bad',125:'c.S',126:'JR 236K',127:'SKIP',
     128:'c.S',129:'5K',130:'632146K',131:'bad',132:'JR f.S',133:'bad',134:'5H',135:'5K',
     136:'OR46D',137:'JR 2K',138:'c.S',139:'236K',140:'f.S',141:'bad',142:'bad',143:'OR46D',
     144:'bad',145:'bad',146:'bad',147:'6H',148:'bad',149:'c.S',150:'2H',151:'5K',152:'f.S',
     153:'bad',154:'j.S',155:'j.S',156:'bad'}


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def resolve(gold: str, cands: list[str], char: str) -> str | None:
    """골드 표기를 후보/참고이미지 표기로 해석. 실패 시 None."""
    if gold in ("bad", "MECH", "SKIP"):
        return None
    if gold == "OR46D":                              # '4d or 6d' -> 6D_or_4D류 후보
        for c in cands:
            if norm(c) in ("6dor4d", "4dor6d"):
                return c
        gold = "6D_or_4D"
    if gold.endswith("X"):                           # 와일드카드: 후보 중 같은 접두 유일 매칭
        pre = norm(gold[:-1])
        hit = [c for c in cands if norm(c).startswith(pre)]
        return hit[0] if len(hit) == 1 else None
    g = norm(gold)
    for c in cands:                                  # 후보 정확/레벨변형 매칭
        if norm(c) == g or norm(c).startswith(g + "level"):
            return c
    d = v.IMG_DIR / char.replace(" ", "_")           # 후보 밖 -> ref 파일명에서 찾기
    if d.is_dir():
        for p in d.glob("*.png"):
            if norm(p.stem) == g or norm(p.stem).startswith(g + "level"):
                return p.stem
    return None


def main():
    SHEETS.mkdir(parents=True, exist_ok=True)
    rows_in = [json.loads(l) for l in (PACK / "pack.jsonl").open(encoding="utf-8") if l.strip()]
    # 수동 val 영상 stem(누수 방지) — convert_manual과 동일 규칙으로 재계산
    import convert_manual as cm
    val_stems = set()
    for p in (ROOT / "dataset").glob("*/*/*.png"):
        m = re.match(r"(.+)_(\d+(?:\.\d+)?)_(left|right)\.png$", p.name)
        if m and cm.is_val_video(m.group(1)):
            val_stems.add(m.group(1)[:24])

    out_rows, stats = [], {"gold": 0, "rebuilt": 0, "skip": 0, "leak": 0, "noresolve": 0}
    for r in rows_in:
        i = r["seq"]
        if i not in G:
            continue
        g = G[i]
        r["gold"] = g
        if g in ("bad", "MECH", "SKIP"):
            stats["skip"] += 1
            continue
        if any(r["video"][:24].startswith(s) or s.startswith(r["video"][:24]) for s in val_stems):
            stats["leak"] += 1
            continue
        mv = resolve(g, r["cands"], r["char"])
        if not mv:
            stats["noresolve"] += 1
            print(f"  ! seq{i}: '{g}' 해석 실패 ({r['char']})", flush=True)
            continue
        if mv in r["cands"]:
            img = str(PACK / r["sheet"])             # 기존 시트 재사용
        else:                                        # 골드 후보 포함 새 시트
            vid = str(ROOT / "reference_video" / r["video"])
            frames = v.event_frames(vid, r["t"])
            if not frames:
                stats["noresolve"] += 1
                continue
            hud = v.hud_strip(vid, r["t"])
            cands_mv = list(dict.fromkeys([mv] + r["cands"]))[:4]
            if len(cands_mv) < 3:                 # 전환행 등 후보 부족 -> 프라이어로 distractor 보충
                import analyze_match as am
                extra = [m for m in am.candidates_for({}, r["char"], None) if m != mv]
                cands_mv = list(dict.fromkeys(cands_mv + extra))[:4]
            sheet = v.build_sheet(frames, [{"char": r["char"], "move": m} for m in cands_mv], hud=hud)
            img = str(SHEETS / f"p{i:04d}.png")
            cv2.imwrite(img, sheet)
            stats["rebuilt"] += 1
        out_rows.append({"image": img, "prompt": v._PROMPT.format(ctx="상황: 학습", atk=r["char"]),
                         "answer": mv, "char": r["char"]})
        stats["gold"] += 1

    (PACK / "pack.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_in), encoding="utf-8")
    tr = ROOT / "train_data" / "train.jsonl"
    old = [json.loads(l) for l in tr.open(encoding="utf-8") if l.strip()]
    old = [r for r in old if "sheets_p" not in r["image"] and str(PACK) not in r["image"]]
    merged = old + out_rows
    tr.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged), encoding="utf-8")
    print(f"\n골드 채택 {stats['gold']} (새시트 {stats['rebuilt']}) | 제외 {stats['skip']} | "
          f"누수차단 {stats['leak']} | 해석실패 {stats['noresolve']}", flush=True)
    print(f"train.jsonl: {len(old)} -> {len(merged)}행", flush=True)


if __name__ == "__main__":
    main()
