"""
rebuild_manifest.py  —  dataset_auto/ 프레임 경로에서 manifest.jsonl 재구성.

프레임은 dataset_auto/<Char_underscore>/<move>/<tag>_f{i}.png 로 저장되어,
char/move/t 가 경로에 인코딩됨. 샤드 manifest가 유실돼도 프레임으로 전체 복구.
"""
from __future__ import annotations
import glob, json, re
from collections import defaultdict
from pathlib import Path

import punish_engine as pe

OUT = pe.app_dir() / "dataset_auto"


def main():
    groups: dict[tuple, list] = defaultdict(list)
    for p in glob.glob(str(OUT / "*" / "*" / "*.png")):
        parts = Path(p).parts
        char = parts[-3].replace("_", " ")
        move = parts[-2]
        m = re.match(r"(.+)_f(\d+)\.png$", parts[-1])
        if not m:
            continue
        groups[(char, move, m.group(1))].append((int(m.group(2)), p))

    rows = []
    for (char, move, tag), fs in groups.items():
        fs.sort()
        tm = re.search(r"_(\d+\.\d+)$", tag)
        t = float(tm.group(1)) if tm else 0.0
        video = tag[: tm.start()] if tm else tag
        rows.append({"char": char, "move": move, "kind": "counter",
                     "t": t, "video": video, "frames": [p for _, p in fs]})

    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as o:
        for r in rows:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"복구: {len(rows)}개 라벨 manifest 재작성", flush=True)


if __name__ == "__main__":
    main()
