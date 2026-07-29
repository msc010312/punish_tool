"""
detect_mislabels.py  —  학습된 CNN으로 오염(잘못 라벨) 의심 크롭 탐지

move_cnn.pt 로 모든 라벨 크롭을 예측 -> 라벨과 다르게 '강하게' 예측되는 것 플래그.
같은 캐릭 내 다른 무브로 예측 = 무브 오라벨 의심 / 다른 캐릭 = 폴더 자체 오류 의심.
자동 삭제 안 함 — 의심 목록 + 컨택트시트만 생성, 사람이 확인 후 조치.

출력: mislabel_suspects.png (시트), mislabel_suspects.txt (경로+라벨+예측)
사용:  python detect_mislabels.py [conf=0.55]
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import punish_engine as pe

ROOT = pe.app_dir() / "dataset"
CKPT = pe.app_dir() / "move_cnn.pt"
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


def main():
    conf_thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev)
    classes = ck["classes"]; cidx = {c: i for i, c in enumerate(classes)}
    m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ck["state"]); m.eval().to(dev)

    suspects = []   # (conf, path, label, pred, same_char)
    for cls in classes:
        folder = ROOT / cls  # cls = "Char/Move"
        for p in folder.glob("*.png"):
            x = TF(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            with torch.no_grad():
                o = torch.softmax(m(x), 1)[0]
            pi = int(o.argmax()); pred = classes[pi]; pc = float(o[pi])
            if pred != cls and pc >= conf_thr:
                lab_conf = float(o[cidx[cls]])
                same = pred.split("/")[0] == cls.split("/")[0]
                # 라벨 자체 확신이 낮을 때만(오염 가능성↑) — 라벨conf < 예측conf
                if lab_conf < pc:
                    suspects.append((pc, str(p), cls, pred, same))
    suspects.sort(reverse=True)

    txt = pe.app_dir() / "mislabel_suspects.txt"
    txt.write_text("\n".join(
        f"{'[같은캐릭]' if s else '[다른캐릭]'} conf={c:.2f} | {lab} -> {pred} | {pth}"
        for c, pth, lab, pred, s in suspects), encoding="utf-8")

    # 컨택트시트 (상위 40)
    cells = []
    for c, pth, lab, pred, s in suspects[:40]:
        im = cv2.resize(cv2.imread(pth), (170, 170))
        cv2.rectangle(im, (0, 132), (170, 170), (0, 0, 0), -1)
        l = lab.split("/")[-1]; pr = pred.split("/")[-1] if s else pred.replace("/", " ")
        cv2.putText(im, f"{l}->{pr}"[:22], (3, 150), 0, 0.4, (0, 230, 255), 1)
        cv2.putText(im, f"{c:.2f}", (3, 165), 0, 0.4, (0, 255, 0), 1)
        cv2.rectangle(im, (0, 0), (169, 169), (0, 200, 0) if s else (0, 0, 230), 2)
        cells.append(im)
    if cells:
        while len(cells) % 6:
            cells.append(np.zeros((170, 170, 3), np.uint8))
        rows = [np.hstack(cells[i:i+6]) for i in range(0, len(cells), 6)]
        cv2.imwrite(str(pe.app_dir() / "mislabel_suspects.png"), np.vstack(rows))

    same = sum(1 for s in suspects if s[4])
    print(f"의심 {len(suspects)}개 (같은캐릭 무브오라벨 {same} / 다른캐릭 {len(suspects)-same})")
    print(f"-> mislabel_suspects.png (상위40), mislabel_suspects.txt")


if __name__ == "__main__":
    main()
