"""
train_dino.py  —  DINOv2(frozen) 특징 + 선형분류기로 무브ID (일반화 개선 시도)

ResNet 직접학습은 25개 레퍼런스 영상에 과적합 -> 새 영상 0.05. 대신:
  DINOv2(메타, 수십억장 학습) 백본을 얼리고 -> 도메인-강건 특징 추출 -> 위에 선형분류기만 학습.
백본 특징이 색·녹화·스테이지에 강건하니 새 영상에도 일반화될 가능성. 라벨 데이터 그대로 재사용.

출력: dino_head.pt {head, classes, model}
사용:  python -u train_dino.py [min=5] [epochs=60]
"""
from __future__ import annotations
import sys, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import punish_engine as pe

ROOT = pe.app_dir()
DINO_NAME = "dinov2_vitb14"
TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_dino(dev):
    m = torch.hub.load("facebookresearch/dinov2", DINO_NAME)
    m.eval().to(dev)
    for p in m.parameters():
        p.requires_grad = False
    return m


def extract(dino, paths, dev, bs=64):
    feats = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            im = cv2.imread(p)
            if im is None:
                im = np.zeros((224, 224, 3), np.uint8)
            batch.append(TF(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))))
        x = torch.stack(batch).to(dev)
        with torch.no_grad():
            feats.append(dino(x).cpu())
        print(f"  특징추출 {min(i+bs,len(paths))}/{len(paths)}", flush=True) if i % (bs*8) == 0 else None
    return torch.cat(feats)


def main():
    mn = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)

    by = {}
    for charf in sorted((ROOT / "dataset").iterdir()):
        if not charf.is_dir():
            continue
        for mvf in charf.iterdir():
            if mvf.is_dir():
                ps = [str(x) for x in mvf.glob("*.png")]
                if len(ps) >= mn:
                    by[f"{charf.name}/{mvf.name}"] = ps
    classes = sorted(by); cidx = {c: i for i, c in enumerate(classes)}
    train_p, train_y, val_p, val_y = [], [], [], []
    for c, ps in by.items():
        ps = ps[:]; random.shuffle(ps); k = max(1, int(len(ps) * 0.2))
        for p in ps[:k]:
            val_p.append(p); val_y.append(cidx[c])
        for p in ps[k:]:
            train_p.append(p); train_y.append(cidx[c])
    print(f"클래스 {len(classes)} / train {len(train_p)} / val {len(val_p)} / {dev}", flush=True)

    print("DINOv2 로드…", flush=True)
    dino = load_dino(dev)
    print("train 특징추출…", flush=True)
    Xtr = extract(dino, train_p, dev); ytr = torch.tensor(train_y)
    print("val 특징추출…", flush=True)
    Xva = extract(dino, val_p, dev); yva = torch.tensor(val_y)

    head = nn.Sequential(nn.Linear(Xtr.shape[1], len(classes))).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xtr, ytr, Xva, yva = Xtr.to(dev), ytr.to(dev), Xva.to(dev), yva.to(dev)
    best = 0.0
    for ep in range(epochs):
        head.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i+256]
            opt.zero_grad(); crit(head(Xtr[idx]), ytr[idx]).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            o = head(Xva)
            t1 = (o.argmax(1) == yva).float().mean().item()
            t3 = sum(int(yva[i] in o[i].topk(3).indices) for i in range(len(yva))) / len(yva)
        best = max(best, t1)
        if (ep + 1) % 10 == 0:
            print(f"ep{ep+1}: val top1={100*t1:.1f}% top3={100*t3:.1f}%", flush=True)
    torch.save({"head": head.state_dict(), "classes": classes, "model": DINO_NAME},
               ROOT / "dino_head.pt")
    print(f"\n최고 val top1={100*best:.1f}% -> dino_head.pt", flush=True)


if __name__ == "__main__":
    main()
