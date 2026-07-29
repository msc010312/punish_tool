"""
train_cnn.py  —  무브 ID CNN 학습 (부분 테스트용)

dataset/<캐릭>/<무브>/ + dataset/__mechanics__/<시스템>/ 의 크롭으로 분류기 학습.
클래스 = "<캐릭>/<무브>" (캐릭+무브 동시 식별). ≥MIN 장인 클래스만 사용.
ResNet18 전이학습(ImageNet) → GPU. top1/top3 정확도 보고.
좌우반전 안 함(기술 방향성 보존). 부분 데이터로 접근법 검증용 — 전체 데이터로 재학습 가능.

사용:  python train_cnn.py [min_samples=5] [epochs=25]
"""
from __future__ import annotations
import sys, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import punish_engine as pe

ROOT = pe.app_dir() / "dataset"
OUT = pe.app_dir() / "move_cnn.pt"
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    # 강한 색증강 + 랜덤흑백 = 컬러팔레트(12색+)·녹화차이 무시, '포즈'로 학습
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.5),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(), NORM,
    transforms.RandomErasing(p=0.25)])
VAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


class DS(Dataset):
    def __init__(self, items, cidx, tf):
        self.items, self.cidx, self.tf = items, cidx, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, c = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), self.cidx[c]


def gather(min_samples):
    by = {}
    for charf in sorted(ROOT.iterdir()):
        if not charf.is_dir():
            continue
        for movef in charf.iterdir():
            if not movef.is_dir() or movef.name == "_review":
                continue                                    # 자동라벨 애매분 제외
            imgs = list(movef.glob("*.png"))
            if len(imgs) >= min_samples:
                by[f"{charf.name}/{movef.name}"] = imgs
    return by


def main():
    min_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    by = gather(min_samples)
    classes = sorted(by)
    cidx = {c: i for i, c in enumerate(classes)}
    random.seed(0)
    train, val = [], []
    for c, ps in by.items():
        ps = ps[:]; random.shuffle(ps)
        nv = max(1, int(len(ps) * 0.2))
        val += [(p, c) for p in ps[:nv]]
        train += [(p, c) for p in ps[nv:]]
    print(f"클래스 {len(classes)} / train {len(train)} / val {len(val)} / device {dev}")

    dl_tr = DataLoader(DS(train, cidx, TRAIN_TF), batch_size=32, shuffle=True, num_workers=0)
    dl_va = DataLoader(DS(val, cidx, VAL_TF), batch_size=64, num_workers=0)

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m = m.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for ep in range(epochs):
        m.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
        sched.step()
        m.eval(); top1 = top3 = n = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                o = m(x); n += y.size(0)
                top1 += (o.argmax(1) == y).sum().item()
                t3 = o.topk(min(3, len(classes)), 1).indices
                top3 += sum(int(y[i] in t3[i]) for i in range(y.size(0)))
        a1, a3 = 100 * top1 / n, 100 * top3 / n
        best = max(best, a1)
        print(f"ep{ep+1:2d}: val top1={a1:.1f}%  top3={a3:.1f}%")
    torch.save({"state": m.state_dict(), "classes": classes}, OUT)
    print(f"\n최고 top1={best:.1f}% (랜덤={100/len(classes):.1f}%) -> {OUT}")


if __name__ == "__main__":
    main()
