"""
train_cnn_mixed.py  —  인게임 크롭 + Dustloop(증강 캐시) 혼합 학습 + 전이 테스트

질문: Dustloop 렌더(증강)만으로 학습한 기술을 인게임 크롭에서 맞출 수 있나? (=미등장 기술 커버)
  - 인게임 ≥MIN 클래스 중 일부(transfer)는 인게임 크롭을 전부 테스트로 빼고 Dustloop-캐시만 학습.
  - 나머지(normal)는 인게임 80/20 + Dustloop-캐시 둘 다 학습.
  - 평가: normal-val(본 기술) vs transfer-test(Dustloop만 본 기술, 인게임으로 테스트).

선행: gen_aug_cache.py 로 dustloop_aug_cache/ 생성해둘 것.
사용:  python -u train_cnn_mixed.py [min=5] [epochs=18]
"""
from __future__ import annotations
import sys, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import punish_engine as pe

ROOT = pe.app_dir()
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
AUG = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ColorJitter(0.2, 0.2, 0.2), transforms.RandomRotation(8),
    transforms.ToTensor(), NORM])
PLAIN = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                            transforms.ToTensor(), NORM])


class DS(Dataset):
    def __init__(self, items, cidx, train):
        self.items, self.cidx, self.train = items, cidx, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        im = cv2.imread(path)
        if im is None:
            im = np.zeros((224, 224, 3), np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return (AUG if self.train else PLAIN)(im), self.cidx[cls]


def gather(base, mn=1):
    d = {}
    for charf in sorted(Path(base).iterdir()):
        if not charf.is_dir():
            continue
        for mvf in charf.iterdir():
            if mvf.is_dir():
                ps = [str(x) for x in mvf.glob("*.png")]
                if len(ps) >= mn:
                    d[f"{charf.name}/{mvf.name}"] = ps
    return d


def main():
    mn = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)

    ing = gather(ROOT / "dataset", mn)
    aug = gather(ROOT / "dustloop_aug_cache", 1)
    classes = sorted(set(ing) | set(aug))
    cidx = {c: i for i, c in enumerate(classes)}

    overlap = [c for c in ing if c in aug]          # 인게임∩Dustloop = 전이 후보
    random.shuffle(overlap)
    transfer = set(overlap[:max(1, len(overlap) // 4)])

    train, val, ttest = [], [], []
    for c, ps in ing.items():
        ps = ps[:]; random.shuffle(ps)
        if c in transfer:
            ttest += [(p, c) for p in ps]            # 인게임 전부 테스트
        else:
            k = max(1, int(len(ps) * 0.2))
            val += [(p, c) for p in ps[:k]]
            train += [(p, c) for p in ps[k:]]
    for c, ps in aug.items():                        # Dustloop 캐시는 전부 학습
        train += [(p, c) for p in ps]

    random.shuffle(train)
    print(f"클래스 {len(classes)} (인게임 {len(ing)} / Dustloop {len(aug)} / 겹침 {len(overlap)})")
    print(f"transfer {len(transfer)}클래스 | train {len(train)} / val {len(val)} / ttest {len(ttest)} | {dev}",
          flush=True)

    nw = 4
    dl_tr = DataLoader(DS(train, cidx, True), batch_size=64, shuffle=True, num_workers=nw)
    dl_va = DataLoader(DS(val, cidx, False), batch_size=128, num_workers=nw)
    dl_tt = DataLoader(DS(ttest, cidx, False), batch_size=128, num_workers=nw)

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(classes)); m = m.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    def ev(loader):
        m.eval(); t1 = t3 = n = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(dev), y.to(dev); o = m(x); n += y.size(0)
                t1 += (o.argmax(1) == y).sum().item()
                t3 += sum(int(y[i] in o[i].topk(3).indices) for i in range(y.size(0)))
        return 100 * t1 / max(n, 1), 100 * t3 / max(n, 1)

    for ep in range(epochs):
        m.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); crit(m(x), y).backward(); opt.step()
        sched.step()
        v1, v3 = ev(dl_va); tt1, tt3 = ev(dl_tt)
        print(f"ep{ep+1:2d}: val(본기술) {v1:.1f}/{v3:.1f}  |  transfer(Dustloop만) {tt1:.1f}/{tt3:.1f}",
              flush=True)
    print(f"\n전이결과: transfer top1={tt1:.1f}% top3={tt3:.1f}% (랜덤={100/len(classes):.2f}%)")
    print("=> transfer 높으면 Dustloop만으로 인게임 기술 식별 가능 = 미등장 기술 커버됨")


if __name__ == "__main__":
    main()
