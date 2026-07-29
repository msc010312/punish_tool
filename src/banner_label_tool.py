# -*- coding: utf-8 -*-
"""banner_label_tool.py — 배너 검수 라벨러 (COUNTER / PUNISH / JUST / none).

banner_ds/_review 를 DINO 유사도 정렬 순서로 보여주고 단축키로 4분류.
답은 즉시 banner_labels.json 에 저장(이어하기 가능). 뒤쪽은 대부분 블롭이라
[A] 로 '이후 전부 none' 일괄처리 가능.

라벨 의미:
  COUNTER/PUNISH/JUST : 그 단어가 보이는 진짜 배너
  none                : 배너가 아님 (노란 이펙트·배경 블롭)   <- '모르겠음'이 아님!
  unsure              : 봤는데 판단 불가 -> 학습에서 제외 (오염 방지)

단축키:  C=COUNTER  P=PUNISH  J=JUST  N=none  U=unsure
        ←/→ 이동  A=이후 전부 none  Q=저장후종료
사용: python banner_label_tool.py      (먼저 presort_banner_review.py 실행 권장)
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QVBoxLayout, QWidget)

import punish_engine as pe

# 모드: 인자 'wake' 면 흰색(기상) REVERSAL 전용 검수(별도 폴더·라벨파일).
WAKE = len(sys.argv) > 1 and sys.argv[1] == "wake"
if WAKE:
    DS = pe.app_dir() / "banner_ds_wake"
    LABELS = pe.app_dir() / "banner_wake_labels.json"
    CLASSES = ["REVERSAL_WAKE", "none", "unsure"]
    KEY = {"REVERSAL_WAKE": "R", "none": "N", "unsure": "U"}
    COLOR = {"REVERSAL_WAKE": "#5fe06a", "none": "#8a8d9c", "unsure": "#c07adf"}
else:
    DS = pe.app_dir() / "banner_ds"
    LABELS = pe.app_dir() / "banner_labels.json"
    CLASSES = ["COUNTER", "PUNISH", "JUST", "REVERSAL", "none", "unsure"]
    KEY = {"COUNTER": "C", "PUNISH": "P", "JUST": "J", "REVERSAL": "R",
           "none": "N", "unsure": "U"}
    COLOR = {"COUNTER": "#ffc400", "PUNISH": "#ff5454", "JUST": "#60b0ff",
             "REVERSAL": "#5fe06a", "none": "#8a8d9c", "unsure": "#c07adf"}


def to_pix(bgr, w=760):
    if bgr is None:
        return QPixmap()
    h = max(1, int(bgr.shape[0] * w / bgr.shape[1]))
    im = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(im.data, im.shape[1], im.shape[0],
                                    im.strides[0], QImage.Format_RGB888).copy())


class Tool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("배너 라벨러 — C/P/J/R/N, ←→ 이동, A=이후 전부 none, Q=저장종료")
        order = DS / "_review_order.json"
        if order.exists():
            self.items = [r["path"] for r in json.loads(order.read_text(encoding="utf-8"))]
            self.pred = {r["path"]: (r["pred"], r["sim"]) for r in json.loads(order.read_text(encoding="utf-8"))}
        else:
            self.items = sorted(glob.glob(str(DS / "_review" / "*.png")))
            self.pred = {}
        self.labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
        # 아직 라벨 안 된 첫 항목부터
        self.i = next((k for k, p in enumerate(self.items) if p not in self.labels), 0)

        self.img = QLabel(alignment=Qt.AlignCenter)
        self.info = QLabel(alignment=Qt.AlignCenter)
        self.info.setStyleSheet("font-size:15px;")
        self.bar = QProgressBar()

        btns = QHBoxLayout()
        for c in CLASSES:
            b = QPushButton(f"{c} [{KEY[c]}]")
            b.setStyleSheet(f"background:{COLOR[c]}; color:#111; font-weight:700; padding:8px;")
            b.clicked.connect(lambda _=False, x=c: self.set_label(x))
            btns.addWidget(b)
        hint = QLabel("REVERSAL = 기상에 필살기/각성기(기본기·특수기 X).  "
                      "none = 배너 아님(이펙트/배경).  판단 불가면 unsure(U) — 학습서 제외.",
                      alignment=Qt.AlignCenter)
        hint.setStyleSheet("color:#888; font-size:12px;")

        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.img, 3)
        lay.addLayout(btns)
        lay.addWidget(hint)
        lay.addWidget(self.bar)

        for c, key in KEY.items():
            QShortcut(QKeySequence(key), self, activated=lambda x=c: self.set_label(x))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.move(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.move(1))
        QShortcut(QKeySequence("A"), self, activated=self.rest_none)
        QShortcut(QKeySequence("Q"), self, activated=self.close)
        self.resize(820, 620)
        self.render()

    # ---------- 상태 ----------
    def save(self):
        LABELS.write_text(json.dumps(self.labels, ensure_ascii=False, indent=1), encoding="utf-8")

    def set_label(self, c):
        """run 대표(_0)에 라벨 -> 같은 run의 형제 크롭(_1,_2)에도 전파."""
        if not self.items:
            return
        p = self.items[self.i]
        self.labels[p] = c
        for sib in glob.glob(p.replace("_0.png", "_*.png")):
            self.labels[sib] = c
        self.save()
        self.move(1)

    def rest_none(self):
        n = len(self.items) - self.i
        if QMessageBox.question(self, "확인", f"현재({self.i+1})부터 끝까지 {n}장을 모두 none 처리?") \
                != QMessageBox.Yes:
            return
        for p in self.items[self.i:]:
            self.labels[p] = "none"
        self.save()
        self.i = len(self.items) - 1
        self.render()

    def move(self, d):
        self.i = max(0, min(len(self.items) - 1, self.i + d))
        self.render()

    def render(self):
        if not self.items:
            self.info.setText("banner_ds/_review 가 비었습니다."); return
        p = self.items[self.i]
        bgr = cv2.imread(p)
        self.img.setPixmap(to_pix(bgr))
        pr, sim = self.pred.get(p, ("?", 0))
        cur = self.labels.get(p, "-")
        done = len(self.labels)
        self.info.setText(f"[{self.i+1}/{len(self.items)}]  DINO추정={pr} (sim {sim})   "
                          f"내 라벨=<b style='color:{COLOR.get(cur,'#888')}'>{cur}</b>   "
                          f"완료 {done}/{len(self.items)}")
        self.bar.setMaximum(len(self.items)); self.bar.setValue(done)

    def closeEvent(self, e):
        self.save(); e.accept()


def main():
    app = QApplication(sys.argv)
    t = Tool(); t.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
