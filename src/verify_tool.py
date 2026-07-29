"""
verify_tool.py — verify_pack2 검수 GUI (label_tool 계승, 번호 타이핑 불필요)

시트를 보여주고:
  Enter        = LoRA 제안 그대로 수락
  기술명 입력   = 그 기술로 확정 (후보+캐릭 전체 무브 자동완성)
  B            = bad (크롭불량/피해자/오버레이 등)
  U            = unsure
  ←/→          = 이전/다음 (답 안 하고 이동)
답은 즉시 pack.jsonl의 gold 필드에 저장(이어하기 가능). 최초 실행 시 .bak 백업.
사용: python verify_tool.py [pack폴더=verify_pack2]
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton,
                               QHBoxLayout, QVBoxLayout, QCompleter)

import punish_engine as pe
import vlm_id as v

PACK = pe.app_dir() / (sys.argv[1] if len(sys.argv) > 1 else "verify_pack2")


class Tool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"검수 — {PACK.name}")
        pj = PACK / "pack.jsonl"
        bak = PACK / "pack.jsonl.bak"
        if not bak.exists():
            shutil.copy(pj, bak)
        self.rows = [json.loads(l) for l in pj.open(encoding="utf-8") if l.strip()]
        self.i = next((k for k, r in enumerate(self.rows) if not r.get("gold")), 0)

        self.img = QLabel(alignment=Qt.AlignCenter)
        self.img.setMinimumSize(1280, 640)
        self.info = QLabel()
        self.info.setStyleSheet("font-size:15px; color:#ddd; background:#222; padding:6px")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Enter=제안 수락 | 기술명 입력 | B=bad | U=unsure")
        self.edit.setStyleSheet("font-size:16px; padding:6px")
        self.edit.returnPressed.connect(self.commit)
        bswap = QPushButton("공격자↔ (S)"); bswap.clicked.connect(self.swap_attacker)
        bbad = QPushButton("BAD (B)"); bbad.clicked.connect(lambda: self.set_gold("bad"))
        buns = QPushButton("unsure (U)"); buns.clicked.connect(lambda: self.set_gold("unsure"))
        bprev = QPushButton("← 이전"); bprev.clicked.connect(lambda: self.move(-1))
        bnext = QPushButton("다음 →"); bnext.clicked.connect(lambda: self.move(1))
        top = QHBoxLayout()
        for w in (bprev, bnext, bswap, bbad, buns):
            top.addWidget(w)
        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addWidget(self.img, 1); lay.addWidget(self.info); lay.addWidget(self.edit)
        QShortcut(QKeySequence("S"), self, activated=self.swap_attacker)
        QShortcut(QKeySequence("B"), self, activated=lambda: self.set_gold("bad"))
        QShortcut(QKeySequence("U"), self, activated=lambda: self.set_gold("unsure"))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.move(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.move(1))
        self.setStyleSheet("background:#181818; color:#eee")
        self.resize(1500, 950)
        self.show_row()

    def cur(self):
        return self.rows[self.i]

    def show_row(self):
        r = self.cur()
        pm = QPixmap(str(PACK / r["sheet"]))
        self.img.setPixmap(pm.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        done = sum(1 for x in self.rows if x.get("gold"))
        g = f"  |  ✔저장됨: {r['gold']}" if r.get("gold") else ""
        self.info.setText(f"[{self.i+1}/{len(self.rows)}]  답변 {done}  |  공격자 {r['attacker']} = {r['char']}"
                          f"  |  LoRA 제안: {r.get('lora')}  |  후보: {' / '.join(r['cands'])}{g}")
        # 자동완성: 후보 + 캐릭 전체 ref 무브(+언더스코어 제거 별칭: _4_6S -> 46S) + bad/unsure
        d = v.IMG_DIR / r["char"].replace(" ", "_")
        stems = ({p.stem for p in d.glob("*.png")} | set(r["cands"])) if d.is_dir() else set(r["cands"])
        pool = sorted(stems | {s.replace("_", "") for s in stems if "_" in s})
        comp = QCompleter(pool + ["bad", "unsure"])
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setFilterMode(Qt.MatchContains)             # 부분일치(46 -> _4_6S 별칭 46S 검색)
        self.edit.setCompleter(comp)
        QTimer.singleShot(0, self.edit.clear)            # 완성 선택 후 텍스트 잔류 버그 방지
        self.edit.setFocus()

    def commit(self):
        t = self.edit.text().strip()
        if not t:
            lora = self.cur().get("lora") or ""
            if lora not in self.cur()["cands"]:          # 오타 제안(239S 등) 수락 방지
                self.info.setText("⚠ 제안이 후보에 없음(오타 가능) — 기술명을 직접 입력해줘")
                return
            t = lora                                     # Enter = 제안 수락
        self.set_gold(t)

    def set_gold(self, val: str):
        self.cur()["gold"] = val
        (PACK / "pack.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in self.rows), encoding="utf-8")
        self.move(1)

    def swap_attacker(self):
        """attribution이 반대일 때: 공격자를 상대 캐릭으로 뒤집고 그 캐릭 기술을 입력받는다.
        (기존 후보는 폐기 — 반영 시 새 캐릭 기준으로 시트 재생성)"""
        r = self.cur()
        tl = pe.app_dir() / "timelines" / (Path(r["video"]).stem + ".json")
        if not tl.exists():
            self.info.setText("⚠ 타임라인 없음 — 전환 불가(bad 처리 권장)")
            return
        d = json.loads(tl.read_text(encoding="utf-8"))
        r["attacker"] = "P1" if r["attacker"] == "P2" else "P2"
        r["char"] = d["p1"] if r["attacker"] == "P1" else d["p2"]
        r["cands"] = []
        r["lora"] = None
        r["swapped"] = True
        self.show_row()
        self.info.setText(self.info.text() + "  |  ↔전환됨: 이 캐릭 기술명을 입력해줘")

    def move(self, d: int):
        self.i = max(0, min(len(self.rows) - 1, self.i + d))
        self.show_row()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.show_row()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Tool(); w.show()
    sys.exit(app.exec())
