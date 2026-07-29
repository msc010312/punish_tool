"""
label_tool.py  —  무브 ID 학습 데이터 '반자동 라벨러' [B안]

트래킹의 불확실성을 우회: YOLO가 주는 깨끗한 두 캐릭 크롭을 사람이 보고
  ① 공격자 크롭 클릭(좌/우)  ② 공격자 캐릭 확인  ③ 데미지 후보에서 무브 확정
한 이벤트 5~10초. 저장 -> dataset/<캐릭>/<무브>/<영상>_<t>.png  (사람 보증 = 라벨 정확)

준비(느림): 영상 스캔으로 카운터/펀ish 이벤트 추출 + 각 이벤트 두 크롭 미리 계산.
사용:  python label_tool.py            # 영상 선택 창
       python label_tool.py <영상.mp4>
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout,
    QFileDialog, QProgressBar, QFrame, QButtonGroup, QCompleter)

import hud_reader as hr
import punish_engine as pe
import analyze_match as am
import localize

OUTROOT = pe.app_dir() / "dataset"
DTS = (-0.10, -0.06, -0.03, 0.0, 0.04, 0.08)   # 이벤트 주변 샘플


def safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s).strip("_") or "x"


def to_pix(bgr, size=None):
    if bgr is None:
        return QPixmap()
    if size:
        bgr = cv2.resize(bgr, size)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


# ─────────────────────────────── 준비 워커: 스캔 + 이벤트별 크롭
class PrepWorker(QObject):
    progress = Signal(int, str)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, video):
        super().__init__(); self.video = video

    def run(self):
        try:
            data = pe.load_framedata_file()
            self.progress.emit(2, "캐릭터 인식…")
            p1, p2 = hr.detect_characters(self.video, list(data))
            charmap = {"P1": p1, "P2": p2}
            cfg = hr.load_config()
            self.progress.emit(5, "이벤트 스캔…(시간 걸림)")
            samples, W, H, fps, rects, truns, wruns = hr.scan(
                self.video, cfg, progress_cb=lambda fi, n: self.progress.emit(5 + int(fi / n * 70), "이벤트 스캔…"))
            te = hr.ocr_text_events(truns) + hr.reversal_wake_events(wruns)
            events, _, _ = hr.build_timeline(samples, te)
            ev = [e.as_dict() for e in events]
            cps = [e for e in ev if e["kind"] in ("counter", "punish")]
            # hit 스타터: 같은 victim에 직전 0.6s 내 다른 hit 없는 hit = 콤보 시동기/단발기.
            # (카운터/펀ish만 모으면 기술이 편향됨 -> 모든 공격 기술 커버하려고 추가)
            hits = [e for e in ev if e["kind"] == "hit"]
            cp_t = [e["t"] for e in cps]
            starters = []
            for h in hits:
                if any(g is not h and g["side"] == h["side"] and 0 < h["t"] - g["t"] <= 0.6 for g in hits):
                    continue                                  # 콤보 중간 타격 제외
                if any(abs(h["t"] - ct) < 0.25 for ct in cp_t):
                    continue                                  # 카운터/펀ish와 중복 제거
                starters.append(h)
            todo = sorted(cps + starters, key=lambda e: e["t"])
            gr = hr.find_game_region(self.video)
            cap = cv2.VideoCapture(self.video)
            vfps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            out = []
            for i, e in enumerate(todo):
                self.progress.emit(75 + int(i / max(1, len(todo)) * 24), f"크롭 계산… {i+1}/{len(todo)}")
                if e["kind"] == "hit":
                    victim = e["side"]; atk = am.OTHER[victim]
                else:
                    atk = am.attacker_of(ev, e); victim = am.OTHER[atk]
                ch = charmap.get(atk)
                if not ch or ch not in data:
                    continue
                md = am.first_hit_damage(ev, e["t"], victim)
                cands = pe.identify_move(data[ch], md * pe.HEALTH) if md is not None else []
                # DTS 프레임들 중 '가장 깨끗한'(신뢰 top-2 박스·분리) 프레임 선택.
                # 맨왼쪽/맨오른쪽이 아니라 신뢰도 top-2 = 진짜 두 캐릭(가짜·스테이지 박스 배제).
                best = None     # (score, left, right, ctx)
                for dt in DTS:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int((e["t"] + dt) * vfps))
                    ok, fr = cap.read()
                    if not ok:
                        continue
                    fr = hr.crop_game(fr, gr)
                    W2 = fr.shape[1]
                    boxes = localize.char_boxes(fr)
                    if not boxes:
                        continue
                    if len(boxes) >= 2:
                        t2 = sorted(boxes, key=lambda b: -b[4])[:2]   # 신뢰 top-2
                        t2 = sorted(t2, key=lambda b: b[0])           # 좌->우
                        sep = abs((t2[0][0] + t2[0][2]) / 2 - (t2[1][0] + t2[1][2]) / 2) > 0.06 * W2
                        score = min(t2[0][4], t2[1][4]) + (0.3 if sep else 0)
                        cand = (score, localize.crop_box(fr, t2[0][:4]),
                                localize.crop_box(fr, t2[1][:4]), fr)
                    else:                                    # 1박스: 화면 위치에 맞는 슬롯에
                        b = boxes[0]; c = localize.crop_box(fr, b[:4])
                        if (b[0] + b[2]) / 2 > 0.5 * W2:
                            cand = (b[4] * 0.3, None, c, fr)  # 오른쪽 슬롯
                        else:
                            cand = (b[4] * 0.3, c, None, fr)  # 왼쪽 슬롯
                    if best is None or cand[0] > best[0]:
                        best = cand
                if best is None:
                    continue
                _, left, right, ctx = best
                other = p2 if atk == "P1" else p1
                out.append(dict(
                    t=e["t"], kind=e["kind"], attacker=atk, char=ch, p1=p1, p2=p2,
                    cands=[m.name for m, _ in cands],
                    movelist=list(data.get(ch, {})),
                    movelist2=list(data.get(other, {})),
                    left=left, right=right, ctx=ctx,
                    guess=("left" if atk == "P1" else "right")))
            cap.release()
            self.done.emit(out)
        except Exception as e:  # noqa
            import traceback; self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ─────────────────────────────── 메인 윈도우
class Labeler(QWidget):
    def __init__(self, video=None):
        super().__init__()
        self.setWindowTitle("Punish Tool — 라벨러")
        self.resize(1080, 720)
        self.setStyleSheet("""
            QWidget{background:#15161c;color:#e8e9ef;font-size:14px;}
            QPushButton{background:#262833;border:1px solid #343748;border-radius:8px;padding:8px 14px;}
            QPushButton:hover{background:#2f3242;}
            QComboBox{background:#1d1f29;border:1px solid #343748;border-radius:8px;padding:6px;}
            #save{background:#2f6df6;border:none;font-weight:bold;}
            #pick{border:3px solid #2f6df6;}
            #saved{color:#5ad27e;}
        """)
        self.events = []; self.idx = 0; self.video = video
        self.mechs = self._load_mechs()

        self.status = QLabel("영상을 여세요."); self.status.setStyleSheet("font-size:16px;")
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.hide()
        open_btn = QPushButton("영상 열기"); open_btn.clicked.connect(self.open_video)

        self.ctx_lbl = QLabel(alignment=Qt.AlignCenter); self.ctx_lbl.setMinimumHeight(300)
        self.left_btn = QPushButton(); self.right_btn = QPushButton()
        for b in (self.left_btn, self.right_btn):
            b.setFixedSize(230, 230); b.setIconSize(b.size())
        self.left_btn.clicked.connect(lambda: self.pick("left"))
        self.right_btn.clicked.connect(lambda: self.pick("right"))
        crops = QHBoxLayout(); crops.addStretch()
        crops.addWidget(self._wrap(self.left_btn, "◀ 왼쪽 캐릭"))
        crops.addWidget(self._wrap(self.right_btn, "오른쪽 캐릭 ▶"))
        crops.addStretch()

        self.char_lbl = QLabel("공격자:")
        self.char_box = QComboBox()
        self.char_box.currentTextChanged.connect(lambda _: self.fill_moves())
        self.move_box = QComboBox(); self.move_box.setMinimumWidth(300)
        self.move_box.setEditable(True)                       # 타이핑 검색
        self.move_box.setInsertPolicy(QComboBox.NoInsert)
        comp = self.move_box.completer()
        comp.setCompletionMode(QCompleter.PopupCompletion)
        comp.setFilterMode(Qt.MatchContains)                  # '236h' 치면 236H 포함 전부
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        self.move_box.lineEdit().setPlaceholderText("무브 검색/선택…")   # 빈칸 시작, 바로 타이핑
        self._items = []
        self.skip_btn = QPushButton("건너뛰기 (S)"); self.skip_btn.clicked.connect(self.skip)
        self.stay_btn = QPushButton("+크롭 저장 (Space)"); self.stay_btn.clicked.connect(lambda: self.save_current(False))
        self.save_btn = QPushButton("저장 & 다음 (Enter)"); self.save_btn.setObjectName("save")
        self.save_btn.clicked.connect(self.save_next)
        row = QHBoxLayout()
        row.addWidget(self.char_lbl); row.addWidget(self.char_box)
        row.addSpacing(16); row.addWidget(QLabel("무브/시스템:")); row.addWidget(self.move_box)
        row.addStretch(); row.addWidget(self.skip_btn); row.addWidget(self.stay_btn); row.addWidget(self.save_btn)

        top = QHBoxLayout(); top.addWidget(open_btn); top.addWidget(self.status, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addWidget(self.bar)
        lay.addWidget(self.ctx_lbl, 1); lay.addLayout(crops); lay.addLayout(row)

        QShortcut(QKeySequence(Qt.Key_Return), self, self.save_next)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.save_next)
        QShortcut(QKeySequence(Qt.Key_S), self, self.skip)
        QShortcut(QKeySequence(Qt.Key_Space), self, lambda: self.save_current(False))
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.pick("left"))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.pick("right"))
        self.pick_side = "right"

        if video:
            self.start(video)

    def _load_mechs(self):
        import json
        p = pe.app_dir() / "mechanics_ggst.json"
        if p.exists():
            try:
                return dict(json.loads(p.read_text(encoding="utf-8")).get("visual_classes", {}))
            except Exception:
                pass
        return {}

    def _wrap(self, btn, caption):
        w = QFrame(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(btn); c = QLabel(caption, alignment=Qt.AlignCenter); v.addWidget(c)
        return w

    def open_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "동영상 (*.mp4 *.mkv *.avi *.mov)")
        if f:
            self.start(f)

    def start(self, video):
        self.video = video; self.bar.show(); self.bar.setValue(0)
        self.status.setText("준비 중…")
        self._thr = QThread(); self._wk = PrepWorker(video)
        self._wk.moveToThread(self._thr)
        self._thr.started.connect(self._wk.run)
        self._wk.progress.connect(lambda v, t: (self.bar.setValue(v), self.status.setText(t)))
        self._wk.done.connect(self.on_ready)
        self._wk.failed.connect(lambda m: self.status.setText("오류: " + m.splitlines()[0]))
        self._thr.start()

    def on_ready(self, events):
        if hasattr(self, "_thr"):
            self._thr.quit()
        self.events = events; self.idx = 0; self.bar.hide()
        if not events:
            self.status.setText("라벨할 이벤트가 없습니다."); return
        self.show_event()

    def pick(self, side):
        e = self.cur()
        if e.get(side) is None:
            return
        self.pick_side = side
        self.left_btn.setObjectName("pick" if side == "left" else "")
        self.right_btn.setObjectName("pick" if side == "right" else "")
        self.left_btn.setStyleSheet(self.left_btn.styleSheet()); self.setStyleSheet(self.styleSheet())
        # 고른 크롭의 캐릭 자동 추정(추정공격자 측이면 공격자, 아니면 상대) — 틀리면 드롭다운 수정
        other = e["p2"] if e["char"] == e["p1"] else e["p1"]
        self.char_box.setCurrentText(e["char"] if side == e["guess"] else other)

    def cur(self):
        return self.events[self.idx]

    def show_event(self):
        e = self.cur()
        n = len(self.events)
        saved = sum(1 for x in self.events if x.get("_done"))
        self.status.setText(f"[{self.idx+1}/{n}]  {e['kind'].upper()}  t={e['t']:.1f}s   "
                            f"(저장 {saved})   매치: {e['p1']} vs {e['p2']}")
        self.ctx_lbl.setPixmap(to_pix(e["ctx"], (760, 428)))
        from PySide6.QtGui import QIcon
        self.left_btn.setIcon(QIcon(to_pix(e.get("left"))) if e.get("left") is not None else QIcon())
        self.right_btn.setIcon(QIcon(to_pix(e.get("right"))) if e.get("right") is not None else QIcon())
        self.left_btn.setEnabled(e.get("left") is not None)
        self.right_btn.setEnabled(e.get("right") is not None)
        # 공격자 후보 캐릭(추정 attacker char 먼저)
        self.char_box.clear()
        chars = [e["char"]] + [c for c in (e["p1"], e["p2"]) if c != e["char"]]
        self.char_box.addItems(chars)
        self.fill_moves()
        self.pick(e["guess"] if e.get(e["guess"]) is not None else
                  ("left" if e.get("left") is not None else "right"))
        self.move_box.setFocus()                 # 바로 타이핑 가능하게 포커스
        self.move_box.lineEdit().selectAll()     # 후보 텍스트 전체선택(타이핑하면 덮어씀)

    def fill_moves(self):
        if not self.events:
            return
        e = self.cur()
        self.move_box.clear()
        ch = self.char_box.currentText()
        ml = e["movelist"] if ch == e["char"] else e["movelist2"]
        cands = e["cands"] if ch == e["char"] else []
        items = (list(dict.fromkeys(cands))
                 + [f"⚙ {m} · {c}" for m, c in self.mechs.items()] + ["──무브──"] + ml)
        self._items = list(dict.fromkeys(items))
        self.move_box.addItems(self._items)
        self.move_box.setCurrentText(cands[0] if cands else "")   # 후보 없으면 빈칸

    def save_next(self):
        self.save_current(True)

    def save_current(self, advance):
        if not self.events:
            return
        if self.move_box.completer().popup().isVisible():
            return                                   # 검색 팝업 떠있으면 = 완성 선택 중, 저장X
        e = self.cur()
        crop = e.get(self.pick_side)
        sel = self.move_box.currentText().strip()
        if sel not in self._items:                   # 타이핑 입력 -> 대소문자 무시 매칭
            sel = next((it for it in self._items if it.lower() == sel.lower()), sel)
        if crop is None or sel in ("──무브──", "") or sel not in self._items:
            self.status.setText("⚠ 목록에 있는 무브/시스템을 선택하세요"); return
        if sel.startswith("⚙ "):                  # 시스템 이펙트(버스트/RC) — 캐릭 무관
            mech = next((m for m in self.mechs if sel[2:].startswith(m)), sel[2:].strip())
            dest = OUTROOT / "__mechanics__" / safe(mech); label = f"[시스템] {mech}"
        else:                                      # 캐릭터 무브
            ch = self.char_box.currentText()
            dest = OUTROOT / safe(ch) / safe(sel); label = f"{ch} {sel}"
        dest.mkdir(parents=True, exist_ok=True)
        stem = safe(Path(self.video).stem)[:40]
        cv2.imwrite(str(dest / f"{stem}_{e['t']:.1f}_{self.pick_side}.png"), crop)
        e["_done"] = True
        if advance:
            self.advance()
        else:                                      # 머무름: 다른 크롭도 라벨(트레이드)
            self.status.setText(f"✔ 저장: {label}   (다른 크롭 라벨하거나 S=다음)")
            other = "right" if self.pick_side == "left" else "left"
            if e.get(other) is not None:
                self.pick(other)

    def skip(self):
        if self.events:
            self.advance()

    def advance(self):
        if self.idx + 1 < len(self.events):
            self.idx += 1; self.show_event()
        else:
            saved = sum(1 for x in self.events if x.get("_done"))
            self.status.setText(f"완료! 저장 {saved}장 -> {OUTROOT}")


def main():
    app = QApplication(sys.argv)
    w = Labeler(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
