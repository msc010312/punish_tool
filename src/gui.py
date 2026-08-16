"""
gui.py  —  격투게임 프레임 분석기 GUI (PySide6)

구조(탭 분리):
  [상단] 게임 선택 · 상태 · 진행바
  [탭1 "AI 코치"] Claude 앱 스타일 채팅. + 버튼으로 영상 올리면 분석 → 코치가 코칭.
  [탭2 "분석"]   좌측 요약·주요 이벤트 / 중앙 영상·타임라인·코칭 리포트. (AI 없이 직접 보기)

두 탭은 같은 분석 결과(timeline)를 공유. 코어는 games.ACTIVE 프로필을 따른다.
실행: python src/gui.py    빌드: build_exe.bat
"""

from __future__ import annotations
import itertools
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QObject, Signal, QUrl, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPen, QTextCursor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGraphicsDropShadowEffect, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QProgressBar, QPushButton,
    QSlider, QStyle, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

import games
import punish_engine as pe


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent

BASE = _app_dir()
KIND_COLOR = {"counter": QColor(255, 196, 0), "punish": QColor(255, 84, 84),
              "just_defense": QColor(96, 176, 255), "reversal_wake": QColor(95, 224, 106),
              "reversal_guard": QColor(255, 150, 40), "burst": QColor(230, 230, 240)}
KIND_LABEL = {"counter": "COUNTER", "punish": "PUNISH", "just_defense": "JUST",
              "reversal_wake": "REVERSAL(기상)", "reversal_guard": "REVERSAL(가드경직)",
              "burst": "버스트"}

ACCENT = "#6c8cff"
HINT = "ℹ  영상을 열면 트레이닝/대전 영상을 자동 판별한 뒤 분석할 수 있어요."
HINT_STYLE = "color:#8a8d9c; font-size:12px; background:transparent;"
QSS = f"""
QWidget {{ background:#15161c; color:#e6e7ee; font-size:13px; }}
QLabel {{ background:transparent; }}
QLabel#title {{ color:#fff; letter-spacing:1px; font-size:30px; }}
QLabel#h {{ color:{ACCENT}; font-weight:700; font-size:12px; }}
QLabel#greet {{ font-size:22px; color:#cfd2e0; }}
QFrame#card {{ background:#1d1f29; border:1px solid #2a2d3c; border-radius:12px; }}
QFrame#bar {{ background:#1a1b22; border-bottom:1px solid #2a2d3c; }}
QPushButton {{ background:#262a3a; border:1px solid #343a4f; border-radius:8px; padding:7px 14px; }}
QPushButton:hover {{ background:#323855; }}
QPushButton#accent {{ background:{ACCENT}; border:none; color:#0d0f17; font-weight:700; }}
QPushButton#accent:hover {{ background:#809bff; }}
QPushButton:disabled {{ color:#5a5d6b; background:#1f212b; }}
QPushButton#seg {{ padding:5px 0; min-width:54px; }}
QPushButton#seg:checked {{ background:{ACCENT}; border:none; color:#0d0f17; font-weight:800; }}
QLineEdit, QComboBox {{ background:#1d1f29; border:1px solid #2f3346; border-radius:8px; padding:6px 12px; }}
QComboBox:hover {{ border:1px solid {ACCENT}; }}
QComboBox::drop-down {{ border:none; width:24px; }}
QComboBox::down-arrow {{ image:none; width:0; height:0;
  border-left:5px solid transparent; border-right:5px solid transparent; border-top:6px solid #9aa0b8; margin-right:10px; }}
QComboBox QAbstractItemView {{ background:#1d1f29; border:1px solid #2f3346; border-radius:8px;
  selection-background-color:#323855; outline:none; padding:4px; }}
QTextEdit, QListWidget {{ background:#1a1b22; border:1px solid #2a2d3c; border-radius:10px; }}
QListWidget::item {{ padding:6px 8px; border-radius:6px; }}
QListWidget::item:selected {{ background:#2c3category; }}
QTabWidget::pane {{ border:none; }}
QTabBar::tab {{ background:transparent; padding:9px 20px; color:#9a9db0; font-weight:600; border-bottom:2px solid transparent; }}
QTabBar::tab:selected {{ color:#fff; border-bottom:2px solid {ACCENT}; }}
QProgressBar {{ background:#1d1f29; border:none; border-radius:7px; height:8px; text-align:center; }}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:7px; }}
QScrollBar:vertical {{ background:transparent; width:10px; }}
QScrollBar::handle:vertical {{ background:#343a4f; border-radius:5px; }}
""".replace("#2c3category", "#2c3350")

STRIVE = None  # Strive 폰트 패밀리명 (main 에서 로드)


def _load_strive():
    global STRIVE
    p = BASE / "Impact - Strive.otf"
    if p.exists():
        fid = QFontDatabase.addApplicationFont(str(p))
        fams = QFontDatabase.applicationFontFamilies(fid)
        if fams:
            STRIVE = fams[0]
    return STRIVE


# ─────────────────────────────────────────────── 코치(LLM) 스트리밍 워커
class CoachWorker(QObject):
    chunk = Signal(str); done = Signal(str)     # done(전체 응답 텍스트)

    def __init__(self, mode, tl, chars, perspective="all",
                 question=None, history=None):
        super().__init__()
        self.mode, self.tl, self.chars = mode, tl, chars
        self.persp, self.q, self.history = perspective, question, history or []

    def run(self):
        buf = []
        try:
            import coach
            if self.mode == "chat":
                gen = coach.chat_stream(self.history, self.tl, self.chars, self.q)
            else:                               # "opening"
                gen = coach.coach_stream(self.tl, self.chars, self.persp)
            for piece in gen:
                buf.append(piece); self.chunk.emit(piece)
        except Exception as e:
            self.chunk.emit(f"(코치 오류: {e})")
        self.done.emit("".join(buf))


# ─────────────────────────────────────────────── 음성 합성 워커
class VoiceWorker(QObject):
    """tts_server에 합성을 요청(블로킹 HTTP)해 wav bytes를 돌려준다. UI 스레드 밖에서."""
    done = Signal(bytes)                          # done(wav bytes; 실패 시 빈 bytes)

    def __init__(self, text, emotion, lang):
        super().__init__(); self.text, self.emotion, self.lang = text, emotion, lang

    def run(self):
        data = b""
        try:
            import tts_client
            data = tts_client.synth(self.text, self.emotion, self.lang) or b""
        except Exception:
            data = b""
        self.done.emit(data)


# ─────────────────────────────────────────────── 분석 워커
class AnalyzeWorker(QObject):
    done = Signal(dict); failed = Signal(str); progress = Signal(int)

    def __init__(self, video: str, p1: str = "", p2: str = ""):
        super().__init__(); self.video = video; self.p1 = p1; self.p2 = p2

    def run(self):
        try:
            import hud_reader as hr, json
            cfg = hr.load_config()
            samples, W, H, fps, rects, text_runs, wake_runs = hr.scan(
                self.video, cfg, progress_cb=lambda fi, n: self.progress.emit(int(fi / n * 88)))
            self.progress.emit(89)
            text_events = hr.ocr_text_events(text_runs) + hr.reversal_wake_events(wake_runs)
            events, l_max, r_max = hr.build_timeline(samples, text_events)
            hr.refine_first_hits(self.video, events, rects, l_max, r_max)  # 60fps 첫히트 정밀화
            ev_dicts = [e.as_dict() for e in events]
            # 입력표시(리플레이) 영상이면 시동기 확정 계층 자동 활성 (없으면 0건, 무해)
            move_stats = link_stats = None
            try:
                import input_moveid, punish_engine as _pe
                fd = _pe.load_framedata_file()
                if fd:
                    ch_map = {"P1": self.p1, "P2": self.p2}
                    n_in = input_moveid.annotate(self.video, ev_dicts, ch_map, fd)
                    if n_in:                       # 입력표시 확인된 영상만 전체 통계 수집
                        self.progress.emit(95)
                        presses = input_moveid.scan_presses(self.video, ch_map, fd)
                        if presses:
                            move_stats = input_moveid.move_stats_from(presses, ev_dicts, ch_map, fd)
                            link_stats = input_moveid.link_stats_from(presses)
            except Exception:
                pass
            # 무브/캐릭/상황 식별: 로컬 VLM이 데미지 후보 중 객관식으로 확정(e['vlm']).
            # 서버(Ollama) 없으면 자동 스킵 -> 데미지 추정 폴백. 분석 1회만 실행, 결과는 캐시.
            try:
                import analyze_match as am, punish_engine as pe
                data = pe.load_framedata_file()
                if data:
                    am.enrich_vlm(ev_dicts, self.video, {"P1": self.p1, "P2": self.p2}, data,
                                  progress_cb=lambda i, n: self.progress.emit(90 + int(i / max(1, n) * 9)))
            except Exception:
                pass
            tl = {"video": Path(self.video).name, "fps": fps, "size": [W, H],
                  "calib": {"l_max_cols": l_max, "r_max_cols": r_max},
                  "events": ev_dicts}
            if move_stats:
                tl["move_stats"] = move_stats
            if link_stats:
                tl["link_stats"] = link_stats
            (BASE / "timeline.json").write_text(json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")
            self.progress.emit(100); self.done.emit(tl)
        except Exception as e:  # noqa
            import traceback; self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────── 타임라인 스트립
class EventStrip(QWidget):
    seek = Signal(int)

    def __init__(self):
        super().__init__(); self.setMinimumHeight(36)
        self.duration_ms = 1; self.position_ms = 0; self.events = []

    def set_events(self, events, duration_ms):
        self.events = [e for e in events if e["kind"] in KIND_COLOR]
        self.duration_ms = max(1, duration_ms); self.update()

    def set_position(self, ms): self.position_ms = ms; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.fillRect(self.rect(), QColor(26, 27, 34)); h = self.height()
        for e in self.events:
            x = int(e["t"] * 1000 / self.duration_ms * self.width())
            p.setPen(QPen(KIND_COLOR[e["kind"]], 2)); p.drawLine(x, 4, x, h - 4)
        cx = int(self.position_ms / self.duration_ms * self.width())
        p.setPen(QPen(QColor(255, 255, 255), 1)); p.drawLine(cx, 0, cx, h); p.end()

    def mousePressEvent(self, ev):
        if self.width():
            self.seek.emit(int(ev.position().x() / self.width() * self.duration_ms))


class ClickSlider(QSlider):
    """클릭한 위치로 바로 이동하는 슬라이더 (기본 QSlider는 페이지 스텝만 움직임)."""
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.maximum() > self.minimum():
            v = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), int(e.position().x()), self.width())
            self.setValue(v)
            self.sliderMoved.emit(v)   # sliderMoved -> player.setPosition 에 연결됨
        super().mousePressEvent(e)


# ─────────────────────────────────────────────── 메인
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Punish Tool")
        ico = BASE / "app.ico"
        if ico.exists(): self.setWindowIcon(QIcon(str(ico)))
        self.resize(1340, 820)
        self.video_path = None
        self.last_tl = None
        self.cur_events = []
        self.perspective = "all"   # 전체 / P1 / P2 관점 필터
        self._load_framedata()
        # 동봉 LLM(llama-server) 예열: 지금 띄워두면 첫 질문 때 로딩이 끝나 있다.
        # 동봉본이 없으면(개발 환경 등) 조용히 무시 — coach가 Ollama 폴백.
        try:
            import llm_server
            if llm_server.available():
                llm_server.ensure()
        except Exception:
            pass
        # 솔 음성 합성 서버(sbv2_env) 예열 — 첫 코칭 때 로딩이 끝나 있게. 없으면 무음 폴백.
        try:
            import tts_client
            if tts_client.available():
                tts_client.ensure()
        except Exception:
            pass

        self.player = QMediaPlayer(); self.audio = QAudioOutput(); self.player.setAudioOutput(self.audio)
        self.audio.setMuted(True)   # 분석 영상은 항상 음소거(요구사항). 코치 목소리는 별도 출력.
        self.video_w = QVideoWidget(); self.player.setVideoOutput(self.video_w)
        self.player.positionChanged.connect(self.on_position)
        self.player.durationChanged.connect(self.on_duration)
        self.player.mediaStatusChanged.connect(self._on_media)
        self._primed = False

        # 코치 목소리 전용 플레이어(분석 영상과 분리). 이게 실제로 솔 목소리를 낸다.
        self.voice_player = QMediaPlayer(); self.voice_audio = QAudioOutput()
        self.voice_player.setAudioOutput(self.voice_audio)
        self.voice_player.mediaStatusChanged.connect(self._on_voice_media)
        self._voice_seq = itertools.count()   # 임시 wav 파일명 회전(윈도 파일락 회피)
        self._pending_emote = None

        self.setCentralWidget(self._build())
        self.setStyleSheet(QSS)
        # 모든 버튼/콤보/탭에 손가락(포인터) 커서
        for b in self.findChildren(QPushButton):
            b.setCursor(Qt.PointingHandCursor)
        for c in self.findChildren(QComboBox):
            c.setCursor(Qt.PointingHandCursor)
        self.tabs.tabBar().setCursor(Qt.PointingHandCursor)
        self.clear_results()
        self._greet()

    # ---------- 구성 ----------
    # ┌─ 레이아웃 조작 가이드 (#5) ─────────────────────────────────────────┐
    # · 화면은 메서드별로 나뉨: _build()=상단바+탭 / _chat_tab()=AI코치 / _analysis_tab()=분석
    # · 양옆/위아래 여백  : 각 레이아웃의 setContentsMargins(왼,위,오,아래)  숫자를 키우면 여유↑
    # · 위젯 사이 간격     : setSpacing(px)
    # · 폭 고정           : 위젯.setFixedWidth(px) / setMaximumWidth(px)   (예: 좌측 패널 left_w)
    # · 비율(늘어나는 정도): addWidget(위젯, 정수)  숫자 클수록 더 넓게 차지 (예: video_w,3 / report,2)
    # · 색·테마           : 맨 위 QSS 문자열 + ACCENT 색
    # · 타임라인 틱 색     : KIND_COLOR / 요약카드 색: _update_summary()
    # └──────────────────────────────────────────────────────────────────┘
    def _build(self):
        # 상단 바
        title = QLabel("Punish Tool"); title.setObjectName("title")
        if STRIVE:
            title.setFont(QFont(STRIVE))   # 패밀리만; 크기는 QSS #title font-size
        self.game_combo = QComboBox()
        for key, name in games.list_games(): self.game_combo.addItem(name, key)
        self.game_combo.currentIndexChanged.connect(self.on_game_changed)
        self.game_combo.setMaximumWidth(180)
        self.status = QLabel("영상을 올려 분석을 시작하세요.")
        self.progress = QProgressBar(); self.progress.setMaximumWidth(200); self.progress.setVisible(False)
        bar = QFrame(); bar.setObjectName("bar")
        bl = QHBoxLayout(bar); bl.setContentsMargins(32, 14, 32, 14); bl.setSpacing(14)
        bl.addWidget(title); bl.addSpacing(24)
        bl.addWidget(QLabel("게임")); bl.addWidget(self.game_combo)
        bl.addWidget(self.status, 1); bl.addWidget(self.progress)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._chat_tab(), "  AI 코치  ")
        self.tabs.addTab(self._analysis_tab(), "  분석  ")

        root = QVBoxLayout(); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(bar); root.addWidget(self.tabs, 1)
        w = QWidget(); w.setLayout(root); return w

    def _chat_tab(self):
        self.greet_lbl = QLabel(); self.greet_lbl.setObjectName("greet")
        self.greet_lbl.setAlignment(Qt.AlignCenter); self.greet_lbl.setWordWrap(True)
        self.chat_log = QTextEdit(); self.chat_log.setReadOnly(True)
        self.chat_log.setFrameShape(QFrame.NoFrame)
        self.chat_stack = QVBoxLayout()
        self.chat_stack.addWidget(self.greet_lbl, 1)
        self.chat_stack.addWidget(self.chat_log, 1)
        self.chat_log.hide()

        up = QPushButton("＋ 영상"); up.clicked.connect(self.pick_and_analyze)
        self.chat_in = QLineEdit(); self.chat_in.setPlaceholderText("코치에게 질문…  (영상은 ＋영상 으로 올리세요)")
        self.chat_in.returnPressed.connect(self.on_chat_send)
        send = QPushButton("전송"); send.setObjectName("accent"); send.clicked.connect(self.on_chat_send)
        inrow = QHBoxLayout(); inrow.addWidget(up); inrow.addWidget(self.chat_in, 1); inrow.addWidget(send)
        inbar = QFrame(); inbar.setObjectName("card"); ib = QVBoxLayout(inbar); ib.addLayout(inrow)

        # 코치 아바타: 실시간 3D(avatar3d/) 우선, 불가 시 스프라이트(avatar/) 폴백
        from avatar3d import make_avatar
        self.avatar = make_avatar(height=300)
        body = QHBoxLayout(); body.setSpacing(12)
        if self.avatar.active:
            av_col = QVBoxLayout(); av_col.addStretch(1)
            av_col.addLayout(self._voice_lang_toggle())   # 아바타 위 음성 언어 전환
            av_col.addWidget(self.avatar)
            body.addLayout(av_col)
        body.addLayout(self.chat_stack, 1)

        lay = QVBoxLayout(); lay.setContentsMargins(80, 28, 80, 28); lay.setSpacing(16)
        lay.addLayout(body, 1); lay.addWidget(inbar)
        w = QWidget(); w.setLayout(lay); return w

    def _voice_lang_toggle(self):
        """아바타 위 음성 언어 전환(한국어/일본어). 코치 답변 TTS의 목소리를 결정한다.
        솔의 일본어는 원본 성우 학습 완료, 한국어는 기반모델 학습 후 연결."""
        from PySide6.QtWidgets import QButtonGroup
        self.voice_lang = "KO"                 # 기본: 한국어(대상 사용자)
        row = QHBoxLayout(); row.setSpacing(0); row.setAlignment(Qt.AlignHCenter)
        row.addWidget(QLabel("🔊"))
        self.lang_btns = {}
        grp = QButtonGroup(self); grp.setExclusive(True)
        for code, label in (("KO", "한국어"), ("JP", "日本語")):
            b = QPushButton(label); b.setObjectName("seg"); b.setCheckable(True)
            b.setChecked(code == self.voice_lang)
            b.clicked.connect(lambda _=False, c=code: self._set_voice_lang(c))
            grp.addButton(b); self.lang_btns[code] = b; row.addWidget(b)
        return row

    def _set_voice_lang(self, code: str):
        """음성 언어 상태 변경. 코치 응답 TTS가 이 값으로 목소리(솔 JP/KO)를 고른다."""
        self.voice_lang = code
        for c, b in self.lang_btns.items():
            b.setChecked(c == code)
        self.status.setText(f"코치 음성: {'한국어' if code == 'KO' else '日本語'}")

    def _analysis_tab(self):
        # 좌측: 요약 카드 + 주요 이벤트
        self.summary_lbl = QLabel(); self.summary_lbl.setTextFormat(Qt.RichText)
        card = QFrame(); card.setObjectName("card"); cl = QVBoxLayout(card); cl.addWidget(self.summary_lbl)
        self.list = QListWidget(); self.list.itemClicked.connect(self.on_event_click)
        # 관점 필터 (전체/P1/P2) — 리포트·타임라인·이벤트가 같이 바뀜
        self.persp_btns = {}
        prow = QHBoxLayout(); prow.setSpacing(6)
        for key, lbl in [("all", "전체"), ("P1", "P1"), ("P2", "P2")]:
            b = QPushButton(lbl); b.setCheckable(True); b.setObjectName("seg")
            b.clicked.connect(lambda _=False, k=key: self._set_perspective(k))
            self.persp_btns[key] = b; prow.addWidget(b)
        prow.addStretch(1)
        self.persp_btns["all"].setChecked(True)

        notice = QLabel("· 이벤트 클릭 시 0.5초 전부터 재생됩니다\n"
                        "· 트레이닝 영상은 분석되지 않습니다\n"
                        "· 게임 화면이 아닌 영상은 분석되지 않습니다\n"
                        "· 분석에는 영상 길이의 절반 정도 시간이 걸립니다\n"
                        "· 배너(카운터 등) 감지율 약 95% — 실측 기준,\n"
                        "  일부 놓칠 수 있으나 없는 것을 만들지는 않습니다")
        notice.setWordWrap(True)
        notice.setStyleSheet("color:#8a8d9c; font-size:11px;")
        ncard = QFrame(); ncard.setObjectName("card")
        ncl = QVBoxLayout(ncard); ncl.addWidget(self._h("안내")); ncl.addWidget(notice)

        left = QVBoxLayout()
        left.addWidget(self._h("관점")); left.addLayout(prow); left.addSpacing(8)
        left.addWidget(self._h("요약")); left.addWidget(card)
        left.addSpacing(8); left.addWidget(self._h("주요 이벤트 (클릭=점프)")); left.addWidget(self.list, 1)
        left.addSpacing(8); left.addWidget(ncard)
        left_w = QWidget(); left_w.setLayout(left); left_w.setFixedWidth(260)

        # 중앙: 영상 + 컨트롤 + 스트립 + 리포트
        self.open_btn = QPushButton("영상 열기"); self.open_btn.clicked.connect(self.pick_video)
        self.analyze_btn = QPushButton("분석 시작"); self.analyze_btn.setObjectName("accent")
        self.analyze_btn.clicked.connect(lambda: self.run_analysis(self.video_path))
        self.analyze_btn.setCursor(Qt.PointingHandCursor)   # 항상 활성 -> 커서 정상
        # P1/P2 는 자동 인식 전용 (빈 채로 시작, 직접 수정 불가)
        self.p1_edit = QLineEdit(); self.p1_edit.setReadOnly(True); self.p1_edit.setMaximumWidth(140)
        self.p1_edit.setPlaceholderText("자동 인식")
        self.p2_edit = QLineEdit(); self.p2_edit.setReadOnly(True); self.p2_edit.setMaximumWidth(140)
        self.p2_edit.setPlaceholderText("자동 인식")
        # 네온 빨강 경고 — 배경 투명이라 '글자'에만 글로우(빈 값이면 아무것도 안 보임)
        self.warn_lbl = QLabel("")
        self.warn_lbl.setStyleSheet("color:#ff3b3b; font-weight:800; background:transparent;")
        glow = QGraphicsDropShadowEffect(self); glow.setBlurRadius(22)
        glow.setColor(QColor(255, 40, 40)); glow.setOffset(0, 0)
        self.warn_lbl.setGraphicsEffect(glow)
        oprow = QHBoxLayout()
        for x in (self.open_btn, QLabel("P1"), self.p1_edit, QLabel("P2"), self.p2_edit, self.analyze_btn):
            oprow.addWidget(x)
        oprow.addWidget(self.warn_lbl); oprow.addStretch(1)

        self.play_btn = QPushButton(); self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setMaximumWidth(44); self.play_btn.clicked.connect(self.toggle_play)
        self.slider = ClickSlider(Qt.Horizontal); self.slider.sliderMoved.connect(self.player.setPosition)
        self.time_lbl = QLabel("0:00 / 0:00")
        self.mute_btn = QPushButton(); self.mute_btn.setMaximumWidth(44)
        self.mute_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaVolumeMuted))  # 기본 뮤트
        self.mute_btn.clicked.connect(self.toggle_mute)
        ctl = QHBoxLayout(); ctl.addWidget(self.play_btn); ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.time_lbl); ctl.addWidget(self.mute_btn)
        self.strip = EventStrip(); self.strip.seek.connect(self.player.setPosition)
        self.report = QTextEdit(); self.report.setReadOnly(True)
        self.report.setStyleSheet("font-family:Consolas,monospace;")

        self.hint_lbl = QLabel(HINT); self.hint_lbl.setStyleSheet(HINT_STYLE)
        center = QVBoxLayout()
        center.addWidget(self.hint_lbl)
        center.addLayout(oprow)
        center.addWidget(self.video_w, 3)
        center.addLayout(ctl)
        center.addWidget(QLabel("타임라인 (노랑=COUNTER 빨강=PUNISH 파랑=JUST · 클릭=점프)"))
        center.addWidget(self.strip)
        center.addWidget(self._h("코칭 리포트")); center.addWidget(self.report, 2)
        center_w = QWidget(); center_w.setLayout(center)

        lay = QHBoxLayout(); lay.setContentsMargins(32, 22, 32, 22); lay.setSpacing(28)
        lay.addWidget(left_w); lay.addWidget(center_w, 1)
        w = QWidget(); w.setLayout(lay); return w

    def _h(self, t): l = QLabel(t); l.setObjectName("h"); return l

    def _greet(self):
        self.greet_lbl.setText("격투게임 프레임 분석기<br><br>"
                               "<span style='font-size:15px;color:#9a9db0'>＋영상 으로 매치를 올리면 "
                               "카운터·확정반격을 분석해주고,<br>"
                               "아래 칸에 길티기어 뭐든 물어보면 코치가 대답해줘요.</span>")

    # ---------- 데이터/게임 ----------
    def _load_framedata(self):
        try: self.framedata = pe.load_framedata_file()
        except Exception: self.framedata = None

    def on_game_changed(self, _):
        key = self.game_combo.currentData()
        if key:
            games.set_active(key); self._load_framedata(); self.clear_results()
            self.status.setText(f"게임: {games.ACTIVE.name}")

    def clear_results(self):
        self.list.clear()
        self.report.setPlainText("분석하면 코칭 리포트가 여기 표시됩니다.")
        self.strip.set_events([], self.player.duration() or 1)
        self.last_tl = None; self._update_summary(None)
        if hasattr(self, "warn_lbl"): self.warn_lbl.setText("")
        if hasattr(self, "hint_lbl"):
            self.hint_lbl.setText(HINT); self.hint_lbl.setStyleSheet(HINT_STYLE)

    # ---------- 영상 열기/분석 ----------
    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "영상 (*.mp4 *.mkv *.mov *.avi)")
        if path: self.open_video(path)

    def pick_and_analyze(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "영상 (*.mp4 *.mkv *.mov *.avi)")
        if not path: return
        v = self.open_video(path)
        if v == "match":
            self.run_analysis(path)
        elif v == "training":
            self._chat("코치", "<b style='color:#ff5b5b'>트레이닝 영상은 분석이 불가능합니다.</b> 실전 매치 영상을 올려주세요.")
        else:
            self._chat("코치", "<b style='color:#ff5b5b'>게임플레이 영상을 올려주세요.</b> 이 영상에선 게임 화면을 찾지 못했어요.")

    def open_video(self, path):
        """영상 열고 검사 -> verdict('match'/'training'/'not_gameplay'). P1/P2 자동 채움."""
        self.video_path = path
        self._primed = False
        self.player.setSource(QUrl.fromLocalFile(path))
        # 이전 분석 결과는 [분석 시작] 전까지 유지 (영상만 바꿔 미리보기 가능)
        self.p1_edit.clear(); self.p2_edit.clear(); self.warn_lbl.setText("")
        self.status.setText("영상 확인 중…"); QApplication.processEvents()
        verdict, p1, p2 = self._inspect(path)
        self._verdict = verdict
        if verdict == "match":
            self.p1_edit.setText(p1 or ""); self.p2_edit.setText(p2 or "")
            self.hint_lbl.setText("✓  대전 영상 확인 — 아래 [분석 시작]을 누르세요.")
            self.hint_lbl.setStyleSheet("color:#7ad17a; font-size:12px; background:transparent;")
            self.status.setText(f"캐릭터: {p1 or '?'} vs {p2 or '?'} — {Path(path).name}")
        elif verdict == "training":
            self.warn_lbl.setText("트레이닝 영상은 분석이 불가능합니다")
            self.status.setText("트레이닝 영상")
        else:
            self.warn_lbl.setText("게임플레이 영상을 올려주세요")
            self.status.setText("게임플레이 영상이 아닙니다")
        return verdict

    def _inspect(self, path):
        """게임플레이/트레이닝/비게임 판별. 캐릭 이름판이 안 잡히면 게임 영상 아님."""
        try:
            import hud_reader as hr
            chars = list(self.framedata) if self.framedata else []
            p1, p2 = hr.detect_characters(path, chars) if chars else (None, None)
            if not p1 and not p2:
                return "not_gameplay", None, None
            if hr.detect_mode(path) == "training":
                return "training", p1, p2
            return "match", p1, p2
        except Exception:
            return "not_gameplay", None, None

    def run_analysis(self, path):
        if not path:
            self.warn_lbl.setText("영상을 먼저 올려주세요"); return
        v = getattr(self, "_verdict", None)
        if v == "training":
            self.warn_lbl.setText("트레이닝 영상은 분석이 불가능합니다"); return
        if v != "match":
            self.warn_lbl.setText("게임플레이 영상을 올려주세요"); return
        self.warn_lbl.setText("")
        self.analyze_btn.setEnabled(False); self.open_btn.setEnabled(False)
        self.progress.setVisible(True); self.progress.setValue(0)
        self.status.setText("분석 중…")
        self._chat("코치", "영상 분석 중이에요… 잠시만요.")
        self.thread = QThread()
        self.worker = AnalyzeWorker(path, self.p1_edit.text().strip(), self.p2_edit.text().strip())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self.on_analyzed); self.worker.failed.connect(self.on_failed)
        self.worker.done.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def on_analyzed(self, tl):
        self.open_btn.setEnabled(True); self.analyze_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.populate(tl); self.status.setText("분석 완료 ✓")
        ev = tl["events"]
        nc = sum(1 for e in ev if e["kind"] == "counter")
        npn = sum(1 for e in ev if e["kind"] == "punish")
        self._chat("코치", f"분석 완료. COUNTER {nc}회 · PUNISH {npn}회 잡았다. 코칭 뽑는 중…")
        self._chat_history = []           # 새 매치 -> 대화 기억 초기화
        self._coach_speak("opening")      # 오프닝 코칭(스트리밍)

    def on_failed(self, msg):
        self.open_btn.setEnabled(True); self.analyze_btn.setEnabled(True)
        self.progress.setVisible(False); self.status.setText("분석 실패")
        self.report.setPlainText(msg)

    # ---------- 재생 ----------
    def toggle_mute(self):
        # 분석 영상은 항상 음소거(요구사항). 이 버튼은 코치 목소리를 켜고 끈다.
        m = not self.voice_audio.isMuted()
        self.voice_audio.setMuted(m)
        self.mute_btn.setIcon(self.style().standardIcon(
            QStyle.SP_MediaVolumeMuted if m else QStyle.SP_MediaVolume))

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause(); self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.player.play(); self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def _on_media(self, st):
        # 영상 로드되면 한 번 play->pause 로 PausedState 진입 -> 재생 안 해도 seek 시 프레임 표시
        if not self._primed and st in (QMediaPlayer.MediaStatus.LoadedMedia,
                                       QMediaPlayer.MediaStatus.BufferedMedia):
            self._primed = True
            self.player.play(); self.player.pause()

    def on_duration(self, dur):
        self.slider.setRange(0, dur); self.strip.set_events(self.strip.events, dur); self._update_time()

    def on_position(self, pos):
        self.slider.setValue(pos); self.strip.set_position(pos); self._update_time()

    def _update_time(self):
        def f(ms): s = ms // 1000; return f"{s//60}:{s%60:02d}"
        self.time_lbl.setText(f"{f(self.player.position())} / {f(self.player.duration())}")

    # ---------- 결과 표시 ----------
    def populate(self, tl):
        self.last_tl = tl
        self.cur_events = tl.get("events", [])
        self._render()

    def _set_perspective(self, key):
        self.perspective = key
        for k, b in self.persp_btns.items():
            b.setChecked(k == key)
        if self.last_tl:
            self._render()

    def _render(self):
        """현재 관점(전체/P1/P2)으로 이벤트 리스트·타임라인·리포트를 다시 그린다."""
        import analyze_match as am
        events = self.cur_events

        def disp_side(e):
            return am.attacker_of(events, e) if e["kind"] in ("counter", "punish") else e["side"]

        ftext = [e for e in events if e["kind"] in KIND_COLOR
                 and (self.perspective == "all" or disp_side(e) == self.perspective)]
        dur = self.player.duration() or int(max([e["t"] for e in events], default=0) * 1000) + 1000
        self.strip.set_events(ftext, dur)
        self.list.clear()
        for e in ftext:
            t = e["t"]; s = int(t); side = disp_side(e)
            extra = f" +{e.get('frame_adv',0)}F" if e["kind"] == "counter" else ""
            if e.get("burst"):
                extra = " (버스트)"                     # 버스트로 뜬 카운터 — 반격 분석 대상 아님
            it = QListWidgetItem(f"{s//60}:{s%60:02d}  {side}  {KIND_LABEL[e['kind']]}{extra}")
            it.setData(Qt.UserRole, int(t * 1000)); it.setData(Qt.UserRole + 1, float(t))
            it.setForeground(KIND_COLOR[e["kind"]]); self.list.addItem(it)
        self._update_summary(events)
        if self.framedata is not None and self.last_tl:
            try:
                self.report.setPlainText(am.build_report(
                    self.last_tl, self.p1_edit.text().strip(), self.p2_edit.text().strip(),
                    self.framedata, perspective=self.perspective, use_vlm=False))  # VLM은 워커서 1회 완료
            except Exception as ex:
                self.report.setPlainText(f"리포트 오류: {ex}")

    def _update_summary(self, events):
        def c(kind, side): return sum(1 for e in (events or []) if e["kind"] == kind and e["side"] == side)
        rows = [("COUNTER", "counter", "#ffc400"), ("PUNISH", "punish", "#ff5454"),
                ("JUST", "just_defense", "#60b0ff"), ("REVERSAL(기상)", "reversal_wake", "#5fe06a"),
                ("REVERSAL(가드경직)", "reversal_guard", "#ff9628")]
        html = "<table cellspacing=7><tr><td></td><td><b>P1</b></td><td><b>P2</b></td></tr>"
        for name, kind, col in rows:
            html += (f"<tr><td><b style='color:{col}'>{name}</b></td>"
                     f"<td align=center>{c(kind,'P1')}</td><td align=center>{c(kind,'P2')}</td></tr>")
        self.summary_lbl.setText(html + "</table>")

    SEEK_LEAD_MS = 500       # 이벤트(배너) 시각보다 앞으로 시크 -> 시동 동작이 보이게

    def on_event_click(self, item):
        ms = item.data(Qt.UserRole)
        if ms is not None: self.player.setPosition(max(0, int(ms) - self.SEEK_LEAD_MS))
        t = item.data(Qt.UserRole + 1)
        if t is not None: self.scroll_report_to(float(t))

    def scroll_report_to(self, t):
        timestr = f"{int(t)//60}:{t%60:05.2f}"
        self.report.moveCursor(QTextCursor.Start)
        if self.report.find(timestr):
            self.report.ensureCursorVisible()
            # 찾은 줄을 화면 '위쪽'으로 올려 해당 블록 전체가 보이게
            sb = self.report.verticalScrollBar()
            sb.setValue(sb.value() + self.report.cursorRect().top() - 12)

    # ---------- 채팅 ----------
    def _render_chat(self):
        html = []
        for who, text in getattr(self, "_messages", []):
            me = who == "나"
            col = ACCENT if me else "#262a3a"
            align = "right" if me else "left"
            fg = "#0d0f17" if me else "#e6e7ee"
            body = text.replace("\n", "<br>")
            html.append(
                f"<table width='100%'><tr><td align='{align}'>"
                f"<table><tr><td style='background:{col};color:{fg};padding:8px 12px;'>"
                f"{body}</td></tr></table></td></tr></table>")
        self.chat_log.setHtml("".join(html))
        self.chat_log.verticalScrollBar().setValue(self.chat_log.verticalScrollBar().maximum())

    def _chat(self, who, text):
        if self.greet_lbl.isVisible():
            self.greet_lbl.hide(); self.chat_log.show()
        if not hasattr(self, "_messages"):
            self._messages = []
        self._messages.append([who, text])
        self._render_chat()

    def _coach_speak(self, mode, question=None):
        """코치 스트리밍 시작. mode='opening'(분석직후) | 'chat'(자유대화)."""
        if getattr(self, "_coach_thr", None) and self._coach_thr.isRunning():
            return                         # 코치 응답 중엔 새 요청 무시
        if getattr(self, "voice_player", None):
            self.voice_player.stop()       # 이전 발화 중단
        self._chat("코치", "…")
        self._coach_buf = ""
        self.avatar.set_state("think")     # 첫 토큰 대기 = 생각 포즈
        persp = getattr(self, "perspective", "all")
        chars = {"P1": self.p1_edit.text().strip() or "P1",
                 "P2": self.p2_edit.text().strip() or "P2"}
        if not hasattr(self, "_chat_history"):
            self._chat_history = []        # [{role,content}] 멀티턴 대화 기억
        self._coach_thr = QThread()
        self._coach_wk = CoachWorker(mode, self.last_tl, chars, persp,
                                     question, list(self._chat_history))
        self._coach_wk.moveToThread(self._coach_thr)
        self._coach_thr.started.connect(self._coach_wk.run)
        self._coach_wk.chunk.connect(self._coach_chunk)
        self._coach_wk.done.connect(self._coach_done)
        self._coach_wk.done.connect(self._coach_thr.quit)
        self._coach_thr.start()

    def _coach_chunk(self, piece):
        if not getattr(self, "_coach_buf", ""):
            self.avatar.set_state("talk")  # 첫 조각 도착 = 말하기 시작
        self._coach_buf = getattr(self, "_coach_buf", "") + piece
        if self._messages:
            self._messages[-1][1] = self._coach_buf
            self._render_chat()

    # 답변 내용 -> 감정 제스처(결정론적 키워드 매핑; 3D 아바타의 emote 지원 시)
    _EMOTES = [
        (("잘했", "올랐네", "축하", "좋네", "제법"), "joy"),
        (("몰라", "없어", "없다", "아니", "지어내", "안 돼", "없는 것"), "shake"),
        (("⚠", "그만", "핑계", "정신"), "angry"),
        (("이기고", "이겨", "덤벼", "가서 이기"), "smirk"),
        (("아쉽", "아깝", "졌구나", "안타"), "sad"),
        (("쉬어", "한숨"), "sigh"),
        (("오,", "대박", "!?"), "surprised"),
        (("그래", "좋아", "맞아", "반갑", "됐다"), "nod"),
    ]

    def _pick_emote(self, text):
        for keys, name in self._EMOTES:
            if any(k in text for k in keys):
                return name
        return None

    def _coach_done(self, full):
        """응답 완료 -> 감정 감지 -> (가능하면)솔 목소리로 발화 + 립싱크 -> 감정 제스처.
        음성 서버가 없거나 실패하면 글자수 기반 talk 유지로 무음 폴백."""
        if not hasattr(self, "_chat_history"):
            self._chat_history = []
        if full.strip():
            self._chat_history.append({"role": "assistant", "content": full})
        emote = self._pick_emote(full)
        self._pending_emote = emote
        if not self._speak(full, emote or "neutral"):
            # 무음 폴백: 말하기 지속시간 = 글자수 비례(1.2~7초)
            hold_ms = int(max(1.2, min(7.0, len(full) * 0.045)) * 1000)
            QTimer.singleShot(hold_ms, lambda: self._finish_talk(emote))

    # ---------- 솔 목소리 발화 ----------
    def _speak(self, text, emotion) -> bool:
        """코치 답변을 솔 목소리로 합성·재생 시작. 서버 준비 안됐으면 False(폴백)."""
        text = (text or "").strip()
        if not text:
            return False
        try:
            import tts_client
            if not tts_client.ready():
                return False
        except Exception:
            return False
        self.voice_player.stop()
        lang = getattr(self, "voice_lang", "KO")
        self._voice_thr = QThread()
        self._voice_wk = VoiceWorker(text, emotion, lang)
        self._voice_wk.moveToThread(self._voice_thr)
        self._voice_thr.started.connect(self._voice_wk.run)
        self._voice_wk.done.connect(self._on_voice_ready)
        self._voice_wk.done.connect(self._voice_thr.quit)
        self._voice_thr.start()
        return True

    def _on_voice_ready(self, data: bytes):
        """합성 완료 -> 임시 wav 재생 시작. 재생 동안 아바타가 말한다."""
        if not data:                              # 합성 실패 -> 무음 폴백(짧은 talk)
            self.avatar.set_state("talk")
            QTimer.singleShot(1200, lambda: self._finish_talk(self._pending_emote))
            return
        p = Path(tempfile.gettempdir()) / f"punish_coach_voice_{next(self._voice_seq)}.wav"
        try:
            p.write_bytes(data)
        except Exception:
            self._finish_talk(self._pending_emote); return
        self._voice_wav = p
        self.avatar.set_state("talk")
        self.voice_player.setSource(QUrl.fromLocalFile(str(p)))
        self.voice_player.play()

    def _on_voice_media(self, st):
        """음성 재생 끝 -> 대기 포즈 + 감정 제스처."""
        if st == QMediaPlayer.MediaStatus.EndOfMedia:
            self._finish_talk(self._pending_emote)

    def _finish_talk(self, emote):
        self.avatar.set_state("idle")
        if emote:
            if hasattr(self.avatar, "play_emote"):            # 시퀀스 아바타
                self.avatar.play_emote(emote)
            elif hasattr(self.avatar, "page"):                # (구)3D 아바타
                self.avatar.page().runJavaScript(
                    f"window.emote && window.emote('{emote}')")

    def on_chat_send(self):
        msg = self.chat_in.text().strip()
        if not msg: return
        self.chat_in.clear(); self._chat("나", msg)
        if not hasattr(self, "_chat_history"):
            self._chat_history = []
        self._chat_history.append({"role": "user", "content": msg})
        self._coach_speak("chat", question=msg)


def main():
    app = QApplication(sys.argv)
    _load_strive()
    ico = BASE / "app.ico"
    if ico.exists(): app.setWindowIcon(QIcon(str(ico)))
    w = Main(); w.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
