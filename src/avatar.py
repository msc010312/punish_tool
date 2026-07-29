# -*- coding: utf-8 -*-
"""avatar.py — 코치 아바타(프리렌더 시퀀스 재생기, 최종 노선).

Blender에서 그림자 캐스팅 포함으로 프리렌더한 PNG 시퀀스를 재생한다.
  idle_00..NN.png        대기 루프(20fps, 깜빡임 포함)
  talk_00..NN.png        말하기 루프
  think_0.png            생각
  emote_<name>_00..NN.png  감정 원샷(재생 후 idle 복귀)
폴더(app_dir/avatar)가 비어 있으면 위젯 자동 숨김(무해).
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel

FPS_MS = 50                       # 20fps
STATES = ("idle", "talk", "think")


def _asset_dir() -> Path:
    import punish_engine as pe
    return pe.app_dir() / "avatar"


class AvatarWidget(QLabel):
    """시퀀스 재생기: 상태 루프 + 감정 원샷."""

    def __init__(self, height: int = 300, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        self._h = height
        self._frames: dict[str, list[QPixmap]] = {}
        self._emotes: dict[str, list[QPixmap]] = {}
        self._state = "idle"
        self._emote_seq: list[QPixmap] | None = None
        self._i = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._load()

    def _load_seq(self, pattern: str) -> list[QPixmap]:
        d = _asset_dir()
        out = []
        for p in sorted(d.glob(pattern)):
            pm = QPixmap(str(p))
            if not pm.isNull():
                out.append(pm.scaledToHeight(self._h, Qt.SmoothTransformation))
        return out

    def _load(self):
        d = _asset_dir()
        if d.is_dir():
            self._frames["idle"] = self._load_seq("idle_*.png")
            self._frames["talk"] = self._load_seq("talk_*.png")
            self._frames["think"] = self._load_seq("think_*.png")
            names = set()
            for p in d.glob("emote_*_*.png"):
                m = re.match(r"emote_(.+)_\d+\.png", p.name)
                if m:
                    names.add(m.group(1))
            for n in names:
                self._emotes[n] = self._load_seq(f"emote_{n}_*.png")
        if not self._frames.get("idle"):
            self.hide()
            return
        for s in ("talk", "think"):
            if not self._frames.get(s):
                self._frames[s] = self._frames["idle"]
        self._timer.start(FPS_MS)
        self._show(self._frames["idle"][0])

    @property
    def active(self) -> bool:
        return bool(self._frames.get("idle"))

    def set_state(self, state: str):
        if not self.active or state not in STATES:
            return
        self._state = state
        # 상태 전환 시 감정 원샷은 유지(끝나면 새 상태로)

    def play_emote(self, name: str):
        seq = self._emotes.get(name)
        if seq:
            self._emote_seq = seq
            self._i = 0

    def _tick(self):
        if not self.active:
            return
        if self._emote_seq is not None:
            if self._i < len(self._emote_seq):
                self._show(self._emote_seq[self._i])
                self._i += 1
                return
            self._emote_seq = None
            self._i = 0
        frames = self._frames[self._state]
        self._i = (self._i + 1) % len(frames)
        self._show(frames[self._i])

    def _show(self, pm: QPixmap):
        self.setPixmap(pm)
