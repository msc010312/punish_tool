# -*- coding: utf-8 -*-
"""avatar3d.py — 실시간 3D 솔 아바타(QWebEngineView + three.js, 오프라인 60fps).

app_dir()/avatar3d/ (index.html + three.min.js + GLTFLoader.js + sol.glb)를
로드해 스탠드 루프를 실시간 재생. 상태는 JS로 전달:
  set_state('idle'|'think'|'talk')
폴더가 없거나 WebEngine 불가면 avatar.AvatarWidget(스프라이트)로 폴백하도록
gui 쪽에서 make_avatar()를 쓴다.
"""
from __future__ import annotations

from pathlib import Path


def _asset_dir() -> Path:
    import punish_engine as pe
    return pe.app_dir() / "avatar3d"


def available() -> bool:
    d = _asset_dir()
    if not (d / "index.html").exists() or not (d / "sol.glb").exists():
        return False
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        return True
    except Exception:
        return False


def make_avatar(height: int = 300):
    """실시간 3D 우선(60fps·본 제어), 불가 시 프리렌더 시퀀스 폴백.
    반환 위젯은 .active(bool)/.set_state(str)(/.play_emote)를 갖는다."""
    if available():
        try:
            return Avatar3DWidget(height)
        except Exception:
            pass
    from avatar import AvatarWidget
    return AvatarWidget(height)


class Avatar3DWidget:  # 실제 클래스는 지연 정의(WebEngine import 회피)
    def __new__(cls, height: int = 300):
        from PySide6.QtCore import QUrl, Qt
        from PySide6.QtGui import QColor
        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView

        class _W(QWebEngineView):
            def __init__(self, h):
                super().__init__()
                self.active = True
                self._state = "idle"
                self.setFixedSize(int(h * 0.9), h)
                self.setAttribute(Qt.WA_TranslucentBackground)
                self.setStyleSheet("background: transparent;")
                self.page().setBackgroundColor(QColor(0, 0, 0, 0))
                s = self.settings()
                s.setAttribute(
                    QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
                self.load(QUrl.fromLocalFile(str(_asset_dir() / "index.html")))

            def set_state(self, state: str):
                if state == self._state:
                    return
                self._state = state
                self.page().runJavaScript(f"window.setState && window.setState('{state}')")

        return _W(height)
