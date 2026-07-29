"""
games  —  게임 프로필 레지스트리

코어는 `games.ACTIVE` (현재 선택된 게임 프로필)만 본다.
게임 추가: games/<game>.py 에 PROFILE 정의 후 아래 REGISTRY 에 등록.
게임 전환: set_active("<key>") (GUI 게임 선택 메뉴에서 호출).
"""

from __future__ import annotations
from .profile import GameProfile, ColorRange
from . import ggst

REGISTRY: dict[str, GameProfile] = {
    ggst.PROFILE.key: ggst.PROFILE,
}

# 현재 활성 게임 (기본: GGST)
ACTIVE: GameProfile = ggst.PROFILE


def set_active(key: str) -> GameProfile:
    global ACTIVE
    if key not in REGISTRY:
        raise KeyError(f"등록되지 않은 게임: {key} (가능: {list(REGISTRY)})")
    ACTIVE = REGISTRY[key]
    return ACTIVE


def list_games() -> list[tuple[str, str]]:
    """(key, 표시이름) 목록 — GUI 게임 선택 메뉴용."""
    return [(p.key, p.name) for p in REGISTRY.values()]
