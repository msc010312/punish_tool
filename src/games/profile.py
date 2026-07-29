"""
games/profile.py  —  '게임 프로필' 추상화

게임마다 다른 것(HUD 좌표, 체력바 색, 텍스트 색/위치, 카운터 레벨, 프레임DB 파일,
캐릭터 명단 등)을 한 객체에 모은다. 코어(hud_reader / punish_engine / analyze_match)는
이 프로필만 보고 동작하므로, 다른 게임을 추가할 때 games/<game>.py 만 새로 만들면 된다.

색 임계는 BGR(OpenCV) 기준, 좌표는 화면 비율(0~1).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ColorRange:
    """채널별 (min,max) 범위. None이면 제한 없음. BGR."""
    r: tuple[int, int] | None = None
    g: tuple[int, int] | None = None
    b: tuple[int, int] | None = None


@dataclass
class GameProfile:
    key: str                      # "ggst"
    name: str                     # "Guilty Gear Strive"
    framedata_file: str           # "framedata_ggst.json"

    # --- 체력바 fill 색 (여러 상태 OR 결합: 평상시 크림 + 저체력 주황 등) ---
    health_colors: list[ColorRange] = field(default_factory=list)

    # --- 캐릭터 이름판 (자동 캐릭터 인식 OCR 영역) ---
    name_left: tuple = (0.08, 0.095, 0.28, 0.14)
    name_right: tuple = (0.72, 0.095, 0.92, 0.14)

    # --- 트레이닝 모드 판별: 맨 왼쪽 프레임카운터 숫자 컬럼(매치엔 없음) ---
    train_col: tuple = (0.0, 0.06, 0.045, 0.55)

    # --- COUNTER/PUNISH 텍스트 ---
    text_left: tuple = (0.03, 0.19, 0.22, 0.33)    # (x0,y0,x1,y1) 비율
    text_right: tuple = (0.78, 0.19, 0.97, 0.33)
    text_yellow: ColorRange = field(default_factory=ColorRange)   # 텍스트 검출용 노랑
    text_words: list[str] = field(default_factory=lambda: ["COUNTER", "PUNISH", "JUST"])

    # --- 카운터 레벨(텍스트 채움 비율 -> 프레임이득) ---
    ctr_noise: float = 0.06       # 이 미만 = 노이즈
    ctr_large: float = 0.25       # 이 이상 = 라지, 사이 = 미디움
    level_adv: dict = field(default_factory=lambda: {"medium": 13, "large": 18})
