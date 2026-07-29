"""
games/ggst.py  —  길티기어 스트라이브 게임 프로필

여기에 GGST 고유의 모든 수치를 모은다. 다른 게임(SF6 등)은 이 파일을 복사해
값만 바꿔 games/<game>.py 로 만들면 코어 수정 없이 추가된다.
근거 수치는 HUD_NOTES.md 참고.
"""

from __future__ import annotations
from .profile import GameProfile, ColorRange

PROFILE = GameProfile(
    key="ggst",
    name="Guilty Gear Strive",
    framedata_file="framedata_ggst.json",

    # 체력바: 평상시 크림(R255/G180/B130) + 저체력 주황(R170/G110/B40)
    health_colors=[
        ColorRange(r=(235, 255), g=(150, 215), b=(100, 185)),   # cream
        ColorRange(r=(150, 255), g=(90, 160), b=(0, 94)),       # orange(저체력)
    ],

    # 캐릭터 이름판: 상단 포트레이트 옆 (흰 글자). 우측은 우정렬이라 넓게.
    name_left=(0.055, 0.092, 0.30, 0.145),
    name_right=(0.70, 0.092, 0.95, 0.145),

    # 트레이닝 판별: 맨 왼쪽 입력 프레임카운터 숫자 컬럼 (매치엔 없음)
    train_col=(0.0, 0.06, 0.045, 0.55),

    # 배너 텍스트: 체력바 아래 양쪽 바깥. 실제 영역은 hud_config.json 의
    # banner_left/right(banner_calibrate.py) 가 있으면 그쪽이 우선.
    text_left=(0.03, 0.19, 0.22, 0.33),
    text_right=(0.78, 0.19, 0.97, 0.33),
    text_yellow=ColorRange(r=(200, 255), g=(170, 255), b=(0, 119)),
    # REVERSAL: 기상 프레임에 필살기/각성필살기를 낸 것(1F 무적기가 많으나 필수는 아님).
    # 기본기(5P)·특수기(6P)에는 안 뜬다. 어휘에서 빠져 있으면 PUNISH 로 오배정된다.
    text_words=["COUNTER", "PUNISH", "JUST", "REVERSAL"],

    # 카운터 레벨: <0.06 노이즈 / 0.06~0.25 미디움(+13) / >=0.25 라지(+18)
    ctr_noise=0.06,
    ctr_large=0.25,
    level_adv={"medium": 13, "large": 18},
)
