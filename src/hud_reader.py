"""
hud_reader.py  —  길티기어 스트라이브 실전 영상 HUD 리더 v0 (①번 벽돌)

목표: 매치 영상(60fps)을 프레임 단위로 훑어, 게임이 화면에 주는 정보만으로
      "언제 누가 데미지를 입었는지" 이벤트 타임라인을 뽑는다. (무브 인식 없음)

핵심 아이디어 — 자가 보정(self-calibration):
  체력바 색은 '밝은 크림색'(BGR≈130,180,255 = R255/G180/B130)이고, 불 배경(빨강)은
  G가 낮아 색으로 분리된다. 단, 바의 정확한 좌우 끝은 영상마다/해상도마다 다르므로
  좌표를 손으로 박지 않는다. 대신 영상 전체에서 각 측 체력바의 '최대 cream 폭'을 찾아
  그걸 100% HP 기준으로 삼는다 -> 이후 각 프레임의 cream 폭 / 최대폭 = HP%.

  이러면 사용자가 좌표를 캘리브레이션할 필요가 없다. (해상도·스테이지 무관하게 동작 지향)

한계(정직하게):
  - PUNISH/COUNTER 텍스트 검출은 아직 골격만(별도 색/위치 보정 필요).
  - 비(非)게임플레이 프레임(메뉴/리플레이UI)은 'cream 양측 존재' 게이트로 대충 거른다.
  - 캐릭터 거리/좌표는 아직 안 읽는다(엔진의 가드백 2차 필터용으로 다음 단계).

사용:
  python hud_reader.py "<영상.mp4>"              # 타임라인 JSON 출력
  python hud_reader.py "<영상.mp4>" --debug       # 보정 검증용 주석 프레임 저장
"""

from __future__ import annotations
import argparse
import difflib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import games  # 활성 게임 프로필(games.ACTIVE) — 게임별 HUD/색/레벨 설정
import reversal_wake  # 흰색(기상) REVERSAL 감지 (torch 불필요, cv2 템플릿)


_GR_CACHE: dict = {}


def find_game_region(video: str, samples: int = 6, thr: float = 22.0):
    """프레임 정규화: GGST 16:9 게임영역을 찾는다.
    검은 띠(레터박스/필러박스)를 제거하고, 남은 콘텐츠가 16:9가 아니면 중앙 16:9로 맞춘다.
    -> 이후 모든 좌표(비율)를 이 영역 기준으로 적용하면 비율 다른 영상도 처리됨.
    16:9 풀프레임이면 (0,0,W,H) = no-op. 반환 (x0,y0,x1,y1) 또는 None.
    같은 영상은 게임영역이 불변 -> 캐싱(이벤트마다 원거리 seek 재계산 방지, 54s/event -> ~1s)."""
    if video in _GR_CACHE:
        return _GR_CACHE[video]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        _GR_CACHE[video] = None; return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    row_max = col_max = None
    W = H = 0
    for i in range(1, samples + 1):
        if n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * i / (samples + 1)))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        rm, cm = g.mean(axis=1), g.mean(axis=0)   # 행/열 평균 밝기
        row_max = rm if row_max is None else np.maximum(row_max, rm)
        col_max = cm if col_max is None else np.maximum(col_max, cm)
    cap.release()
    if row_max is None:
        _GR_CACHE[video] = None; return None
    ys = np.where(row_max > thr)[0]   # 어느 프레임에서든 밝았던 행 = 콘텐츠(띠는 항상 어두움)
    xs = np.where(col_max > thr)[0]
    if len(ys) == 0 or len(xs) == 0:
        _GR_CACHE[video] = (0, 0, W, H); return _GR_CACHE[video]
    x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
    cw, ch, tgt = x1 - x0, y1 - y0, 16 / 9
    if cw / ch > tgt * 1.02:          # 너무 넓음(울트라와이드 등) -> 중앙 16:9 폭
        nw = int(ch * tgt); x0 += (cw - nw) // 2; x1 = x0 + nw
    elif cw / ch < tgt * 0.98:        # 너무 높음 -> 중앙 16:9 높이
        nh = int(cw / tgt); y0 += (ch - nh) // 2; y1 = y0 + nh
    _GR_CACHE[video] = (x0, y0, x1, y1); return _GR_CACHE[video]


def crop_game(frame: np.ndarray, gr):
    """게임영역(gr)으로 프레임 크롭. gr=None이면 원본."""
    if not gr:
        return frame
    x0, y0, x1, y1 = gr
    return frame[y0:y1, x0:x1]


def color_mask(bgr: np.ndarray, ranges) -> np.ndarray:
    """ColorRange 목록(OR 결합)으로 마스크 생성. 게임 프로필의 색 정의를 그대로 적용."""
    b = bgr[:, :, 0].astype(int); g = bgr[:, :, 1].astype(int); r = bgr[:, :, 2].astype(int)
    out = None
    for cr in ranges:
        m = np.ones(bgr.shape[:2], dtype=bool)
        if cr.r: m &= (r >= cr.r[0]) & (r <= cr.r[1])
        if cr.g: m &= (g >= cr.g[0]) & (g <= cr.g[1])
        if cr.b: m &= (b >= cr.b[0]) & (b <= cr.b[1])
        out = m if out is None else (out | m)
    return out if out is not None else np.zeros(bgr.shape[:2], dtype=bool)


# ---------------------------------------------------------------------------
# OCR (tesseract) — COUNTER / PUNISH / JUST 글자 구분
# ---------------------------------------------------------------------------
_ocr = None  # None=미초기화, False=불가, else=pytesseract 모듈


def _tess_cmd():
    """배포본 우선: exe 옆 tesseract/ 폴더 → 시스템 설치 경로 순으로 탐색."""
    cand = [
        _pe.app_dir() / "tesseract" / "tesseract.exe",   # 번들(배포본)
        Path(r"C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path(r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ]
    return next((str(p) for p in cand if p.exists()), None)


def _init_ocr():
    global _ocr
    if _ocr is None:
        try:
            import pytesseract
            cmd = _tess_cmd()
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
            pytesseract.get_tesseract_version()
            _ocr = pytesseract
        except Exception:
            _ocr = False
    return _ocr


def _ocr_variants(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """OCR에 넣을 전처리 후보들. 노랑색 글자가 진하면 색마스크가, 흰 글자(PUNISH)는
    그레이스케일 Otsu가 더 잘 잡혀 둘 다 시도한다."""
    b = crop_bgr[:, :, 0].astype(int); g = crop_bgr[:, :, 1].astype(int); r = crop_bgr[:, :, 2].astype(int)
    mask = ((r >= 200) & (g >= 150) & (b < 170)).astype(np.uint8) * 255
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs = []
    for m in (mask, otsu):
        outs.append(cv2.bitwise_not(cv2.resize(m, None, fx=2, fy=2)))
    return outs


def _ocr_one(crop_bgr: np.ndarray) -> tuple[str, float]:
    """크롭 1장 OCR -> (단어, 매칭비율). 속도 위해 psm 7 단일, 색마스크 먼저·Otsu는 보조."""
    pt = _init_ocr()
    if not pt or crop_bgr is None or crop_bgr.size == 0:
        return "?", 0.0
    best_word, best_ratio = "?", 0.0
    for img in _ocr_variants(crop_bgr):   # [색마스크, Otsu]
        try:
            txt = pt.image_to_string(
                img, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ).strip().upper()
        except Exception:
            continue
        txt = "".join(c for c in txt if c.isalpha())
        if txt:
            cand = max(games.ACTIVE.text_words, key=lambda w: difflib.SequenceMatcher(None, txt, w).ratio())
            ratio = difflib.SequenceMatcher(None, txt, cand).ratio()
            if ratio > best_ratio:
                best_word, best_ratio = cand, ratio
        if best_ratio >= 0.8:           # 색마스크에서 충분하면 Otsu 생략
            break
    return best_word, best_ratio


def ocr_best_word(crops: list[np.ndarray]) -> str:
    """텍스트 구간의 여러 프레임을 OCR해, 가장 잘 읽힌(최고 비율) 단어를 채택.
    텍스트가 0.5~1초 떠 있으므로 배경이 깨끗한 프레임이 섞여 인식률이 오른다."""
    if not crops:
        return "?"
    if len(crops) > 8:
        step = len(crops) // 8
        crops = crops[::step][:8]
    best_word, best_ratio = "?", 0.0
    for c in crops:
        w, ratio = _ocr_one(c)
        if ratio > best_ratio:
            best_word, best_ratio = w, ratio
        if best_ratio >= 0.85:       # 확실하면 조기 종료
            break
    # _ocr_one 은 읽은 글자를 text_words 중 '최근접'으로 강제 배정한다. 문턱이 낮으면
    # 어휘 밖 단어나 와이프 프레임의 첫 글자('P')까지 단어로 확정된다 -> 단어별 하한.
    need = {"JUST": 0.78, "REVERSAL": 0.75}.get(best_word, 0.6)
    return best_word if best_ratio >= need else "?"


# ---------------------------------------------------------------------------
# 튜닝 상수 (TUNE)
# ---------------------------------------------------------------------------
# 권장: hud_calibrate.py 로 만든 hud_config.json 의 '고정 박스'를 쓴다.
# config 가 없으면 아래 대략 영역으로 폴백(부정확).
import punish_engine as _pe  # app_dir() 재사용 (.exe 경로 처리)
CONFIG = _pe.app_dir() / "hud_config.json"
L_SEARCH = (0.04, 0.46)   # 폴백용 왼쪽 바 탐색 x범위
R_SEARCH = (0.54, 0.96)   # 폴백용 오른쪽 바 탐색 x범위
STRIP_TOP = 0.0
STRIP_BOT = 0.10

COL_COVER = 0.30          # 한 x열이 'cream'으로 인정되려면 세로 픽셀의 이 비율 이상
FRAME_STRIDE = 4          # 몇 프레임마다 한 번 샘플(60fps -> 4면 15Hz, 수집 가속·윈도우0.32s가 흡수)
RUN_MAX_CROPS = 24        # 텍스트 런 1개에서 보관할 최대 연속 프레임(15Hz -> 1.6s, 배너보다 김)
DAMAGE_DROP = 0.04        # HP%가 한 스텝에 이만큼 떨어지면 '피격' 이벤트로 본다
GAMEPLAY_MIN_HP = 0.15    # 양측 cream 폭이 최대폭 대비 이 이상이면 게임플레이로 간주
FLASH_COVER = 0.85        # 박스 위/주변까지 cream 이 이만큼 차면 '플래시 연출' 프레임 -> 스킵
# COUNTER 텍스트 영역/색/레벨 임계는 게임 프로필(games.ACTIVE)에서 가져온다.


def load_config() -> dict | None:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    return None


def cream_mask(strip_bgr: np.ndarray) -> np.ndarray:
    """체력바 fill 마스크. 색 정의는 활성 게임 프로필(health_colors)에서 가져온다.
    (GGST: 평상시 크림 + 저체력 주황. 잃은 체력/불 배경은 색 범위 밖이라 배제됨)"""
    return color_mask(strip_bgr, games.ACTIVE.health_colors)


def rect_px(reg: dict, W: int, H: int) -> tuple[int, int, int, int]:
    """비율 박스 -> 픽셀 (x0,y0,x1,y1)."""
    return (int(reg["x0"] * W), int(reg["y0"] * H),
            int(reg["x1"] * W), int(reg["y1"] * H))


def cream_cols_in_rect(frame: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    """고정 박스 안에서 'cream 열'(세로 COL_COVER 이상)의 개수 = 현재 HP 길이."""
    x0, y0, x1, y1 = rect
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        return 0
    m = cream_mask(sub)
    col_cov = m.mean(axis=0)
    return int((col_cov > COL_COVER).sum())


def yellow_text_cover(frame: np.ndarray, reg: tuple[float, float, float, float]) -> float:
    """COUNTER/PUNISH 노랑 텍스트 영역의 노랑 픽셀 비율. 텍스트가 뜨면 확 올라간다."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = int(reg[0]*W), int(reg[1]*H), int(reg[2]*W), int(reg[3]*H)
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    yellow = color_mask(sub, [games.ACTIVE.text_yellow])
    return float(yellow.mean())


def crop_region(frame: np.ndarray, reg: tuple[float, float, float, float]) -> np.ndarray:
    H, W = frame.shape[:2]
    return frame[int(reg[1]*H):int(reg[3]*H), int(reg[0]*W):int(reg[2]*W)].copy()


def split_text_run(crops: list) -> list[tuple[int, list]]:
    """배너 2개가 연달아 떠 한 run으로 묶인 경우 분리 -> [(시작오프셋, crops), ...].

    배너 사이클은 판때기(노란픽셀 피크) -> 글자 -> 디졸브. 두 배너가 붙으면 노란 곡선에
    피크-골-피크가 생기는데, 안 쪼개면 이벤트 하나가 통째로 사라진다(실측 버그).
    골이 피크의 60% 미만으로 꺼졌다가 다시 골의 1.6배 & 피크의 55% 이상으로 오르면
    두 번째 판때기로 보고 골에서 분리. 단일 배너의 글자/디졸브 구간은 재상승이 없어 안전.
    MIN_SEG=6(15Hz 기준 0.4s): 배너 1개 표시가 최소 그 이상이라, 더 짧은 분리는 과분리."""
    MIN_SEG = 6
    if len(crops) < 2 * MIN_SEG:
        return [(0, crops)]
    ys = [float(color_mask(c, [games.ACTIVE.text_yellow]).sum()) for c in crops]
    segs: list[tuple[int, list]] = []
    s = 0
    peak = ys[0]
    valley = None; v_idx = 0
    for i in range(1, len(ys)):
        y = ys[i]
        if valley is None:
            peak = max(peak, y)
            if peak > 0 and y < 0.6 * peak:
                valley, v_idx = y, i
        else:
            if y < valley:
                valley, v_idx = y, i
            elif y > 1.6 * max(valley, 1.0) and y > 0.55 * peak:
                if v_idx - s >= MIN_SEG and len(ys) - v_idx >= MIN_SEG:
                    segs.append((s, crops[s:v_idx]))
                    s = v_idx
                peak, valley = y, None
    segs.append((s, crops[s:]))
    return segs


def _read_name(frame: np.ndarray, reg, char_list: list[str]) -> tuple[str | None, float]:
    """이름판 OCR -> 캐릭 명단 퍼지매칭. 여러 임계 시도해 '가장 잘 읽힌' 결과 채택.
    (우측 이름은 우정렬이라 HP바 위에 일부 겹쳐 저대비 -> 단일 임계로는 부분만 읽힘)."""
    pt = _init_ocr()
    if not pt:
        return None, 0.0
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = int(reg[0]*W), int(reg[1]*H), int(reg[2]*W), int(reg[3]*H)
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        return None, 0.0
    gray = cv2.resize(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3)
    best_name, best_ratio = None, 0.0
    for thr in (170, 195, 150):    # 밝기 다른 배경(크림바 겹침 등) 대응
        inv = cv2.bitwise_not((gray > thr).astype(np.uint8) * 255)
        try:
            txt = pt.image_to_string(inv, config="--psm 7").strip().upper()
        except Exception:
            continue
        txt = "".join(c for c in txt if c.isalpha() or c == " ").strip()
        if len(txt) < 2:
            continue
        cand = max(char_list, key=lambda c: difflib.SequenceMatcher(None, txt, c.upper()).ratio())
        r = difflib.SequenceMatcher(None, txt, cand.upper()).ratio()
        if r > best_ratio:
            best_name, best_ratio = cand, r
    return best_name, best_ratio


def detect_mode(video: str, samples: int = 5) -> str:
    """영상이 '트레이닝'인지 '실전(match)'인지 판별. 트레이닝은 맨 왼쪽에 입력 프레임카운터
    숫자 컬럼이 있고 매치엔 없다. 여러 프레임 OCR해서 다수결. ('training'/'match'/'unknown')"""
    pt = _init_ocr()
    if not pt:
        return "unknown"
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return "unknown"
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    reg = games.ACTIVE.train_col
    gr = find_game_region(video)   # 16:9 게임영역 정규화
    train_hits = checked = 0
    for i in range(1, samples + 1):
        if n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * i / (samples + 1)))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = crop_game(frame, gr)
        checked += 1
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = int(reg[0]*W), int(reg[1]*H), int(reg[2]*W), int(reg[3]*H)
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        inv = cv2.bitwise_not(cv2.resize(m, None, fx=3, fy=3))
        try:
            txt = pt.image_to_string(inv, config="--psm 6 -c tessedit_char_whitelist=0123456789")
        except Exception:
            continue
        if sum(c.isdigit() for c in txt) >= 3:   # 숫자 컬럼 존재 = 트레이닝
            train_hits += 1
    cap.release()
    if checked == 0:
        return "unknown"
    return "training" if train_hits >= max(1, checked // 2) else "match"


def detect_characters(video: str, char_list: list[str], samples: int = 10) -> tuple[str | None, str | None]:
    """영상 여러 지점에서 좌우 이름판을 OCR해 P1/P2 캐릭터를 자동 인식.
    HP바 겹침 등으로 프레임마다 읽기 품질이 들쭉날쭉 -> 비율 가중 합산으로 '가장 잘 읽힌' 캐릭 채택.
    이름은 매치 내내 고정이라 몇 장만 봐도 됨. 못 읽으면 (None, None)."""
    if not _init_ocr() or not char_list:
        return None, None
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    prof = games.ACTIVE
    gr = find_game_region(video)   # 16:9 게임영역 정규화
    score = {"P1": {}, "P2": {}}   # 캐릭 -> 비율 합(가중 투표)
    for i in range(1, samples + 1):
        if n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * i / (samples + 1)))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = crop_game(frame, gr)
        for side, reg in [("P1", prof.name_left), ("P2", prof.name_right)]:
            name, ratio = _read_name(frame, reg, char_list)
            if name and ratio >= 0.55:        # 부분 읽기도 반영
                score[side][name] = score[side].get(name, 0.0) + ratio
    cap.release()
    def winner(side):
        v = score[side]
        return max(v, key=v.get) if v else None
    return winner("P1"), winner("P2")


def word_to_event(t: float, side: str, word: str, peak_cov: float) -> Event | None:
    """OCR 단어 + peak 채움 -> Event. 노이즈면 None."""
    res = classify_counter(peak_cov)
    if res is None:
        return None
    level, adv = res
    if word == "JUST":
        # 직전가드: 방어 이벤트(맞은 게 아니라 막은 쪽). 프레임 이득 의미 없음.
        return Event(t, side, "just_defense")
    if word == "PUNISH":
        # 펀ish(확정반격): 카운터레벨/프레임이득 개념 없음 — 반격 성립 자체가 정보.
        return Event(t, side, "punish", word=word)
    if word == "REVERSAL":
        # 주황 REVERSAL(사이드 배너) = 가드경직 중 필살기/무적기. side=지른 쪽.
        # 흰색(기상) REVERSAL 은 별도 큰 배너(reversal_wake)로 감지.
        return Event(t, side, "reversal_guard", word=word)
    kind = {"COUNTER": "counter"}.get(word, "text")
    return Event(t, side, kind, level=level, frame_adv=adv, word=word)


def flash_coverage(frame: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    """바 '바로 위' 같은 크기 영역의 cream 비율. 평소엔 어두워 0에 가깝고,
    초필살기 흰 플래시 땐 여기까지 차서 1에 가까워진다 -> 플래시 프레임 판별용."""
    x0, y0, x1, y1 = rect
    h = y1 - y0
    yy0, yy1 = max(0, y0 - h - 2), max(1, y0 - 2)
    sub = frame[yy0:yy1, x0:x1]
    if sub.size == 0:
        return 0.0
    return float(cream_mask(sub).mean())


@dataclass
class Sample:
    t: float            # 초
    frame: int
    l_cols: int
    r_cols: int
    flash: bool = False
    ctr_l_cov: float = 0.0   # 좌측 COUNTER 노랑 텍스트 영역 채움 비율
    ctr_r_cov: float = 0.0   # 우측
    b1: float | None = None  # P1 버스트 게이지 채움(0~1) — 급락+지속저하 = 버스트 사용
    b2: float | None = None  # P2


def classify_counter(peak_cov: float) -> tuple[str, int] | None:
    """텍스트 peak 채움비율 -> (레벨, 프레임이득). 노이즈면 None. 임계는 게임 프로필에서."""
    p = games.ACTIVE
    if peak_cov < p.ctr_noise:
        return None
    level = "large" if peak_cov >= p.ctr_large else "medium"
    return level, p.level_adv[level]




@dataclass
class Event:
    t: float
    side: str           # "P1"(왼쪽) / "P2"(오른쪽)
    kind: str           # hit/counter/punish/just_defense/reversal_guard(주황)/reversal_wake(흰색)/text
    hp_from: float = 0.0
    hp_to: float = 0.0
    level: str = ""     # counter/punish 전용: "medium"/"large"
    frame_adv: int = 0  # counter/punish 전용: 프레임 이득
    word: str = ""      # OCR 원문 단어(COUNTER/PUNISH 등)
    victim: str = ""    # counter/punish: HP 델타로 본 피격측(hit 이벤트 없어도 부착)
    hp_drop: float = 0.0    # victim의 측정 HP 감소(배너-확인된 히트)
    other_drop: float = 0.0  # 반대측 HP 감소(트레이드 판정용)
    fh_damage: float | None = None  # 60fps 재판독한 '첫 히트' 데미지(HP%). 15Hz 스캔은 다단히트가 뭉개짐
    fh_hp_from: float = 1.0         # 그 히트 '직전' victim HP(0~1) — Guts 역산용
    fh_risc: float | None = None    # 히트 직전 victim R.I.S.C. 채움(0~1) — 데미지 최대 2배 역산용
    burst: bool = False             # counter/punish가 '버스트로 인한 것'(±1s 내 게이지 급락) 표시
    atk_tension: float | None = None  # 이벤트 직전 공격자 텐션(0~1) — RC/각성기 콤보 추천 필터용

    def as_dict(self) -> dict:
        d = {"t": round(self.t, 3), "side": self.side, "kind": self.kind}
        if self.kind == "hit":
            d.update(hp_from=round(self.hp_from, 3), hp_to=round(self.hp_to, 3),
                     damage_pct=round(self.hp_from - self.hp_to, 3))
        elif self.kind in ("counter", "punish", "text"):
            d.update(level=self.level, frame_adv=self.frame_adv, word=self.word)
            if self.kind in ("counter", "punish"):
                d.update(victim=self.victim, hp_drop=round(self.hp_drop, 3),
                         other_drop=round(self.other_drop, 3))
                if self.burst:
                    d.update(burst=True)
                if self.atk_tension is not None:
                    d.update(atk_tension=round(self.atk_tension, 3))
                if self.fh_damage is not None:      # 60fps 정밀 재판독된 '진짜 첫 히트'
                    d.update(fh_damage=round(self.fh_damage, 4),
                             fh_hp_from=round(self.fh_hp_from, 3))
                    if self.fh_risc is not None:
                        d.update(fh_risc=round(self.fh_risc, 3))
        elif self.kind in ("reversal_wake", "reversal_guard"):
            d.update(word="REVERSAL")
        elif self.kind == "burst_missed":
            d.update(hp_drop=round(self.hp_drop, 3))
        return d


def resolve_rects(cfg: dict | None, W: int, H: int) -> tuple[tuple, tuple]:
    """config 의 고정 박스를 픽셀로. 없으면 폴백(상단 strip + L/R 탐색범위)."""
    if cfg and "regions" in cfg and "p1_hp" in cfg["regions"]:
        return (rect_px(cfg["regions"]["p1_hp"], W, H),
                rect_px(cfg["regions"]["p2_hp"], W, H))
    y0, y1 = int(STRIP_TOP * H), int(STRIP_BOT * H)
    return ((int(L_SEARCH[0] * W), y0, int(L_SEARCH[1] * W), y1),
            (int(R_SEARCH[0] * W), y0, int(R_SEARCH[1] * W), y1))


def banner_regions(cfg: dict | None) -> tuple[tuple, tuple]:
    """배너(COUNTER/PUNISH/JUST) 크롭 영역. banner_calibrate.py 가 저장한 값 우선.
    반환은 게임영역 기준 비율 (x0,y0,x1,y1) — 프로파일 기본값과 같은 형식."""
    r = (cfg or {}).get("regions", {})
    if "banner_left" in r and "banner_right" in r:
        f = lambda b: (b["x0"], b["y0"], b["x1"], b["y1"])
        return f(r["banner_left"]), f(r["banner_right"])
    return games.ACTIVE.text_left, games.ACTIVE.text_right


def wake_regions(cfg: dict | None):
    """흰색(기상) REVERSAL 크롭 영역(좌/우). 없으면 None (감지 생략)."""
    r = (cfg or {}).get("regions", {})
    if "reversal_wake_left" in r and "reversal_wake_right" in r:
        f = lambda b: (b["x0"], b["y0"], b["x1"], b["y1"])
        return f(r["reversal_wake_left"]), f(r["reversal_wake_right"])
    return None


def scan(video: str, cfg: dict | None, stride: int = FRAME_STRIDE, progress_cb=None):
    """1패스: cream 폭/플래시 기록 + 노랑 텍스트 구간을 OCR해 COUNTER/PUNISH/JUST 분류.
    progress_cb(현재프레임, 총프레임): 진행률 콜백(GUI 진행바용)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"영상 못 엶: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    samples: list[Sample] = []
    text_runs: list[dict] = []        # OCR은 스캔 후 '병렬'로 (디코딩과 분리해 속도↑)
    runs = {"P1": dict(in_run=False), "P2": dict(in_run=False)}

    def finalize(side):
        st = runs[side]
        if st["in_run"]:
            # 균등 샘플(stride)로 줄이면 시간해상도가 뭉개져 '글자 완성' 프레임을 놓친다.
            # 배너는 판때기->와이프->글자->디졸브 순이라 '연속' 프레임이 있어야 고를 수 있음.
            # 배너 2개가 붙어 한 run이 된 경우 분리(안 하면 이벤트 하나가 사라짐).
            dt = stride / fps
            for off, seg in split_text_run(st["crops"]):
                text_runs.append(dict(start_t=st["start_t"] + off * dt, side=side,
                                      peak=st["peak"], crops=seg[:RUN_MAX_CROPS]))
            st["in_run"] = False

    # --- 흰색(기상) REVERSAL 런 추적 (큰 중앙 영역, 흰 커버리지 트리거) ---
    wreg = wake_regions(cfg)
    wake_runs: list[dict] = []
    wruns = {"P1": dict(in_run=False), "P2": dict(in_run=False)}

    def finalize_wake(side):
        st = wruns[side]
        if st["in_run"]:
            wake_runs.append(dict(start_t=st["start_t"], side=side,
                                  crops=st["crops"][:RUN_MAX_CROPS]))
            st["in_run"] = False

    gr = find_game_region(video)   # 16:9 게임영역 정규화
    bl, br = banner_regions(cfg)
    fi = 0
    W = H = 0
    l_rect = r_rect = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if fi % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = crop_game(frame, gr)
            if l_rect is None:
                H, W = frame.shape[:2]
                l_rect, r_rect = resolve_rects(cfg, W, H)
            flash = (flash_coverage(frame, l_rect) > FLASH_COVER or
                     flash_coverage(frame, r_rect) > FLASH_COVER)
            samples.append(Sample(
                t=fi / fps, frame=fi,
                l_cols=cream_cols_in_rect(frame, l_rect),
                r_cols=cream_cols_in_rect(frame, r_rect),
                flash=flash,
                b1=None if flash else burst_fill(frame, "P1"),
                b2=None if flash else burst_fill(frame, "P2"),
            ))
            # --- 노랑 텍스트 런 추적 (플래시 프레임은 무시) ---
            for side, reg in [("P1", bl), ("P2", br)]:
                cov = 0.0 if flash else yellow_text_cover(frame, reg)
                st = runs[side]
                if cov >= games.ACTIVE.ctr_noise:
                    if not st["in_run"]:
                        st.update(in_run=True, start_t=fi / fps, peak=cov,
                                  crops=[crop_region(frame, reg)])
                    else:
                        st["peak"] = max(st["peak"], cov)
                        st["crops"].append(crop_region(frame, reg))
                elif st["in_run"]:
                    finalize(side)
            # --- 흰색(기상) REVERSAL 런 추적 (큰 중앙 영역) ---
            if wreg is not None:
                for side, reg in [("P1", wreg[0]), ("P2", wreg[1])]:
                    c = crop_region(frame, reg)
                    cov = reversal_wake.white_cover(c)
                    st = wruns[side]
                    if cov >= reversal_wake.WHITE_COV:
                        if not st["in_run"]:
                            st.update(in_run=True, start_t=fi / fps, crops=[c])
                        else:
                            st["crops"].append(c)
                    elif st["in_run"]:
                        finalize_wake(side)
        fi += 1
        if n_total and fi % (stride * 120) == 0:
            if progress_cb:
                progress_cb(fi, n_total)
            else:
                print(f"  ...{fi}/{n_total} 프레임 ({fi/n_total*100:.0f}%)", file=sys.stderr)
    for side in ("P1", "P2"):
        finalize(side)
        finalize_wake(side)
    cap.release()
    return samples, W, H, fps, (l_rect, r_rect), text_runs, wake_runs


def ocr_text_events(text_runs: list[dict], workers: int = 6) -> list[Event]:
    """스캔에서 모은 텍스트 구간들을 '병렬'로 분류해 Event 로 변환.
    OCR(tesseract) 우선, 미스('?')는 배너 템플릿매칭(banner_match)으로 폴백 -> 놓친 배너 회수.
    OCR 불가여도 템플릿만으로 분류 가능(배포본에서 Tesseract 없이도 배너 감지)."""
    if not text_runs:
        return []
    import banner_match
    ocr_on = bool(_init_ocr())
    tpl_on = banner_match.available()

    # NCC 폴백: OCR·banner_match 둘 다 놓친 '?'만 회수. 이미 트리거된 구간이라 진짜 배너
    # 가능성이 높음 -> 점수 게이트(0.40)로 블롭 거르고 회수. (전체 대체 아님 = 저위험)
    try:
        import banner_ncc
        _ncc_T = banner_ncc._tpl() if banner_ncc.TPL.exists() else None
    except Exception:
        banner_ncc = None; _ncc_T = None
    NCC_GATE = 0.40

    def _ncc_word(crops):
        if not _ncc_T:
            return "?"
        best_c, best_s = "?", -9.0
        for c in crops:
            lab, s, _m = banner_ncc.classify(c, _ncc_T)
            if s > best_s:
                best_c, best_s = lab, s
        return best_c if best_s >= NCC_GATE else "?"

    def work(r):
        word = ocr_best_word(r["crops"]) if ocr_on else "?"
        if word == "?" and tpl_on:                       # OCR 미스/불가 -> 템플릿 폴백
            word = banner_match.classify(r["crops"])
        if word == "?" and _ncc_T:                       # 그래도 미상 -> NCC 폴백(점수 게이트)
            word = _ncc_word(r["crops"])
        # NOTE: OCR 결과에 시각 게이트를 걸었더니 진짜 배너 대부분이 탈락(counter 27->6).
        #       OCR 자체의 블롭 오독은 남아있으나, 게이트는 라벨된 blob/text 데이터로
        #       제대로 보정하기 전까지 적용하지 않는다(과교정 방지). 폴백엔 이미 적용됨.
        return word_to_event(r["start_t"], r["side"], word, r["peak"])

    from concurrent.futures import ThreadPoolExecutor
    events: list[Event] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ev in ex.map(work, text_runs):
            if ev:
                events.append(ev)
    return events


def reversal_wake_events(wake_runs: list[dict]) -> list[Event]:
    """흰색(기상) REVERSAL 런 -> Event. side=리버설한 쪽(P1=좌영역/P2=우영역).
    템플릿 NCC 문턱(0.50, 오검 0%)을 넘는 것만 채택 -> 오정보 방지."""
    if not wake_runs or not reversal_wake.available():
        return []
    hits = sorted((r for r in wake_runs if reversal_wake.is_wake(r["crops"])),
                  key=lambda r: r["start_t"])
    evs = []
    last = {"P1": -10.0, "P2": -10.0}       # 같은 배너가 여러 런으로 쪼개진 것 병합
    for r in hits:
        if r["start_t"] - last[r["side"]] < 1.5:
            continue
        last[r["side"]] = r["start_t"]
        evs.append(Event(r["start_t"], r["side"], "reversal_wake", word="REVERSAL"))
    return evs


def build_timeline(samples: list[Sample], text_events: list[Event]) -> tuple[list[Event], int, int]:
    """2패스: 자가 보정(최대폭=100%) 후 HP% 시계열로 피격 이벤트 추출 + 텍스트 이벤트 병합."""
    if not samples:
        return list(text_events), 0, 0
    # 자가보정 최대폭은 '플래시 아닌' 프레임에서만 (플래시가 100% 기준을 오염시키지 않게)
    clean = [s for s in samples if not s.flash] or samples
    l_max = max(s.l_cols for s in clean) or 1
    r_max = max(s.r_cols for s in clean) or 1

    events: list[Event] = []
    gp_times: list[float] = []       # 게임플레이(양쪽 HP바 존재) 시각 -> 리버설 배너 오검 게이트
    bser = {"P1": [], "P2": []}      # 게임플레이 중 버스트 게이지 (t, fill)
    # --- 피격(HP 감소) 이벤트 ---
    prev_l = prev_r = None
    for s in samples:
        if s.flash:                  # 플래시 연출 프레임은 통째로 스킵 + 추적 리셋
            prev_l = prev_r = None
            continue
        l_hp = s.l_cols / l_max
        r_hp = s.r_cols / r_max
        gameplay = (l_hp > GAMEPLAY_MIN_HP) and (r_hp > GAMEPLAY_MIN_HP)
        if gameplay:
            gp_times.append(s.t)
            if s.b1 is not None:
                bser["P1"].append((s.t, s.b1))
            if s.b2 is not None:
                bser["P2"].append((s.t, s.b2))
        if gameplay and prev_l is not None:
            if prev_l - l_hp >= DAMAGE_DROP:
                events.append(Event(s.t, "P1", "hit", prev_l, l_hp))
            if prev_r - r_hp >= DAMAGE_DROP:
                events.append(Event(s.t, "P2", "hit", prev_r, r_hp))
        prev_l, prev_r = (l_hp, r_hp) if gameplay else (None, None)

    # --- 버스트 사용 감지: 게이지 급락 + '지속 저하'(3초 중앙값<0.3) ---
    # 진짜 버스트는 사용 후 게이지가 수 초간 바닥(느린 재충전). 이펙트가 게이지를 씻는
    # 프레임은 곧 원복되므로 지속 조건이 걸러낸다. 게임플레이 게이트는 bser 수집 시 적용.
    burst_uses: list[tuple[float, str]] = []
    for side, ser in bser.items():
        last = -10.0
        for i in range(4, len(ser)):
            t, f = ser[i]
            # 비교창(직전 4샘플) 전체가 시간 연속이어야 함 — 라운드 전환/암전 갭을
            # 건너뛴 비교는 급락이 아니다(갭 건너편 높은 값이 중앙값에 남는 것 방지).
            if t - ser[i - 4][0] > 0.8:
                continue
            prev = sorted(x for _, x in ser[i - 4:i])
            a = prev[len(prev) // 2]
            if a > 0.5 and f < 0.25 and t - last > 3.0:
                after = [x for tt, x in ser[i:] if tt - t <= 3.0]
                # 라운드 종료 페이드 가드: 이후 3초에 게임플레이 샘플이 계속 있어야 함
                # (15Hz면 ~45개 기대. 페이드/전환이면 뚝 끊긴다)
                if len(after) < 32:
                    continue
                sa = sorted(after)
                if sa[len(sa) // 2] < 0.3:
                    burst_uses.append((t, side))
                    last = t
    # 양쪽이 0.4s 안에 동시 급락 = 화면 단위 아티팩트(암전/전환) -> 둘 다 기각
    burst_uses = [(t, s) for t, s in burst_uses
                  if not any(s2 != s and abs(t2 - t) < 0.4 for t2, s2 in burst_uses)]
    for t, side in burst_uses:
        events.append(Event(t, side, "burst"))

    # --- 버스트 코칭: 큰 콤보(>=25% HP)를 '버스트 보유' 상태로 끝까지 맞은 순간 ---
    # 버스트는 콤보 탈출 자원 — 가득한데 치명 콤보를 그냥 맞았다면 코칭 포인트.
    hits_sorted = sorted([e for e in events if e.kind == "hit"], key=lambda x: x.t)
    cur = None
    combos_big = []
    for h in hits_sorted:
        if cur and h.side == cur["side"] and h.t - cur["last"] <= 1.0:
            cur["last"] = h.t; cur["end"] = h.hp_to
        else:
            if cur:
                combos_big.append(cur)
            cur = dict(side=h.side, start=h.t, last=h.t, s_hp=h.hp_from, end=h.hp_to)
    if cur:
        combos_big.append(cur)
    for c in combos_big:
        dmg = max(0.0, c["s_hp"] - c["end"])
        if dmg < 0.25:
            continue
        key = "b1" if c["side"] == "P1" else "b2"
        near = [getattr(s, key) for s in samples
                if abs(s.t - c["start"]) <= 0.4 and getattr(s, key) is not None]
        if not near or sorted(near)[len(near) // 2] < 0.9:
            continue                          # 버스트가 없었으면 코칭 대상 아님
        if any(bs == c["side"] and c["start"] - 0.5 <= bt <= c["last"] + 0.3
               for bt, bs in burst_uses):
            continue                          # 실제로 버스트 썼음
        ev_b = Event(c["start"], c["side"], "burst_missed")
        ev_b.hp_drop = dmg
        events.append(ev_b)

    # --- 슈퍼 플래시 런 -> 'super' 이벤트 (가드된 슈퍼 펀니쉬 놓침 분석용) ---
    fstart = None; last_super = -10.0
    for s in samples:
        if s.flash:
            if fstart is None:
                fstart = s.t
        elif fstart is not None:
            if fstart - last_super > 1.0:        # 근접 플래시 런 중복 제거
                events.append(Event(fstart, "?", "super"))
                last_super = fstart
            fstart = None

    # --- 텍스트 이벤트(COUNTER/PUNISH/JUST, scan 에서 OCR 완료)를 병합 ---
    # counter/punish는 배너가 '히트 있었음'을 보증 -> hit 이벤트가 없어도(빈사<15%·경타<4%·
    # 플래시로 누락) HP 델타를 직접 측정해 victim/데미지 부착 -> 다운스트림서 스킵 안 하고 분석.
    series = [(s.t, s.l_cols / l_max, s.r_cols / r_max) for s in samples if not s.flash]

    def _max_drop(t: float, side: str) -> float:
        col = 1 if side == "P1" else 2
        win = [row[col] for row in series if -0.3 <= row[0] - t <= 0.6]
        return max((win[i - 1] - win[i] for i in range(1, len(win))), default=0.0)

    # 리버설 배너(reversal_wake)는 게임플레이 구간에서만 인정 -> 로딩/인트로 화면의
    # 흰 글자("Player 1 / UNIKA" 등)를 리버설로 오검하는 것 차단.
    import bisect
    gp_sorted = sorted(gp_times)
    def _near_gameplay(t: float, win: float = 1.2) -> bool:
        if not gp_sorted:
            return False
        i = bisect.bisect_left(gp_sorted, t)
        for j in (i - 1, i):
            if 0 <= j < len(gp_sorted) and abs(gp_sorted[j] - t) <= win:
                return True
        return False

    merged = []
    last_by = {}                         # (side, kind) -> t : 물리적 불가 중복(같은 배너 재카운트) 제거
    for e in sorted(text_events, key=lambda x: x.t):
        if e.kind in ("counter", "punish"):
            dl, dr = _max_drop(e.t, "P1"), _max_drop(e.t, "P2")
            if dl >= dr:
                e.victim, e.hp_drop, e.other_drop = "P1", dl, dr
            else:
                e.victim, e.hp_drop, e.other_drop = "P2", dr, dl
            # 버스트로 뜬 카운터/펀ish: 사용자 보고 오정보 케이스. 버스트는 데미지 0이라
            # HP 단서가 없어 side 대조 불가 -> 시간 근접(±1.2s)만으로 표시(버스트 희소).
            if any(-1.0 <= e.t - tb <= 1.2 for tb, _ in burst_uses):
                e.burst = True
        if e.kind == "reversal_wake" and not _near_gameplay(e.t):
            continue                     # 게임플레이 아님(로딩/인트로) -> 버림
        key = (e.side, e.kind)
        if e.kind != "text" and e.t - last_by.get(key, -9.0) < 1.25:
            continue                     # 배너 1개 수명(~1.1s)보다 짧은 동종 반복 = 과분리/중복(실측)
        last_by[key] = e.t
        merged.append(e)
    events += merged

    events.sort(key=lambda e: e.t)
    return events, l_max, r_max


# R.I.S.C. 게이지(버스트 게이지 아래 얇은 바). 게임영역 비율, 실측(2560x1440 DVR 기준
# 좌우 완전대칭 확인). 빈 상태=회색, 차면 핑크/마젠타, 만땅=빨강.
RISC_P1 = (0.3359, 0.1706, 0.4523, 0.1785)
RISC_P2 = (0.5477, 0.1706, 0.6637, 0.1785)

# 버스트 게이지(HP바 아래, RISC 위). 채워진 부분=라임/블루(유채색), 빈 부분=어두운 보라.
BURST_P1 = (0.3355, 0.144, 0.4969, 0.162)
BURST_P2 = (0.5031, 0.144, 0.6645, 0.162)

# 텐션 게이지(화면 하단). P1은 중앙쪽(오른끝) 앵커에서 왼쪽으로, P2는 반대로 채워짐.
TENSION_P1 = (0.100, 0.928, 0.395, 0.948)
TENSION_P2 = (0.605, 0.928, 0.900, 0.948)


def tension_fill(frame: np.ndarray, side: str) -> float | None:
    """텐션 채움(0~1). 채움=초록~노랑 유채색, 빈 부분=반투명 회색(저채도).
    'TENSION' 글자가 색 마스크를 찢으므로 평균이 아니라 앵커에서 가장 먼 채움 지점으로 잰다."""
    x0, y0, x1, y1 = TENSION_P1 if side == "P1" else TENSION_P2
    H, W = frame.shape[:2]
    sub = frame[int(y0 * H):max(int(y0 * H) + 2, int(y1 * H)), int(x0 * W):int(x1 * W)]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 85) &
         (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 140)).mean(0)
    k = max(5, len(m) // 40)
    sm = np.convolve(m, np.ones(k) / k, mode="same")
    on = np.where(sm > 0.35)[0]
    if len(on) == 0:
        return 0.0
    n = len(m)
    if side == "P1":                      # 오른끝 앵커 -> 왼쪽으로: 가장 왼쪽 채움까지
        return float((n - on[0]) / n)
    return float((on[-1] + 1) / n)        # P2: 왼끝 앵커 -> 오른쪽으로


def burst_fill(frame: np.ndarray, side: str) -> float | None:
    """버스트 게이지 채움(0~1). 흰 'BURST' 글자는 저채도라 제외, 보라(빈)는 H>=125라 제외."""
    x0, y0, x1, y1 = BURST_P1 if side == "P1" else BURST_P2
    H, W = frame.shape[:2]
    sub = frame[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    filled = (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 110)
    return float(filled.mean())


def risc_fill(frame: np.ndarray, side: str) -> float | None:
    """victim 측 R.I.S.C. 채움 비율(0~1). 이펙트가 바를 가리면 None(판독 불가).
    데미지 배율(Dustloop): 2% 채움당 방어 1% 감소 -> 타격 데미지 x 1/(1-0.5f), 풀=2배."""
    x0, y0, x1, y1 = RISC_P1 if side == "P1" else RISC_P2
    H, W = frame.shape[:2]
    sub = frame[int(y0 * H):max(int(y0 * H) + 2, int(y1 * H)), int(x0 * W):int(x1 * W)]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    b, g, r = sub[:, :, 0].astype(int), sub[:, :, 1].astype(int), sub[:, :, 2].astype(int)
    pink = (r > 140) & (g < 110)                                   # 마젠타(핑크)·빨강 공통
    gray = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 110) & (hsv[:, :, 2] < 210)
    valid = pink | gray
    if valid.mean() < 0.6:                                         # 이펙트로 바가 가려짐
        return None
    return float(pink.sum() / valid.sum())


def refine_first_hits(video: str, events: list[Event], rects: tuple,
                      l_max: int, r_max: int, pre: float = 0.15, post: float = 0.7) -> int:
    """카운터/펀ish 이벤트 주변만 60fps '전 프레임' 재판독 -> 진짜 첫 히트 데미지.

    15Hz 스캔은 빠른 다단히트를 한 번의 HP 하락으로 뭉개서(측정이 실제 첫히트의 2~4배)
    데미지 기반 무브 후보가 틀어진다. HP 하락은 프레임 단위 순간 적용이므로,
    이벤트당 ~50프레임만 다시 읽으면 히트별 정확한 델타가 나온다. 반환=보정한 이벤트 수."""
    todo = [e for e in events if e.kind in ("counter", "punish") and e.victim in ("P1", "P2")]
    if not todo:
        return 0
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    gr = find_game_region(video)
    l_rect, r_rect = rects
    done = 0
    K = 5                                     # 미디언 창(히트스파크가 바를 스치는 1~2프레임 노이즈 제거)
    HOLD = 6                                  # 하락이 이만큼 지속돼야 '진짜 히트'(노이즈 리젝트)
    for e in todo:
        # --- 피해자 재판정 (60fps, 양쪽 동시 판독) ---
        # 15Hz 순간하락 비교는 카운터 연출·이펙트 가림이 양쪽 바를 눌러 오귀속했다.
        # 바 판독은 '가려져서 낮게'는 나와도 높게는 안 나옴 -> 창 내 max = 진짜 수위.
        #   victim = (before창 max) - (after창 max) 가 큰 쪽.
        # KO(콤보 사망 후 리필) 케이스: 한쪽 hp<0.07 인데 그 순간 상대는 >0.15 -> 그쪽 victim.
        # (라운드 전환은 양쪽이 같이 0으로 사라지므로 교차 조건이 걸러냄)
        bmax = {"P1": 0.0, "P2": 0.0}
        amax = {"P1": 0.0, "P2": 0.0}
        ko = {"P1": False, "P2": False}
        tens = {"P1": [], "P2": []}       # 이벤트 직전 텐션 표본(공격자 확정 후 부착)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((e.t - 1.1) * fps))
        for j in range(int(4.0 * fps)):
            ok, fr = cap.read()
            if not ok:
                break
            g = crop_game(fr, gr)
            hp = {"P1": cream_cols_in_rect(g, l_rect) / (l_max or 1),
                  "P2": cream_cols_in_rect(g, r_rect) / (r_max or 1)}
            tt = e.t - 1.1 + j / fps
            if e.t - 0.5 <= tt <= e.t - 0.05:
                for s in ("P1", "P2"):
                    tv = tension_fill(g, s)
                    if tv is not None:
                        tens[s].append(tv)
            for s in ("P1", "P2"):
                if tt <= e.t - 0.05:
                    bmax[s] = max(bmax[s], hp[s])
                elif e.t + 0.6 <= tt <= e.t + 1.8:
                    amax[s] = max(amax[s], hp[s])
                o = "P2" if s == "P1" else "P1"
                if tt >= e.t and hp[s] < 0.07 and hp[o] > 0.15:
                    ko[s] = True
        ko_sides = [s for s in ("P1", "P2") if ko[s]]
        if len(ko_sides) == 1:
            e.victim = ko_sides[0]
        else:
            d1 = bmax["P1"] - amax["P1"]
            d2 = bmax["P2"] - amax["P2"]
            if max(d1, d2) > 0.01:
                e.victim = "P1" if d1 >= d2 else "P2"
        atk_side = "P2" if e.victim == "P1" else "P1"
        if tens[atk_side]:                # 공격자 텐션(직전 0.5s 중앙값) — RC/각성기 필터용
            e.atk_tension = sorted(tens[atk_side])[len(tens[atk_side]) // 2]

        rect = l_rect if e.victim == "P1" else r_rect
        mx = l_max if e.victim == "P1" else r_max
        f0 = int((e.t - pre) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        raw = []
        riscs = []                                # 프레임별 victim RISC (판독 실패=None)
        for _ in range(int((pre + post) * fps)):
            ok, fr = cap.read()
            if not ok:
                break
            g = crop_game(fr, gr)
            if flash_coverage(g, rect) > FLASH_COVER:  # 슈퍼 플래시 프레임은 신뢰 불가
                raw.append(None)
                riscs.append(None)
                continue
            raw.append(cream_cols_in_rect(g, rect) / (mx or 1))
            riscs.append(risc_fill(g, e.victim))
        # 미디언 필터(None 무시). 원시 60fps 판독은 이펙트로 ±3%씩 출렁여 프레임델타로는 못 잰다.
        med = []
        for i in range(len(raw)):
            win = [x for x in raw[max(0, i - K // 2):i + K // 2 + 1] if x is not None]
            med.append(sorted(win)[len(win) // 2] if win else None)
        # 첫 '지속 하락' 탐색: hp_from=직전 평탄값, hp_to=하락 후 HOLD프레임 유지 확인된 값.
        for i in range(1, len(med) - HOLD):
            a, b = med[i - 1], med[i]
            if a is None or b is None or a - b < 0.006:
                continue
            after = [x for x in med[i:i + HOLD] if x is not None]
            if not after or max(after) > a - 0.005:     # 도로 올라오면 노이즈/리필
                continue
            hp_to = min(after)                          # 같은 히트의 안정값(다음 히트 전)
            if a - hp_to > 0.45:                        # 라운드 전환급 폭락은 측정 아님
                continue
            e.fh_damage = a - hp_to
            e.fh_hp_from = a
            # 히트 '직전' victim RISC: 직전 판독 성공값(이펙트로 절반이 가려지므로,
            # 최대 60프레임(1초)까지 거슬러 최근 유효값 5개의 중앙값 — RISC는 완만히 변함)
            pre_r = [x for x in riscs[max(0, i - 60):i] if x is not None][-5:]
            if pre_r:
                e.fh_risc = sorted(pre_r)[len(pre_r) // 2]
            done += 1
            break
    cap.release()
    return done


def save_debug_frames(video: str, samples: list[Sample], l_max: int, r_max: int,
                      rects: tuple, out_dir: Path, n: int = 8) -> None:
    """보정 검증: 고정 박스와 HP% 수치를 프레임에 그려 저장. 박스가 HP바에 딱 맞는지 눈으로 확인용."""
    out_dir.mkdir(parents=True, exist_ok=True)
    l_rect, r_rect = rects
    gameplay = [s for s in samples if not s.flash and s.l_cols > 0.2 * l_max and s.r_cols > 0.2 * r_max]
    if not gameplay:
        gameplay = [s for s in samples if not s.flash] or samples
    picks = gameplay[:: max(1, len(gameplay) // n)][:n]
    cap = cv2.VideoCapture(video)
    for s in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s.frame)
        ok, frame = cap.read()
        if not ok:
            continue
        # 중요: seek 로 읽은 '이 프레임'에서 HP를 다시 측정해야 라벨과 그림이 일치한다.
        # (DVR 가변프레임에선 seek 결과가 순차 스캔 프레임과 어긋날 수 있어, 저장값을 그대로 쓰면 안 됨)
        for rect, mx, col in [(l_rect, l_max, (0, 255, 0)),
                              (r_rect, r_max, (255, 0, 255))]:
            x0, y0, x1, y1 = rect
            cols = cream_cols_in_rect(frame, rect)
            cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
            hp = cols / (mx or 1)
            cv2.putText(frame, f"HP {hp*100:.0f}%", (x0, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        p = out_dir / f"debug_{s.frame:06d}.png"
        cv2.imwrite(str(p), frame)
        print(f"  debug 저장: {p.name}")
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--debug", action="store_true", help="보정 검증용 주석 프레임 저장")
    ap.add_argument("--stride", type=int, default=FRAME_STRIDE)
    ap.add_argument("--out", default="timeline.json")
    args = ap.parse_args()

    cfg = load_config()
    ocr_on = bool(_init_ocr())
    print(f"[1/2] 영상 스캔: {Path(args.video).name}  (HUD config: {'있음' if cfg else '없음->폴백'} / OCR: {'ON' if ocr_on else 'OFF'})")
    samples, W, H, fps, rects, text_runs, wake_runs = scan(args.video, cfg, args.stride)
    n_flash = sum(s.flash for s in samples)
    print(f"  샘플 {len(samples)}개 / {W}x{H} / {fps:.1f}fps / 플래시 스킵 {n_flash}개")
    print(f"  텍스트 구간 {len(text_runs)}개 OCR(병렬) 중...")
    wake_ev = reversal_wake_events(wake_runs)
    if wake_ev:
        print(f"  기상 REVERSAL {len(wake_ev)}개 감지(흰색 배너)")
    text_events = ocr_text_events(text_runs) + wake_ev

    print("[2/2] 자가 보정 + 타임라인 추출")
    events, l_max, r_max = build_timeline(samples, text_events)
    print(f"  100% 기준 폭: L={l_max}px R={r_max}px")
    n_ref = refine_first_hits(args.video, events, rects, l_max, r_max)
    print(f"  첫히트 60fps 정밀화: {n_ref}건")

    out = {
        "video": Path(args.video).name,
        "fps": fps, "size": [W, H],
        "calib": {"l_max_cols": l_max, "r_max_cols": r_max},
        "events": [e.as_dict() for e in events],
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {args.out}")

    from collections import Counter as _C
    kinds = _C(e.kind for e in events)
    print(f"  이벤트: 피격 {kinds['hit']} / COUNTER {kinds['counter']} / PUNISH {kinds['punish']} "
          f"/ JUST {kinds['just_defense']} / REVERSAL 기상{kinds['reversal_wake']}·가드경직{kinds['reversal_guard']} "
          f"/ 미상텍스트 {kinds['text']}")
    # 텍스트 이벤트 위주 미리보기
    for e in [e for e in events if e.kind != "hit"][:20]:
        d = e.as_dict()
        tag = {"counter": "카운터", "punish": "★PUNISH", "just_defense": "직전가드(JUST)",
               "reversal_wake": "기상 REVERSAL", "reversal_guard": "가드경직 REVERSAL",
               "text": "텍스트?", "super": "슈퍼플래시"}.get(e.kind, e.kind)
        extra = f" {d.get('level','').upper()} (+{d.get('frame_adv',0)}F)" if e.kind in ("counter", "punish") else ""
        print(f"   {d['t']:>7.2f}s  {d['side']} ▶ {tag}{extra}")

    if args.debug:
        dbg = Path("debug_frames")
        print(f"\n[debug] 주석 프레임 저장 -> {dbg}/")
        save_debug_frames(args.video, samples, l_max, r_max, rects, dbg)


if __name__ == "__main__":
    main()
