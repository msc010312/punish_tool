"""
vlm_id.py  —  로컬 VLM(멀티모달 LLM) 기반 무브/캐릭/상황 식별

CNN 학습은 새 영상 일반화 실패(81%->~0%). 대신 이미 일반화된 오픈 VLM에게 **객관식으로** 묻는다:
  인게임 공격 프레임 + 데미지로 좁힌 후보 기술의 Dustloop 참고이미지 + 상황 컨텍스트
  -> {캐릭터, 무브, 상황(클린/카운터/트레이드), 확신도} JSON.

배포 비용 0원을 위해 **로컬 Ollama**(유저 GPU)가 기본 백엔드. Claude는 옵션(고정밀/저사양 폴백).
멀티이미지 미지원 런타임도 있어, 모든 이미지를 **라벨 붙은 한 장(시트)** 으로 합쳐 1장만 보낸다.

백엔드:  env VLM_BACKEND = "lora"(어댑터 있으면 기본) | "ollama" | "claude"
  - lora = 사람 골드라벨로 파인튜닝한 Qwen2.5-VL-3B (사람라벨 val 46% vs 7b 27% — 주력)
모델:    env VLM_MODEL   = "qwen2.5vl:7b"(ollama) / "claude-opus-4-8"(claude)
Ollama:  http://localhost:11434  (env OLLAMA_URL 로 변경)

사용: from vlm_id import identify, event_frames, ref_path, build_sheet
"""
from __future__ import annotations
import base64
import json
import os
import re as _re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "dustloop_images"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LORA_DIR = ROOT / "qwen_lora"
LORA_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"

_SCHEMA = {
    "type": "object",
    "properties": {
        "character": {"type": "string"},
        "role": {"type": "string", "enum": ["attacking", "being_hit", "unclear"]},
        "move": {"type": "string"},
        "situation": {"type": "string",
                      "enum": ["clean_hit", "counter_hit", "trade", "blocked", "whiff", "unclear"]},
        "confidence": {"type": "number"},
        "note": {"type": "string"},
    },
    "required": ["character", "role", "move", "situation", "confidence", "note"],
    "additionalProperties": False,
}

_PROMPT = (
    "Guilty Gear Strive(GGST) 대전 한 장면이다. 첨부 이미지는 분석용 시트 한 장:\n"
    "- (있으면) HUD 줄 = 게임 상단 HUD. **캐릭터 신원은 옷/머리 색이 아니라 여기 초상화·이름으로 판별해라** "
    "(인게임 색은 커스텀 컬러라 공식 이미지와 완전히 다를 수 있다. 색으로 캐릭터를 판단하지 마라)\n"
    "- 윗줄 IN-GAME = 공격 순간 프레임(시간순). 각 프레임에 두 인물이 좌/우로 보인다.\n"
    "- 아랫줄 CANDIDATES = 후보 기술의 공식 참고이미지(각 라벨=캐릭/기술). "
    "**주의: 참고이미지는 그 기술의 대표 모션 한 장면일 뿐이다.** 실제 기술은 준비→발동→회수 모션이 "
    "다 다르고 인게임 프레임은 어느 단계든 걸릴 수 있다 — 포즈를 그대로 대조하지 말고 "
    "무기·이펙트(색/모양/궤적)·타격 위치로 판단해라\n\n"
    "{ctx}\nHP 변화상 공격자는 '{atk}'로 **추정**되지만 틀릴 수 있다(트레이드·오검출). 프레임으로 직접 검증해라.\n\n"
    "참고이미지의 외형을 보고 판단해라:\n"
    "1) character = 두 인물 중 '{atk}'로 보이는 쪽을 찾아라(참고이미지 외형과 대조). "
    "두 인물 다 '{atk}'가 아니면(미러가 아닌데) 실제로 본 공격 캐릭명을 적고 move='unsure' "
    "— 절대 후보에 억지로 맞추지 마라(이게 attribution 오류 검출의 핵심)\n"
    "1-2) role = '{atk}'가 이 장면에서 **공격을 내미는 쪽(attacking)** 인지, "
    "**맞고 있는/경직당한 쪽(being_hit)** 인지 골라라. '{atk}'가 기술을 뻗는 게 아니라 "
    "히트당하거나 넘어지는 중이면 being_hit. 애매하면 unclear. "
    "being_hit/unclear면 move는 'unsure'로 둬라(공격자가 아니면 그 기술은 '{atk}' 것이 아니다)\n"
    "2) move = '{atk}'가 쓰는 기술을 후보 표기 중에서 정확히 하나. **색(컬러)은 무시하고 "
    "실루엣·포즈·무기 모양으로만** 비교해라(인게임은 색 커스텀이라 참고이미지와 색이 다를 수 있다). "
    "각 후보 라벨 아래 영문 속성이 적혀있다: low=발밑(스윕류)/mid/high=상단, 숫자F=발생프레임, "
    "far=원거리, special=특수기. 인게임에서 그 기술이 '발밑을 치는지/사거리가 먼지'를 보고 대조해라. "
    "확실하지 않으면 반드시 'unsure'. 후보에 없는 기술명을 지어내지 마라\n"
    "3) situation: clean_hit/counter_hit/trade/blocked/whiff/unclear\n"
    "4) confidence(0~1)와 note(한 줄 근거). 애매하면 confidence를 낮춰라. JSON으로만 답해라.")


# ---------- 참고이미지 ----------

def _folder(char: str) -> str:
    return char.replace(" ", "_")


def ref_path(char: str, move: str) -> Path | None:
    """캐릭+무브의 Dustloop 참고이미지 경로. 없으면 None."""
    d = IMG_DIR / _folder(char)
    if not d.is_dir():
        return None
    for name in (move, move.replace(".", ""), move.upper(), move.lower()):
        p = d / f"{name}.png"
        if p.exists():
            return p
    want = move.lower().replace(".", "")
    for p in d.glob("*.png"):
        if p.stem.lower().replace(".", "") == want:
            return p
    return None


def _load_ref(char: str, move: str, size: int = 200) -> np.ndarray | None:
    p = ref_path(char, move)
    if not p:
        return None
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3 and im.shape[2] == 4:               # 알파 -> 흰 배경
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        im = (im[:, :, :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.resize(im, (size, size))


# ---------- 시트 합성(이미지 1장) ----------

def _tile(img: np.ndarray, label: str, w: int = 200, h: int | None = None, sub: str = "") -> np.ndarray:
    """이미지 아래 라벨(+속성) 띠를 붙인 타일. w!=h면 비율 유지(인게임 400x200 왜곡 방지)."""
    h = h or w
    img = cv2.resize(img, (w, h))
    bh = 26 if not sub else 44
    bar = np.full((bh, w, 3), 30, np.uint8)
    cv2.putText(bar, label[:30], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if sub:
        cv2.putText(bar, sub[:44], (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 215, 255), 1)
    return np.vstack([img, bar])


def _row(tiles: list[np.ndarray], header: str, width: int) -> np.ndarray:
    """타일들을 한 줄로 잇고 위에 헤더 띠. 폭 부족분은 패딩."""
    body = np.hstack(tiles) if tiles else np.full((226, width, 3), 50, np.uint8)
    if body.shape[1] < width:
        body = np.hstack([body, np.full((body.shape[0], width - body.shape[1], 3), 50, np.uint8)])
    hd = np.full((24, body.shape[1], 3), 70, np.uint8)
    cv2.putText(hd, header, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return np.vstack([hd, body])


def build_sheet(ingame: list[np.ndarray], candidates: list[dict], size: int = 200,
                hud: np.ndarray | None = None) -> np.ndarray:
    """인게임 프레임 + 후보 참고이미지를 라벨 붙여 한 장으로 합친다.
    인게임은 두 인물(400x200, 2:1)이라 비율 유지해 크게(왜곡·뭉개짐 방지) — 판별 핵심.
    hud가 있으면 맨 위에 HUD 명판 줄 추가(캐릭터 신원은 색이 아니라 이걸로 판별)."""
    # 인게임 타일: 원본 비율에 맞춰 2:1(두인물) 또는 1:1(단일 크롭) — 왜곡 방지
    def _igw(im):
        return size * 2 if im.shape[1] >= 1.5 * im.shape[0] else size
    ig = [_tile(im, f"in-game {i+1}", _igw(im), size) for i, im in enumerate(ingame)]
    igw = max((_igw(im) for im in ingame), default=size * 2)
    cand = []
    for c in candidates:
        ref = _load_ref(c["char"], c["move"], size)
        if ref is None:
            ref = np.full((size, size, 3), 90, np.uint8)
            cv2.putText(ref, "no ref", (size // 4, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cand.append(_tile(ref, f"{c['move']} ({c['char'][:8]})", size, size, sub=c.get("desc", "")))
    width = max(len(ig) * igw, len(cand) * size, 1)
    rows = []
    if hud is not None:                                  # HUD 명판(초상화·캐릭명) = 신원 근거
        hh = max(48, int(hud.shape[0] * width / max(1, hud.shape[1])))
        rows.append(_row([cv2.resize(hud, (width, hh))], "HUD (identify characters HERE, not by color)", width))
    rows.append(_row(ig, "IN-GAME (time order)", width))
    for ci in range(0, len(cand), 6):                    # 후보 많으면(v2, ~11개) 6개씩 줄바꿈
        rows.append(_row(cand[ci:ci + 6], "CANDIDATES (label = move)" if ci == 0 else "CANDIDATES (cont.)", width))
    return np.vstack(rows)


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------- 인게임 프레임 추출(공격자 크롭) ----------

def event_frames(video: str, t: float, side: str | None = None, n: int = 4, span: float = 0.32):
    """t 전후 n프레임 -> 게임영역 정규화 -> **두 인물 다** 좌/우로 크롭해 한 셀(400x200)에 담는다.
    크로스업/오attribution에 안전 — VLM이 두 인물 중 공격자를 직접 찾게 한다. side는 무시(호환용).
    span 좁게(0.32s, 히트 t-0.1~+0.22): 여러 무브 걸침(멀티무브 라벨) 방지."""
    import hud_reader as hr
    import localize
    gr = hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, t0 = [], max(0.0, t - span * 0.3)
    # 4프레임이 ~0.32s 내라, 각각 seek(키프레임 재디코딩·느림) 대신 첫 프레임만 seek 후 순차 grab
    targets = [int((t0 + span * i / max(1, n - 1)) * fps) for i in range(n)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
    cur = targets[0]
    for tgt in targets:
        while cur < tgt:                             # 목표까지 디코딩만(순차, 빠름)
            cap.grab(); cur += 1
        ok, fr = cap.retrieve(); cur += 1
        if not ok:
            continue
        fr = hr.crop_game(fr, gr)
        W = fr.shape[1]
        boxes = localize.char_boxes(fr)
        # 신뢰도 상위 2개만(램리썰 대검·딜라일라 등 오검출 배제) -> 좌우 정렬
        boxes = sorted(sorted(boxes, key=lambda b: -b[4])[:2], key=lambda b: b[0] + b[2])
        if len(boxes) >= 2:
            crops = [localize.crop_box(fr, boxes[0][:4]), localize.crop_box(fr, boxes[-1][:4])]
        elif len(boxes) == 1:
            b = boxes[0]; left = (b[0] + b[2]) / 2 <= W / 2
            other = fr[:, W // 2:] if left else fr[:, :W // 2]
            cb = localize.crop_box(fr, b[:4])
            crops = [cb, other] if left else [other, cb]
        else:
            crops = [fr[:, :W // 2], fr[:, W // 2:]]
        cell = np.hstack([cv2.resize(c, (200, 200)) for c in crops])  # [좌인물 | 우인물]
        out.append(cell)
    cap.release()
    return out


def attacker_frames(video: str, t: float, attacker_side: str, n: int = 4, span: float = 0.14):
    """공격자 단일 크롭 n프레임 (RAG 쿼리용). dataset 인덱스와 동일 크롭 파이프라인:
    게임리전 -> YOLO 공격자박스(주력) -> 없으면 반쪽 폴백 -> 224². 창=t-0.10~+0.04(액티브 순간).
    NOTE: 실배포 무브ID 정확도는 ~25%(5지선다 random 20%)로 신뢰 낮음 — 확신 단정 금물."""
    import hud_reader as hr
    import localize
    gr = hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, t0 = [], max(0.0, t - 0.10)
    targets = [int((t0 + span * i / max(1, n - 1)) * fps) for i in range(n)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0]); cur = targets[0]
    for tgt in targets:
        while cur < tgt:
            cap.grab(); cur += 1
        ok, fr = cap.retrieve(); cur += 1
        if not ok:
            continue
        fr = hr.crop_game(fr, gr)
        box = localize.attacker_box(fr, attacker_side)
        if box is not None:
            crop = localize.crop_box(fr, box)
        else:                                          # 반쪽 폴백(extract_move_dataset.attacker_crop 동일)
            H, W = fr.shape[:2]
            x0, x1 = (0.04, 0.66) if attacker_side == "P1" else (0.34, 0.96)
            crop = fr[int(0.18 * H):int(0.92 * H), int(x0 * W):int(x1 * W)]
        if crop.size:
            out.append(cv2.resize(crop, (224, 224)))
    cap.release()
    return out


def hud_strip(video: str, t: float) -> np.ndarray | None:
    """t 시점 게임 상단 HUD 밴드(양측 초상화·캐릭명·닉네임 포함).
    캐릭터 신원은 옷 색(커스텀 컬러)이 아니라 이 명판으로 판별해야 한다 — 시트에 넣어 VLM에 제공."""
    import hud_reader as hr
    gr = hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    fr = hr.crop_game(fr, gr)
    return fr[: int(fr.shape[0] * 0.17)]           # 상단 17% = 양측 명판(초상화·캐릭명·HP바)


# ---------- 백엔드 ----------

def _ask_ollama(sheet_b64: str, prompt: str, model: str, schema: dict) -> dict:
    import urllib.request
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [sheet_b64]}],
        "stream": False,
        "format": schema,                       # Ollama 구조화 출력(JSON 스키마, move enum 포함)
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(OLLAMA_URL + "/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode("utf-8"))
    return json.loads(d["message"]["content"])


def _ask_claude(sheet_b64: str, prompt: str, model: str, schema: dict) -> dict:
    import anthropic
    cl = anthropic.Anthropic()
    resp = cl.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": sheet_b64}},
            {"type": "text", "text": prompt}]}],
        output_config={"format": {"type": "json_schema", "schema": schema}})
    txt = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(txt)


_LORA_CACHE: dict = {}


def _lora_model():
    """골드 파인튜닝 LoRA(Qwen2.5-VL-3B 4bit + qwen_lora) 지연 로딩(1회)."""
    if "model" not in _LORA_CACHE:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from peft import PeftModel
        proc = AutoProcessor.from_pretrained(LORA_BASE, min_pixels=256 * 256, max_pixels=512 * 512)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            LORA_BASE, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base, str(LORA_DIR))
        model.eval()
        _LORA_CACHE["model"], _LORA_CACHE["proc"] = model, proc
    return _LORA_CACHE["model"], _LORA_CACHE["proc"]


def _lora_score(img, prompt: str, cand: str) -> float:
    """target {"move":"cand"}의 평균 토큰 로그우도(=-loss). labels 마스킹으로 계산 —
    전체 vocab log_softmax를 안 만들어 메모리 안정적(반복 채점시 단편화 방지)."""
    import torch
    from qwen_vl_utils import process_vision_info
    model, proc = _lora_model()
    m = [{"role": "user", "content": [{"type": "image", "image": img},
                                      {"type": "text", "text": prompt}]},
         {"role": "assistant", "content": [{"type": "text",
          "text": json.dumps({"move": cand}, ensure_ascii=False)}]}]
    full = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
    pr = proc.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(m)
    enc = proc(text=[full], images=imgs, videos=vids, return_tensors="pt").to("cuda")
    plen = proc(text=[pr], images=imgs, videos=vids, return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100                              # assistant(정답 무브) 토큰만 채점
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss          # 평균 CE = -평균 로그우도
    return float(-loss.item())


def _ask_lora(sheet_bgr: np.ndarray, prompt: str, candidates: list[str] | None = None) -> dict:
    """LoRA 백엔드. 후보가 주어지면 **확률채점**(closed-set, 자유생성보다 정확·+3pt),
    없으면 자유생성 폴백. 반환은 표준 스키마로 래핑."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    model, proc = _lora_model()
    img = Image.fromarray(cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2RGB))
    if candidates:                                       # closed-set 채점
        scored = sorted(((_lora_score(img, prompt, c), c) for c in candidates), reverse=True)
        best_lp, mv = scored[0]
        # 신뢰: 1등-2등 로그우도 격차(margin)를 확신도로(0~1 근사)
        margin = best_lp - scored[1][0] if len(scored) > 1 else 1.0
        conf = float(min(1.0, max(0.0, margin)))
        return {"character": "", "role": "attacking", "move": mv,
                "situation": "unclear", "confidence": round(conf, 2), "note": "lora-score"}
    m = [{"role": "user", "content": [{"type": "image", "image": img},
                                      {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(m)
    enc = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=48, do_sample=False)
    gen = proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    try:
        mv = json.loads(gen).get("move", "")
    except Exception:
        mm = _re.search(r'"move"\s*:\s*"([^"]+)"', gen)
        mv = mm.group(1) if mm else gen.strip()[:16]
    return {"character": "", "role": "attacking", "move": mv,
            "situation": "unclear", "confidence": 0.0, "note": "lora"}


def is_burst(frames: list[np.ndarray], thr: float = 0.20) -> bool:
    """프레임 중 하나라도 큰 파랑(블루버스트)/금색(골드버스트) bloom이면 버스트로 판정.
    버스트는 방어기라 카운터/펀니쉬 분석·학습에서 제외해야 한다.
    임계 0.20: 버스트 per-frame 23~41% vs 일반/파란기술 ≤19% — 오검출(Dizzy 등 파란캐릭) 방지."""
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        blue = ((H > 95) & (H < 130) & (S > 120) & (V > 120)).mean()
        gold = ((H > 15) & (H < 35) & (S > 120) & (V > 160)).mean()
        if blue > thr or gold > thr:
            return True
    return False


def default_backend() -> str:
    """RAG(참조이미지 인덱스) 우선 — 경량·오프라인·정확도↑. 없으면 lora, 그다음 ollama."""
    env = os.environ.get("VLM_BACKEND")
    if env:
        return env
    if (ROOT / "rag_index.npz").exists() or IMG_DIR.is_dir():   # RAG 소스 존재
        return "rag"
    return "lora" if (LORA_DIR / "adapter_model.safetensors").exists() else "ollama"


def available(backend: str | None = None) -> bool:
    """식별 백엔드가 지금 쓸 수 있나? (lora 어댑터 있나 / ollama 서버 떠있나 / claude 키 있나)."""
    backend = backend or default_backend()
    if backend == "lora":
        try:
            import torch  # noqa: F401
            import peft  # noqa: F401
            return (LORA_DIR / "adapter_model.safetensors").exists()
        except Exception:
            return False
    if backend == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import urllib.request
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2):
            return True
    except Exception:
        return False


def _snap_move(raw: str, candidates: list[dict]) -> str:
    """모델이 'unsure'/라벨통째로 답해도 후보 기술 표기로 스냅. 미일치는 원문 유지."""
    s = (raw or "").lower().replace(".", "")
    if "unsure" in s or not s:
        return "unsure"
    moves = [c["move"] for c in candidates]
    for m in sorted(moves, key=len, reverse=True):   # 긴 표기 우선(623S가 23S보다 먼저)
        if m.lower().replace(".", "") in s:
            return m
    return raw.split("(")[0].strip() or raw.strip()  # 후보 밖이면 괄호앞 토큰만(라벨 제거)


def identify(ingame: list[np.ndarray], candidates: list[dict], context: str,
             attacker_char: str | None = None,
             backend: str | None = None, model: str | None = None,
             hud: np.ndarray | None = None) -> dict:
    """공격 장면(두 인물)을 VLM으로 식별. 반환: {character, move, situation, confidence, note}.
    attacker_char: 공격자 캐릭명(VLM이 두 인물 중 이걸 찾음). backend: 'ollama'(기본)|'claude'.
    hud: 상단 명판 크롭(초상화·캐릭명) — 색 아닌 HUD로 신원 판별하게 시트에 포함."""
    backend = backend or default_backend()
    atk = attacker_char or (candidates[0]["char"] if candidates else "?")
    sheet = build_sheet(ingame, candidates, hud=hud)
    prompt = _PROMPT.format(ctx=f"상황: {context}", atk=atk)
    # move를 후보표기로 제한(enum) — 모델이 후보 밖 기술을 지어내지 못하게
    schema = json.loads(json.dumps(_SCHEMA))
    schema["properties"]["move"]["enum"] = [c["move"] for c in candidates] + ["unsure"]
    try:
        if backend == "rag":                                    # DINOv2 임베딩 검색(경량·오프라인)
            import rag_id
            r = rag_id.identify_rag(ingame, candidates, atk)
            return r                                             # 이미 후보표기·confidence 완비
        if backend == "lora":
            r = _ask_lora(sheet, prompt, [c["move"] for c in candidates])  # 후보 확률채점
            r["character"] = atk
        elif backend == "claude":
            r = _ask_claude(_b64(sheet), prompt, model or os.environ.get("VLM_MODEL", "claude-opus-4-8"), schema)
        else:
            r = _ask_ollama(_b64(sheet), prompt, model or os.environ.get("VLM_MODEL", "qwen2.5vl:7b"), schema)
        r["move"] = _snap_move(r.get("move", ""), candidates)   # 후보 표기로 정규화
        if r.get("role") in ("being_hit", "unclear"):           # 공격자가 아니면 그 무브는 무효
            r["move"] = "unsure"
        return r
    except Exception as e:
        return {"character": "", "role": "unclear", "move": "unsure", "situation": "unclear",
                "confidence": 0.0, "note": f"VLM 오류({backend}): {e}"}


# ---------- CLI 테스트 ----------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("사용: python vlm_id.py <video> <t초> <P1|P2> <Char> [move1 move2 ...] [--sheet]")
        print('예: python vlm_id.py video.mp4 59.4 P2 "Axl Low" 2H 2D 214S 5H')
        print("    --sheet : API 호출 없이 합성 시트만 저장(시각 점검용)")
        raise SystemExit(1)
    args = [a for a in sys.argv if a != "--sheet"]
    sheet_only = "--sheet" in sys.argv
    video, t, side, char = args[1], float(args[2]), args[3], args[4]
    moves = args[5:]
    frames = event_frames(video, t, side)
    cands = [{"char": char, "move": m} for m in moves]
    if sheet_only:
        out = build_sheet(frames, cands)
        cv2.imwrite("vlm_sheet.png", out)
        print("시트 저장 -> vlm_sheet.png", out.shape)
    else:
        print(json.dumps(identify(frames, cands, f"{side} 공격 장면(t={t}s)", attacker_char=char),
                         ensure_ascii=False, indent=2))
