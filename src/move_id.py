"""
move_id.py  —  학습된 CNN으로 무브 식별 (분석 파이프라인 통합용)

이벤트에서 YOLO로 두 캐릭 크롭 -> CNN을 둘 다 돌려 '공격자 캐릭으로 예측되는 크롭'을 공격자로
선택(크로스업 박스선택 문제 회피) -> 그 무브 top-k 예측 + 데미지 후보 결합.

torch/모델/YOLO 없으면 None 반환 -> 호출측은 데미지 기반으로 폴백.
"""
from __future__ import annotations

import punish_engine as pe

_model = None
_classes = None
_tf = None
_dev = None
CKPT = pe.app_dir() / "move_cnn.pt"


def safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s).strip("_") or "x"


def _load():
    """모델 1회 로드. 실패(미설치 등)면 False."""
    global _model, _classes, _tf, _dev
    if _model is not None:
        return True
    if _model is False:
        return False
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        if not CKPT.exists():
            _model = False; return False
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(CKPT, map_location=_dev)
        _classes = ck["classes"]
        m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, len(_classes))
        m.load_state_dict(ck["state"]); m.eval().to(_dev)
        _model = m
        _tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return True
    except Exception:
        _model = False; return False


def available() -> bool:
    return _load()


def predict(crop_bgr, topk: int = 5):
    """크롭 -> [(class='Char/Move', prob), ...] top-k. 모델 없으면 []."""
    if not _load():
        return []
    import torch, cv2
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = _tf(rgb).unsqueeze(0).to(_dev)
    with torch.no_grad():
        o = torch.softmax(_model(x), 1)[0]
    p, idx = o.topk(min(topk, len(_classes)))
    return [(_classes[int(i)], float(v)) for v, i in zip(p, idx)]


def attacker_move(frame, attacker_char: str, topk: int = 3):
    """프레임에서 공격자(=attacker_char) 크롭을 CNN으로 골라 무브 top-k 반환.
    반환: [(move_name, prob), ...] (해당 캐릭 클래스 한정) / 실패시 []."""
    if not _load():
        return []
    import localize
    boxes = localize.char_boxes(frame)
    if not boxes:
        return []
    sc = safe(attacker_char)
    best, best_score = None, -1.0
    for b in boxes:
        crop = localize.crop_box(frame, b[:4])
        preds = predict(crop, topk=8)
        # 이 크롭이 공격자 캐릭일 확률(해당 캐릭 클래스 확률 합)
        score = sum(p for c, p in preds if c.split("/")[0] == sc)
        if score > best_score:
            best_score, best = score, preds
    if best is None:
        return []
    moves = [(c.split("/", 1)[1], p) for c, p in best if c.split("/")[0] == sc]
    return moves[:topk]


DTS = (-0.10, -0.06, -0.03, 0.0, 0.04, 0.08)


def event_move(cap, t: float, vfps: float, gr, attacker_char: str,
               attacker_side: str | None = None, topk: int = 3):
    """이벤트 주변 여러 프레임 샘플 -> 공격자 크롭의 무브 top-k.
    크롭 선택 = 공격자 캐릭 매칭 점수 + 공격자 측(side) 보너스.
    (미러전은 양쪽 캐릭이 같아 캐릭매칭이 모호 -> side 보너스가 공격자 크롭을 결정)."""
    if not _load():
        return []
    import cv2, hud_reader as hr, localize
    sc = safe(attacker_char)
    best, best_score = [], -1.0
    for dt in DTS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + dt) * vfps))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = hr.crop_game(fr, gr)
        W = fr.shape[1]
        for b in localize.char_boxes(fr):
            preds = predict(localize.crop_box(fr, b[:4]), topk=8)
            score = sum(p for c, p in preds if c.split("/")[0] == sc)
            if attacker_side:                         # 공격자 측 박스에 보너스
                cx = (b[0] + b[2]) / 2
                on_side = (cx < W / 2) if attacker_side == "P1" else (cx >= W / 2)
                if on_side:
                    score += 0.35
            if score > best_score:
                best_score = score
                best = [(c.split("/", 1)[1], p) for c, p in preds if c.split("/")[0] == sc][:topk]
    return best


def enrich_events(video: str, events: list, chars: dict, gr=None, progress_cb=None):
    """counter/punish 이벤트에 CNN 무브ID 부착(in-place): e['cnn_move'],['cnn_conf'],['cnn_topk'].
    모델 없으면 아무것도 안 함(데미지 폴백)."""
    if not _load():
        return
    import cv2, hud_reader as hr, analyze_match as am
    if gr is None:
        gr = hr.find_game_region(video)
    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    targets = [e for e in events if e.get("kind") in ("counter", "punish")]
    for i, e in enumerate(targets):
        atk = am.attacker_of(events, e)
        ch = chars.get(atk)
        if not ch:
            continue
        cnn = event_move(cap, e["t"], vfps, gr, ch, attacker_side=atk)
        if cnn:
            e["cnn_move"] = cnn[0][0]
            e["cnn_conf"] = round(cnn[0][1], 3)
            e["cnn_topk"] = [(m, round(p, 3)) for m, p in cnn]
        if progress_cb:
            progress_cb(i + 1, len(targets))
    cap.release()


def combine(cnn_moves, dmg_cands):
    """CNN 예측 + 데미지 후보 결합. 반환 (best_move, confidence, agree)."""
    dmg_names = {safe(m.name) for m, _ in dmg_cands} if dmg_cands else set()
    if not cnn_moves:
        return (dmg_cands[0][0].name if dmg_cands else None, 0.0, False)
    # CNN top 중 데미지에도 맞는 게 있으면 그것 우선(신뢰↑)
    for mv, p in cnn_moves:
        if mv in dmg_names:
            return (mv, p, True)
    # 없으면 CNN top-1
    return (cnn_moves[0][0], cnn_moves[0][1], False)
