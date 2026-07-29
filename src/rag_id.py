# -*- coding: utf-8 -*-
"""rag_id.py — 멀티모달 RAG 무브ID 엔진 (DINOv2 임베딩 + 최근접검색).

Qwen 3B VLM 대체: 인게임 프레임을 DINOv2로 임베딩 -> 결합 인덱스(참조이미지 + 골드크롭)에서
후보별 최근접 유사도로 무브 판정. 경량(~330MB)·CPU가능·오프라인. 배포 땐 인덱스 캐시만 번들.

인덱스 구성:
  - 참조이미지(dustloop_images/): 무브당 1장, **전 무브 커버**(희소 무브도 검색 가능)
  - 골드크롭(dataset/): 실제 인게임, 있으면 정확도↑ (검증 96.9%)
판정: 후보 무브별 점수 = 그 무브 인덱스 항목들의 (쿼리 프레임 최대유사도) 상위 k평균.
      confidence = 1등-2등 점수차(margin). 배포는 새 영상이라 인덱스 누수 없음.
"""
from __future__ import annotations
import functools
import re
from pathlib import Path

import numpy as np
import cv2

import punish_engine as pe

DINO = "dinov2_vitb14"
IDX_CACHE = pe.app_dir() / "rag_index.npz"
DATASET_EMB = pe.app_dir() / "rag_emb.npz"        # dataset 크롭 임베딩(rag_moveid.py 산출)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
TOPK = 3


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def _nchar(s: str) -> str:
    """캐릭명 정규화(공백/언더바 무시) — dataset('Jam_Kuradoberi')·framedata('Jam Kuradoberi') 통일."""
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


@functools.lru_cache(maxsize=1)
def _dino():
    """DINOv2 로드. 배포본은 번들된 torch_hub/ 에서 오프라인 로드, 없으면 온라인 hub(개발)."""
    import torch
    bundled = pe.app_dir() / "torch_hub"                     # 배포: 여기에 repo+가중치 번들
    repo = bundled / "facebookresearch_dinov2_main"
    if repo.is_dir():
        torch.hub.set_dir(str(bundled))                      # 가중치도 여기서 찾음(오프라인)
        m = torch.hub.load(str(repo), DINO, source="local")
    else:
        m = torch.hub.load("facebookresearch/dinov2", DINO)  # 개발 머신(캐시/다운로드)
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev), dev


def _embed(imgs_bgr: list[np.ndarray]) -> np.ndarray:
    """BGR 이미지들 -> DINOv2 정규화 임베딩(N,768)."""
    import torch
    model, dev = _dino()
    ts = []
    for im in imgs_bgr:
        rgb = cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2RGB)
        t = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - _MEAN) / _STD
        ts.append(t)
    x = torch.from_numpy(np.stack(ts)).to(dev)
    with torch.no_grad():
        f = model(x).float()
        f = torch.nn.functional.normalize(f, dim=1)
    return f.cpu().numpy()


@functools.lru_cache(maxsize=1)
def _index():
    """결합 인덱스 로드/구축: (E[N,768] 정규화, chars[N], moves[N]). 캐시 우선."""
    if IDX_CACHE.exists():
        d = np.load(IDX_CACHE, allow_pickle=True)
        return d["embs"], d["chars"], d["moves"]
    embs, chars, moves = [], [], []
    # 1) dataset 골드크롭 임베딩(있으면 재사용)
    if DATASET_EMB.exists():
        d = np.load(DATASET_EMB, allow_pickle=True)
        embs.append(d["embs"]); chars.append(d["chars"]); moves.append(d["moves"])
    # 2) 참조이미지(전 무브 커버)
    import vlm_id
    img_dir = vlm_id.IMG_DIR
    ref_imgs, ref_ch, ref_mv = [], [], []
    for cdir in sorted(Path(img_dir).glob("*")):
        if not cdir.is_dir():
            continue
        char = cdir.name.replace("_", " ")
        for p in cdir.glob("*.png"):
            im = cv2.imread(str(p))
            if im is None:
                continue
            ref_imgs.append(im); ref_ch.append(char); ref_mv.append(p.stem)
    if ref_imgs:
        RE = np.concatenate([_embed(ref_imgs[i:i + 64]) for i in range(0, len(ref_imgs), 64)], 0)
        embs.append(RE); chars.append(np.array(ref_ch)); moves.append(np.array(ref_mv))
    E = np.concatenate(embs, 0).astype(np.float32)
    C = np.concatenate(chars); M = np.concatenate(moves)
    np.savez(IDX_CACHE, embs=E, chars=C, moves=M)
    return E, C, M


def available() -> bool:
    try:
        return bool(len(_index()[0]))
    except Exception:
        return False


def score_candidates(ingame: list[np.ndarray], char: str, cand_moves: list[str]) -> dict:
    """후보별 점수(dict move->score). 점수 = 그 무브 인덱스 항목의 (프레임 최대유사도) 상위 k평균."""
    E, chars, moves = _index()
    Q = _embed(ingame)                                 # (F,768)
    cn = {_norm(m) for m in cand_moves}
    nc = _nchar(char)
    mask = np.array([_nchar(chars[i]) == nc and _norm(moves[i]) in cn for i in range(len(moves))])
    if not mask.any():
        return {}
    sub = np.where(mask)[0]
    sims = (E[sub] @ Q.T).max(axis=1)                  # 각 인덱스항목: 프레임 최대유사도
    by_move: dict[str, list[float]] = {}
    for j, i in enumerate(sub):
        by_move.setdefault(_norm(moves[i]), []).append(float(sims[j]))
    return {m: float(np.mean(sorted(v, reverse=True)[:TOPK])) for m, v in by_move.items()}


def identify_rag(ingame: list[np.ndarray], candidates: list[dict], attacker_char: str) -> dict:
    """RAG 무브ID. 반환: {character, move, situation, confidence, note}."""
    cand_moves = [c["move"] for c in candidates]
    scores = score_candidates(ingame, attacker_char, cand_moves)
    if not scores:
        return {"character": attacker_char, "role": "attacking", "move": "unsure",
                "situation": "attack", "confidence": 0.0, "note": "RAG: 후보 참조 없음"}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    # 원문 후보표기로 복원(정규화 역매핑)
    best_norm = ranked[0][0]
    move = next((c["move"] for c in candidates if _norm(c["move"]) == best_norm), best_norm)
    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    # cosine margin -> 기존 게이팅 스케일(conf>=0.20=확정, 0.10=유력)로 보정.
    #   보정근거(val 검증): margin 0.06->73%, 0.08->79%, 0.10->80% (VLM 게이트 67% 상회)
    conf = min(1.0, max(0.0, margin / 0.30))
    return {"character": attacker_char, "role": "attacking", "move": move,
            "situation": "attack", "confidence": round(conf, 3),
            "note": f"RAG sim={ranked[0][1]:.2f} margin={margin:.3f}"}
