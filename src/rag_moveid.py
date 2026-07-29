# -*- coding: utf-8 -*-
"""rag_moveid.py — 멀티모달 RAG(DINOv2 임베딩 + 최근접검색) 무브ID 프로토타입 평가.

목적: Qwen 3B VLM(~48%, 수 GB·GPU·오프라인 불가) 대신 DINOv2(~330MB, CPU가능) 임베딩
검색으로 무브ID가 되는지 '측정'. 골드 크롭(dataset/)을 인덱스로, val.jsonl의
(char, answer, cands)를 그대로 써서 VLM과 동일 난이도(후보 제한)로 비교.

- 인덱스: dataset/{char}/{move}/*.png 를 DINOv2로 임베딩(L2정규화), (char,move) 라벨.
- 쿼리: 각 val 행의 (char, answer) 크롭 1장을 held-out(인덱스서 제외; 누수 방지).
- 판정: 같은 char & move∈cands 인 인덱스 중 코사인 최근접 k개 투표 -> 예측.
사용: python rag_moveid.py [--k 5]
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import punish_engine as pe

_MARGINS: list = []
ROOT = pe.app_dir() / "dataset"
VAL = pe.app_dir() / "train_data" / "val.jsonl"
EMB_CACHE = pe.app_dir() / "rag_emb.npz"
DINO = "dinov2_vitb14"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def _vid(path: str) -> str:
    """크롭 파일명에서 출처 영상 ID 추출(끝의 _<타임스탬프>_<side>.png 제거)."""
    return re.sub(r"_[0-9.]+_(left|right)\.png$", "", Path(path).name)


def _load_dino():
    m = torch.hub.load("facebookresearch/dinov2", DINO)
    m.eval().cuda()
    return m


def _prep(path: str) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


@torch.no_grad()
def build_index():
    """dataset/ 전 크롭 임베딩(캐시). 반환: embs(N,D) 정규화, chars[N], moves[N], paths[N]."""
    if EMB_CACHE.exists():
        d = np.load(EMB_CACHE, allow_pickle=True)
        return d["embs"], d["chars"], d["moves"], d["paths"]
    model = _load_dino()
    crops = sorted(ROOT.rglob("*.png"))
    embs, chars, moves, paths = [], [], [], []
    batch, meta = [], []
    def flush():
        if not batch:
            return
        x = torch.stack(batch).cuda()
        f = model(x).float()
        f = torch.nn.functional.normalize(f, dim=1).cpu().numpy()
        embs.append(f)
    for i, p in enumerate(crops):
        rel = p.relative_to(ROOT).parts
        if len(rel) < 2:
            continue
        try:
            batch.append(_prep(str(p)))
        except Exception:
            continue
        meta.append((rel[0], rel[1], str(p)))
        chars.append(rel[0]); moves.append(rel[1]); paths.append(str(p))
        if len(batch) == 64:
            flush(); batch = []
        if (i + 1) % 512 == 0:
            print(f"  임베딩 {i+1}/{len(crops)}", flush=True)
    flush()
    E = np.concatenate(embs, 0)
    chars, moves, paths = map(np.array, (chars, moves, paths))
    np.savez(EMB_CACHE, embs=E, chars=chars, moves=moves, paths=paths)
    print(f"인덱스 저장 {E.shape} -> {EMB_CACHE}", flush=True)
    return E, chars, moves, paths


@torch.no_grad()
def _ref_embeddings(needed):
    """needed={(char, move_raw)} 참조이미지 임베딩. 반환 (E, chars, moves)."""
    import vlm_id
    model = _load_dino()
    embs, chs, mvs = [], [], []
    for ch, mv in needed:
        p = vlm_id.ref_path(ch, mv)
        if not p:
            continue
        try:
            im = Image.open(str(p)).convert("RGB").resize((224, 224), Image.BILINEAR)
            t = ((torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0) - MEAN) / STD
            f = model(t.unsqueeze(0).cuda()).float()
            f = torch.nn.functional.normalize(f, dim=1)[0].cpu().numpy()
        except Exception:
            continue
        embs.append(f); chs.append(ch); mvs.append(mv)
    if not embs:
        return np.zeros((0, 768), np.float32), np.array([]), np.array([])
    return np.stack(embs), np.array(chs), np.array(mvs)


def evaluate(k: int = 5, use_refs: bool = False):
    E, chars, moves, paths = build_index()
    vids = np.array([_vid(p) for p in paths])
    if use_refs:                           # 참조이미지를 인덱스에 추가(전 무브 커버 보강)
        rows0 = [json.loads(l) for l in VAL.open(encoding="utf-8") if l.strip()]
        needed = {(r["char"], c) for r in rows0 for c in (r.get("cands") or [])}
        RE, rc, rm = _ref_embeddings(needed)
        E = np.concatenate([E, RE], 0)
        chars = np.concatenate([chars, rc]); moves = np.concatenate([moves, rm])
        paths = np.concatenate([paths, np.array([f"__ref__/{c}/{m}" for c, m in zip(rc, rm)])])
        vids = np.concatenate([vids, np.array(["__ref__"] * len(rc))])  # ref는 영상 아님(제외 안 됨)
    # (char,move) -> 인덱스 행들
    by_cls = defaultdict(list)
    for i in range(len(paths)):
        by_cls[(chars[i], _norm(moves[i]))].append(i)

    rows = [json.loads(l) for l in VAL.open(encoding="utf-8") if l.strip()]
    hit = n = fair_hit = fair_n = 0
    for r in rows:
        ch, ans, cands = r["char"], r["answer"], r.get("cands") or []
        key = (ch, _norm(ans))
        pool = [i for i in by_cls.get(key, []) if not paths[i].startswith("__ref__")]
        if not pool:
            continue                       # 이 클래스의 게임플레이 골드 크롭이 없음 -> 쿼리 불가
        q = pool[0]                        # 쿼리 1장(게임플레이)
        qvid = vids[q]
        n += 1
        cand_norm = {_norm(c) for c in cands}
        # 인덱스: 같은 char & move∈cands & **같은 출처영상 통째 제외**(near-dup 누수 차단)
        idx = [i for i in range(len(paths))
               if chars[i] == ch and _norm(moves[i]) in cand_norm and vids[i] != qvid]
        if not idx:
            continue
        sims = E[idx] @ E[q]
        # 후보별 점수 = 그 무브 인덱스 항목들의 상위 유사도 평균(min(k,보유수))
        by_move = defaultdict(list)
        for j, s in enumerate(sims):
            by_move[_norm(moves[idx[j]])].append(float(s))
        scores = {m: float(np.mean(sorted(v, reverse=True)[:k])) for m, v in by_move.items()}
        srt = sorted(scores.values(), reverse=True)
        margin = srt[0] - (srt[1] if len(srt) > 1 else 0.0)
        pred = max(scores, key=scores.get)
        ok = pred == _norm(ans)
        _MARGINS.append((margin, bool(ok)))
        hit += ok
        # 공정조건: 정답 클래스가 (다른 영상에) exemplar ≥1 존재 (검색이 이론상 가능)
        if any(_norm(moves[i]) == _norm(ans) for i in idx):
            fair_n += 1; fair_hit += ok
    print(f"\n=== RAG(DINOv2 k={k}) 무브ID - 같은영상 제외(정직) ===")
    print(f"전체: {hit}/{n} = {hit/max(n,1)*100:.1f}%  (VLM 채점 ~48% 대비)")
    print(f"공정(정답 exemplar 타영상 존재): {fair_hit}/{fair_n} = {fair_hit/max(fair_n,1)*100:.1f}%")
    print(f"  (exemplar 없어 구조적 미스: {n - fair_n}건 = 데이터 희소 한계)")
    # margin(1등-2등 유사도차) -> 정확도 (confidence 게이팅 보정용)
    print("\n  margin 임계별 정확도·커버리지(confidence 게이트 후보):")
    for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        sel = [ok for m, ok in _MARGINS if m >= thr]
        if sel:
            print(f"    margin>={thr:.2f}: {sum(sel)}/{len(sel)} = {sum(sel)/len(sel)*100:.0f}%  "
                  f"(커버 {len(sel)/len(_MARGINS)*100:.0f}%)")


@torch.no_grad()
def evaluate_refindex():
    """인덱스=Dustloop 참조이미지(무브당 1장, 전 무브 커버), 쿼리=인게임 골드크롭.
    크로스도메인(인게임->위키sprite) 검색이 되면 데이터희소 문제 없이 전 무브 커버 가능."""
    import vlm_id
    E, chars, moves, paths = build_index()
    by_cls = defaultdict(list)
    for i in range(len(paths)):
        by_cls[(chars[i], _norm(moves[i]))].append(i)
    model = _load_dino()
    # 참조이미지 임베딩
    ref_emb, ref_char, ref_move = [], [], []
    batch, meta = [], []
    rows = [json.loads(l) for l in VAL.open(encoding="utf-8") if l.strip()]
    need = {(r["char"], _norm(c)) for r in rows for c in (r.get("cands") or [])}
    seen = set()
    for ch, mv in need:
        # cands 원문 move 를 찾아 ref_path 로 로드
        pass
    # 각 val 캐릭의 전체 후보 무브에 대한 ref 로드 (원문 표기 필요 -> rows 순회)
    ref_cache = {}
    def emb_ref(ch, move_raw):
        key = (ch, _norm(move_raw))
        if key in ref_cache:
            return ref_cache[key]
        p = vlm_id.ref_path(ch, move_raw)
        if not p:
            ref_cache[key] = None; return None
        try:
            im = Image.open(str(p)).convert("RGB").resize((224, 224), Image.BILINEAR)
            t = ((torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0) - MEAN) / STD
            f = model(t.unsqueeze(0).cuda()).float()
            f = torch.nn.functional.normalize(f, dim=1)[0].cpu().numpy()
        except Exception:
            f = None
        ref_cache[key] = f; return f

    hit = n = 0
    for r in rows:
        ch, ans, cands = r["char"], r["answer"], r.get("cands") or []
        pool = by_cls.get((ch, _norm(ans)), [])
        if not pool:
            continue
        q = E[pool[0]]
        # 후보 ref 임베딩(무브당 1장)
        cand_refs = [(c, emb_ref(ch, c)) for c in cands]
        cand_refs = [(c, f) for c, f in cand_refs if f is not None]
        if not cand_refs:
            continue
        n += 1
        pred = max(cand_refs, key=lambda cf: float(cf[1] @ q))[0]
        hit += (_norm(pred) == _norm(ans))
    print(f"\n=== RAG 크로스도메인(인덱스=참조이미지) 무브ID ===")
    print(f"전체: {hit}/{n} = {hit/max(n,1)*100:.1f}%  (VLM ~48% 대비, 전 무브 커버)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--refindex", action="store_true", help="인덱스를 참조이미지로")
    ap.add_argument("--refs", action="store_true", help="게임플레이 인덱스+참조이미지 결합")
    a = ap.parse_args()
    if a.refindex:
        evaluate_refindex()
    else:
        evaluate(a.k, use_refs=a.refs)
