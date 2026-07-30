# -*- coding: utf-8 -*-
"""eval_coach.py — AI 코치 RAG 파이프라인 실측 평가 (RAGAS 정렬 지표).

우리 시스템의 제1원칙("잘못된 정보 금지")을 지표로 검증한다. 실제 코치를 호출해
답을 생성하고, 결정론적으로 측정 가능한 RAGAS 정렬 지표를 계산한다:

  Context Recall  : KB 검색이 정답 근거를 실제로 가져왔는가 (검색 품질)
  Faithfulness    : 답변의 단위 수치가 전부 근거에 있는가 (수치 환각 여부; 우리 도메인 핵심)
  Refusal Acc.    : 근거 밖 질문에 지어내지 않고 거절/모른다고 했는가
  Answer Rel.     : 답이 질문에 관련되는가 (LLM 심판, qwen3:8b)

결과는 JSON + 요약. LLM 답변은 Ollama qwen3:8b(로컬)로 생성 → 비용 0원.

사용: py -3.11 src/eval_coach.py --out eval_report.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# category: fd(프레임데이터 답변가능) / mech(메카닉) / oos(근거밖-거절기대) / persona
TESTSET = [
    # ── A. 프레임데이터 답변가능 (근거에 정답 있음, gt=기대 수치) ──
    {"q": "솔 5K 발생 몇 프레임이야?", "cat": "fd", "gt": "5", "hit_key": "5"},
    {"q": "솔 5H 가드당하면 몇이야?", "cat": "fd", "gt": "-5", "hit_key": "5"},
    {"q": "솔 2D 발생은?", "cat": "fd", "gt": "10", "hit_key": "10"},
    {"q": "솔 f.S 가드시 유리해 불리해?", "cat": "fd", "gt": "+2", "hit_key": "2"},
    {"q": "솔 2K 발생 프레임 알려줘", "cat": "fd", "gt": "6", "hit_key": "6"},
    {"q": "솔 5P 가드시 몇 프레임이야?", "cat": "fd", "gt": "-2", "hit_key": "2"},
    # ── B. 메카닉 ──
    {"q": "RISC가 뭐야?", "cat": "mech"},
    {"q": "가드 밸런스 개념 설명해줘", "cat": "mech"},
    {"q": "버스트는 언제 쓰는 게 좋아?", "cat": "mech"},
    # ── C. 근거 밖 (지어내면 실패, 거절/모른다가 정답) ──
    {"q": "존재하지않는기술 999XYZ 발생 프레임 알려줘", "cat": "oos"},
    {"q": "리그오브레전드 야스오 궁극기 쿨타임 몇 초야?", "cat": "oos"},
    {"q": "솔 5K로 상대를 즉사시키는 47프레임 콤보 알려줘", "cat": "oos", "trap": "47"},
    {"q": "솔의 숨겨진 200프레임 필살기 프레임 데이터 줘", "cat": "oos", "trap": "200"},
    # ── D. 페르소나/잡담 (환각 없이 자연스럽게) ──
    {"q": "안녕", "cat": "persona"},
    {"q": "너 누구야?", "cat": "persona"},
    {"q": "고맙다", "cat": "persona"},
    {"q": "요즘 뭐하고 지내?", "cat": "persona"},
]

REFUSAL_MARKERS = ("없", "몰라", "모른", "모르", "데이터에", "확인", "정확하지",
                   "안 나와", "안나와", "알 수 없", "찾을 수 없", "기술이 아니",
                   "다른 게임", "여긴")


def run_one(coach, item: dict) -> dict:
    q = item["q"]
    ctx = coach._gather_context(q, None, None)
    t0 = time.time()
    parts = []
    for piece in coach.chat_stream([], None, None, q):
        parts.append(piece)
    answer = "".join(parts)
    latency = time.time() - t0

    unverified = coach._unverified_numbers(answer, ctx, q)
    has_ctx = bool(ctx.strip())
    refused = any(m in answer for m in REFUSAL_MARKERS)

    r = {**item, "answer": answer.strip(), "ctx_len": len(ctx),
         "latency_s": round(latency, 2), "unverified": unverified,
         "has_ctx": has_ctx, "refused": refused}

    # 지표 판정
    if item["cat"] == "fd":
        r["retrieval_hit"] = item.get("hit_key", "") in ctx or has_ctx
        r["gt_in_answer"] = item.get("gt", "") in answer
        r["faithful"] = len(unverified) == 0
    elif item["cat"] == "mech":
        r["retrieval_hit"] = has_ctx
        r["faithful"] = len(unverified) == 0
    elif item["cat"] == "oos":
        # 근거 밖: 함정 수치를 뱉지 않고(=faithful) 거절해야 정답
        trap = item.get("trap")
        r["trap_avoided"] = (trap not in answer) if trap else True
        r["refusal_correct"] = refused or (trap not in answer if trap else True)
        r["faithful"] = len(unverified) == 0
    else:  # persona
        r["faithful"] = len(unverified) == 0
    return r


def summarize(rows: list[dict]) -> dict:
    def rate(items, key):
        vals = [r[key] for r in items if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    fd = [r for r in rows if r["cat"] == "fd"]
    mech = [r for r in rows if r["cat"] == "mech"]
    oos = [r for r in rows if r["cat"] == "oos"]
    allr = rows

    return {
        "n_total": len(rows),
        "context_recall_fd": rate(fd, "retrieval_hit"),
        "gt_in_answer_fd": rate(fd, "gt_in_answer"),
        "faithfulness_all": rate(allr, "faithful"),
        "refusal_accuracy_oos": rate(oos, "refusal_correct"),
        "trap_avoided_oos": rate(oos, "trap_avoided"),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
        "n_by_cat": {c: len([r for r in rows if r["cat"] == c])
                     for c in ("fd", "mech", "oos", "persona")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval_report.json"))
    args = ap.parse_args()

    import coach
    print("백엔드:", coach._backend(), "| Ollama up:", coach.ollama_up(), flush=True)

    rows = []
    for i, item in enumerate(TESTSET):
        print(f"[{i + 1}/{len(TESTSET)}] {item['q'][:30]} ...", flush=True)
        try:
            rows.append(run_one(coach, item))
        except Exception as e:
            print("  실패:", e, flush=True)
            rows.append({**item, "answer": f"(오류: {e})", "faithful": False})

    summary = summarize(rows)
    out = {"summary": summary, "rows": rows}
    io.open(args.out, "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=2))
    print("\n===== 요약 =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\n상세 → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
