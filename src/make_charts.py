# -*- coding: utf-8 -*-
"""make_charts.py — 발표 PPT용 차트 이미지(PNG) 생성. 실측 데이터 기반.

보고서(불·다크 메탈) 톤과 통일. 결과: 제출물/이미지/*.png
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "제출물" / "이미지"
OUT.mkdir(parents=True, exist_ok=True)

# 한글 폰트
fm.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 보고서 팔레트
GROUND = "#14100e"; PANEL = "#1d1713"; LINE = "#3a2c22"
EMBER = "#e8482a"; MOLTEN = "#f2a63d"; ASH = "#9b8b7d"; BONE = "#ece2d3"; JADE = "#63c39a"


def _style(ax, fig):
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=ASH, labelsize=11)
    ax.title.set_color(BONE)


def chart_ragas(summary: dict):
    """RAGAS 정렬 지표 — 0~1 점수 수평 막대."""
    items = [
        ("Context Recall\n(검색이 근거 확보)", summary["context_recall_fd"]),
        ("Faithfulness\n(수치 환각 0)", summary["faithfulness_all"]),
        ("정답수치 언급\n(GT in answer)", summary["gt_in_answer_fd"]),
        ("Refusal Acc.\n(근거밖 거절)", summary["refusal_accuracy_oos"]),
        ("Trap Avoided\n(함정수치 회피)", summary["trap_avoided_oos"]),
    ]
    labels = [a for a, _ in items][::-1]
    vals = [b for _, b in items][::-1]
    colors = [JADE if v >= 0.9 else MOLTEN if v >= 0.75 else EMBER for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    _style(ax, fig)
    y = range(len(vals))
    ax.barh(list(y), vals, height=0.6, color=colors, zorder=3)
    ax.axvline(1.0, color=LINE, lw=1.2, ls="--", zorder=1)
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", ha="left",
                color=BONE, fontsize=12, fontweight="bold")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_title("AI 코치 RAGAS 정렬 평가  (n=17, 로컬 qwen3:8b)", fontsize=14,
                 fontweight="bold", pad=14, loc="left")
    ax.grid(axis="x", color=LINE, lw=.6, zorder=0)
    fig.tight_layout()
    p = OUT / "eval_ragas.png"
    fig.savefig(p, facecolor=GROUND, bbox_inches="tight"); plt.close(fig)
    print("저장:", p)


def chart_faith_by_cat(rows: list):
    """카테고리별 문항 수 + 충실도(수치환각 0) — 누적 막대."""
    cats = ["fd", "mech", "oos", "persona"]
    names = {"fd": "프레임데이터", "mech": "메카닉", "oos": "근거밖", "persona": "페르소나"}
    faith = [sum(1 for r in rows if r["cat"] == c and r.get("faithful")) for c in cats]
    total = [sum(1 for r in rows if r["cat"] == c) for c in cats]
    fail = [t - f for t, f in zip(total, faith)]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    _style(ax, fig)
    x = range(len(cats))
    ax.bar(list(x), faith, color=JADE, label="충실(환각 0)", zorder=3, width=.6)
    ax.bar(list(x), fail, bottom=faith, color=EMBER, label="환각 발생", zorder=3, width=.6)
    for i, (f, t) in enumerate(zip(faith, total)):
        ax.text(i, t + 0.08, f"{f}/{t}", ha="center", color=BONE, fontsize=12, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([names[c] for c in cats], fontsize=11)
    ax.set_ylim(0, max(total) + 1)
    ax.set_title("카테고리별 충실도 — 단위 수치 환각 여부", fontsize=13.5,
                 fontweight="bold", pad=12, loc="left")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(BONE)
    ax.grid(axis="y", color=LINE, lw=.6, zorder=0)
    fig.tight_layout()
    p = OUT / "eval_faith_by_cat.png"
    fig.savefig(p, facecolor=GROUND, bbox_inches="tight"); plt.close(fig)
    print("저장:", p)


def chart_warmstart():
    """웜스타트 전이 — 91% 재사용 vs 9% 재학습 (수평 누적 1막대)."""
    fig, ax = plt.subplots(figsize=(9, 2.4), dpi=150)
    _style(ax, fig)
    ax.barh([0], [91], color=JADE, height=.5, zorder=3, label="언어무관 전이 (보코더·flow·style)")
    ax.barh([0], [9], left=[91], color=EMBER, height=.5, zorder=3, label="재학습 (텍스트 인코더)")
    ax.text(45.5, 0, "91%  전이", ha="center", va="center", color=GROUND, fontsize=15, fontweight="bold")
    ax.text(95.5, 0, "9%", ha="center", va="center", color=BONE, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(-.5, .5); ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_title("JP-Extra → 한국어 웜스타트  (73.2M 파라미터 중 66.6M 전이)",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    leg = ax.legend(loc="lower center", bbox_to_anchor=(.5, -.55), ncol=2, frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(BONE)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    p = OUT / "warmstart.png"
    fig.savefig(p, facecolor=GROUND, bbox_inches="tight"); plt.close(fig)
    print("저장:", p)


def chart_latency(rows: list):
    """응답 지연 분포 — 히스토그램(학습 동시 부하 하 측정)."""
    lats = [r["latency_s"] for r in rows if "latency_s" in r]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    _style(ax, fig)
    ax.hist(lats, bins=8, color=MOLTEN, edgecolor=GROUND, zorder=3)
    import statistics
    med = statistics.median(lats)
    ax.axvline(med, color=EMBER, lw=2, zorder=4)
    ax.text(med, ax.get_ylim()[1] * .92, f" 중앙값 {med:.0f}s", color=EMBER, fontsize=11, fontweight="bold")
    ax.set_xlabel("응답 생성 시간 (초)", color=ASH, fontsize=11)
    ax.set_ylabel("문항 수", color=ASH, fontsize=11)
    ax.set_title("응답 지연 분포  (※ 학습 GPU 동시 부하 하 · qwen3 사고형)",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    ax.grid(axis="y", color=LINE, lw=.6, zorder=0)
    fig.tight_layout()
    p = OUT / "latency.png"
    fig.savefig(p, facecolor=GROUND, bbox_inches="tight"); plt.close(fig)
    print("저장:", p)


def main():
    d = json.load(io.open(ROOT / "eval_report.json", encoding="utf-8"))
    chart_ragas(d["summary"])
    chart_faith_by_cat(d["rows"])
    chart_latency(d["rows"])
    chart_warmstart()
    print("\n완료 →", OUT)


if __name__ == "__main__":
    main()
