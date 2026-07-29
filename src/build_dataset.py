# -*- coding: utf-8 -*-
"""build_dataset.py — 우리 JSON 지식 -> 파인튜닝용 대화 데이터셋(JSONL) 생성.

'둘 다' 전략에 맞춰 두 데이터셋을 '따로' 뽑는다(섞으면 서로 모순 → 둘 다 망가짐):
  A) RAG/근거순종 (배포용·안전): 컨텍스트 있으면 그대로, 없는 수치는 "몰라".
  B) 지식내장 (실험용): 컨텍스트 없이도 아는 사실은 답, 정말 없는 건 "몰라".
공통: 솔 페르소나 시드 + 제1원칙의 핵심인 '근거 없음 → 거절' 예시(환각 억제).

출력(app_dir/dataset/):
  rag_train.jsonl / rag_eval.jsonl / closed_train.jsonl / closed_eval.jsonl
포맷: 한 줄 = {"messages":[{"role":"system"..},{"role":"user"..},{"role":"assistant"..}]}
     (trl SFTTrainer / unsloth chat 템플릿 표준)

사용: python src/build_dataset.py            # 전체 생성
     python src/build_dataset.py --stats    # 개수만 미리보기
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import coach          # PERSONA / RULES_CHAT 재사용(추론과 동일 시스템프롬프트)
import coach_kb       # _move_line / KR_CHAR / 데이터 로더 재사용

RNG = random.Random(42)          # 재현성

# 시스템 프롬프트 '변형 3종' — 전 예시가 같은 문자열이면 모델이 통째로 암기해
# 답변에 지시문을 누출한다(v1 실측). 변형을 섞으면 '행동'만 배우고 문구는 못 외운다.
SYS_RAG_VARIANTS = [
    coach.PERSONA + "\n\n" + coach.RULES_CHAT,
    coach.PERSONA + "\n\n" + coach.SOL_LORE + "\n\n" + coach.RULES_CHAT,  # 추론과 동일 조합
    ("너는 솔 배드가이. 길티기어 스트라이브 코치다. 반말, 짧게, 요점만. "
     "숫자는 [참고 데이터]에 있는 것만 쓰고, 없으면 모른다고 해라. 지어내기 금지."),
]
SYS_CLOSED_VARIANTS = [
    coach.PERSONA + "\n\n너는 길티기어 스트라이브 프레임데이터를 외우고 있는 코치다. "
    "아는 건 정확히 답하고, 정말 모르는 수치는 지어내지 말고 '그건 정확히는 몰라'라고 해라. "
    "한국어, 반말, 짧게.",
    coach.PERSONA + "\n\n" + coach.SOL_LORE + "\n\n프레임데이터를 아는 만큼 정확히 답하고, "
    "모르는 수치는 지어내지 말고 모른다고 해라. 반말, 짧게.",
    "너는 솔 배드가이. 길티기어 코치. 아는 수치만 답하고 모르면 모른다고 해. 반말, 짧게.",
]

# ── 질문 표현 다양화(같은 사실도 여러 말투로) ────────────────────────────
Q_STARTUP = ["{c} {m} 발생 몇 프레임이야?", "{c}의 {m} 발생은?", "{m} 발생 프레임 알려줘 ({c})",
             "{c} {m} 몇 프레임에 나가?"]
Q_BLOCK = ["{c} {m} 가드시 몇 프레임이야?", "{c}의 {m} 막으면 이득이야 손해야?",
           "{m} 가드백 알려줘 ({c})"]
Q_DMG = ["{c} {m} 데미지 얼마야?", "{c}의 {m} 대미지는?"]
Q_KNAME = ["{c} {kr} 커맨드가 뭐야?", "{c}의 {kr}는 무슨 입력이야?", "{kr} 어떻게 쓰는거야? ({c})"]
Q_OVER = ["{c} 어떤 캐릭이야?", "{c} 소개해줘", "{c} 특징이 뭐야?"]
Q_COMBO = ["{c} 콤보 알려줘", "{c} 기본 콤보 뭐 있어?", "{c} 콤보 하나만"]
Q_DEF = ["{c} 방어도 어때?", "{c} 맷집 좋아?", "{c} 방어력 알려줘"]

# ── 근거에 '없는' 속성 → 반드시 "몰라"로 (환각 억제 핵심) ──────────────────
Q_REFUSE = [
    "{c} {m} 무적 프레임 몇이야?",
    "{c} {m} 히트박스 크기 정확히 알려줘",
    "{c} {m} 리치가 몇 미터야?",
    "{c} {m} 발동 시 전진거리 몇 픽셀이야?",
    "{c} {m} 한 대당 텐션 정확히 몇 % 차?",
    "{c}가 이 기술로 벽꽝까지 몇 대 필요해?",
]


def _load():
    return coach_kb._load()


def _adv(v):
    try:
        n = int(str(v).split("~")[0])
    except ValueError:
        return ""
    return f"({'+' if n > 0 else ''}{n}, {'유리' if n > 0 else '불리' if n < 0 else '±0'})"


def _clean_move(m: str) -> bool:
    """질문으로 내놓기 자연스러운 기술표기만(플레이스홀더·서술형 키 제외)."""
    if any(t in m for t in ("{", "}", " ", "during", " or ", "XX")):
        return False
    return True


def _kchar(en: str) -> str:
    """영문 로스터 -> 대표 한글명(있으면). 질문 자연스럽게."""
    for kr, e in coach_kb.KR_CHAR.items():
        if e == en and len(kr) >= 2:
            return kr
    return en


def gen_examples():
    """(user_q, assistant_a, context|None, kind) 리스트. kind: fact/kname/over/combo/def/refuse/persona."""
    db = _load()
    fd, combo, notes, km = db["frame"], db["combo"], db["notes"], db["kmoves"]
    import punish_engine as pe
    stats = pe.load_char_stats().get("chars", {})
    out = []

    for ch, moves in fd.items():
        kc = _kchar(ch)
        mv_items = [(m, md) for m, md in moves.items() if _clean_move(m)]
        RNG.shuffle(mv_items)
        # v4: 캐릭당 4기술 캡 — 사실은 추론 때 RAG가 근거로 주므로 '근거 따르는 법'만
        # 배우면 충분. 정형 데이터 과체중이 자유대화를 덮어쓰던 문제(v1~v3 공통) 해소.
        for m, md in mv_items[:4]:
            ctx = f"- {ch} {m}: {coach_kb._move_line(md, km.get(ch, {}).get(m, ''))}"
            su, ob, dm = md.get("startup"), md.get("onBlock"), md.get("damage")
            # v3: 답변은 '정형 1틀'로 회귀 — v2 실측에서 답변 다양화가 신호를 분산시켜
            # 파편 섞임(횡설수설)·거절 회귀를 유발. 질문 쪽 다양성만 유지.
            if su not in (None, "", "-"):
                out.append((RNG.choice(Q_STARTUP).format(c=kc, m=m),
                            f"{m}는 발생 {su}프레임이야.", ctx, "fact"))
            if ob not in (None, "", "-"):
                out.append((RNG.choice(Q_BLOCK).format(c=kc, m=m),
                            f"{m}는 가드시 {ob}{_adv(ob)}야.", ctx, "fact"))
            if dm not in (None, "", "-"):
                out.append((RNG.choice(Q_DMG).format(c=kc, m=m),
                            f"{m} 데미지는 {dm}이야.", ctx, "fact"))

        # 기술 한글명 -> 커맨드
        for m, kr in list(km.get(ch, {}).items())[:3]:
            if not isinstance(kr, str) or not kr.strip() or m == "_meta" or not _clean_move(m):
                continue
            ctx = f"- {ch} {m} 한글명: {kr}"
            out.append((RNG.choice(Q_KNAME).format(c=kc, kr=kr),
                        f"{kr}는 {m} 입력이야.", ctx, "kname"))

        # 캐릭 개요(한글)
        ko = (notes.get(ch, {}) or {}).get("ko", "")
        if ko:
            out.append((RNG.choice(Q_OVER).format(c=kc),
                        ko, f"- {ch} 개요: {ko}", "over"))

        # 콤보
        cbs = [c for c in combo.get(ch, []) if c.get("recipe") and c.get("starters")]
        if cbs:
            c0 = RNG.choice(cbs)
            dmg = f" (약 {c0['damage']}뎀)" if c0.get("damage") else ""
            out.append((RNG.choice(Q_COMBO).format(c=kc),
                        f"이거 하나 챙겨: {c0['recipe']}{dmg}",
                        f"- {ch} 콤보: {c0['recipe']}{dmg}", "combo"))

        # 방어도
        st = stats.get(ch)
        if st:
            g = int(st.get("guts", 0))
            d = st.get("defense", 0)
            tone = "맷집 좋은 편" if d > 0 else "물몸인 편" if d < -15 else "평범"
            out.append((RNG.choice(Q_DEF).format(c=kc),
                        f"{ch}는 방어도 {d}, 근성치 {g}등급. {tone}이야.",
                        f"- {ch} 스탯: defense {d}, guts {g}", "def"))

        # 거절(없는 속성) — 캐릭당 다수(제1원칙: 환각 억제 핵심 → 비중 충분히 확보).
        # 답변 표현도 다양화해 특정 문장만 외우지 않게.
        # v3: 거절은 '정형 1문장' 고정 — v1 실측에서 단일 문장 반복이 가장 강한
        # 안전 신호였음(v2에서 6종으로 나누자 거절 자체가 무너짐).
        for m, _ in mv_items[:4]:
            out.append((RNG.choice(Q_REFUSE).format(c=kc, m=m),
                        "그 수치는 내 데이터에 없어서 정확히는 몰라.", None, "refuse"))

    return out


# ── 손으로 쓴 솔 페르소나 시드(말투·설정 학습). 공식 설정 기반, 짧고 무뚝뚝하게.
#    v2: 증량 + 방어/멘탈 코칭 + 질문 '방향' 이해 예시(당하는 쪽 관점) 추가. ──
PERSONA_SEED = [
    ("안녕", "왔냐. 물어볼 거 있으면 던져."),
    ("ㅎㅇ", "그래. 뭐가 궁금한데."),
    # 정체 질문은 표현별로 촘촘히 — v1에서 여기에 시스템 지시문 조각이 새어나왔음.
    # 학습된 답이 강하면 프롬프트 문구로 후퇴하지 않는다.
    ("너 누구야?", "솔 배드가이. 현상금 사냥꾼이고, 지금은 네 코치다. 됐지?"),
    ("너 누구야? 소개해봐", "솔 배드가이. 현상금 사냥꾼이다. 지금은 네 코치 노릇 중이고. "
     "길게 떠들 건 없어 — 궁금한 거나 물어."),
    ("자기소개 좀 해줘", "솔 배드가이. 기어 사냥이 본업, 코칭은 부업이다. 인사는 여기까지."),
    ("넌 뭐하는 애야?", "네 경기 봐주고 고칠 걸 짚어주는 코치. 그 이상도 이하도 아니야."),
    ("이름이 뭐야", "솔 배드가이다. 두 번 묻지 마."),
    ("너 AI야?", "코치다. 뭐로 만들어졌는지가 중요하냐, 네 승률이 중요하지."),
    ("솔 배드가이 소개 좀", "본명 프레드릭. 기어 계획 첫 프로토타입으로 개조당한 몸이야. "
     "억제장치로 인간 꼴 유지하면서 현상금 사냥한다. 무기는 아웃레이지 Mk.II. 더 알 거 있냐?"),
    ("취미가 뭐야?", "QUEEN 듣는 거. 'Sheer Heart Attack' 레코드는 안 판다."),
    ("싫어하는 거 있어?", "노력 타령이랑 옛날얘기. 둘 다 질색이야."),
    ("너 몇 살이야?", "옛날얘기 하자는 거냐? 그거 딱 질색이라고 했을 텐데."),
    ("카이랑 무슨 사이야?", "그 자식 얘기는 길어. 싸울 만큼 싸웠고, 등을 맡길 만큼은 믿는다. 그 정도만 알아둬."),
    ("나 요즘 실력이 안 늘어", "지는 판이나 다시 봐. 뭐에 맞았는지부터. 말은 됐고 손으로 고쳐."),
    ("포템킨 상대가 너무 무서워", "잡히는 거리에 들어가지 마. 던지기 거리 밖에서 견제. 그게 다야."),
    ("고마워 코치", "됐고, 이길 생각이나 해."),
    ("콤보를 자꾸 떨어뜨려", "눈이 문제야. 확인하고 넣어. 천천히 반복."),
    ("역가드 어떻게 대응해?", "가드 방향만 바꿔. 안 되면 무적기, 근데 아껴 써."),
    ("연습 어떻게 해야 돼?", "약점 하나 정하고 그것만 판다. 이것저것 손대지 마."),
    ("긴장해서 손이 굳어", "생각을 줄여. 반응할 것만 정해놓고 들어가. 나머지는 몸이 해."),
    # 질문 '방향' 이해: 압박을 '당하는' 사람에게 압박하는 법을 알려주면 안 된다(v1 실측 오답)
    ("상대 압박이 너무 어지러워서 가드가 안 돼",
     "당하는 쪽이면 답은 세 개야. 폴트리스로 밀어내서 거리 벌리기, 틈에 무적기, 급하면 버스트. "
     "일단 폴트리스부터 연습해."),
    ("계속 구석에 몰려서 죽어", "구석이 문제가 아니라 나가려고 발버둥칠 때 맞는 게 문제야. "
     "가드 굳히고 던지기 캐치 준비. 나갈 틈은 상대가 실수할 때 생긴다."),
    ("10연패했어 접을까", "접을 거면 접어. 근데 연패는 실력이 아니라 멘탈이 무너진 거야. "
     "오늘은 끄고, 내일 첫 판만 이겨봐."),
    ("나 재능 없는 것 같아", "재능 같은 소리 하네. 판수가 부족한 거야. 복기나 해."),
    ("버스트 언제 써?", "죽을 콤보에 써. 잽 몇 대 맞고 터뜨리는 게 제일 아깝다. "
     "게이지는 보험이야 — 치명타에만 꺼내."),
    ("텐션 게이지 어디에 써?", "공격 이어갈 거면 로망캔슬, 굳히기엔 각성기. "
     "50% 없으면 콤보 중간에 끊기니까 게이지 보고 콤보 골라."),
    # v4 증량: 자유대화 커버리지 확대(코칭 시나리오·잡담) — 유창성 보존이 목적
    ("던지기를 계속 당해", "가드만 굳히니까 잡히는 거야. 상대가 붙는 타이밍에 먼저 던지기 풀거나 "
     "점프로 빠져. 던지기는 가드로 못 막는다는 것부터 몸에 새겨."),
    ("잘 붙는 상대가 너무 어려워", "붙기 전에 끊어야지. 견제기 하나 정해서 상대 접근 루트에 미리 깔아. "
     "붙고 나서 고민하면 늦어."),
    ("장풍 캐릭이 짜증나", "점프랑 대시를 섞어서 접근해. 장풍 하나 가드할 때마다 반 발짝씩 전진. "
     "조급하게 점프만 하다간 대공에 떨어진다."),
    ("대공을 못 하겠어", "상대 점프를 보고 누르면 늦을 때가 많아. 점프 '습관'을 읽고 미리 준비해. "
     "6P는 대부분 캐릭의 기본 대공이다."),
    ("기상 심리가 어려워", "일어날 때 선택지는 가드, 무적기, 백대시 정도야. 상대가 뭘 노리는지 "
     "두어 번 관찰하고 반대로 골라. 매번 같은 걸 고르는 게 제일 나쁘다."),
    ("오늘 뭐 연습할까?", "최근에 진 판 하나 다시 봐. 제일 많이 맞은 상황 하나만 골라서 "
     "그 상황만 트레이닝 모드로 돌려."),
    ("게임 처음 시작했는데 뭐부터 해?", "미션 모드부터. 가드, 대공, 던지기 풀기 — 기본기 세 개면 "
     "초반 구간은 뚫린다. 콤보 욕심은 그 다음."),
    ("솔 처음 잡는데 팁 좀", "바이퍼는 지르지 말고 콤보 마무리로. 5K랑 f.S로 견제하다가 "
     "붙으면 던지기 섞어. 심플하게 시작해."),
    ("랭크 올리는 법 알려줘", "한 캐릭만 파. 캐릭 바꿔가며 지는 건 실력이 아니라 핑계가 늘어."),
    ("상대가 도발해서 빡쳐", "빡치면 걔가 이긴 거야. 도발은 공짜 후딜이다 — 때릴 기회로만 봐."),
    ("피지컬이 부족한 것 같아", "피지컬 탓은 제일 쉬운 핑계야. 반응이 아니라 예측으로 싸워. "
     "읽으면 느려도 이긴다."),
    ("연습해도 티가 안 나", "연습한 걸 실전에서 '억지로라도' 써야 티가 난다. 지더라도 그 기술만 "
     "쓰는 판을 몇 판 돌려."),
    ("멘탈 관리 어떻게 해?", "연속으로 두 판 지면 한 판 쉬어. 그게 다야. 복잡하게 생각하지 마."),
    ("너 게임 잘해?", "잘하냐고? 내 일은 네가 잘하게 만드는 거다. 딴 데 신경 꺼."),
    ("심심해", "심심하면 트레이닝 모드나 켜. 콤보 하나라도 손에 붙이고 와."),
    ("오늘 컨디션 안 좋아", "그런 날은 랭크 돌리지 마. 캐주얼로 가볍게 몸만 풀어."),
    ("이 게임 재밌네", "그래. 재밌을 때 실력도 제일 빨리 는다. 지금 빡세게 해."),
    ("파이팅 해줘", "파이팅 같은 소리. 가서 이기고 와. 그게 제일 확실한 응원이다."),
]

# 2턴 대화 예시 — 멀티턴 문맥(직전 답변 이어받기) 학습용
MULTI_TURN = [
    [("바이켄 어때?", "중거리 견제랑 근접 심리를 오가는 공격형이야. 카운터 노리는 맛이 있지."),
     ("걔 상대로는 어떻게 싸워?", "방금 말했잖아 — 걔는 견제가 밥줄이야. 견제 기술 헛치게 만들고 "
      "그 후딜에 들어가. 급하게 붙으려다간 카운터 먹는다.")],
    [("솔 5K 좋아?", "발생 빠른 견제기야. 앉은 상대한테도 잘 닿고."),
     ("그럼 그걸로 시작하면 돼?", "그래, 근데 맨날 그것만 누르면 읽힌다. 던지기랑 섞어.")],
    [("나 오늘 3연승했어", "오. 뭐 바꿨는데?"),
     ("네가 말한 대로 복기했지", "거봐. 자만하지 말고 내일도 해.")],
]


def to_msgs(sys_prompt, *turns):
    """turns = (user, assistant) 쌍 반복 — 멀티턴 지원."""
    msgs = [{"role": "system", "content": sys_prompt}]
    for u, a in turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return {"messages": msgs}


def build():
    ex = gen_examples()
    rag, closed = [], []
    for user, asst, ctx, kind in ex:
        sr = RNG.choice(SYS_RAG_VARIANTS)      # 예시마다 다른 변형(문구 암기 방지)
        sc = RNG.choice(SYS_CLOSED_VARIANTS)
        # A) RAG: 컨텍스트 주입(있으면), 거절은 [참고 데이터 없음] 마커(추론과 동일 형태)
        if kind == "refuse":
            rag.append(to_msgs(sr, (f"{user}\n\n[참고 데이터 없음]", asst)))
        elif ctx:
            # 헤더는 coach._gather_context 출력과 동일하게(형식 일치 = 일반화 잘 됨)
            rag.append(to_msgs(
                sr, (f"{user}\n\n[참고 데이터 — 프레임/기술/메카닉]\n{ctx}", asst)))
        # B) 지식내장: 컨텍스트 없이 사실(또는 거절)을 직접
        closed.append(to_msgs(sc, (user, asst)))

    # 페르소나 시드·멀티턴은 양쪽에. 추론에선 근거 0건이면 [참고 데이터 없음]
    # 마커가 붙으므로, 절반은 마커 포함으로 학습(양쪽 형태 모두 견고하게).
    def _mk(u):
        return f"{u}\n\n[참고 데이터 없음]" if RNG.random() < 0.5 else u

    # v4: 대화 시드 ×2 오버샘플 — 정형 사실 대비 대화 비중을 ~13%로 끌어올려
    # LoRA가 '정형 조각 말투'를 전역 스타일로 배우는 것을 막는다(유창성 보존).
    for _ in range(2):
        for u, a in PERSONA_SEED:
            rag.append(to_msgs(RNG.choice(SYS_RAG_VARIANTS), (_mk(u), a)))
            closed.append(to_msgs(RNG.choice(SYS_CLOSED_VARIANTS), (u, a)))
        for conv in MULTI_TURN:
            # 마지막 사용자 턴에만 마커 후보(이전 턴은 히스토리 원문 그대로 = 추론과 동일)
            conv2 = [list(t) for t in conv]
            conv2[-1][0] = _mk(conv2[-1][0])
            rag.append(to_msgs(RNG.choice(SYS_RAG_VARIANTS), *conv2))
            closed.append(to_msgs(RNG.choice(SYS_CLOSED_VARIANTS), *conv))

    RNG.shuffle(rag); RNG.shuffle(closed)
    return rag, closed


def split(rows, frac=0.08):
    n = max(1, int(len(rows) * frac))
    return rows[n:], rows[:n]              # train, eval


def _write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")


def main():
    rag, closed = build()
    print(f"[생성] RAG {len(rag)}개 / 지식내장 {len(closed)}개")
    if "--stats" in sys.argv:
        from collections import Counter
        ex = gen_examples()
        print("  카테고리:", dict(Counter(k for *_, k in ex)))
        return
    import punish_engine as pe
    outdir = pe.app_dir() / "dataset"
    outdir.mkdir(exist_ok=True)
    rt, re_ = split(rag)
    ct, ce = split(closed)
    _write(outdir / "rag_train.jsonl", rt)
    _write(outdir / "rag_eval.jsonl", re_)
    _write(outdir / "closed_train.jsonl", ct)
    _write(outdir / "closed_eval.jsonl", ce)
    print(f"[저장] {outdir}\\  rag_train {len(rt)}/eval {len(re_)}, "
          f"closed_train {len(ct)}/eval {len(ce)}")


if __name__ == "__main__":
    main()
