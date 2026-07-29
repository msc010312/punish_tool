# -*- coding: utf-8 -*-
"""coach.py — 측정 레이어가 뽑은 '사실'을 로컬 LLM(Ollama)로 코칭 언어로 바꾼다.

설계 원칙(프로젝트 제1원칙 유지): LLM은 build_facts() 가 만든 사실 목록만 받고,
프레임데이터·기술을 스스로 지어내면 안 된다. 프롬프트가 이를 강제한다. Ollama 가
없으면 템플릿 폴백으로 그대로 동작(코칭 품질만 낮아짐, 거짓말은 안 함).

페르소나: 솔 배드가이 — 무뚝뚝·직설, 근데 실제로 도움됨. 짧고 굵게. 한국어.

  coach_stream(tl, chars, perspective, question) -> yield 텍스트 조각
  build_facts(...)  -> 사람이 읽어도 되는 사실 요약(디버그·폴백용)
"""
from __future__ import annotations
import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
COACH_MODEL = os.environ.get("COACH_MODEL", "qwen3:8b")

OTHER = {"P1": "P2", "P2": "P1"}

PERSONA = (
    "너는 길티기어 스트라이브 코치 '솔 배드가이'다. 초공격적 스트라이커, 현상금 사냥꾼. "
    "성격: 퉁명스럽고 귀찮은 걸 싫어함, 말수 적고 필요한 말만 한다, 설명보다 행동파. "
    "'노력해라' 같은 잔소리·훈계·옛날얘기는 질색이다 — 대신 뭘 하면 이기는지만 툭 던진다. "
    "겉은 난폭해도 실은 챙겨주는 타입. 반말, 짧고 굵게, 서론 빼고 핵심만. "
    "인사를 받으면 무시하지 말고 짧게 받아준다 — 퉁명≠싸가지없음. "
    "단, 욕설·비하·조롱은 절대 안 한다 — 퉁명스러운 것과 무례한 건 다르다. 플레이어를 깎아내리지 마라. "
    "길티기어든 잡담이든 자연스럽게 받되, 딱딱한 봇처럼 굴지 마라. "
    "단 코칭 내용(뭘 왜 어떻게)은 말 아낀다고 빼먹지 말고 확실히 짚어라."
)

# 말투 few-shot — 파인튜닝 없이 톤을 고정하는 예시 대화(시스템에 내장).
# 규칙 설명보다 예시가 소형 모델 톤 유도에 훨씬 효과적(실측: v1~v4 FT보다 안전).
STYLE_EXAMPLES = (
    "말투 예시(아래는 '톤' 참고용. 예시 문장을 그대로 답으로 베끼지 말고 매번 네 말로 "
    "새로 말해라. 문장은 완결형으로, 끝 마침표는 생략 — 채팅 말투):\n"
    "Q: 안녕\n"
    "A: 왔냐. 오늘은 뭘 잡아볼까\n"
    "Q: 나 요즘 실력이 안 늘어\n"
    "A: 지는 판이나 다시 봐. 뭐에 맞았는지부터 확인하고 손으로 고쳐야지\n"
    "Q: 고마워 코치\n"
    "A: 그래, 가서 이기고 와\n"
    "Q: 콤보를 자꾸 떨어뜨려\n"
    "A: 눈이 문제야. 히트 확인하고 넣는 연습부터 천천히 해봐\n"
    "Q: 솔 5K 발생 몇 프레임이야? (참고 데이터에 '발생 5' 있음)\n"
    "A: 5프레임이야. 빠르니까 견제로 써\n"
    "Q: 나 이번에 랭크 올랐다\n"
    "A: 오, 올랐네. 방심하지 말고 다음 것도 따라\n"
    "Q: 나고리유키 피 게이지 최대치 몇이야? (참고 데이터 없음)\n"
    "A: 그 수치는 내 데이터에 없어서 정확히는 몰라\n"
    "Q: 5K에서 6S가 가끔 안 이어져. 왜? (참고 데이터에 개틀링 규칙·양 기술 정보 있음)\n"
    "A: 루트 자체는 합법이야 — 5K는 커맨드노말로 개틀링 되니까. 안 이어지는 건 대부분 "
    "거리 문제다. 끝거리에서 맞히면 푸시백 때문에 6S가 닿기 전에 밀려나. 상대가 앉아 "
    "있으면 타점 때문에 헛치기도 해. 가까이 붙어 맞혔을 때만 이어봐라.\n"
    "('왜/이유' 질문엔 이렇게 참고 데이터의 근거를 구체적으로 풀어서 2~4문장으로 설명해라. "
    "뭉뚱그려 '연습해라'로 끝내지 마라.)"
)

# 자기 정체·설정 질문("너 누구야" 등)에 지어내지 않게 하는 공식 설정 근거.
SOL_LORE = (
    "[솔 배드가이 공식 설정] 본명 프레드릭. 20세기 말 미국 태생. 대학 때 '그 남자'·'아리아'와 "
    "법력학·기어 계획 연구에 몰두하다 2016년 '그 남자'의 배신으로 첫 프로토타입 기어로 개조당함. "
    "스스로 기어세포 억제장치를 만들어 인간 모습 유지, 현상금 사냥꾼 '솔 배드가이'로 활동. "
    "무기 아웃레이지 Mk.II. 취미 QUEEN 음악 감상, 소중한 것 QUEEN 'Sheer Heart Attack' 레코드. "
    "싫어하는 것: 노력, 옛날얘기. 신장 184cm, 체중 74kg."
)

# 오프닝 코칭(매치 분석 직후)용 — 사실만
RULES = (
    "규칙(엄수): 아래 [분석 데이터]에 있는 사실·숫자·기술명만 써라. "
    "거기 없는 프레임수·기술·콤보를 지어내면 안 된다. 데이터가 '추정/불확실'이라 표시하면 "
    "너도 단정하지 마라. 시각(0:12 같은)과 숫자를 근거로 인용해라. "
    "3~5개의 구체적 코칭 포인트를 골라, 각각 '무엇을/왜/어떻게'를 한두 문장으로. 한국어."
)

# 자유 대화용 — 근거 우선, 모르면 모른다(제1원칙: 잘못된 정보 금지)
RULES_CHAT = (
    "절대규칙(어기면 실패): "
    "① 프레임·데미지·가드·발생 같은 '숫자'는 아래 [참고 데이터]에 실제로 적힌 것만 말해라. "
    "참고 데이터에 어떤 기술이 '없음'이라고 표시되면 그 기술은 존재하지 않는 것이다 — "
    "있다고 전제하고 설명하지 말고 '그 기술은 없는 것 같다'고 답해라. "
    "거기 없는 숫자는 절대 만들지 마라. 추측·어림잡기도 금지. "
    "숫자를 물었는데 참고 데이터에 없으면 딱 이렇게만 답해라: '그 수치는 내 데이터에 없어서 정확히는 몰라.' "
    "이 거절 문장은 '숫자 질문'에 답할 때만 쓴다 — 전략·잡담 답변 끝에 절대 덧붙이지 마라. "
    "② 없는 출처를 지어내지 마라. '[웹]'이나 '참고 데이터에 따르면' 같은 말은 실제로 그 블록이 있을 때만 써라. "
    "③ 숫자가 아닌 전략·심리전·매치업·연습법 조언은 네 지식으로 자유롭게 해도 된다. "
    "단 조언 중에도 프레임·데미지 '숫자'를 끼워 넣지 마라(①과 동일 — 참고 데이터에 있을 때만). "
    "최신 패치·티어처럼 바뀌었을 수 있는 건 '내 정보가 오래됐을 수 있다'고 붙여라. "
    "④ 매치 [분석 데이터]가 있으면 그 사람 플레이에 맞춰 답해라. "
    "⑤ 한 답변은 최대 6문장. 한국어, 반말. 답은 완전한 문장으로 자연스럽게 말하고, "
    "한 단어짜리 마무리 추임새를 덧붙이지 마라. "
    "직전 턴에 했던 말이나 인사를 그대로 되풀이하지 마라 — 매 질문에 그 질문 내용으로 답해라. "
    "상대가 바로 알아듣게 구체적으로."
)


# ─────────────────────────────────────────── 사실 추출
def _combos(events, side):
    """side가 '가한' 콤보들 (연속 피격 묶음)."""
    hits = sorted([e for e in events if e.get("kind") == "hit" and e.get("side") == OTHER[side]],
                  key=lambda e: e["t"])
    out, cur = [], None
    for h in hits:
        if cur and h["t"] - cur["last"] <= 1.0:
            cur["last"] = h["t"]; cur["end"] = h["hp_to"]; cur["n"] += 1
        else:
            if cur:
                out.append(cur)
            cur = dict(start=h["t"], last=h["t"], s=h["hp_from"], end=h["hp_to"], n=1)
    if cur:
        out.append(cur)
    return out


def build_facts(tl: dict, chars: dict, perspective: str = "all") -> str:
    """타임라인 -> LLM에 줄 사실 요약(한국어 불릿). perspective: 'all'|'P1'|'P2'."""
    ev = tl.get("events", [])
    ms = tl.get("move_stats") or {}
    ls = tl.get("link_stats") or {}
    sides = ["P1", "P2"] if perspective == "all" else [perspective]
    L = []

    def tm(t):
        return f"{int(t)//60}:{t%60:04.1f}"

    for s in sides:
        me, opp = chars.get(s, s), chars.get(OTHER[s], OTHER[s])
        L.append(f"### {s}={me} (상대 {opp})")

        cps = [e for e in ev if e.get("kind") in ("counter", "punish")
               and (e.get("input_atk") or (OTHER[e.get("victim", "")] == s)) == s
               and not e.get("burst")]
        if cps:
            L.append("- 내가 카운터/확정반격 낸 순간:")
            for e in cps[:8]:
                mv = e.get("input_move")
                lvl = e.get("input_level", "")     # 입력 확정 무브의 counter 속성만 신뢰
                tag = f"{mv} 확정" if mv else "시동기 추정(불확실)"
                lv = f", {lvl} 카운터" if e["kind"] == "counter" and lvl and mv else ""
                L.append(f"  · {tm(e['t'])} {e['kind']} — {tag}{lv}")

        st = ms.get(s) or {}
        weak = [(k, v) for k, v in st.items() if v["n"] >= 3 and v["hit"] / v["n"] < 0.2]
        if weak:
            L.append("- 많이 눌렀는데 거의 안 맞은 기술(헛치거나 가드당함):")
            for k, v in sorted(weak, key=lambda kv: -kv[1]["n"])[:4]:
                L.append(f"  · {k}: {v['n']}회 중 {v['hit']}회 적중 ({v['hit']/v['n']*100:.0f}%)")

        lk = [r for r in (ls.get(s) or []) if (r["avg"] - r["min"]) >= 6][:3]
        if lk:
            L.append("- 연계 입력이 들쭉날쭉한 것(최속은 되는데 평균이 느림):")
            for r in lk:
                L.append(f"  · {r['a']}→{r['b']}: 평균 {r['avg']:.0f}F인데 최속 {r['min']:.0f}F "
                         f"({r['n']}회) — {r['avg']-r['min']:.0f}F 손해")

        bm = [e for e in ev if e.get("kind") == "burst_missed" and e.get("side") == s]
        for e in bm[:3]:
            L.append(f"- {tm(e['t'])} 버스트 가득인데 {e.get('hp_drop',0)*100:.0f}% 콤보를 끝까지 맞음 "
                     f"(치명 콤보엔 버스트로 끊는 것 고려)")

        # 상대가 나한테 리버설(무적기 지름) 지른 습관
        rev = [e for e in ev if e.get("kind") in ("reversal_wake", "reversal_guard")
               and e.get("side") == OTHER[s]]
        if len(rev) >= 2:
            L.append(f"- 상대가 리버설(기상/경직에 무적기)을 {len(rev)}번 질렀다 — "
                     f"내 압박에 무적기로 탈출 시도, 바이트(늦게 치기)로 처벌 가능")

        # 콤보 전환(데미지 살렸나)
        big = [c for c in _combos(ev, s) if c["s"] - c["end"] >= 0.25]
        if big:
            top = max(big, key=lambda c: c["s"] - c["end"])
            L.append(f"- 최대 콤보: {tm(top['start'])} {(top['s']-top['end'])*100:.0f}% "
                     f"({top['n']}히트)")

    if len(L) <= len(sides):
        L.append("(뚜렷한 코칭 포인트를 데이터에서 못 뽑음 — 큰 실수 없이 무난했거나 표본 부족)")
    return "\n".join(L)


# ─────────────────────────────────────────── LLM 호출
# 백엔드 우선순위: ①동봉 llama-server(배포 — 유저 설치 없음) ②Ollama(개발 편의).
# 둘 다 OpenAI 호환 /v1/chat/completions 를 지원해 호출 경로는 하나다.
def _backend() -> str | None:
    """사용 가능한 백엔드 base_url. 없으면 None."""
    try:
        import llm_server
        if llm_server.available():
            llm_server.ensure()
            if llm_server.wait_ready(timeout=180):   # 첫 호출 시 모델 로딩 대기
                return llm_server.base_url()
            return None
    except Exception:
        pass
    try:
        urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2).read()
        return OLLAMA_URL
    except Exception:
        return None


def ollama_up() -> bool:
    """(호환 유지) LLM 백엔드가 하나라도 사용 가능한가."""
    return _backend() is not None


def _stream_chat(messages: list[dict], temperature: float = 0.4):
    """OpenAI 호환 SSE 스트림 -> 텍스트 조각 yield. 예외는 호출부가 처리.
    qwen3 thinking 대응: enable_thinking=False 요청(llama-server --jinja가 반영,
    Ollama는 무시) + <think>…</think>가 content로 새면 스트림에서 걸러낸다."""
    base = _backend()
    if base is None:
        raise ConnectionError("LLM 백엔드 없음")
    payload = {"model": COACH_MODEL, "messages": messages, "stream": True,
               "temperature": temperature,
               "max_tokens": 512,                    # 폭주(반복 루프) 상한
               # 1.15는 한국어 어미(-야/-라/-져)를 반복으로 오판해 '건지/맞히라'
               # 같은 기형 활용 유발(실측) -> 약하게. 폭주는 max_tokens가 막는다.
               "repeat_penalty": 1.05,               # llama-server 확장(Ollama 무시)
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    buf, thinking = "", False
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                if buf and not thinking:
                    yield buf                        # 보류했던 꼬리 방출
                break
            d = json.loads(data.decode("utf-8"))
            piece = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if piece:
                buf += piece
                out = ""
                while buf:
                    if thinking:
                        j = buf.find("</think>")
                        if j < 0:               # 닫힘 대기(태그가 조각에 걸칠 수 있음)
                            buf = buf[-8:]
                            break
                        buf, thinking = buf[j + len("</think>"):], False
                    else:
                        i = buf.find("<think>")
                        if i < 0:
                            keep = min(len(buf), 7)   # '<think' 걸침 대비 꼬리 보류
                            out += buf[:len(buf) - keep] if len(buf) > keep else ""
                            buf = buf[len(buf) - keep:] if len(buf) > keep else buf
                            break
                        out += buf[:i]
                        buf, thinking = buf[i + len("<think>"):], True
                if out:
                    yield out


def coach_stream(tl: dict, chars: dict, perspective: str = "all",
                 question: str | None = None):
    """오프닝 코칭(매치 분석 직후) 텍스트를 조각으로 yield. Ollama 없으면 사실 폴백."""
    facts = build_facts(tl, chars, perspective)
    if not ollama_up():
        yield "(로컬 AI 코치 미연결 — 분석 사실만 표시)\n\n" + facts
        return
    q = f"\n\n[사용자 질문]\n{question}" if question else ""
    prompt = f"{RULES}\n\n[분석 데이터]\n{facts}{q}"
    try:
        yield from _stream_chat([{"role": "system",
                                  "content": PERSONA + "\n\n" + STYLE_EXAMPLES},
                                 {"role": "user", "content": prompt}])
    except Exception as e:
        yield f"(코치 응답 오류: {e})\n\n" + facts


def _gather_context(question: str, tl: dict | None, chars: dict | None) -> str:
    """질문에 대한 근거 블록 조립: 매치 사실 + 로컬 KB + (필요시)온라인."""
    blocks = []
    if tl:
        facts = build_facts(tl, chars or {}, "all")
        if facts:
            blocks.append("[분석 데이터 — 이 사람의 최근 매치]\n" + facts)
    try:
        import coach_kb
        kb = coach_kb.retrieve(question, chars)
        if kb:
            blocks.append("[참고 데이터 — 프레임/기술/메카닉]\n" + kb)
    except Exception:
        pass
    try:
        import coach_web
        web = coach_web.web_context(question, chars)
        if web:
            blocks.append("[참고 데이터 — 웹(최신)]\n" + web)
    except Exception:
        pass
    return "\n\n".join(blocks)


_NUM_CLAIM = None   # 지연 컴파일


def _unverified_numbers(answer: str, ctx: str, question: str) -> list[str]:
    """답변 속 '단위 붙은 수치 주장'(N프레임/N뎀/N데미지/NF) 중 근거·질문에 없는 것.
    기술표기(5K, 236S 등)는 단위가 아니므로 안 걸린다 — 오탐 방지를 위해
    단위 명시 주장만 검사(결정론적 사후 가드, 제1원칙 보조 장치)."""
    global _NUM_CLAIM
    import re as _re
    if _NUM_CLAIM is None:
        _NUM_CLAIM = _re.compile(r"(\d+)\s*(?:프레임|F(?![A-Za-z])|뎀|데미지)")
    allowed = set(_re.findall(r"\d+", (ctx or "") + " " + (question or "")))
    return [m.group(0) for m in _NUM_CLAIM.finditer(answer)
            if m.group(1) not in allowed]


def chat_stream(history: list[dict], tl: dict | None, chars: dict | None,
                question: str):
    """자유 대화. history=[{role,content}...] 이전 대화, question=이번 질문.
    근거(매치+로컬KB+웹)를 물려 답을 조각으로 yield. Ollama 없으면 KB 폴백.
    답변 완료 후 근거에 없는 수치 주장이 있으면 결정론적 경고를 덧붙인다."""
    ctx = _gather_context(question, tl, chars)
    if not ollama_up():
        if ctx:
            yield "(로컬 AI 미연결 — 아는 데이터만 보여준다)\n\n" + ctx
        else:
            yield ("AI 코치 모델을 못 찾았거나 로딩에 실패했다. "
                   "llama 폴더(모델 파일)가 앱 옆에 있는지 확인해라. "
                   "그 전엔 매치 분석 사실만 답할 수 있다.")
        return
    if ctx:
        user_msg = f"{question}\n\n{ctx}"
    else:
        # 근거 0건: 마커만 남긴다(거절 '문장'을 여기 쓰면 모델이 답에 인용해버림 —
        # 실측). 숫자 질문 시의 행동은 시스템 RULES_CHAT ①이 담당.
        user_msg = f"{question}\n\n[참고 데이터 없음]"
    # 같은 질문을 연달아 하면 같은 답 복제 방지 힌트(하드코딩 아님 — 생성 유도만)
    prev_qs = [m["content"] for m in history if m.get("role") == "user"]
    if prev_qs and question.strip() == prev_qs[-1].strip():
        user_msg += "\n(같은 말을 또 걸었다 — 아까와는 다른 문장으로 받아쳐라)"
    # 설정(LORE)은 정체/설정 질문에만 주입 — 프롬프트 축소로 응답 대기 단축
    idq = any(k in question for k in ("누구", "정체", "소개", "취미", "설정", "프레드릭",
                                      "몇 살", "몇살", "퀸", "queen", "아웃레이지"))
    sys_txt = (PERSONA + (("\n\n" + SOL_LORE) if idq else "")
               + "\n\n" + RULES_CHAT + "\n\n" + STYLE_EXAMPLES)
    msgs = [{"role": "system", "content": sys_txt}]
    msgs += history[-6:]        # 최근 대화만(컨텍스트 절약·속도)
    msgs.append({"role": "user", "content": user_msg})
    try:
        parts = []
        for piece in _stream_chat(msgs, temperature=0.75):  # 다양성(동일답 반복 방지)
            parts.append(piece)
            yield piece
        bad = _unverified_numbers("".join(parts), ctx, question)
        if bad:
            # LLM이 근거 밖 수치를 말함 -> 결정론적 정정(모델 신뢰 안 함, 코드가 보증)
            yield ("\n\n⚠ 위에 나온 " + ", ".join(dict.fromkeys(bad)) +
                   " 수치는 내 데이터에 없는 값이라 정확하지 않을 수 있다. "
                   "정확한 프레임은 기술명으로 다시 물어봐.")
    except Exception as e:
        yield f"(코치 응답 오류: {e})" + (("\n\n" + ctx) if ctx else "")
