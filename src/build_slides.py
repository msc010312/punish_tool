# -*- coding: utf-8 -*-
"""build_slides.py — 제출물 md 내용을 PPT형 슬라이드 덱 HTML로 생성.

제품명: Punish Tool (솔 배드가이는 코치 페르소나). 불·다크 메탈 톤.
스크롤 스냅 + 키보드(←→↑↓/Space) 네비게이션. 차트/음성/스크린샷 base64 임베드.
결과: 제출물/프로젝트_최종보고서.html(음성 포함) + _공개용.html(음성 제외).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMGD = ROOT / "제출물" / "이미지"
SC = Path(r"C:/Users/msc01/AppData/Local/Temp/claude/d--punishTool"
         r"/69027c21-2e41-4ebc-bce5-d7ae82c47055/scratchpad")


def img_uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((IMGD / name).read_bytes()).decode()


def audio_uri(name: str) -> str:
    return "data:audio/mpeg;base64," + base64.b64encode((SC / name).read_bytes()).decode()


RAGAS = img_uri("eval_ragas.png")
FAITH = img_uri("eval_faith_by_cat.png")
WARM = img_uri("warmstart.png")
LAT = img_uri("latency.png")
TOGGLE = img_uri("gui_lang_toggle.png")

HEAD = """<title>Punish Tool — 발표자료</title>
<style>
  :root{
    --ground:#14100e; --ground-2:#1d1713; --panel:#241c17; --line:#3a2c22;
    --ember:#e8482a; --molten:#f2a63d; --ash:#9b8b7d; --bone:#ece2d3; --jade:#63c39a; --white:#fff;
    --disp:"Arial Narrow","Roboto Condensed",system-ui,sans-serif;
    --body:system-ui,-apple-system,"Segoe UI",sans-serif;
    --mono:ui-monospace,"Cascadia Code","Consolas",monospace;
  }
  *{box-sizing:border-box}
  html{scroll-snap-type:y mandatory;scroll-behavior:smooth;height:100%}
  body{margin:0;background:var(--ground);color:var(--bone);font-family:var(--body);
    height:100%;overflow-y:scroll;
    background-image:radial-gradient(120% 80% at 50% -10%, #2a1a12 0%, var(--ground) 60%);}
  h1,h2,h3{text-wrap:balance;margin:0;line-height:1.05}
  a{color:var(--molten)}
  code{font-family:var(--mono);color:var(--molten);font-size:.92em}

  .slide{min-height:100vh;scroll-snap-align:start;display:flex;flex-direction:column;
    justify-content:center;padding:clamp(40px,7vh,90px) clamp(28px,7vw,120px);
    border-bottom:1px solid var(--line);position:relative}
  .slide .inner{max-width:1040px;margin:0 auto;width:100%}

  .kicker{display:inline-block;border:1px solid var(--line);border-left:3px solid var(--ember);
    padding:.35em .9em;font-family:var(--disp);text-transform:uppercase;letter-spacing:.2em;
    font-size:.72rem;color:var(--molten);margin-bottom:1.3rem}
  .snum{font-family:var(--disp);color:var(--ember);font-weight:800;letter-spacing:.05em;font-size:1.05rem}
  .shead{display:flex;align-items:baseline;gap:1rem;margin-bottom:1.6rem}
  .shead h2{font-family:var(--disp);text-transform:uppercase;font-weight:800;
    font-size:clamp(1.7rem,4.4vw,2.7rem);letter-spacing:.02em;color:var(--white)}
  .sub{color:var(--ash);max-width:70ch;margin:-.6rem 0 1.6rem;font-size:1.02rem}
  .lead{font-size:clamp(1.05rem,2.3vw,1.35rem);color:#d3c6b6;max-width:62ch}
  b.hl{color:var(--bone)}

  /* 타이틀 슬라이드 */
  .title h1{font-family:var(--disp);font-weight:800;text-transform:uppercase;
    font-size:clamp(3rem,11vw,7rem);letter-spacing:.01em;color:var(--white)}
  .title h1 em{font-style:normal;color:var(--ember);text-shadow:0 0 44px rgba(232,72,42,.5)}
  .title .tagline{margin-top:1.2rem;font-size:clamp(1.1rem,2.6vw,1.5rem);color:#d3c6b6}
  .principle{margin-top:2rem;padding:1.1rem 1.3rem;background:var(--ground-2);
    border:1px solid var(--line);border-left:4px solid var(--molten);max-width:64ch}
  .principle b{color:var(--molten)}

  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
    background:var(--line);border:1px solid var(--line);margin:1.4rem 0}
  .stat{background:var(--ground-2);padding:1.3rem 1.1rem}
  .stat .v{font-family:var(--mono);font-variant-numeric:tabular-nums;font-weight:700;
    font-size:clamp(1.7rem,4.5vw,2.4rem);color:var(--molten);line-height:1}
  .stat .l{font-size:.8rem;color:var(--ash);margin-top:.45rem}

  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem}
  .card{background:var(--ground-2);border:1px solid var(--line);padding:1.2rem}
  .card h3{font-family:var(--disp);text-transform:uppercase;letter-spacing:.06em;font-size:1.02rem;
    color:var(--molten);margin-bottom:.5rem}
  .card p{color:#cabeae;font-size:.94rem;margin:.2rem 0 0}
  .card.j h3{color:var(--jade)}

  .pipe{display:grid;gap:.55rem}
  .pnode{background:var(--ground-2);border:1px solid var(--line);border-left:3px solid var(--ember);padding:.75rem 1rem}
  .pnode .t{font-family:var(--disp);text-transform:uppercase;letter-spacing:.07em;font-weight:700;color:var(--bone)}
  .pnode .d{color:var(--ash);font-size:.9rem;margin-top:.15rem}
  .pnode.brain{border-left-color:var(--molten)} .pnode.out{border-left-color:var(--jade)}
  .parrow{color:var(--ember);text-align:center;font-family:var(--mono);opacity:.7;font-size:.85rem}

  table{width:100%;border-collapse:collapse;font-size:.92rem}
  th,td{text-align:left;padding:.6rem .7rem;border-bottom:1px solid var(--line);vertical-align:top}
  th{font-family:var(--disp);text-transform:uppercase;letter-spacing:.09em;font-size:.75rem;color:var(--ash)}
  .st{font-family:var(--disp);font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:.78rem}
  .st.done{color:var(--jade)} .st.run{color:var(--molten)} .st.plan{color:var(--ash)}

  pre{background:#0f0b09;border:1px solid var(--line);border-left:3px solid var(--molten);
    padding:.9rem 1rem;overflow-x:auto;font-family:var(--mono);font-size:.82rem;color:#d8cbb8;margin:.6rem 0}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:1.3rem;align-items:start}
  @media(max-width:760px){.two{grid-template-columns:1fr}}
  img.fig{width:100%;border:1px solid var(--line);display:block}
  .tag{display:inline-block;font-family:var(--disp);text-transform:uppercase;letter-spacing:.12em;
    font-size:.66rem;color:var(--ground);background:var(--molten);padding:.2em .6em;font-weight:700}
  .voices{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.8rem;margin-top:1rem}
  .voice{background:var(--ground-2);border:1px solid var(--line);padding:.85rem 1rem}
  .voice .jp{font-size:1rem;color:var(--bone)} .voice .ro{font-family:var(--mono);font-size:.75rem;color:var(--ash);margin:.1rem 0 .5rem}
  .voice audio{width:100%;height:32px}
  ul.tight{margin:.4rem 0 0;padding-left:1.1rem} ul.tight li{margin:.3rem 0;color:#cabeae}
  .muted{color:var(--ash);font-size:.82rem}

  /* 네비게이션 */
  .nav{position:fixed;bottom:18px;right:22px;font-family:var(--mono);font-size:.85rem;
    color:var(--ash);background:rgba(20,16,14,.8);border:1px solid var(--line);padding:.35em .7em;z-index:20}
  .dots{position:fixed;top:50%;right:14px;transform:translateY(-50%);display:flex;flex-direction:column;gap:7px;z-index:20}
  .dots a{width:8px;height:8px;border-radius:50%;background:var(--line);display:block;transition:background .2s}
  .dots a.on{background:var(--ember)}
  .hint{position:fixed;bottom:18px;left:22px;color:var(--ash);font-size:.78rem;z-index:20}
  @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  @media(max-width:760px){.dots,.hint{display:none}}
</style>
"""


def slide(n, body, cls=""):
    tag = f'<span class="snum">{n:02d}</span>' if n else ""
    return f'<section class="slide {cls}" id="s{n}"><div class="inner">{body}</div></section>\n'


def build(audio: bool) -> str:
    A = [audio_uri(f"s{i}.mp3") for i in range(4)] if audio else [None] * 4
    voices = ""
    lines = [("そっちの近況が先だ。", "sotchi no kinkyō ga saki da"),
             ("くたばりやがれ。", "kutabari yagare"),
             ("俺の獲物に手を出すな。", "ore no emono ni te o dasu na"),
             ("まだ終わってねえぞ。", "mada owatte nee zo")]
    for i, (jp, ro) in enumerate(lines):
        player = (f'<audio controls preload="none" src="{A[i]}"></audio>' if audio
                  else '<div class="muted" style="border:1px dashed var(--line);padding:.4rem .6rem">▶ 로컬 발표본에서 재생</div>')
        voices += f'<div class="voice"><div class="jp">{jp}</div><div class="ro">{ro}</div>{player}</div>'

    S = []
    # 00 타이틀
    S.append(slide(0,
        '<span class="kicker">길티기어 스트라이브 · AI 코치</span>'
        '<h1>Punish <em>Tool</em></h1>'
        '<p class="tagline">대전격투 리플레이를 넣으면, AI 코치가 "그 순간 뭘 했어야 이겼는지"를 '
        '프레임 단위로 짚어 주는 <b class="hl">로컬·오프라인</b> 도구.</p>'
        '<div class="principle">제1원칙 — <b>사용자에게 잘못된 정보를 주지 않는다.</b> 애매하면 침묵한다. '
        '격투 코칭에서 틀린 프레임 수치는 나쁜 습관을 심으므로, 시스템 전체가 "측정된 사실만 말한다"를 코드로 보증한다.</div>',
        "title"))

    # 01 문제
    S.append(slide(1,
        '<div class="shead"><span class="snum">01</span><h2>문제</h2></div>'
        '<p class="lead">"막혔을 때, 나는 무엇으로 <b class="hl">확정 반격</b>이 되는가?" — 이 한 가지가 격투게임 실력의 핵심이다.</p>'
        '<p class="sub" style="margin-top:1.4rem">그 답은 Dustloop 같은 방대한 프레임 데이터표에 흩어져 있어, 초·중급자는 '
        '대전 중은커녕 리플레이를 봐도 못 찾는다. → <b class="hl">영상에서 그 순간을 자동으로 찾아 해답을 주는</b> 코치를 만든다.</p>'))

    # 02 접근 · 파이프라인
    S.append(slide(2,
        '<div class="shead"><span class="snum">02</span><h2>접근 — 측정과 언어의 분리</h2></div>'
        '<p class="sub">영상에서 뽑은 <b class="hl">사실</b>만 LLM에 넘긴다. LLM은 프레임 수치를 스스로 지어낼 수 없다.</p>'
        '<div class="pipe">'
        '<div class="pnode"><div class="t">① HUD 리더 · hud_reader.py</div><div class="d">60fps 영상을 체력바 색 자가보정으로 훑어 피격·카운터·펀ish 이벤트 추출 (비전)</div></div>'
        '<div class="parrow">▼ timeline.json</div>'
        '<div class="pnode brain"><div class="t">② 펀ish 엔진 · punish_engine.py</div><div class="d">프레임데이터만으로 "막혔을 때 무엇으로 확정 반격이 되는가" 즉답 (두뇌)</div></div>'
        '<div class="parrow">＋ ③ 무브ID (DINOv2 RAG / VLM / 입력표시)</div>'
        '<div class="pnode out"><div class="t">분석 리포트 → AI 코치</div><div class="d">사실을 자연어로 정리 → 코치가 페르소나로 코칭</div></div>'
        '</div>'))

    # 03 기획 · 페르소나
    S.append(slide(3,
        '<div class="shead"><span class="snum">03</span><h2>기획 — 페르소나 카드</h2></div>'
        '<p class="sub">코치 페르소나: <b class="hl">솔 배드가이</b> (게임 원작 캐릭터). 제품명은 Punish Tool, 코치가 솔이다.</p>'
        '<div class="cards">'
        '<div class="card"><h3>성격</h3><p>퉁명·직설, 반말, 서론 없이 핵심만. "노력해라" 잔소리 금지 — "뭘 하면 이기는지"만 툭. 겉은 난폭, 속은 챙김.</p></div>'
        '<div class="card"><h3>선은 지킨다</h3><p>욕설·비하·조롱 절대 없음. 플레이어를 깎아내리지 않는다 (퉁명 ≠ 무례).</p></div>'
        '<div class="card j"><h3>톤 고정</h3><p>파인튜닝 없이 시스템 프롬프트 few-shot으로 말투 고정. (실측: QLoRA v1~v4보다 톤·사실성 안전)</p></div>'
        '</div>'
        '<div class="cards" style="margin-top:1rem"><div class="card"><h3>행동 규칙</h3><p>① 숫자는 근거에 있는 것만 ② 모르면 "데이터에 없다" '
        '③ \'왜\' 질문엔 근거를 2~4문장 설명 ④ 코칭 핵심은 확실히.</p></div></div>'))

    # 04 데이터
    S.append(slide(4,
        '<div class="shead"><span class="snum">04</span><h2>데이터</h2></div>'
        '<table><thead><tr><th>구분</th><th>내용</th></tr></thead><tbody>'
        '<tr><td>사전 데이터셋</td><td><code>framedata·mechanics·combos·char_notes·char_stats·move_names</code> JSON</td></tr>'
        '<tr><td>청킹·전처리</td><td><code>coach_kb.py</code>(검색) · <code>tts_prep.py</code>·<code>tts_aihub.py</code>(음성 전처리)</td></tr>'
        '<tr><td>벡터 DB</td><td><code>rag_emb.npz</code>·<code>rag_index.npz</code> (DINOv2 무브ID). ※코치 텍스트 RAG는 구조화 키워드 검색(수치 정확 매칭 우선)</td></tr>'
        '<tr><td>TTS 데이터셋</td><td>한국어 215 · 일본어 155 · AI Hub 다화자 13.2시간(14화자·7발화체)</td></tr>'
        '</tbody></table>'))

    # 05 AI 엔진
    S.append(slide(5,
        '<div class="shead"><span class="snum">05</span><h2>AI 엔진 — 근거 RAG + 숫자 가드</h2></div>'
        '<div class="two"><div>'
        '<div class="card"><h3>시스템 프롬프트</h3><p>PERSONA + 절대규칙(RULES) + few-shot + (정체질문시) LORE. '
        '숫자는 [참고 데이터]에 있는 것만, 없으면 "데이터에 없다".</p></div>'
        '<div class="card j" style="margin-top:1rem"><h3>결정론적 숫자 가드</h3><p>근거·질문에 없는 단위 수치를 후처리 정규식으로 잡아 경고. '
        '<b class="hl">코드가 제1원칙을 최종 보증.</b></p></div></div>'
        '<div><pre>def _unverified_numbers(answer, ctx, q):\n  allowed = set(re.findall(r"\\d+", ctx+q))\n  return [m.group(0) for m in\n    NUM.finditer(answer)\n    if m.group(1) not in allowed]\n# 있으면 "※ 근거 데이터 없음" 부착</pre>'
        '<p class="muted">모델: qwen3:8b(llama.cpp 동봉). QLoRA v1~v4는 자유대화 손상 → 베이스+가드로 결론.</p></div></div>'))

    # 06 평가 RAGAS
    S.append(slide(6,
        '<div class="shead"><span class="snum">06</span><h2>평가 — RAGAS 정렬 (실측)</h2></div>'
        '<p class="sub">실제 코치를 로컬 qwen3:8b로 호출해 채점 (n=17, fabrication 없음).</p>'
        '<div class="stats">'
        '<div class="stat"><div class="v">1.00</div><div class="l">Faithfulness<br>수치 환각 0건</div></div>'
        '<div class="stat"><div class="v">1.00</div><div class="l">Context Recall<br>검색 근거 확보</div></div>'
        '<div class="stat"><div class="v">0.75</div><div class="l">Refusal Acc.<br>근거밖 거절</div></div>'
        '<div class="stat"><div class="v">$0</div><div class="l">1콜 비용<br>로컬 추론</div></div></div>'
        f'<img class="fig" src="{RAGAS}" alt="RAGAS 평가" style="max-width:820px;margin-top:.6rem">'))

    # 07 평가 · 정직한 한계
    S.append(slide(7,
        '<div class="shead"><span class="snum">07</span><h2>평가 — 정직한 한계</h2></div>'
        '<div class="two"><div>'
        '<div class="card j"><h3>핵심 검증</h3><p>정상 사용에서 코치는 근거 밖 수치를 <b class="hl">단 한 번도 지어내지 않았다.</b> Faithfulness 1.00.</p></div>'
        '<div class="card" style="margin-top:1rem"><h3>재현된 약점 (개선 과제)</h3><p>"47프레임 즉사 콤보"처럼 <b class="hl">질문에 거짓 수치를 심으면</b> '
        '가드가 "질문에 있음→허용"으로 처리해 놓쳤다(4중 1). → 근거에 없고 질문에만 있는 수치를 경고하도록 확장 예정.</p></div>'
        '<p class="muted" style="margin-top:.8rem">실패 케이스를 드러내는 것이 이 평가의 신뢰 근거다.</p>'
        f'</div><div><img class="fig" src="{FAITH}" alt="카테고리별 충실도"></div></div>'))

    # 08 호출 로그 / 비용
    S.append(slide(8,
        '<div class="shead"><span class="snum">08</span><h2>응답 엔진 · 호출 로그</h2></div>'
        '<div class="two"><div>'
        '<table><tbody>'
        '<tr><td>엔진</td><td>llama.cpp (동봉 llama-server, Vulkan)</td></tr>'
        '<tr><td>모델</td><td>Qwen3-8B (Q4_K_M GGUF)</td></tr>'
        '<tr><td>API</td><td>OpenAI 호환 REST + 스트리밍</td></tr>'
        '<tr><td><b class="hl">1콜 비용</b></td><td><b class="hl">$0</b> (로컬 추론)</td></tr>'
        '<tr><td>TTFB 중앙</td><td>~8초 <span class="muted">(학습 동시부하 최악조건)</span></td></tr>'
        '</tbody></table>'
        '<p class="muted" style="margin-top:.7rem">클라우드 API 대신 로컬 GGUF 동봉 → 배포·운영 비용 0원. '
        '지연은 TTS 학습과 GPU 동시점유 시 측정값(단독 실행 시 수 초).</p>'
        f'</div><div><img class="fig" src="{LAT}" alt="응답 지연"></div></div>'))

    # 09 백엔드 · 클라이언트
    S.append(slide(9,
        '<div class="shead"><span class="snum">09</span><h2>백엔드 · 클라이언트</h2></div>'
        '<div class="two"><div>'
        '<div class="card"><h3>백엔드</h3><p><code>llm_server.py</code> — 동봉 llama-server 자동 기동/관리, OpenAI 호환 REST·스트리밍. '
        'Windows Job Object로 고아 프로세스 정리.</p></div>'
        '<div class="card" style="margin-top:1rem"><h3>클라이언트</h3><p><code>gui.py</code> (PySide6) — [AI 코치] 채팅 탭 + [분석] 탭. '
        '답변에 근거 반영 + 숫자 가드 경고. <b class="hl">아바타 위 한국어/日本語 음성 전환 토글.</b></p></div>'
        f'</div><div><img class="fig" src="{TOGGLE}" alt="한국어/일본어 토글"></div></div>'))

    # 10 멀티모달 · 음성/아바타
    S.append(slide(10,
        '<div class="shead"><span class="snum">10</span><h2>멀티모달 — 음성 · 비전 · 아바타</h2></div>'
        '<div class="cards">'
        '<div class="card"><h3>비전</h3><p>HUD 색 자가보정, 배너 감지 95%·오검 0, DINOv2 무브ID, YOLO pose.</p></div>'
        '<div class="card"><h3>음성 · SBV2 개조</h3><p>Style-Bert-VITS2 한국어 개조. 일본어 솔 파인튜닝 완료.</p></div>'
        '<div class="card"><h3>아바타 · three.js</h3><p>게임 모델 추출→실시간 툰셰이더. 립싱크·표정·깜빡임.</p></div></div>'
        '<p style="margin-top:1.4rem"><span class="tag">JP 솔 음성 · 실제 합성</span></p>'
        f'<div class="voices">{voices}</div>'
        '<p class="muted" style="margin-top:.7rem">※ 게임 음성 기반 합성 (연구·교육용).</p>'))

    # 11 한국어 TTS 개조
    S.append(slide(11,
        '<div class="shead"><span class="snum">11</span><h2>한국어 TTS 개조 — 91%를 물려받다</h2></div>'
        '<p class="sub">SBV2는 한국어 미지원. NDC26(넥슨게임즈) 방식으로 개조 — 언어 의존부는 <b class="hl">G2P·BERT 두 곳뿐</b>.</p>'
        '<div class="stats">'
        '<div class="stat"><div class="v">91%</div><div class="l">사전학습 전이<br>66.6M/73.2M</div></div>'
        '<div class="stat"><div class="v">112→125</div><div class="l">음소 심볼<br>112 전이+13 신규</div></div>'
        '<div class="stat"><div class="v">13.2h</div><div class="l">KO 코퍼스<br>14화자·7발화체</div></div>'
        '<div class="stat"><div class="v">1024</div><div class="l">KLUE-RoBERTa<br>히든 차원</div></div></div>'
        f'<img class="fig" src="{WARM}" alt="웜스타트 91% 전이" style="max-width:760px;margin-top:.4rem">'))

    # 12 인프라 · 결과
    S.append(slide(12,
        '<div class="shead"><span class="snum">12</span><h2>인프라 · 결과 · 현황</h2></div>'
        '<div class="two" style="align-items:start"><div>'
        '<div class="card j"><h3>배포 (설치·터미널 없음)</h3><p>통짜 배포(<code>dist/</code>, PyInstaller). 더블클릭 실행. '
        'llama.cpp·모델·전부 동봉 → 완전 로컬·오프라인·무료.</p></div></div>'
        '<div><table><tbody>'
        '<tr><td>영상 분석·펀ish 엔진</td><td><span class="st done">동작</span></td></tr>'
        '<tr><td>AI 코치 (RAG·가드)</td><td><span class="st done">동작</span></td></tr>'
        '<tr><td>3D 아바타·언어 토글</td><td><span class="st done">동작</span></td></tr>'
        '<tr><td>JP 솔 음성</td><td><span class="st done">완료</span></td></tr>'
        '<tr><td>KO 기반모델</td><td><span class="st run">학습중</span></td></tr>'
        '<tr><td>통짜 배포</td><td><span class="st done">동작</span></td></tr>'
        '</tbody></table></div></div>'))

    # 13 역할 분담
    S.append(slide(13,
        '<div class="shead"><span class="snum">13</span><h2>역할 분담</h2></div>'
        '<p class="sub">개인(1인) 프로젝트 — AI 보조 개발(페어). 담당 모듈·책임 단위로 정리.</p>'
        '<div class="cards">'
        '<div class="card"><h3>비전</h3><p>hud_reader·punish_engine·무브ID(DINOv2/VLM/입력표시)</p></div>'
        '<div class="card"><h3>코치</h3><p>coach(프롬프트·가드)·coach_kb(RAG)·coach_web</p></div>'
        '<div class="card"><h3>음성·아바타</h3><p>SBV2 개조·tts_*·three.js 아바타</p></div>'
        '<div class="card"><h3>백엔드·배포</h3><p>llm_server·gui·통짜 배포</p></div></div>'
        '<p class="muted" style="margin-top:1rem">프로세스: 측정 우선(실측) → 실패 시 빠른 피벗 · 제1원칙을 전 트랙 공통 제약.</p>'))

    # 14 회고
    S.append(slide(14,
        '<div class="shead"><span class="snum">14</span><h2>회고</h2></div>'
        '<div class="two"><div>'
        '<div class="card j"><h3>잘된 점</h3><ul class="tight">'
        '<li>측정 우선 원칙이 방향을 정함 (감 아닌 수치)</li>'
        '<li>실패 시 빠른 피벗 (CNN→DINOv2, QLoRA→가드, Ollama→llama.cpp)</li>'
        '<li>제1원칙을 코드로 강제 (숫자 가드)</li>'
        '<li>선행 사례를 실측 재현 (웜스타트 91% 검증)</li></ul></div></div>'
        '<div><div class="card"><h3>개선점</h3><ul class="tight">'
        '<li>값비싼 실패 전 더 일찍 검증 (CNN)</li>'
        '<li>의존성 관리 (TTS 환경 버전 충돌)</li>'
        '<li>스코프: 4축 동시 진행으로 마무리 지연</li>'
        '<li>재현성·로그 (중단/재개, 리소스 경합)</li></ul></div></div></div>'
        '<p class="muted" style="margin-top:1rem">한 줄 교훈: <b class="hl">측정이 방향을 정한다.</b> 싸게 측정하고, 실패는 빨리 인정하고 피벗한다.</p>'))

    # 15 자기평가 · 로드맵
    S.append(slide(15,
        '<div class="shead"><span class="snum">15</span><h2>자기평가 · 로드맵</h2></div>'
        '<div class="two"><div>'
        '<div class="card"><h3>자기평가 A−</h3><p>핵심 원칙·파이프라인 완성·검증. 미완(KO 음성·통합 마감)이 명확하고 재개 경로 준비됨 → 완결성 높음.</p></div>'
        '</div><div>'
        '<div class="card j"><h3>로드맵</h3><ul class="tight">'
        '<li>KO 기반모델 학습 완료 → 한국어 솔 파인튜닝 → 코치 응답 실시간 TTS</li>'
        '<li>무브ID 정밀도(57%) 향상 · 히트 귀속</li>'
        '<li>TTS 음소 타이밍 ↔ 아바타 립싱크 동기화</li></ul></div></div></div>'))

    # 16 마무리
    S.append(slide(16,
        '<span class="kicker">Punish Tool</span>'
        '<h1 style="font-family:var(--disp);text-transform:uppercase;font-weight:800;'
        'font-size:clamp(2.2rem,6vw,4rem);color:var(--white)">측정은 정확하게,<br><em style="color:var(--ember)">코칭은 사람처럼.</em></h1>'
        '<p class="lead" style="margin-top:1.4rem">정확한 측정과 사람 같은 코칭을 분리해, 틀린 정보 없이 실력 향상을 돕는 로컬 AI 코치.</p>'
        '<p class="muted" style="margin-top:1.6rem">github.com/msc010312/punish_tool · 비영리 교육 목적. '
        'Guilty Gear Strive 및 솔 배드가이 관련 자산·음성은 Arc System Works의 지식재산이며 상업적으로 이용하지 않는다. '
        '아바타는 게임 모델 추출, 음성은 게임 내 음성 기반 합성(연구·교육용).</p>',
        "title"))

    dots = "".join(f'<a href="#s{i}" data-i="{i}"></a>' for i in range(len(S)))
    body = "".join(S)
    nav = (f'<div class="dots">{dots}</div>'
           f'<div class="nav" id="nav">01 / {len(S):02d}</div>'
           '<div class="hint">↓ 스크롤 · ←→↑↓/Space 이동</div>')
    js = f'''<script>
      const slides=[...document.querySelectorAll('.slide')];
      const nav=document.getElementById('nav'), dots=[...document.querySelectorAll('.dots a')];
      const N={len(S)};
      function cur(){{let b=0,bd=1e9;slides.forEach((s,i)=>{{const d=Math.abs(s.getBoundingClientRect().top);if(d<bd){{bd=d;b=i;}}}});return b;}}
      function upd(){{const i=cur();nav.textContent=String(i+1).padStart(2,'0')+' / '+String(N).padStart(2,'0');
        dots.forEach((d,j)=>d.classList.toggle('on',j===i));}}
      function go(d){{const i=Math.min(N-1,Math.max(0,cur()+d));slides[i].scrollIntoView({{behavior:'smooth'}});}}
      addEventListener('scroll',upd,{{passive:true}}); upd();
      addEventListener('keydown',e=>{{
        if(['ArrowDown','ArrowRight','PageDown',' '].includes(e.key)){{e.preventDefault();go(1);}}
        else if(['ArrowUp','ArrowLeft','PageUp'].includes(e.key)){{e.preventDefault();go(-1);}}
      }});
    </script>'''
    return HEAD + nav + body + js


for fn, aud in (("프로젝트_최종보고서.html", True), ("프로젝트_최종보고서_공개용.html", False)):
    html = build(aud)
    io.open(ROOT / "제출물" / fn, "w", encoding="utf-8").write(html)
    print(f"{fn}: {round(len(html)/1024,1)}KB")
