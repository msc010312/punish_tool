# GitHub 링크 정리 (샘플 소스 포함)

> 저장소 게시 후 아래 `<링크>`를 채운다. 현재는 로컬 프로젝트(`d:/punishTool`) 기준
> 파일 경로와 핵심 샘플 소스를 정리한다.

## 저장소 링크

| 항목 | 링크 |
|---|---|
| 메인 저장소 | `<github.com/USER/punish-tool>` |
| 릴리스(배포본) | `<.../releases>` |
| 데모 영상 | `<.../프로젝트시연_동영상.mp4>` |

### 게시 방법 (아직 git 저장소 아님)
```bash
cd d:/punishTool
git init && git add . && git commit -m "Initial commit: Punish Tool"
git branch -M main
git remote add origin https://github.com/USER/punish-tool.git
git push -u origin main
# 대용량(모델 가중치 *.pt/*.gguf/*.safetensors, dist/, tts/)은 .gitignore 또는 Git LFS 권장
```

## 소스 트리 (핵심)

```
punishTool/
├─ src/
│  ├─ hud_reader.py       ① 비전: HUD 자가보정 이벤트 추출 (1,109줄)
│  ├─ punish_engine.py    ② 두뇌: 프레임데이터 확정반격 계산 (472줄)
│  ├─ vlm_id.py / rag_moveid.py / input_moveid.py   ③ 무브ID 3갈래
│  ├─ analyze_match.py    분석 리포트 생성 (701줄)
│  ├─ coach.py            AI 코치(페르소나·스트리밍·숫자가드) (362줄)
│  ├─ coach_kb.py         오프라인 RAG 근거 검색 (288줄)
│  ├─ coach_web.py        Dustloop 온라인 보강
│  ├─ llm_server.py       동봉 llama.cpp 서버 관리 (176줄)
│  ├─ gui.py              PySide6 클라이언트 (734줄)
│  ├─ tts_prep.py / tts_aihub.py / tts_warmstart.py / tts_train.py / tts_synth.py   TTS 파이프라인
│  └─ ...
├─ avatar3d/index.html    three.js 실시간 3D 아바타 뷰어
├─ tts/sbv2/              Style-Bert-VITS2 한국어 개조 (nlp/korean/ 추가)
├─ *.json                 도메인 데이터(framedata/mechanics/combos/...)
├─ dist/FrameAnalyzer/    통짜 배포본
├─ requirements.txt
└─ 제출물/                 과제 산출물(README·설명서·보고서·시연)
```

## 샘플 소스 ①: 측정→언어 분리의 핵심 (숫자 가드)

> `src/coach.py` — LLM이 근거에 없는 프레임 수치를 지어내면 후처리로 잡아낸다.
> **제1원칙("잘못된 정보 금지")을 코드가 최종 보증하는 지점.**

```python
# coach.py (요지)
def _unverified_numbers(answer: str, context: str, question: str) -> list[str]:
    """근거/질문에 없는 단위 숫자(N프레임/N뎀/NF)를 잡아 경고 대상으로 반환."""
    allowed = set(re.findall(r"\d+", context + " " + question))
    flagged = []
    for m in re.finditer(r"(\d+)\s*(프레임|F|뎀|데미지|dmg)", answer):
        if m.group(1) not in allowed:
            flagged.append(m.group(0))
    return flagged
# → flagged가 있으면 답변에 "※ 근거 데이터 없음" 경고를 덧붙인다.
```

## 샘플 소스 ②: 한국어 G2P (SBV2 개조)

> `tts/sbv2/style_bert_vits2/nlp/korean/g2p.py` — 표기형 한글을 발음형으로 바꾼 뒤
> 자모 음소열로 변환. **음절 수 보존**으로 word2ph(문자↔음소 정렬)를 정확히 만든다.

```python
# g2p.py (요지)
def g2p(norm_text: str) -> tuple[list[str], list[int], list[int]]:
    spoken = hangul_to_pronunciation(norm_text)   # 삼일전→사밀전 (연음·경음화·구개음화)
    phones, word2ph = ["_"], [1]
    for src_char, spk_char in zip(norm_text, spoken):
        got = __syllable_to_phonemes(spk_char) if is_hangul(spk_char) else [spk_char]
        phones += got
        word2ph.append(len(got))          # 원문 문자 기준 정렬(BERT 특징과 일치)
    phones.append("_"); word2ph.append(1)
    tones = [0] * len(phones)             # 한국어는 성조 없음
    return phones, tones, __redistribute_empty(word2ph)
```

## 샘플 소스 ③: 웜스타트 가중치 전이 (91% 재사용)

> `src/tts_warmstart.py` — JP-Extra 사전학습에서 언어무관 91%를 그대로 물려받고,
> 음소 임베딩은 **심볼 '이름'으로 대응**시켜 인덱스가 밀리지 않게 전이.

```python
# tts_warmstart.py (요지)
row_map = {i: old_index[s] for i, s in enumerate(SYMBOLS) if s in old_index}
sd["enc_p.emb.weight"] = grow_rows(old_emb, len(SYMBOLS), row_map)  # 112 전이 + 13 신규
# 나머지(보코더/flow/duration/style, 66.6M)는 값 변경 0 → 그대로 재사용
```

## 산출물 파일 위치

| 산출물 | 경로 |
|---|---|
| JP 솔 음성 샘플 | `tts/samples/sol_jp/sample_00~03.wav` |
| 학습된 JP 모델 | `tts/sbv2/Data/SOL_JP/models/G_3800.pth` |
| 아바타 뷰어 | `avatar3d/index.html` |
| 배포본 | `dist/FrameAnalyzer/` |
| 도메인 데이터 | `framedata_ggst.json` 외 `*.json` |
