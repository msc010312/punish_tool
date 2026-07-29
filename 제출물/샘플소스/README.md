# 샘플 소스 — 한국어 SBV2 개조 (핵심 novel 코드)

Style-Bert-VITS2(SBV2)는 한국어를 지원하지 않는다. NDC26(넥슨게임즈) 방식을 따라
**언어 의존부(G2P·BERT)만 한국어로 개조**한 핵심 소스다. 원본 SBV2 클론(수 GB, 서드파티)은
저장소에 포함하지 않고, 우리가 작성한 부분만 여기 보존한다.

## sbv2_korean/ — 원 위치: `tts/sbv2/style_bert_vits2/nlp/korean/`
| 파일 | 역할 |
|---|---|
| `normalizer.py` | 숫자·기호 → 한글 정규화 (예: `6F` → `육 에프`) |
| `g2p.py` | 발음 변환(g2pkk+mecab-ko) + 자모 음소열. 음절 수 보존으로 word2ph 정렬 |
| `bert_feature.py` | KLUE-RoBERTa-large(1024) 서브워드 → 문자 단위 정렬 |
| `__init__.py` | 패키지 |

## SBV2 코어에 가한 패치(라인 수준, 원본 수정)
| 파일 | 변경 |
|---|---|
| `nlp/symbols.py` | `KO_SYMBOLS` 추가, 112→125 심볼, `Languages.KO` 톤/언어맵 |
| `constants.py` | `Languages.KO` + KLUE-RoBERTa 경로 |
| `nlp/__init__.py` | `clean_text`/`extract_bert_feature`에 KO 분기 |
| `data_utils.py` | `get_text`에 KO 분기(ja_bert 채널 매핑) — 학습 시 `bert` 미할당 버그 수정 |

## 파이프라인 스크립트 (원 위치: `src/`)
`tts_prep.py`(전처리·프로파일) · `tts_aihub.py`(AI Hub→코퍼스) ·
`tts_warmstart.py`(JP→KO 가중치 91% 전이) · `tts_train.py`(학습 러너) ·
`tts_synth.py`(합성) · `ko_base_train.sh`(중단/재개 학습).
