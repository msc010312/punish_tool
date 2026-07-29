# 20. 음성 클로닝 · 감정 TTS — 30초 샘플로 목소리 복제하기 (XTTS-v2)

> 선수지식: 03차시(실시간 음성 STT/TTS), Python, 코랩 GPU 사용법
>
> 오늘 한 줄: **3차시엔 '고정 화자' 목소리로 말했다 → 오늘은 참조 음성 30초로 "그 사람 목소리"를 복제(클로닝)해 한국어를 읽게 만든다.**

## 🎯 학습목표
- **음성 클로닝 = 스피커 임베딩**(목소리 지문) 원리를 이해하고, **제로샷**으로 즉시 복제할 수 있다
- **Coqui TTS의 XTTS-v2**(`tts_models/multilingual/multi-dataset/xtts_v2`)를 로드해 **한국어를 합성**할 수 있다
- **참조 음성(`speaker_wav`)으로 목소리를 클로닝**해 임의 문장을 그 목소리로 읽게 만들 수 있다(language='ko')
- **Gradio 데모**로 참조 음성 업로드 + 텍스트 → 복제 음성 출력을 직접 만들 수 있다

> ⚖️ **라이선스 한 줄 고지:** XTTS-v2 모델은 **Coqui Public Model License(CPML) — 비상업(non-commercial)** 입니다. 본 실습은 교육 목적입니다. **상업 사용 시 라이선스를 반드시 확인**하고, **본인 또는 동의받은 사람의 목소리만** 복제하세요.

---

## 🔧 환경 설정 (가장 먼저 실행 — 첫 셀 일괄 설치)

아래 셀을 **노트북 맨 처음에 한 번만** 실행하세요. 이후 모든 예제는 여기서 만든 `tts` 객체를 그대로 사용합니다. (중간 설치 금지 — 설치는 전부 이 셀에서)

## 0. 환경 준비 — Colab/Kaggle 공통 + 모델 캐시(재다운로드 방지)

> 코랩은 GPU 한도/세션 종료가 잦아 **매번 대형 모델을 다시 받는** 참사가 납니다. 아래 셀이 **모델 캐시를 영구 저장소**(코랩=구글드라이브, 캐글=작업폴더)로 돌려 **1회만 다운로드**하게 합니다. **캐글에서도 그대로 실행**됩니다 — 우측 *Settings → Internet ON*, *Accelerator = GPU(T4 x2 / P100)*. 캐글은 주당 GPU 시간이 넉넉(약 30h)해 코랩 한도의 대안입니다.

```python
# ── 0. 환경 자동 감지 + 모델 캐시(재다운로드 방지) — Colab / Kaggle / 로컬 공통 ──
import os, sys

def _detect_env():
    if "google.colab" in sys.modules: return "colab"
    if os.path.exists("/kaggle"):      return "kaggle"
    return "local"
ENV = _detect_env()

if ENV == "colab":
    try:
        from google.colab import drive
        drive.mount("/content/drive")                       # 한 번 인증
        CACHE = "/content/drive/MyDrive/ai_human_models"    # ★ 드라이브에 모델 캐시(영구)
    except Exception as e:
        print("드라이브 마운트 생략:", e); CACHE = "/content/hf_cache"
    WORK = "/content"
elif ENV == "kaggle":
    CACHE = "/kaggle/working/hf_cache"                       # 작업폴더(출력으로 보존)
    WORK  = "/kaggle/working"
    if not os.path.exists("/content"):                       # /content 하드코딩 호환 시도
        try: os.symlink("/kaggle/working", "/content")
        except Exception: pass
else:
    CACHE = os.path.expanduser("~/ai_human_models"); WORK = os.getcwd()

# HF 캐시를 영구 위치로 고정 → from_pretrained 가 같은 모델을 두 번 받지 않음
os.environ["HF_HOME"]      = CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE, "hub")
os.makedirs(CACHE, exist_ok=True); os.makedirs(WORK, exist_ok=True)
print(f"환경={ENV} · 모델캐시={CACHE} · 작업폴더(WORK)={WORK}")
print("※ 이후 경로는 WORK 변수를 쓰세요(예: f'{WORK}/avatar'). 코랩=/content, 캐글=/kaggle/working")
```

```python
# ============================================================
#  공통 환경 설정  (이 셀을 가장 먼저 실행하세요)
#  - 코랩 GPU(T4)에서 XTTS-v2 를 '직접' 로드 (별도 서버 불필요)
#  - 설치는 전부 이 첫 셀에서 일괄 처리한다 (중간 설치 금지)
# ============================================================
# 런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU(T4) 로 먼저 설정하세요.

# coqui-tts: 현재 유지보수되는 Coqui TTS 패키지(예전 'TTS' 의 후속). import 는 동일하게 'from TTS.api import TTS'
# ★ transformers 5.x 는 coqui-tts 와 비호환(isin_mps_friendly 제거) → 4.57대로 핀해야 클로닝·파인튜닝 둘 다 동작
%pip install -q coqui-tts "transformers>=4.57,<5"
%pip install -q gradio          # ★ 데모용 패키지도 첫 셀에서 미리 (중간 설치 금지)

import os
# XTTS-v2 는 로드 시 라이선스 동의를 요구한다 → 비대화형(코랩)에서 미리 동의 처리
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.api import TTS

# GPU 사용 여부 확인 (CPU는 매우 느리므로 GPU 권장)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("사용 디바이스:", DEVICE)
if DEVICE == "cpu":
    print("⚠️ GPU가 아닙니다. 런타임 유형을 GPU(T4)로 바꾸면 훨씬 빠릅니다.")

# XTTS-v2 모델 로드 (첫 실행 시 모델 가중치 수 GB 다운로드 → 몇 분 소요)
# 다국어/다화자 제로샷 클로닝 모델: 클로닝 + 합성이 한 모델에 통합
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(MODEL_NAME).to(DEVICE)

print("✅ XTTS-v2 로드 완료. 이후 셀에서 tts 객체를 그대로 사용합니다.")
```

---

## 🗣️ 비유로 이해하기 — 목소리의 '지문'을 뜨는 성대모사

성대모사 배우는 30초만 들어도 그 사람 목소리를 따라 합니다. **음성 클로닝**이 바로 이것 — 짧은 참조 음성에서 **'목소리의 숫자 지문(스피커 임베딩)'** 을 뽑아, 전혀 다른 문장에 입힙니다.

| 구분 | 3차시 기본 TTS | 오늘 클로닝 TTS |
|---|---|---|
| **목소리** | 모델에 내장된 고정 화자 | 참조 음성으로 지정한 임의 화자 |
| **새 목소리 추가** | 불가(재학습 필요) | 샘플 1개로 즉시(**제로샷**) |
| **감정·다국어** | 제한적 | 참조·파라미터로 제어, 다국어 |

> 🔑 핵심 = **"무엇을 말하나(텍스트)"** 와 **"누구 목소리로 말하나(스피커 임베딩)"** 를 **분리**해서 자유롭게 조합한다.

### 클로닝의 흐름

```
  ① 참조 음성(30초)  →  ② 스피커 인코더(음색→임베딩)  ┐
                                                       ├→ ④ 그 목소리로 음성 합성
  ③ 읽을 텍스트(무엇을)  ────────────────────────────┘
```

---

## ⌨️ 따라하기 ① — XTTS-v2로 기본 한국어 합성 (환경 점검)

먼저 **참조 음성 없이** 기본 화자로 한국어를 합성해 봅니다. (환경이 정상인지 먼저 확인 → 그다음 클로닝으로)

```python
# ------------------------------------------------------------
# 기본 화자로 한국어 합성 → 환경/모델이 정상 동작하는지 먼저 확인
# XTTS-v2 는 화자 정보가 필요하므로, 내장 스튜디오 화자 한 명을 골라 사용한다.
# ------------------------------------------------------------
from IPython.display import Audio, display

# XTTS-v2 에 내장된 스튜디오 화자 목록(없을 수도 있으므로 방어적으로 접근)
try:
    speakers = tts.synthesizer.tts_model.speaker_manager.speaker_names
    print("내장 화자 수:", len(speakers), "| 예시:", speakers[:3])
except Exception as e:
    speakers = []
    print("내장 화자 목록을 못 읽음:", e)

BASE_TEXT = "안녕하세요. 이것은 기본 화자로 만든 한국어 음성입니다."
OUT_BASE = "out_base.wav"

if speakers:
    # 내장 화자 한 명을 골라 한국어 합성
    tts.tts_to_file(text=BASE_TEXT, speaker=speakers[0],
                    language="ko", file_path=OUT_BASE)
else:
    # 내장 화자가 없으면 다음 셀의 클로닝 폴백을 사용하세요
    print("내장 화자가 없으니, 따라하기 ②(클로닝)에서 참조 음성/폴백으로 진행합니다.")

if os.path.exists(OUT_BASE):
    print("✅ 기본 합성 완료:", OUT_BASE)
    display(Audio(OUT_BASE))   # 노트북에서 바로 재생
```

> 💡 **환경 점검이 먼저.** 기본 합성이 되면 "모델·GPU는 정상"이라는 뜻 → 다음 단계(클로닝)에서 문제가 생기면 '클로닝 로직'만 의심하면 됩니다.

---

## ⌨️ 따라하기 ② — 참조 음성으로 목소리 클로닝 (핵심)

이제 오늘의 핵심입니다. **참조 음성(`speaker_wav`)** 을 주면, 그 목소리로 한국어 문장을 읽습니다. 참조가 없으면 **기본 화자로 폴백**합니다.

```python
# ------------------------------------------------------------
# speaker_wav(참조 음성)로 '그 목소리'를 복제해 한국어를 읽게 한다.
# 참조 음성 준비 방법(둘 중 하나):
#   (A) 코랩 왼쪽 파일창에 본인 목소리 wav 업로드 후 경로 지정
#   (B) 아래 files.upload() 주석을 풀어 바로 업로드
# 참조가 없으면 → 기본 화자로 폴백(데모가 멈추지 않게)
# ------------------------------------------------------------
from IPython.display import Audio, display

# (B) 직접 업로드하려면 아래 두 줄의 주석을 푸세요 (10~30초, 잡음 적은 한국어 음성 권장)
# from google.colab import files
# REF_WAV = list(files.upload().keys())[0]

REF_WAV = "reference.wav"          # ← 본인 참조 음성 경로로 바꾸세요
CLONE_TEXT = "안녕하세요. 지금 들리는 목소리는 참조 음성을 복제한 결과입니다."
OUT_CLONE = "out_clone.wav"

def synth_korean(text, ref_wav=None, out_path="out.wav"):
    """참조 음성이 있으면 클로닝, 없으면 기본 화자로 폴백해 한국어 합성."""
    if ref_wav and os.path.exists(ref_wav):
        # ★ 목소리 클로닝: speaker_wav 에 참조 음성을 넘기면 그 음색으로 합성
        tts.tts_to_file(text=text, speaker_wav=ref_wav,
                        language="ko", file_path=out_path)
        used = f"클로닝(참조: {ref_wav})"
    else:
        # 폴백: 참조가 없으면 내장 기본 화자로
        try:
            spk = tts.synthesizer.tts_model.speaker_manager.speaker_names[0]
            tts.tts_to_file(text=text, speaker=spk, language="ko", file_path=out_path)
            used = f"폴백(기본 화자: {spk})"
        except Exception:
            tts.tts_to_file(text=text, language="ko", file_path=out_path)
            used = "폴백(모델 기본값)"
    return used

used = synth_korean(CLONE_TEXT, ref_wav=REF_WAV, out_path=OUT_CLONE)
print("합성 방식:", used)
display(Audio(OUT_CLONE))           # 복제된 목소리 재생 → 참조와 비교해 보세요
```

> 🔑 **딱 한 줄이 클로닝의 전부:** `tts.tts_to_file(text=..., speaker_wav=참조음성, language="ko", ...)`.
> `speaker_wav` 가 참조 음성이고, 모델이 거기서 **스피커 임베딩(목소리 지문)** 을 자동으로 뽑아 그 음색으로 읽어 줍니다.

> ⚠️ **윤리·동의:** `speaker_wav` 에는 **본인 또는 명시적으로 동의받은 사람의 목소리만** 넣으세요. 타인 목소리 무단 복제는 권리 침해입니다.

---

## ⌨️ 따라하기 ③ — 감정·스타일·다국어 살짝 맛보기

"참조가 곧 연출"입니다. 또 `language` 만 바꾸면 **같은 목소리로 다른 언어**(크로스링구얼)도 됩니다.

```python
# ------------------------------------------------------------
# (1) 속도 조절: 같은 목소리, 다른 말 빠르기  (speed 지원 시)
# (2) 다국어: language 코드만 바꿔 '같은 목소리로 영어' (크로스링구얼)
# ------------------------------------------------------------
from IPython.display import Audio, display

# (1) 속도(말 빠르기) 조절 — 파라미터 미지원 환경이면 except 로 일반 합성
try:
    if os.path.exists(REF_WAV):
        tts.tts_to_file(text="조금 더 천천히 말해 보겠습니다.",
                        speaker_wav=REF_WAV, language="ko",
                        speed=0.9, file_path="out_slow.wav")   # 0.8~1.2 권장
        print("✅ 속도 조절 합성 완료"); display(Audio("out_slow.wav"))
except Exception as e:
    print("speed 파라미터 미지원 또는 참조 없음:", e)

# (2) 다국어(크로스링구얼): 한국어 참조 목소리로 영어 말하기 (language="en")
try:
    if os.path.exists(REF_WAV):
        tts.tts_to_file(text="Hello! This is the same voice speaking English.",
                        speaker_wav=REF_WAV, language="en",
                        file_path="out_en.wav")
        print("✅ 영어(크로스링구얼) 합성 완료"); display(Audio("out_en.wav"))
    else:
        print("참조 음성이 없어 다국어 데모는 건너뜁니다.")
except Exception as e:
    print("다국어 합성 실패:", e)
```

> 💡 **참조가 곧 연출.** 밝게 녹음한 참조를 주면 밝게, 차분하게 녹음한 참조를 주면 차분하게 읽습니다.
> 감정을 바꾸고 싶으면 **그 감정으로 녹음한 참조 음성**을 쓰는 게 가장 확실합니다.

---

## 🚀 직접 써보는 데모 — 참조 음성 업로드 → 복제 음성 (Gradio)

참조 음성을 **업로드**하고 **텍스트**를 입력하면, 그 목소리로 복제된 한국어 음성을 **들려주는** 웹앱입니다.
참조 음성이 없으면 **기본 화자로 폴백**합니다. (위에서 만든 `synth_korean` 을 그대로 호출)

```python
# [첫 셀에서 일괄 설치] %pip install -q gradio
import gradio as gr
import tempfile, os

def clone_demo(ref_audio, text, language):
    """참조 음성(파일 경로) + 텍스트 → 복제 음성 wav 경로 반환."""
    text = (text or "").strip()
    if not text:
        return None, "(읽을 텍스트를 입력하세요)"
    # 출력 임시 파일
    out_path = tempfile.mktemp(suffix=".wav")
    try:
        if ref_audio and os.path.exists(ref_audio):
            # ★ 참조 음성으로 클로닝
            tts.tts_to_file(text=text, speaker_wav=ref_audio,
                            language=language, file_path=out_path)
            status = f"클로닝 완료 (참조 음성 사용 / {language})"
        else:
            # 참조 없음 → 기본 화자 폴백
            try:
                spk = tts.synthesizer.tts_model.speaker_manager.speaker_names[0]
                tts.tts_to_file(text=text, speaker=spk,
                                language=language, file_path=out_path)
            except Exception:
                tts.tts_to_file(text=text, language=language, file_path=out_path)
            status = f"참조 없음 → 기본 화자로 폴백 ({language})"
        return out_path, status
    except Exception as e:
        return None, f"(에러) 합성 실패: {e}"

with gr.Blocks(title="음성 클로닝 데모 (XTTS-v2)") as demo:
    gr.Markdown("## 🎙️ 음성 클로닝 — 참조 음성 + 텍스트 → 그 목소리로 읽기")
    gr.Markdown("⚖️ **본인 또는 동의받은 목소리만** 사용하세요. (XTTS-v2: 비상업 CPML)")
    with gr.Row():
        with gr.Column():
            ref = gr.Audio(label="참조 음성 업로드(10~30초, 없으면 기본 화자)",
                           type="filepath", sources=["upload", "microphone"])
            txt = gr.Textbox(label="읽을 텍스트", lines=3,
                             value="안녕하세요. 제 목소리를 복제한 AI 휴먼입니다.")
            lang = gr.Dropdown(["ko", "en", "ja"], value="ko", label="언어")
            btn = gr.Button("🔊 복제 음성 생성", variant="primary")
        with gr.Column():
            out_audio = gr.Audio(label="✅ 복제된 음성", type="filepath")
            out_status = gr.Textbox(label="상태", lines=2)
    btn.click(fn=clone_demo, inputs=[ref, txt, lang], outputs=[out_audio, out_status])

# share=True 로 외부 공유 링크 생성, debug=True 로 에러 로그 표시 (코랩에서도 동작)
demo.launch(share=True, debug=True)
```

> 💡 참조 음성을 올리고 텍스트를 넣은 뒤 **복제 음성 생성**을 누르면, 오른쪽에 그 목소리로 읽은 음성이 나옵니다.
> 참조를 안 올리면 **기본 화자로 폴백**해 그래도 동작합니다. 언어를 `en` 으로 바꾸면 같은 목소리로 영어도 시도해 보세요(크로스링구얼).

---

## 🏋️ 현업 한 걸음 더 — XTTS-v2 **파인튜닝** (제로샷 → 화자 전용 모델)

제로샷은 30초로 즉시 되지만 **발음·일관성·억양**이 참조에 휘둘립니다. 현업에서 "그 사람 전용 고품질 보이스"가 필요하면 **소량 데이터로 XTTS의 GPT 디코더를 미세조정(파인튜닝)** 합니다. 여기서는 *원리를 끝까지 한 바퀴 돌리는* 축소판으로 **데이터셋 → 베이스 체크포인트 → 트레이너 → 파인튜닝 모델 합성**까지 직접 실행합니다. (RTX4070 로컬 검증 완료)

> ⏱️ **무거운 셀.** 베이스 체크포인트(~2GB) 다운로드 + 소량 학습으로 코랩 T4에서 수 분~십수 분. 데이터·에폭을 늘리면 품질↑.
> ⚙️ **버전 핀(필수):** `coqui-tts` 는 **transformers 5.x 와 비호환**(`isin_mps_friendly` 제거)이라 첫 셀에서 `transformers>=4.57,<5` 로 핀합니다(이미 반영). 안 하면 클로닝·파인튜닝 둘 다 깨집니다.
> ⚖️ **동의·라이선스:** 본인/동의받은 목소리만, XTTS-v2 비상업(CPML).

### ① 학습 데이터셋 만들기 — 참조 목소리로 N개 클립 + 전사
제로샷으로 **참조 목소리가 학습 문장을 읽게** 해서 소량 데이터셋을 부트스트랩합니다(참조 없으면 내장 화자 폴백). 실제 현업에선 **본인이 직접 녹음한 5~10분**을 metadata 로 만들어 씁니다.

```python
# 위 따라하기에서 만든 tts 객체와 REF_WAV 를 그대로 사용 (제로샷 → 데이터셋 부트스트랩)
import os, random, torch, torchaudio
DATA = f"{WORK}/xtts_ft_data"; WAVS = f"{DATA}/wavs"; os.makedirs(WAVS, exist_ok=True)

SENTS = [
    "안녕하세요. 오늘은 음성 합성 모델을 직접 학습해 보겠습니다.",
    "데이터가 적어도 파인튜닝으로 목소리를 적응시킬 수 있습니다.",
    "제로샷 클로닝은 즉시 되지만 일관성은 학습이 더 좋습니다.",
    "이 문장은 학습용으로 만든 짧은 예시 음성입니다.",
    "현업에서는 오 분에서 십 분 분량의 깨끗한 음성을 씁니다.",
    "스피커 임베딩과 지피티 디코더를 함께 미세 조정합니다.",
    "오늘 강의에서는 원리를 끝까지 돌려보는 것이 목표입니다.",
    "마이크 잡음이 적을수록 합성 품질이 좋아집니다.",
    "같은 문장도 화자에 따라 분위기가 달라집니다.",
    "콜랩 무료 지피유에서도 소량 학습은 가능합니다.",
    "학습이 끝나면 새 문장을 그 목소리로 읽게 만듭니다.",
    "감정과 억양은 참조 음성과 데이터에 좌우됩니다.",
    "이번 실습은 교육 목적의 축소판 파인튜닝입니다.",
    "동의받은 목소리만 사용하는 것이 원칙입니다.",
    "검증이 끝나면 강의 노트북에 그대로 넣습니다.",
    "수고하셨습니다. 다음 단계로 넘어가겠습니다.",
]

use_ref = bool(REF_WAV) and os.path.exists(REF_WAV)
spk0 = None
if not use_ref:
    try: spk0 = tts.synthesizer.tts_model.speaker_manager.speaker_names[0]
    except Exception: spk0 = None

rows = []
for i, s in enumerate(SENTS):
    rel = f"wavs/clip_{i:03d}.wav"; ab = f"{WAVS}/clip_{i:03d}.wav"
    if use_ref:   tts.tts_to_file(text=s, speaker_wav=REF_WAV, language="ko", file_path=ab)
    elif spk0:    tts.tts_to_file(text=s, speaker=spk0,        language="ko", file_path=ab)
    else:         tts.tts_to_file(text=s,                      language="ko", file_path=ab)
    w, sr = torchaudio.load(ab)
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)          # 모노로
    if sr != 22050:    w = torchaudio.functional.resample(w, sr, 22050)
    torchaudio.save(ab, w, 22050)                           # 22.05kHz 모노 통일
    rows.append((rel, s, "my_voice"))

random.seed(0); random.shuffle(rows); ev, tr = rows[:2], rows[2:]
for path, items in [(f"{DATA}/metadata_train.csv", tr), (f"{DATA}/metadata_eval.csv", ev)]:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("audio_file|text|speaker_name\n")            # coqui formatter: 파이프 구분 + 헤더
        for r in items: f.write("|".join(r) + "\n")
print(f"데이터셋 준비 완료: {len(tr)} train / {len(ev)} eval | 참조 사용: {use_ref}")
```

### ② 베이스 체크포인트 받기 (파인튜닝 전용 DVAE·mel_stats 포함)
파인튜닝엔 추론본에 없는 **DVAE·mel_stats** 가 추가로 필요합니다.

```python
import os, torch
CKPT = f"{WORK}/xtts_ckpt"; os.makedirs(CKPT, exist_ok=True)
BASE = "https://huggingface.co/coqui/XTTS-v2/resolve/main"
for fn in ["config.json", "vocab.json", "mel_stats.pth", "dvae.pth", "model.pth"]:
    dst = f"{CKPT}/{fn}"
    if (not os.path.exists(dst)) or os.path.getsize(dst) < 1024:
        print("다운로드", fn, "..."); torch.hub.download_url_to_file(f"{BASE}/{fn}", dst)
print("✅ 베이스 체크포인트 준비:", CKPT)
```

### ③ XTTS GPT 파인튜닝 실행
소량·few-epoch 로 *원리를 끝까지* 돌립니다. (`num_loader_workers=0` 은 코랩/윈도우 공통 안전값)

```python
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs
import os
OUT = f"{WORK}/xtts_ft_out"; os.makedirs(OUT, exist_ok=True)

dataset_config = BaseDatasetConfig(formatter="coqui", dataset_name="my_ft", path=DATA,
    meta_file_train="metadata_train.csv", meta_file_val="metadata_eval.csv", language="ko")
model_args = GPTArgs(
    max_conditioning_length=132300, min_conditioning_length=66150, debug_loading_failures=False,
    max_wav_length=255995, max_text_length=200,
    mel_norm_file=f"{CKPT}/mel_stats.pth", dvae_checkpoint=f"{CKPT}/dvae.pth",
    xtts_checkpoint=f"{CKPT}/model.pth", tokenizer_file=f"{CKPT}/vocab.json",
    gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True)
audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
config = GPTTrainerConfig(
    output_path=OUT, model_args=model_args, audio=audio_config,
    run_name="xtts_my_ft", project_name="xtts_ft",
    epochs=4, batch_size=2, eval_batch_size=2, batch_group_size=0,
    num_loader_workers=0, num_eval_loader_workers=0, eval_split_max_size=2,
    print_step=5, plot_step=100000, log_model_step=100000, save_step=100000,
    save_n_checkpoints=1, save_checkpoints=True, print_eval=True,
    optimizer="AdamW", optimizer_wd_only_on_weights=True,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-6, lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [900000, 2700000, 5400000], "gamma": 0.5, "last_epoch": -1})

model = GPTTrainer.init_from_config(config)
train_samples, eval_samples = load_tts_samples([dataset_config], eval_split=True,
                                               eval_split_max_size=2, eval_split_size=2)
print("samples:", len(train_samples), "train /", len(eval_samples), "eval")
Trainer(TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=1),
        config, output_path=OUT, model=model,
        train_samples=train_samples, eval_samples=eval_samples).fit()
print("✅ 파인튜닝 완료 → 체크포인트:", OUT)
```

> 📉 학습 로그에서 **eval `avg_loss` 가 에폭마다 내려가면** 학습이 되는 것입니다. (로그의 "not a git repository" 메시지는 무해)

### ④ 파인튜닝 모델로 합성 — 학습에 없던 새 문장
```python
import glob, torch, torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from IPython.display import Audio, display

run_dir = sorted(glob.glob(f"{OUT}/xtts_my_ft-*"))[-1]
ft = sorted(glob.glob(f"{run_dir}/best_model_*.pth"))[-1]
print("파인튜닝 체크포인트:", ft)

cfg = XttsConfig(); cfg.load_json(f"{CKPT}/config.json")
ftm = Xtts.init_from_config(cfg)
ftm.load_checkpoint(cfg, checkpoint_path=ft, vocab_path=f"{CKPT}/vocab.json", use_deepspeed=False)
ftm.cuda().eval()

ref = f"{WAVS}/clip_000.wav"                                  # 화자 조건(참조 1개)
gpt_lat, spk_emb = ftm.get_conditioning_latents(audio_path=[ref])
TEXT = "이 문장은 학습에 없던 새 문장을, 파인튜닝된 목소리로 읽은 결과입니다."
out = ftm.inference(TEXT, "ko", gpt_lat, spk_emb, temperature=0.7)

torchaudio.save(f"{WORK}/ft_sample.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
print("✅ 합성 완료 — 제로샷 결과와 같은 문장으로 비교해 보세요")
display(Audio(f"{WORK}/ft_sample.wav"))
```

> 💡 **제로샷 vs 파인튜닝**: 데이터가 적으면 차이가 작지만, **5~10분 + 에폭↑** 이면 발음·억양 일관성이 눈에 띄게 좋아집니다 — 그게 현업이 파인튜닝을 쓰는 이유입니다.

---

## ✏️ 연습문제

### 문제 1 — 감정 비교
**같은 텍스트**를 (a) 밝게 녹음한 참조, (b) 차분하게 녹음한 참조로 각각 클로닝해 두 음성을 만들고, 감정이 어떻게 달라지는지 비교하세요.
힌트: `synth_korean(text, ref_wav=...)` 를 참조 파일만 바꿔 두 번 호출.

### 문제 2 — 다국어 안내방송
참조 음성 하나로 **한국어 + 영어** 안내 문구를 각각 합성해, 같은 목소리가 두 언어를 말하는지 확인하세요.
힌트: `language="ko"` 와 `language="en"` 으로 각각 `tts.tts_to_file` 호출.

---

## ✅ 해답

### 해답 1 — 감정(참조) 비교

```python
# 두 참조 음성 경로(본인이 밝게/차분하게 녹음한 파일로 바꾸세요)
REF_BRIGHT = "ref_bright.wav"   # 밝게 녹음
REF_CALM   = "ref_calm.wav"     # 차분하게 녹음
TEXT = "오늘 하루도 정말 수고 많으셨습니다."

for tag, ref in [("밝게", REF_BRIGHT), ("차분하게", REF_CALM)]:
    out = f"emo_{tag}.wav"
    used = synth_korean(TEXT, ref_wav=ref, out_path=out)   # 참조 없으면 폴백
    print(f"[{tag}] {used}")
    display(Audio(out))
```

> 포인트: 텍스트는 같아도 **참조의 감정**에 따라 합성 결과의 분위기가 달라집니다 — "참조가 곧 연출".

### 해답 2 — 다국어 안내방송

```python
REF = REF_WAV if os.path.exists(REF_WAV) else None
items = [
    ("ko", "잠시 후 3번 출구에서 안내를 시작하겠습니다."),
    ("en", "We will begin the announcement at Exit 3 shortly."),
]
for lang, text in items:
    out = f"announce_{lang}.wav"
    if REF:
        tts.tts_to_file(text=text, speaker_wav=REF, language=lang, file_path=out)
    else:
        # 참조 없으면 기본 화자 폴백
        try:
            spk = tts.synthesizer.tts_model.speaker_manager.speaker_names[0]
            tts.tts_to_file(text=text, speaker=spk, language=lang, file_path=out)
        except Exception:
            tts.tts_to_file(text=text, language=lang, file_path=out)
    print(f"[{lang}] 합성 완료"); display(Audio(out))
```

> 포인트: `language` 코드만 바꾸면 **같은 목소리로 여러 언어**(크로스링구얼). 음색(임베딩)은 언어와 독립이라 가능합니다.

---

## 🧾 한 장 정리
- **음성 클로닝 = 스피커 임베딩(목소리 지문)** 으로 내용·화자를 분리·재조합한다.
- **제로샷** — 참조 음성 30초로 학습 없이 즉시 복제. 핵심 한 줄: `tts.tts_to_file(text, speaker_wav, language="ko", ...)`.
- **참조가 곧 연출** — 감정·스타일은 참조 음성으로, 다국어는 `language` 코드로.
- **폴백** — 참조가 없으면 기본 화자로 동작하게 설계(데모가 멈추지 않게).
- ⚖️ **동의·고지·라이선스 필수** — 본인/동의자 목소리만, AI 합성 고지, XTTS-v2는 비상업(CPML).
