"""
zeroshot_tts.py  —  20강 md '따라하기②' 그대로: XTTS-v2 제로샷 클로닝 (추가 파라미터 없음)

md 핵심 한 줄: tts.tts_to_file(text, speaker_wav=참조, language="ko", file_path=...)
참조 = 성우 최한(그레고르) 깨끗한 클립 몇 개. 파인튜닝/speed/penalty 안 씀 — md 방식 그대로.
사용: tts_env/Scripts/python.exe src/zeroshot_tts.py [--text "..."]
"""
import argparse, glob, os
os.environ["COQUI_TOS_AGREED"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="안녕하세요. 반갑습니다.")
ap.add_argument("--ref", default="D:/punishTool/voice/dataset/wavs/greg_0015.wav")
ap.add_argument("--out", default="D:/punishTool/voice/zeroshot_output.wav")
a = ap.parse_args()

import torch
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
print("참조 클립:", os.path.basename(a.ref), flush=True)

# md 그대로 — 단일 참조, 추가 파라미터 없이
tts.tts_to_file(text=a.text, speaker_wav=a.ref, language="ko", file_path=a.out)
print(f"✅ 제로샷 합성 완료 -> {a.out}\n   문장: {a.text}", flush=True)
