"""
clone_voice.py  —  XTTS-v2 제로샷 음성 클로닝 (base 모델 + 참조 음성, 안정 파라미터)

md '따라하기②' 제로샷에 늘어짐/반복 방지 파라미터(enable_text_splitting=True + penalty, speed 미사용)를
적용한 안정판. 파인튜닝 불필요 — 참조 음성만 있으면 그 목소리로 한국어 합성.

사용: tts_env/Scripts/python.exe src/clone_voice.py --ref <참조.wav> --text "..." --out <out.wav>
"""
import argparse, os
os.environ["COQUI_TOS_AGREED"] = "1"

CKPT = r"D:/punishTool/voice/xtts_ckpt"

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--text", default="안녕하세요. 반갑습니다.")
ap.add_argument("--out", default="D:/punishTool/voice/my_voice_output.wav")
ap.add_argument("--temp", type=float, default=0.65)
a = ap.parse_args()

import torch, torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

cfg = XttsConfig(); cfg.load_json(f"{CKPT}/config.json")
m = Xtts.init_from_config(cfg)
m.load_checkpoint(cfg, checkpoint_path=f"{CKPT}/model.pth", vocab_path=f"{CKPT}/vocab.json", use_deepspeed=False)
m.cuda().eval()
print("참조:", a.ref, flush=True)
lat, spk = m.get_conditioning_latents(audio_path=[a.ref])
out = m.inference(a.text, "ko", lat, spk, temperature=a.temp,
                  length_penalty=0.8, repetition_penalty=10.0, enable_text_splitting=True)
torchaudio.save(a.out, torch.tensor(out["wav"]).unsqueeze(0), 24000)
print(f"✅ 클로닝 완료 -> {a.out}\n   문장: {a.text}", flush=True)
