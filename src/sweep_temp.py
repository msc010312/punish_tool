"""파인튜닝 모델로 temperature를 낮춰가며 합성 — 끝음 인토네이션 완만한 것 찾기."""
import glob, torch, torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CKPT = r"D:/punishTool/voice/xtts_ckpt"
OUT = r"D:/punishTool/voice/xtts_ft_out"
TEXT = "안녕하세요. 반갑습니다."
REF = r"D:/punishTool/voice/ref_choihan.wav"

run = sorted(glob.glob(f"{OUT}/xtts_solft-*"))[-1]
ft = sorted(glob.glob(f"{run}/best_model*.pth"))[-1]
cfg = XttsConfig(); cfg.load_json(f"{CKPT}/config.json")
m = Xtts.init_from_config(cfg)
m.load_checkpoint(cfg, checkpoint_path=ft, vocab_path=f"{CKPT}/vocab.json", use_deepspeed=False)
m.cuda().eval()
lat, spk = m.get_conditioning_latents(audio_path=[REF])

for t in [0.35, 0.45, 0.55]:
    o = m.inference(TEXT, "ko", lat, spk, temperature=t,
                    length_penalty=0.8, repetition_penalty=10.0, enable_text_splitting=True)
    p = f"D:/punishTool/voice/temp_{int(t*100)}.wav"
    torchaudio.save(p, torch.tensor(o["wav"]).unsqueeze(0), 24000)
    print(f"temp={t} -> {p}", flush=True)
print("끝음 가장 자연스러운(덜 올라가는) 거 골라봐: temp_35 / temp_45 / temp_55", flush=True)
