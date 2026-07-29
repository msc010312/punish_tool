"""파인튜닝 모델로 repetition/length penalty 조합을 쓸어보며 정상 길이 합성 찾기 (speed 미사용)."""
import glob, torch, torchaudio, subprocess
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CKPT = r"D:/punishTool/voice/xtts_ckpt"
OUT = r"D:/punishTool/voice/xtts_ft_out"
DATA = r"D:/punishTool/voice/dataset"
TEXT = "안녕하세요. 반갑습니다."
REF = r"D:/punishTool/voice/ref_choihan.wav"   # 그레고르 29초 참조(깨끗)

run = sorted(glob.glob(f"{OUT}/xtts_solft-*"))[-1]
ft = sorted(glob.glob(f"{run}/best_model*.pth"))[-1]
cfg = XttsConfig(); cfg.load_json(f"{CKPT}/config.json")
m = Xtts.init_from_config(cfg)
m.load_checkpoint(cfg, checkpoint_path=ft, vocab_path=f"{CKPT}/vocab.json", use_deepspeed=False)
m.cuda().eval()
lat, spk = m.get_conditioning_latents(audio_path=[REF])

def dur(p):
    return float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=nw=1:nk=1",p],capture_output=True,text=True).stdout.strip() or 0)

combos = [
    dict(rep=5.0, len=1.0, temp=0.7, split=True),
    dict(rep=10.0, len=0.8, temp=0.65, split=True),
    dict(rep=10.0, len=1.0, temp=0.6, split=False),
    dict(rep=8.0, len=0.7, temp=0.6, split=True),
]
for i, c in enumerate(combos):
    o = m.inference(TEXT, "ko", lat, spk, temperature=c["temp"],
                    repetition_penalty=c["rep"], length_penalty=c["len"],
                    enable_text_splitting=c["split"])
    p = f"D:/punishTool/voice/sweep_{i}.wav"
    torchaudio.save(p, torch.tensor(o["wav"]).unsqueeze(0), 24000)
    print(f"[{i}] rep={c['rep']} len={c['len']} temp={c['temp']} split={c['split']} -> {dur(p):.1f}s", flush=True)
print("목표: '안녕하세요. 반갑습니다.' ≈ 2~3초", flush=True)
