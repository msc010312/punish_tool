"""
train_xtts.py  —  XTTS-v2 GPT 파인튜닝 (강의 20강 섹션 ③④ 레시피 그대로) + 합성

강사 노트북의 수동 GPTTrainer 레시피를 따른다. 다른 점: 부트스트랩(제로샷 생성) 대신
**실제 솔 한국어 더빙 클립(voice/dataset)** 을 학습 데이터로 사용. eval avg_loss 가 에폭마다
내려가는지가 학습 확인 포인트(과제 제출용).  RTX4070 로컬 검증된 config.

venv: tts_env.  사용: tts_env/Scripts/python.exe src/train_xtts.py [--epochs 6]
출력: voice/xtts_ft_out/xtts_solft-*/  (best_model·loss) + voice/sol_clone_output.wav
"""
from __future__ import annotations
import argparse, glob, os

os.environ["COQUI_TOS_AGREED"] = "1"

WORK = r"D:/punishTool/voice"
DATA = r"D:/punishTool/voice/dataset"
CKPT = r"D:/punishTool/voice/xtts_ckpt"
OUT = r"D:/punishTool/voice/xtts_ft_out"


def download_base():
    import torch
    os.makedirs(CKPT, exist_ok=True)
    base = "https://huggingface.co/coqui/XTTS-v2/resolve/main"
    for fn in ["config.json", "vocab.json", "mel_stats.pth", "dvae.pth", "model.pth"]:
        dst = f"{CKPT}/{fn}"
        if (not os.path.exists(dst)) or os.path.getsize(dst) < 1024:
            print(f"다운로드 {fn} ...", flush=True)
            torch.hub.download_url_to_file(f"{base}/{fn}", dst)
    print("✅ 베이스 체크포인트 준비:", CKPT, flush=True)


def train(epochs):
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs
    os.makedirs(OUT, exist_ok=True)

    dataset_config = BaseDatasetConfig(formatter="coqui", dataset_name="sol_ft", path=DATA,
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
        run_name="xtts_solft", project_name="xtts_ft",
        epochs=epochs, batch_size=2, eval_batch_size=2, batch_group_size=0,
        num_loader_workers=0, num_eval_loader_workers=0, eval_split_max_size=3,
        print_step=10, plot_step=100000, log_model_step=100000, save_step=410,
        save_n_checkpoints=6, save_checkpoints=True, print_eval=True,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [900000, 2700000, 5400000], "gamma": 0.5, "last_epoch": -1})

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples([dataset_config], eval_split=True,
                                                   eval_split_max_size=3, eval_split_size=3)
    print(f"samples: {len(train_samples)} train / {len(eval_samples)} eval", flush=True)
    Trainer(TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=1),
            config, output_path=OUT, model=model,
            train_samples=train_samples, eval_samples=eval_samples).fit()
    print("✅ 파인튜닝 완료 →", OUT, flush=True)


def synth(text, speed=1.0, temp=0.65, ref=None):
    import glob, os, torch, torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    run_dir = max(glob.glob(f"{OUT}/xtts_solft-*"), key=os.path.getmtime)  # 최신=수정시간 기준(문자열정렬 버그 회피)
    import re
    cks = glob.glob(f"{run_dir}/best_model*.pth") + glob.glob(f"{run_dir}/checkpoint_*.pth")
    def step_of(p):
        m = re.search(r"_(\d+)\.pth$", p); return int(m.group(1)) if m else 0
    ft = max(cks, key=step_of)                    # 가장 많이 학습된(스텝 최대) 체크포인트
    print("파인튜닝 체크포인트:", ft, flush=True)

    cfg = XttsConfig(); cfg.load_json(f"{CKPT}/config.json")
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=ft, vocab_path=f"{CKPT}/vocab.json", use_deepspeed=False)
    m.cuda().eval()
    if not ref:
        greg = sorted(glob.glob(f"{DATA}/wavs/greg_*.wav"))
        ref = greg[len(greg)//2] if greg else sorted(glob.glob(f"{DATA}/wavs/*.wav"))[0]
    print(f"참조={ref} speed={speed} temp={temp}", flush=True)
    gpt_lat, spk = m.get_conditioning_latents(audio_path=[ref])
    # 자연스러운 합성: text_splitting=True(늘어짐 방지) + 적당한 penalty(과하면 로봇음). speed 미사용
    out = m.inference(text, "ko", gpt_lat, spk, temperature=temp,
                      length_penalty=1.0, repetition_penalty=3.0, enable_text_splitting=True)
    outwav = f"{WORK}/sol_clone_output.wav"
    torchaudio.save(outwav, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"\n✅ 합성 완료 -> {outwav}\n   문장: {text}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--text", default="안녕하세요. 에이아이로 만든 음성 샘플입니다. 감사합니다.")
    ap.add_argument("--speed", type=float, default=1.0)   # 1.0=원음(>1.0이면 피치 왜곡)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    if not a.skip_train:
        download_base()
        train(a.epochs)
    synth(a.text, speed=a.speed, temp=a.temp, ref=a.ref)


if __name__ == "__main__":
    main()
