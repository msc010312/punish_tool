# -*- coding: utf-8 -*-
"""merge_gguf.py — LoRA 병합 -> GGUF 변환 -> Q4_K_M 양자화 (학습 후 1회 실행).

산출: lora_out/coach_ragA-Q4_K_M.gguf
배포 반영: llama/ 폴더의 gguf와 교체(llm_server가 *.gguf 첫 파일을 집음).

필요 도구(스크래치패드에 준비됨):
  <scratch>/llama.cpp/convert_hf_to_gguf.py
  <scratch>/lltools/llama-quantize.exe
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER = ROOT / "lora_out" / "coach_ragA"
MERGED = ROOT / "lora_out" / "merged_fp16"
SCRATCH = Path(os.environ.get(
    "PUNISH_SCRATCH",
    r"C:\Users\msc01\AppData\Local\Temp\claude\d--punishTool"
    r"\69027c21-2e41-4ebc-bce5-d7ae82c47055\scratchpad"))
CONVERT = SCRATCH / "llama.cpp" / "convert_hf_to_gguf.py"
QUANTIZE = SCRATCH / "lltools" / "llama-quantize.exe"
F16_GGUF = ROOT / "lora_out" / "coach_ragA-F16.gguf"
OUT_GGUF = ROOT / "lora_out" / "coach_ragA-Q4_K_M.gguf"


def merge():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[merge] base fp16 로드(CPU RAM 사용)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model = model.merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED))
    AutoTokenizer.from_pretrained(str(ADAPTER)).save_pretrained(str(MERGED))
    print(f"[merge] -> {MERGED}", flush=True)


def convert():
    print("[convert] HF -> GGUF f16", flush=True)
    subprocess.run([sys.executable, str(CONVERT), str(MERGED),
                    "--outfile", str(F16_GGUF), "--outtype", "f16"], check=True)
    print("[quantize] f16 -> Q4_K_M", flush=True)
    subprocess.run([str(QUANTIZE), str(F16_GGUF), str(OUT_GGUF), "Q4_K_M"],
                   check=True)
    print(f"[done] {OUT_GGUF}", flush=True)


if __name__ == "__main__":
    if not (ADAPTER / "adapter_model.safetensors").exists():
        sys.exit(f"어댑터 없음: {ADAPTER} — train_sft.py 먼저")
    if "--convert-only" not in sys.argv:
        merge()
    convert()
