#!/usr/bin/env bash
# sol_ko_finetune.sh — 솔 KO 파인튜닝 시작/재개.
#
# Epoch70 KO 기반(base_ko G_221500)에서 emb_g를 벗긴 G_0(1화자)로부터 출발한다.
# train_ms_jp_extra가 Data/SOL/models/의 최신 G_*.pth를 자동으로 찾아 재개한다.
# eval_interval(200스텝)마다 체크포인트를 저장하므로 언제 멈춰도 손실이 적다.
#
# 멈추기: 이 프로세스(python.exe) 종료. 다시 이 스크립트를 실행하면 이어서 계속.
# 사용: bash src/sol_ko_finetune.sh
cd "$(dirname "$0")/.."
SBV2="tts/sbv2"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p tts/logs
echo "[$(date '+%F %T')] 솔 KO 파인튜닝 시작/재개"
LATEST=$(ls -1 "$SBV2/Data/SOL/models/"G_[1-9]*.pth 2>/dev/null | tail -1)
[ -n "$LATEST" ] && echo "  최신 체크포인트에서 재개: $(basename "$LATEST")" || echo "  G_0(Epoch70 기반)에서 처음부터"
( cd "$SBV2" && ../sbv2_env/Scripts/python.exe _run_noworker.py train_ms_jp_extra.py \
    --config Data/SOL/config.json --model Data/SOL \
    --assets_root model_assets --not_use_custom_batch_sampler ) \
    > tts/logs/phase3_sol_ko_finetune.log 2>&1
echo "[$(date '+%F %T')] 파인튜닝 종료 (exit $?)"
