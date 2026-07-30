#!/usr/bin/env bash
# ko_base_train.sh — KO 기반모델 학습 시작/재개 (중단해도 이 명령으로 이어서 계속).
#
# train_ms_jp_extra는 Data/base_ko/models/의 최신 G_*.pth를 자동으로 찾아 그 지점부터
# 재개한다. config의 eval_interval(500스텝)마다 체크포인트를 저장하므로, 언제 멈춰도
# 최대 ~5분치만 손실된다.
#
# 멈추기: 이 프로세스를 종료(Ctrl+C 또는 작업관리자에서 python.exe). 다음에 이 스크립트를
#         다시 실행하면 마지막 체크포인트에서 이어서 학습한다.
#
# 사용: bash src/ko_base_train.sh
cd "$(dirname "$0")/.."
SBV2="tts/sbv2"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
# 12GB GPU에서 가변 길이 배치가 카싱 할당자를 단편화시켜 수십 스텝 후 near-OOM 스래싱을
# 일으킨다(GPU 99%인데 60W대·무진행). expandable_segments가 이 단편화를 해소한다.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p tts/logs
echo "[$(date '+%F %T')] KO 기반모델 학습 시작/재개"
LATEST=$(ls -1 "$SBV2/Data/base_ko/models/"G_*.pth 2>/dev/null | tail -1)
[ -n "$LATEST" ] && echo "  최신 체크포인트에서 재개: $(basename "$LATEST")" || echo "  체크포인트 없음 → 웜스타트 G_0에서 처음부터"
( cd "$SBV2" && ../sbv2_env/Scripts/python.exe _run_noworker.py train_ms_jp_extra.py \
    --config Data/base_ko/config.json --model Data/base_ko \
    --assets_root model_assets --not_use_custom_batch_sampler ) \
    > tts/logs/phase2_ko_base_train.log 2>&1
echo "[$(date '+%F %T')] 학습 종료 (exit $?)"
