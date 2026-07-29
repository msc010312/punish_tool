#!/usr/bin/env bash
# overnight_run.sh — JP 솔 파인튜닝 → KO 기반모델 학습을 순차로 밤새 돌린다.
#
# GPU는 한 번에 하나라 직렬로 진행한다. 각 단계는 eval_interval(1000스텝)마다 체크포인트를
# 저장하므로 중간에 멈춰도 다음 세션에서 이어서 학습할 수 있다.
#
# 사전조건(이 스크립트 실행 전 이미 완료됨):
#   - SOL_JP: stage/preprocess 완료(bert/style 포함)
#   - base_ko: stage/resample/preprocess_text 완료(train.list/val.list/config.json 존재)
#
# 사용: bash src/overnight_run.sh
set -u
cd "$(dirname "$0")/.."   # 프로젝트 루트
ROOT="$(pwd)"
SBV2="$ROOT/tts/sbv2"
VENV_PY="$ROOT/tts/sbv2_env/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY="py -3.11"
LOG_DIR="tts/logs"
mkdir -p "$LOG_DIR"
ts() { date "+%F %T"; }

# _run_noworker 래퍼로 SBV2 스크립트 실행(shim 3종 적용).
sbv2() { ( cd "$SBV2" && "$VENV_PY" _run_noworker.py "$@" ); }

echo "[$(ts)] ===== 밤샘 학습 시작 ====="

# ── Phase 1: JP 솔 파인튜닝 (데이터 전처리 완료됨) ──────────────────────
echo "[$(ts)] Phase 1: JP 솔 파인튜닝 (100 epochs)"
$PY src/tts_train.py train --profile sol_jp --epochs 100 --batch 4 \
    > "$LOG_DIR/phase1_jp_train.log" 2>&1
echo "[$(ts)] Phase 1 종료 (exit $?)"

# ── Phase 2: KO 기반모델 남은 전처리(bert/style) + 학습 ─────────────────
CFG="Data/base_ko/config.json"
if [ -f "$SBV2/$CFG" ]; then
    echo "[$(ts)] Phase 2a: base_ko BERT 특징 생성"
    sbv2 bert_gen.py -c "$CFG" > "$LOG_DIR/phase2a_bert.log" 2>&1
    echo "[$(ts)]   bert_gen 종료 (exit $?)"

    echo "[$(ts)] Phase 2b: base_ko 스타일 벡터 생성"
    sbv2 style_gen.py -c "$CFG" --num_processes 2 > "$LOG_DIR/phase2b_style.log" 2>&1
    echo "[$(ts)]   style_gen 종료 (exit $?)"

    echo "[$(ts)] Phase 2c: 사전학습 스테이징 + 웜스타트 주입"
    # initialize가 하던 일(pretrained 복사)을 직접: models/에 JP-Extra 사전학습 3개 복사
    mkdir -p "$SBV2/Data/base_ko/models"
    cp "$SBV2/pretrained_jp_extra/D_0.safetensors" "$SBV2/Data/base_ko/models/" 2>/dev/null
    cp "$SBV2/pretrained_jp_extra/WD_0.safetensors" "$SBV2/Data/base_ko/models/" 2>/dev/null
    cp "$SBV2/pretrained_ko/G_0.safetensors" "$SBV2/Data/base_ko/models/" 2>/dev/null
    echo "[$(ts)]   웜스타트 G_0(125심볼) + D_0/WD_0 주입 완료"

    echo "[$(ts)] Phase 2d: base_ko 기반모델 학습 (200 epochs 상한, 체크포인트 자동저장)"
    sbv2 train_ms_jp_extra.py --config "$CFG" --model Data/base_ko \
        --assets_root model_assets --not_use_custom_batch_sampler \
        > "$LOG_DIR/phase2d_ko_base_train.log" 2>&1
    echo "[$(ts)]   base_ko 학습 종료 (exit $?)"
else
    echo "[$(ts)] Phase 2 건너뜀: $CFG 없음"
fi

echo "[$(ts)] ===== 밤샘 학습 종료 ====="
