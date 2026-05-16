#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 4: Ablation study
#
# Dataset: ACM
# Model: HAN
# Attacks: uba, relation, clean_label, cba
# Seeds: 1..5
# Variants: all
#
# Total:
#   4 attacks × 5 seeds × 10 variants = 200 defense trainings
#
# Usage:
#   bash experiments/launch_ablation_acm_han_5seeds.sh
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_ablation_acm_han_5seeds"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON="--dataset ACM --model HAN --epochs 200 \
--output_dir ${OUT_DIR} \
--variants all \
--min_weight 0.0 --max_downweight 1.0 \
--unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2 \
--save_results"

SEEDS=(1 2 3 4 5)

run_one() {
  local SEED="$1"
  local ATTACK="$2"
  shift 2
  local EXTRA_ARGS="$*"

  echo "============================================================"
  echo "[ABLATION] ACM HAN ${ATTACK} seed=${SEED}"
  echo "============================================================"

  python -m experiments.run_ablation_suite \
    ${COMMON} \
    --attack "${ATTACK}" \
    ${EXTRA_ARGS} \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/ablation_${ATTACK}_seed${SEED}.log"
}

for SEED in "${SEEDS[@]}"; do
  run_one "${SEED}" "uba" \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2

  run_one "${SEED}" "relation" \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2

  run_one "${SEED}" "clean_label" \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2

  run_one "${SEED}" "cba" \
    --poison_rate 0.075 \
    --trigger_size 12 \
    --target_feature_strength 1.5 \
    --aux_feature_strength 3.5 \
    --detection_ratio 0.075
done

python tools/aggregate_ablation.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished Experiment 4 ablation."
echo "Outputs:"
echo "  ${OUT_DIR}/tables/ablation_all_runs.csv"
echo "  ${OUT_DIR}/tables/ablation_mean_std.csv"
echo "  ${OUT_DIR}/tables/ablation_latex.tex"
echo "============================================================"
