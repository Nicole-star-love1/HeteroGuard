#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 5: Parameter sensitivity
#
# Dataset: ACM
# Model: HAN
# Defense: HeteroGuard-Unlearn
# Seeds: 1..5
#
# Groups:
#   poison_rate        ∈ {0.05, 0.075, 0.10, 0.15, 0.20}
#   trigger_size       ∈ {4, 8, 10, 12, 16}
#   unlearn_lambda     ∈ {0, 0.25, 0.5, 1.0, 2.0}
#   target_suppression ∈ {0, 0.1, 0.2, 0.5, 1.0}
#
# Attacks:
#   relation, uba, cba
#
# Total:
#   4 groups × 3 attacks × 5 values × 5 seeds = 300 runs
#
# This script explicitly passes --sensitivity_group and --param_value so default
# sweep points are not incorrectly grouped as "base".
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_sensitivity_acm_han_5seeds"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON="--dataset ACM --model HAN --epochs 200 \
--output_dir ${OUT_DIR} \
--defenses unlearn \
--min_weight 0.0 --max_downweight 1.0 \
--save_results"

SEEDS=(1 2 3 4 5)
POISON_RATES=(0.05 0.075 0.10 0.15 0.20)
TRIGGER_SIZES=(4 8 10 12 16)
UNLEARN_LAMBDAS=(0 0.25 0.5 1.0 2.0)
TARGET_SUPPRESSIONS=(0 0.1 0.2 0.5 1.0)

run_one() {
  local GROUP="$1"
  local PARAM_NAME="$2"
  local PARAM_VALUE="$3"
  local SEED="$4"
  local ATTACK="$5"
  shift 5
  local EXTRA_ARGS="$*"

  echo "============================================================"
  echo "[SENS] group=${GROUP} ${PARAM_NAME}=${PARAM_VALUE} attack=${ATTACK} seed=${SEED}"
  echo "============================================================"

  python -m experiments.run_defense_suite \
    ${COMMON} \
    --sensitivity_group "${GROUP}" \
    --param_value "${PARAM_VALUE}" \
    --attack "${ATTACK}" \
    ${EXTRA_ARGS} \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/${GROUP}_${ATTACK}_${PARAM_NAME}${PARAM_VALUE}_seed${SEED}.log"
}

# poison_rate
for SEED in "${SEEDS[@]}"; do
  for R in "${POISON_RATES[@]}"; do
    run_one "poison_rate" "poison_rate" "${R}" "${SEED}" "relation" \
      --poison_rate "${R}" --trigger_size 10 --detection_ratio "${R}" \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2

    run_one "poison_rate" "poison_rate" "${R}" "${SEED}" "uba" \
      --poison_rate "${R}" --trigger_size 10 --detection_ratio "${R}" \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2

    run_one "poison_rate" "poison_rate" "${R}" "${SEED}" "cba" \
      --poison_rate "${R}" --trigger_size 12 \
      --target_feature_strength 1.5 --aux_feature_strength 3.5 \
      --detection_ratio "${R}" \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2
  done
done

# trigger_size
for SEED in "${SEEDS[@]}"; do
  for T in "${TRIGGER_SIZES[@]}"; do
    run_one "trigger_size" "trigger_size" "${T}" "${SEED}" "relation" \
      --poison_rate 0.2 --trigger_size "${T}" --detection_ratio 0.2 \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2

    run_one "trigger_size" "trigger_size" "${T}" "${SEED}" "uba" \
      --poison_rate 0.2 --trigger_size "${T}" --detection_ratio 0.2 \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2

    run_one "trigger_size" "trigger_size" "${T}" "${SEED}" "cba" \
      --poison_rate 0.075 --trigger_size "${T}" \
      --target_feature_strength 1.5 --aux_feature_strength 3.5 \
      --detection_ratio 0.075 \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2
  done
done

# unlearn_lambda
for SEED in "${SEEDS[@]}"; do
  for L in "${UNLEARN_LAMBDAS[@]}"; do
    run_one "unlearn_lambda" "unlearn_lambda" "${L}" "${SEED}" "relation" \
      --poison_rate 0.2 --trigger_size 10 --detection_ratio 0.2 \
      --unlearn_lambda "${L}" --unlearn_samples 256 --target_suppression 0.2

    run_one "unlearn_lambda" "unlearn_lambda" "${L}" "${SEED}" "uba" \
      --poison_rate 0.2 --trigger_size 10 --detection_ratio 0.2 \
      --unlearn_lambda "${L}" --unlearn_samples 256 --target_suppression 0.2

    run_one "unlearn_lambda" "unlearn_lambda" "${L}" "${SEED}" "cba" \
      --poison_rate 0.075 --trigger_size 12 \
      --target_feature_strength 1.5 --aux_feature_strength 3.5 \
      --detection_ratio 0.075 \
      --unlearn_lambda "${L}" --unlearn_samples 256 --target_suppression 0.2
  done
done

# target_suppression
for SEED in "${SEEDS[@]}"; do
  for B in "${TARGET_SUPPRESSIONS[@]}"; do
    run_one "target_suppression" "target_suppression" "${B}" "${SEED}" "relation" \
      --poison_rate 0.2 --trigger_size 10 --detection_ratio 0.2 \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression "${B}"

    run_one "target_suppression" "target_suppression" "${B}" "${SEED}" "uba" \
      --poison_rate 0.2 --trigger_size 10 --detection_ratio 0.2 \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression "${B}"

    run_one "target_suppression" "target_suppression" "${B}" "${SEED}" "cba" \
      --poison_rate 0.075 --trigger_size 12 \
      --target_feature_strength 1.5 --aux_feature_strength 3.5 \
      --detection_ratio 0.075 \
      --unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression "${B}"
  done
done

python tools/aggregate_sensitivity.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished Experiment 5 sensitivity."
echo "Outputs:"
echo "  ${OUT_DIR}/tables/sensitivity_all_runs.csv"
echo "  ${OUT_DIR}/tables/sensitivity_mean_std.csv"
echo "  ${OUT_DIR}/tables/sensitivity_latex.tex"
echo "============================================================"
