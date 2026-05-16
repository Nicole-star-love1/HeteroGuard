#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 6: Robustness analysis
#
# Dataset: ACM
# Model: HAN
# Defense: HeteroGuard-Unlearn
# Seeds: 1..5
#
# Part 1: inaccurate detection ratio
#   detection_ratio = poison_rate × {0.5, 0.75, 1.0, 1.25, 1.5}
#   Attacks: relation, uba, cba
#
# Part 2: partial detection failure
#   drop detected suspicious nodes by {0, 0.1, 0.2, 0.3, 0.4}
#   Attacks: relation, uba, cba
#
# Part 3: adaptive attacks
#   low_feature_relation
#   sparse_relation
#   stealthy_clean_label
#
# Total:
#   75 + 75 + 15 = 165 runs
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_robustness_acm_han_5seeds"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON="--dataset ACM --model HAN --epochs 200 \
--output_dir ${OUT_DIR} \
--min_weight 0.0 --max_downweight 1.0 \
--unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2 \
--save_results"

SEEDS=(1 2 3 4 5)
RATIO_FACTORS=(0.5 0.75 1.0 1.25 1.5)
DROP_RATIOS=(0 0.1 0.2 0.3 0.4)

run_one() {
  local NAME="$1"
  local SEED="$2"
  shift 2

  echo "============================================================"
  echo "[ROBUST] ${NAME} seed=${SEED}"
  echo "============================================================"

  python -m experiments.run_robustness_suite \
    ${COMMON} \
    "$@" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/${NAME}_seed${SEED}.log"
}

# Helper: detection_ratio = poison_rate * factor.
ratio_value() {
  python - "$1" "$2" <<'PY'
import sys
p=float(sys.argv[1])
f=float(sys.argv[2])
print(f"{p*f:.6f}")
PY
}

# ------------------------------------------------------------------
# Part 1: inaccurate detection ratio
# ------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  for F in "${RATIO_FACTORS[@]}"; do
    DR=$(ratio_value 0.2 "${F}")
    run_one "ratio_relation_factor${F}" "${SEED}" \
      --robustness_group inaccurate_detection_ratio \
      --robustness_value "${F}" \
      --attack relation \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio "${DR}"

    run_one "ratio_uba_factor${F}" "${SEED}" \
      --robustness_group inaccurate_detection_ratio \
      --robustness_value "${F}" \
      --attack uba \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio "${DR}"

    DR_CBA=$(ratio_value 0.075 "${F}")
    run_one "ratio_cba_factor${F}" "${SEED}" \
      --robustness_group inaccurate_detection_ratio \
      --robustness_value "${F}" \
      --attack cba \
      --poison_rate 0.075 \
      --trigger_size 12 \
      --target_feature_strength 1.5 \
      --aux_feature_strength 3.5 \
      --detection_ratio "${DR_CBA}"
  done
done

# ------------------------------------------------------------------
# Part 2: partial detection failure
# ------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  for D in "${DROP_RATIOS[@]}"; do
    run_one "partial_relation_drop${D}" "${SEED}" \
      --robustness_group partial_detection_failure \
      --robustness_value "${D}" \
      --drop_detected_ratio "${D}" \
      --attack relation \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio 0.2

    run_one "partial_uba_drop${D}" "${SEED}" \
      --robustness_group partial_detection_failure \
      --robustness_value "${D}" \
      --drop_detected_ratio "${D}" \
      --attack uba \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio 0.2

    run_one "partial_cba_drop${D}" "${SEED}" \
      --robustness_group partial_detection_failure \
      --robustness_value "${D}" \
      --drop_detected_ratio "${D}" \
      --attack cba \
      --poison_rate 0.075 \
      --trigger_size 12 \
      --target_feature_strength 1.5 \
      --aux_feature_strength 3.5 \
      --detection_ratio 0.075
  done
done

# ------------------------------------------------------------------
# Part 3: adaptive attacks
# ------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  run_one "adaptive_low_feature_relation" "${SEED}" \
    --robustness_group adaptive_attack \
    --robustness_value low_feature \
    --adaptive_variant low_feature_relation \
    --attack relation \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --target_feature_strength 1.0 \
    --aux_feature_strength 1.0

  run_one "adaptive_sparse_relation" "${SEED}" \
    --robustness_group adaptive_attack \
    --robustness_value sparse_relation \
    --adaptive_variant sparse_relation \
    --attack relation \
    --poison_rate 0.2 \
    --trigger_size 4 \
    --detection_ratio 0.2 \
    --target_feature_strength 2.0 \
    --aux_feature_strength 2.0 \
    --no_aux_clique

  run_one "adaptive_stealthy_clean_label" "${SEED}" \
    --robustness_group adaptive_attack \
    --robustness_value stealthy_clean_label \
    --adaptive_variant stealthy_clean_label \
    --attack clean_label \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --target_feature_strength 1.5 \
    --aux_feature_strength 2.0
done

python tools/aggregate_robustness.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished Experiment 6 robustness."
echo "Outputs:"
echo "  ${OUT_DIR}/tables/robustness_all_runs.csv"
echo "  ${OUT_DIR}/tables/robustness_mean_std.csv"
echo "  ${OUT_DIR}/tables/robustness_latex.tex"
echo "============================================================"
