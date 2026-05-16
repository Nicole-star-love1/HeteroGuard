#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ACM + HAN | Six Attacks | 5 Seeds
# Defense: HeteroGuard-Unlearn
#
# Usage:
#   cd /home/HeterogeneousGraphTasks
#   bash experiments/launch_acm_han_5seeds.sh
#
# Outputs:
#   ./results_acm_han_5seeds/
#   ./results_acm_han_5seeds/tables/
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_acm_han_5seeds"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS="\
--dataset ACM \
--model HAN \
--epochs 200 \
--run_defense \
--output_dir ${OUT_DIR} \
--min_weight 0.0 \
--max_downweight 1.0 \
--hard_remove_suspicious \
--use_trigger_unlearning \
--unlearn_lambda 1.0 \
--unlearn_samples 256 \
--target_suppression 0.2 \
--save_results"

echo "============================================================"
echo "ACM + HAN | Six attacks | 5 seeds"
echo "Output dir: ${OUT_DIR}"
echo "============================================================"

for SEED in 1 2 3 4 5; do
  echo "============================================================"
  echo "[SEED ${SEED}] FeatureAttack"
  echo "============================================================"
  python -m experiments.run_integrated \
    ${COMMON_ARGS} \
    --attack feature \
    --poison_rate 0.1 \
    --trigger_size 10 \
    --trigger_strength 3.0 \
    --detection_ratio 0.1 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/ACM_HAN_feature_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] SBA-Hybrid"
  echo "============================================================"
  python -m experiments.run_integrated \
    ${COMMON_ARGS} \
    --attack sba \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/ACM_HAN_sba_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] UBA-Hybrid"
  echo "============================================================"
  python -m experiments.run_integrated \
    ${COMMON_ARGS} \
    --attack uba \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/ACM_HAN_uba_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] Relation-Hybrid"
  echo "============================================================"
  python -m experiments.run_integrated \
    ${COMMON_ARGS} \
    --attack relation \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/ACM_HAN_relation_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] CleanLabel-Hybrid"
  echo "============================================================"
  python -m experiments.run_integrated \
    ${COMMON_ARGS} \
    --attack clean_label \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/ACM_HAN_clean_label_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] CBA-Hybrid"
  echo "============================================================"
  python -m experiments.run_integrated \
    ${COMMON_ARGS} \
    --attack cba \
    --poison_rate 0.075 \
    --trigger_size 12 \
    --target_feature_strength 1.5 \
    --aux_feature_strength 3.5 \
    --detection_ratio 0.075 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/ACM_HAN_cba_seed${SEED}.log"
done

echo "============================================================"
echo "Validating result JSON schema..."
echo "============================================================"
python tools/validate_result_schema.py \
  --results_dir "${OUT_DIR}"

echo "============================================================"
echo "Aggregating results..."
echo "============================================================"
python tools/aggregate_results.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished."
echo "Main outputs:"
echo "  ${OUT_DIR}/tables/all_runs_flat.csv"
echo "  ${OUT_DIR}/tables/main_table_mean_std.csv"
echo "  ${OUT_DIR}/tables/main_table_latex.tex"
echo "  ${OUT_DIR}/tables/attack_validity.csv"
echo "  ${OUT_DIR}/tables/defense_effectiveness.csv"
echo "============================================================"