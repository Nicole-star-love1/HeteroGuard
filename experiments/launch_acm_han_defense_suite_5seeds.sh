#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_acm_han_defense_suite"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON="--dataset ACM --model HAN --epochs 200 \
--output_dir ${OUT_DIR} \
--defenses none,prune,isolate,retraining,hr,unlearn \
--min_weight 0.0 --max_downweight 1.0 \
--unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2 \
--save_results"

for SEED in 1 2 3 4 5; do
  echo "============================================================"
  echo "[SEED ${SEED}] Feature | defense suite"
  echo "============================================================"
  python -m experiments.run_defense_suite \
    ${COMMON} \
    --attack feature \
    --poison_rate 0.1 \
    --trigger_size 10 \
    --trigger_strength 3.0 \
    --detection_ratio 0.1 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/suite_feature_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] SBA | defense suite"
  echo "============================================================"
  python -m experiments.run_defense_suite \
    ${COMMON} \
    --attack sba \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/suite_sba_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] UBA | defense suite"
  echo "============================================================"
  python -m experiments.run_defense_suite \
    ${COMMON} \
    --attack uba \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/suite_uba_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] Relation | defense suite"
  echo "============================================================"
  python -m experiments.run_defense_suite \
    ${COMMON} \
    --attack relation \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/suite_relation_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] CleanLabel | defense suite"
  echo "============================================================"
  python -m experiments.run_defense_suite \
    ${COMMON} \
    --attack clean_label \
    --poison_rate 0.2 \
    --trigger_size 10 \
    --detection_ratio 0.2 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/suite_clean_label_seed${SEED}.log"

  echo "============================================================"
  echo "[SEED ${SEED}] CBA | defense suite"
  echo "============================================================"
  python -m experiments.run_defense_suite \
    ${COMMON} \
    --attack cba \
    --poison_rate 0.075 \
    --trigger_size 12 \
    --target_feature_strength 1.5 \
    --aux_feature_strength 3.5 \
    --detection_ratio 0.075 \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/suite_cba_seed${SEED}.log"
done

python tools/aggregate_defense_suite.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished defense suite."
echo "Outputs:"
echo "  ${OUT_DIR}/tables/defense_suite_all_runs.csv"
echo "  ${OUT_DIR}/tables/defense_suite_mean_std.csv"
echo "  ${OUT_DIR}/tables/defense_suite_latex.tex"
echo "============================================================"
