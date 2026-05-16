#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 2: Cross-dataset generalization
# Datasets: ACM, DBLP, IMDB, Freebase
# Model: HAN
# Attacks: feature, sba, uba, relation, clean_label, cba
# Defenses: NoDefense, HeteroGuard-HR, HeteroGuard-Unlearn
# Seeds: 1..5
#
# Usage:
#   cd /home/HeterogeneousGraphTasks
#   chmod +x experiments/launch_han_cross_dataset_5seeds.sh
#   bash experiments/launch_han_cross_dataset_5seeds.sh
#
# Outputs:
#   ./results_han_cross_dataset_5seeds/
#   ./results_han_cross_dataset_5seeds/tables/
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_han_cross_dataset_5seeds"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON="--model HAN --epochs 200 \
--output_dir ${OUT_DIR} \
--defenses none,hr,unlearn \
--min_weight 0.0 --max_downweight 1.0 \
--unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2 \
--save_results"

DATASETS=("ACM" "DBLP" "IMDB" "Freebase")
SEEDS=(1 2 3 4 5)

run_one() {
  local DATASET="$1"
  local SEED="$2"
  local ATTACK="$3"
  shift 3
  local EXTRA_ARGS="$*"

  echo "============================================================"
  echo "[${DATASET}] [SEED ${SEED}] [${ATTACK}]"
  echo "============================================================"

  python -m experiments.run_defense_suite \
    ${COMMON} \
    --dataset "${DATASET}" \
    --attack "${ATTACK}" \
    ${EXTRA_ARGS} \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/${DATASET}_HAN_${ATTACK}_seed${SEED}.log"
}

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    # FeatureAttack: lower poison rate to keep the attack reasonably stealthy.
    run_one "${DATASET}" "${SEED}" "feature" \
      --poison_rate 0.1 \
      --trigger_size 10 \
      --trigger_strength 3.0 \
      --detection_ratio 0.1

    # SBA-Hybrid.
    run_one "${DATASET}" "${SEED}" "sba" \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio 0.2

    # UBA-Hybrid.
    run_one "${DATASET}" "${SEED}" "uba" \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio 0.2

    # Relation-Hybrid.
    run_one "${DATASET}" "${SEED}" "relation" \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio 0.2

    # CleanLabel-Hybrid.
    run_one "${DATASET}" "${SEED}" "clean_label" \
      --poison_rate 0.2 \
      --trigger_size 10 \
      --detection_ratio 0.2

    # CBA-Hybrid.
    # This is tuned to be less destructive than the strong version.
    run_one "${DATASET}" "${SEED}" "cba" \
      --poison_rate 0.075 \
      --trigger_size 12 \
      --target_feature_strength 1.5 \
      --aux_feature_strength 3.5 \
      --detection_ratio 0.075

  done
done

echo "============================================================"
echo "Aggregating cross-dataset defense suite results..."
echo "============================================================"

python tools/aggregate_defense_suite.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished Experiment 2."
echo "Main outputs:"
echo "  ${OUT_DIR}/tables/defense_suite_all_runs.csv"
echo "  ${OUT_DIR}/tables/defense_suite_mean_std.csv"
echo "  ${OUT_DIR}/tables/defense_suite_latex.tex"
echo "Logs:"
echo "  ${LOG_DIR}/"
echo "============================================================"
