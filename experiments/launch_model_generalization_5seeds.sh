#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 3: Model generalization
#
# Datasets: ACM, DBLP
# Models: HAN, HGT, RGCN, HeteroSAGE
# Attacks: relation, uba, clean_label
# Defense: HeteroGuard-Unlearn
# Seeds: 1..5
#
# Total:
#   2 datasets × 4 models × 3 attacks × 5 seeds = 120 runs
#
# If Experiment 2 is still using GPU 0 and a second GPU is available:
#   CUDA_VISIBLE_DEVICES=1 bash experiments/launch_model_generalization_5seeds.sh
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_model_generalization_5seeds"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON="--epochs 200 \
--output_dir ${OUT_DIR} \
--defenses unlearn \
--min_weight 0.0 --max_downweight 1.0 \
--unlearn_lambda 1.0 --unlearn_samples 256 --target_suppression 0.2 \
--save_results"

DATASETS=("ACM" "DBLP")
MODELS=("HAN" "HGT" "RGCN" "HeteroSAGE")
SEEDS=(1 2 3 4 5)

run_one() {
  local DATASET="$1"
  local MODEL="$2"
  local SEED="$3"
  local ATTACK="$4"
  shift 4
  local EXTRA_ARGS="$*"

  echo "============================================================"
  echo "[DATASET ${DATASET}] [MODEL ${MODEL}] [SEED ${SEED}] [ATTACK ${ATTACK}]"
  echo "============================================================"

  python -m experiments.run_defense_suite \
    ${COMMON} \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --attack "${ATTACK}" \
    ${EXTRA_ARGS} \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/${DATASET}_${MODEL}_${ATTACK}_seed${SEED}.log"
}

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      run_one "${DATASET}" "${MODEL}" "${SEED}" "relation" \
        --poison_rate 0.2 \
        --trigger_size 10 \
        --detection_ratio 0.2

      run_one "${DATASET}" "${MODEL}" "${SEED}" "uba" \
        --poison_rate 0.2 \
        --trigger_size 10 \
        --detection_ratio 0.2

      run_one "${DATASET}" "${MODEL}" "${SEED}" "clean_label" \
        --poison_rate 0.2 \
        --trigger_size 10 \
        --detection_ratio 0.2

    done
  done
done

echo "============================================================"
echo "Aggregating model generalization results..."
echo "============================================================"

python tools/aggregate_defense_suite.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished Experiment 3."
echo "Main outputs:"
echo "  ${OUT_DIR}/tables/defense_suite_all_runs.csv"
echo "  ${OUT_DIR}/tables/defense_suite_mean_std.csv"
echo "  ${OUT_DIR}/tables/defense_suite_latex.tex"
echo "Logs:"
echo "  ${LOG_DIR}/"
echo "============================================================"
