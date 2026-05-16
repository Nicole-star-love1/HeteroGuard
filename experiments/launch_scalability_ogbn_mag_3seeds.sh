#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment 7: Scalability study
#
# Dataset: OGBN-MAG
# Model: HeteroSAGE by default. You may add HGT if resources allow.
# Attacks: relation, uba
# Defenses: HeteroGuard-HR, HeteroGuard-Unlearn
# Seeds: 1, 2, 3
#
# Main metrics:
#   Defense ASR
#   Defense Clean Acc
#   Defense time
#   Peak GPU memory
#   Number of processed nodes / edges
#
# Note:
#   node_budget_per_type=50000 uses a large induced OGBN-MAG subgraph.
#   Set --node_budget_per_type 0 in COMMON for full graph if your server can
#   handle full-batch training on OGBN-MAG.
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_DIR="./results_scalability_ogbn_mag_3seeds"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

COMMON="--dataset OGBN-MAG \
--epochs 100 \
--pretrain_epochs 30 \
--defense_epochs 50 \
--output_dir ${OUT_DIR} \
--node_budget_per_type 50000 \
--poison_rate 0.05 \
--trigger_size 5 \
--detection_ratio 0.05 \
--unlearn_lambda 1.0 \
--unlearn_samples 512 \
--target_suppression 0.2 \
--num_inject 500 \
--defenses hr,unlearn \
--save_results"

# Default: HeteroSAGE is the safer scalable backbone.
# To also test HGT, change MODELS=("HeteroSAGE" "HGT").
MODELS=("HeteroSAGE")
ATTACKS=("relation" "uba")
SEEDS=(1 2 3)

run_one() {
  local MODEL="$1"
  local ATTACK="$2"
  local SEED="$3"

  echo "============================================================"
  echo "[SCALABILITY] OGBN-MAG ${MODEL} ${ATTACK} seed=${SEED}"
  echo "============================================================"

  python -m experiments.run_scalability_suite \
    ${COMMON} \
    --model "${MODEL}" \
    --attack "${ATTACK}" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/OGBNMAG_${MODEL}_${ATTACK}_seed${SEED}.log"
}

for MODEL in "${MODELS[@]}"; do
  for ATTACK in "${ATTACKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_one "${MODEL}" "${ATTACK}" "${SEED}"
    done
  done
done

python tools/aggregate_scalability.py \
  --results_dir "${OUT_DIR}" \
  --out_dir "${OUT_DIR}/tables"

echo "============================================================"
echo "Finished Experiment 7 scalability."
echo "Outputs:"
echo "  ${OUT_DIR}/tables/scalability_all_runs.csv"
echo "  ${OUT_DIR}/tables/scalability_mean_std.csv"
echo "  ${OUT_DIR}/tables/scalability_latex.tex"
echo "============================================================"
