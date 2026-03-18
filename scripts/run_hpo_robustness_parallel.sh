#!/usr/bin/env bash
# HPO Robustness experiment — parallel waves grouped by dataset size.
#
# Runs all nine sparsification methods for each dataset:
#   Standard (Threshold — keep highest-scoring edges):
#     jaccard, adamic_adar, approx_er, feature_cosine
#   InverseThreshold (keep lowest-scoring edges):
#     jaccard_inv, adamic_adar_inv, approx_er_inv, feature_cosine_inv
#   Control:
#     random — uniform random edge removal (density-only baseline)
#
# Each (dataset, metric) pair writes its own results/hpo/<dataset>_hpo_<metric>.json,
# so there are no file conflicts between parallel processes.
#
# Estimated wall time per metric per dataset (Apple M4 MPS, 100 HP configs × 3 seeds
# × 5 retention rates = 1500 training runs):
#   Wave 1 (cornell/texas/wisconsin): ~35 min   (1.3s/run)
#   Wave 2 (cora/citeseer):          ~60 min   (2.0s/run)
#   Wave 3 (actor/cora_ml):          ~2.1h     (4.5s/run)
#   Wave 4 (pubmed):                 ~8h       (19s/run)  ← run overnight
#   NOT included: squirrel/dblp (>10h each) → use Kaggle if needed
#
# Concurrency notes:
#   Waves 1–3: all dataset×method pairs run in parallel within each wave.
#   Wave 4 (pubmed): methods run in pairs to avoid MPS thrashing.
#   approx_er scores are computed once before the HP loop,
#   so CG solve overhead (~few minutes per dataset) is a one-time cost.
#
# Memory note:
#   Wave 1: 3 datasets × 9 methods = 27 concurrent Python processes.
#   Wave 2: 2 × 9 = 18 processes.  Wave 3: 2 × 9 = 18 processes.
#   Each process uses ~200 MB RAM overhead → ~5 GB for Wave 1.
#   Reduce parallelism here if the machine runs out of memory.
#
# Usage:
#   caffeinate -s bash scripts/run_hpo_robustness_parallel.sh > results/logs/hpo_run.log 2>&1 &
#
# To resume after interruption (skips retention rates already in JSON):
#   caffeinate -s bash scripts/run_hpo_robustness_parallel.sh --resume > results/logs/hpo_run.log 2>&1 &
#
# Per-dataset logs: results/logs/hpo_<dataset>_<metric>.log

set -uo pipefail
cd "$(dirname "$0")/.."

# Use the project venv Python (has torch-geometric, etc.)
PYTHON="$(pwd)/.venv/bin/python"

RESUME_FLAG="${1:-}"   # pass --resume as first arg to propagate it
LOGDIR="results/logs"
mkdir -p "$LOGDIR" "results/hpo"

# 4 standard + 4 inverse threshold + 1 random control = 9 methods total
METHODS="jaccard adamic_adar approx_er feature_cosine jaccard_inv adamic_adar_inv approx_er_inv feature_cosine_inv random"

run_dataset() {
    local ds=$1
    local metric=$2
    echo "[$(date '+%H:%M:%S')] START  $ds  [$metric]"
    "$PYTHON" -u scripts/run_hpo_robustness.py --dataset "$ds" --metric "$metric" $RESUME_FLAG \
        > "$LOGDIR/hpo_${ds}_${metric}.log" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE   $ds  [$metric]"
    else
        echo "[$(date '+%H:%M:%S')] FAILED $ds [$metric] (exit $status) — see $LOGDIR/hpo_${ds}_${metric}.log"
    fi
}

echo "=== HPO Robustness run started at $(date) ==="
echo "=== Resume flag: '${RESUME_FLAG}' ==="
echo "=== Methods (nb04 SMART_METRICS + random): ${METHODS} ==="

# ── Wave 1 — tiny WebKB (~300 nodes, ~300 edges) ─────────────────────────────
# Run one dataset at a time with all 9 methods in parallel → 9 concurrent max.
# (Previously 27 concurrent caused RAM exhaustion on MPS.)
echo "--- Wave 1: cornell  texas  wisconsin  (9 methods each, one dataset at a time) ---"
for ds in cornell texas wisconsin; do
    echo "--- Wave 1: $ds ---"
    for metric in $METHODS; do
        run_dataset "$ds" "$metric" &
    done
    wait
done
echo "--- Wave 1 complete ---"

# ── Wave 2 — small citation graphs (~2–3k nodes, ~10k edges) ─────────────────
echo "--- Wave 2: cora  citeseer  (9 methods each, one dataset at a time) ---"
for ds in cora citeseer; do
    echo "--- Wave 2: $ds ---"
    for metric in $METHODS; do
        run_dataset "$ds" "$metric" &
    done
    wait
done
echo "--- Wave 2 complete ---"

# ── Wave 3 — medium graphs (~8k nodes, ~30k edges) ───────────────────────────
echo "--- Wave 3: actor  cora_ml  (9 methods each, one dataset at a time) ---"
for ds in actor cora_ml; do
    echo "--- Wave 3: $ds ---"
    for metric in $METHODS; do
        run_dataset "$ds" "$metric" &
    done
    wait
done
echo "--- Wave 3 complete ---"

# ── Wave 4 — pubmed: run in pairs to avoid MPS thrashing ─────────────────────
# pubmed: ~20k nodes, ~88k edges → ~8h per metric → run overnight.
# Running all 9 in parallel would thrash MPS; pairs (threshold+inverse) give
# natural grouping and control over memory.
echo "--- Wave 4: pubmed (9 methods in pairs) ---"
run_dataset pubmed jaccard &
run_dataset pubmed jaccard_inv &
wait
run_dataset pubmed adamic_adar &
run_dataset pubmed adamic_adar_inv &
wait
run_dataset pubmed approx_er &
run_dataset pubmed approx_er_inv &
wait
run_dataset pubmed feature_cosine &
run_dataset pubmed feature_cosine_inv &
wait
run_dataset pubmed random
echo "--- Wave 4 complete ---"

echo "=== All HPO robustness experiments complete at $(date) ==="
