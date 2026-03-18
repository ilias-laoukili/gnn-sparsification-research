#!/usr/bin/env bash
# HPO Transfer v2 — parallel launcher (3-condition design).
#
# 7 datasets × 9 methods. Each (dataset, metric) pair runs:
#   oracle  : 20 TPE trials on full graph  (once per pair)
#   proxy   : 20 TPE trials on sparse graph (× 4 retention rates)
#   baseline: 20 RandomSampler trials on sparse graph (× 4 retention rates)
#
# All three conditions use val-acc objective; evaluated on full graph (3 seeds).
# Primary metric: normalized_proxy = (tpe_acc - rnd_acc) / (oracle_acc - rnd_acc).
#
# Model: FlexibleGCN (GCNConv, binary adjacency, 5D HP space).
#
# Estimated wall time per (dataset, metric) — Apple M4 MPS:
#   Wave 1 (cornell/texas/wisconsin): ~5–8 min per pair
#   Wave 2 (cora/citeseer/chameleon): ~15–25 min per pair
#   Wave 3 (actor):                  ~20–30 min per pair
#   Grand total (7 datasets, 9 methods parallel): ~2–2.5h
#
# Usage:
#   caffeinate -s bash scripts/run_hpo_transfer.sh          # fresh run
#   caffeinate -s bash scripts/run_hpo_transfer.sh --resume # resume after interruption
#
# Per-pair logs: results/logs/transfer_<dataset>_<metric>.log

set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="$(pwd)/.venv/bin/python"
RESUME_FLAG="${1:-}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR" "results/hpo_transfer"

METHODS="jaccard adamic_adar approx_er feature_cosine jaccard_inv adamic_adar_inv approx_er_inv feature_cosine_inv random"

run_pair() {
    local ds=$1
    local metric=$2
    echo "[$(date '+%H:%M:%S')] START  $ds  [$metric]"
    "$PYTHON" -u scripts/run_hpo_transfer.py \
        --dataset "$ds" --metric "$metric" $RESUME_FLAG \
        > "$LOGDIR/transfer_${ds}_${metric}.log" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE   $ds  [$metric]"
    else
        echo "[$(date '+%H:%M:%S')] FAILED $ds  [$metric] (exit $status)" \
             "— see $LOGDIR/transfer_${ds}_${metric}.log"
    fi
}

echo "=== HPO Transfer v2 started at $(date) ==="
echo "=== Resume flag: '${RESUME_FLAG}' ==="
echo "=== Methods: ${METHODS} ==="

# ── Wave 1 — WebKB datasets (~5–8 min/pair) ───────────────────────────────────
for ds in cornell texas wisconsin; do
    echo "--- Wave 1: $ds (9 methods in parallel) ---"
    for metric in $METHODS; do
        run_pair "$ds" "$metric" &
    done
    wait
    echo "--- Wave 1: $ds complete ---"
done

# ── Wave 2 — Medium graphs (~15–25 min/pair) ──────────────────────────────────
for ds in cora citeseer chameleon; do
    echo "--- Wave 2: $ds (9 methods in parallel) ---"
    for metric in $METHODS; do
        run_pair "$ds" "$metric" &
    done
    wait
    echo "--- Wave 2: $ds complete ---"
done

# ── Wave 3 — Actor (~20–30 min/pair) ──────────────────────────────────────────
echo "--- Wave 3: actor (9 methods in parallel) ---"
for metric in $METHODS; do
    run_pair "actor" "$metric" &
done
wait
echo "--- Wave 3: actor complete ---"

echo "=== All HPO Transfer v2 experiments complete at $(date) ==="
