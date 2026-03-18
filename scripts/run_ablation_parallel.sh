#!/usr/bin/env bash
# Run sparsification experiments in parallel waves grouped by dataset size.
# Each dataset writes its own results/<dataset>_results.csv — no file conflicts.
#
# Usage:
#   caffeinate -s bash scripts/run_parallel.sh > results/run_log.txt 2>&1 &
#
# Per-dataset logs: results/logs/<dataset>.log

set -uo pipefail
cd "$(dirname "$0")/.."

LOGDIR="results/logs"
mkdir -p "$LOGDIR"

run_dataset() {
    local ds=$1
    echo "[$(date '+%H:%M:%S')] START  $ds"
    python -u scripts/run_ablation.py --dataset "$ds" \
        > "$LOGDIR/${ds}.log" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE   $ds"
    else
        echo "[$(date '+%H:%M:%S')] FAILED $ds (exit $status) — see $LOGDIR/${ds}.log"
    fi
}

echo "=== Run started at $(date) ==="

# Wave 1 — tiny WebKB (< 200 nodes): all 3 in parallel
echo "--- Wave 1: cornell texas wisconsin ---"
run_dataset cornell &
run_dataset texas &
run_dataset wisconsin &
wait
echo "--- Wave 1 complete ---"

# Wave 2 — small citation graphs (~2–3k nodes): 2 in parallel
echo "--- Wave 2: cora citeseer ---"
run_dataset cora &
run_dataset citeseer &
wait
echo "--- Wave 2 complete ---"

# Wave 3 — medium graphs: 2 in parallel
echo "--- Wave 3: cora_ml actor ---"
run_dataset cora_ml &
run_dataset actor &
wait
echo "--- Wave 3 complete ---"

# Wave 4 — large graphs: sequential to avoid memory pressure
for ds in chameleon pubmed squirrel dblp; do
    echo "--- Wave 4: $ds ---"
    run_dataset "$ds"
done

echo "=== All experiments complete at $(date) ==="
