#!/usr/bin/env bash
# MLP Control Sweep — validate GNN(sparse) > MLP on heterophilous cSBM.
#
# Mirrors the h-sweep from Sweep 1 of run_synthetic_sweeps.sh.
# Each job = one (h, graph_seed, arch) pair; trains conditions inside it:
#   {arch}_full, {arch}_sparse × 3 r-values, {arch}_random × 3 r-values, mlp
#
# All jobs write to the single shared results/mlp_control/mlp_control.csv
# (file-locked inside the Python script — safe for parallel writes).
#
# Architecture sweep:
#   gcn (default), graphsage, gat — each × 12 h-values × 3 seeds = 108 jobs
#
# Total: 3 archs × 12 h-values × 3 seeds = 108 jobs
# Per job: 8 conditions × 20 HP × 3 seeds = 480 training runs (f=512, MPS ~8-15 min/job)
# Total runtime: ~8-12h with 3 parallel jobs per h-value (one h at a time per arch group)
#
# Usage:
#   caffeinate -s bash scripts/nb07_synthetic/run_mlp_control_sweeps.sh \
#       > results/logs/mlp_control.log 2>&1 &
#
#   # Resume interrupted run:
#   caffeinate -s bash scripts/nb07_synthetic/run_mlp_control_sweeps.sh --resume \
#       > results/logs/mlp_control.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/../.."

PYTHON="$(pwd)/.venv/bin/python"
RESUME_FLAG="${1:-}"
LOGDIR="results/logs"
FINAL_CSV="$(pwd)/results/mlp_control/mlp_control.csv"
mkdir -p "$LOGDIR" "results/mlp_control" "/tmp/mlp_control"

# Write to /tmp/ during the run to avoid cloud-sync race conditions (ProtonDrive
# can restore older file versions when many processes append simultaneously).
export MLP_CSV_PATH="/tmp/mlp_control/mlp_control.csv"

# If resuming and the final CSV already has data, seed /tmp/ with it so the
# resume logic can detect already-completed rows.
if [ -n "$RESUME_FLAG" ] && [ -f "$FINAL_CSV" ]; then
    cp "$FINAL_CSV" "$MLP_CSV_PATH"
    echo "=== Seeded /tmp from ${FINAL_CSV} ($(wc -l < "$FINAL_CSV") lines) ==="
fi

# Same h-values as Sweep 1 for direct comparability
H_VALUES="0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95"
GRAPH_SEEDS="0 1 2"
ARCHS="gcn graphsage gat"

run_mlp_control() {
    local h=$1
    local gs=$2
    local arch=$3
    local logfile="$LOGDIR/mlp_control_${arch}_h${h}_gs${gs}.log"
    echo "[$(date '+%H:%M:%S')] START  arch=$arch  h=$h  gs=$gs"
    "$PYTHON" -u scripts/nb07_synthetic/run_mlp_control_sweep.py \
        --h "$h" --graph_seed "$gs" --arch "$arch" $RESUME_FLAG \
        > "$logfile" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE   arch=$arch  h=$h  gs=$gs"
    else
        echo "[$(date '+%H:%M:%S')] FAILED arch=$arch  h=$h  gs=$gs — see $logfile"
    fi
}

echo "=== MLP Control Sweep started at $(date) ==="
echo "=== Resume flag: '${RESUME_FLAG}' ==="
echo "=== Architectures: $ARCHS ==="
echo ""

for arch in $ARCHS; do
    echo "=== Architecture: $arch ==="
    for h in $H_VALUES; do
        echo "--- arch=$arch  h=$h ---"
        for gs in $GRAPH_SEEDS; do
            run_mlp_control "$h" "$gs" "$arch" &
        done
        wait
        echo "--- arch=$arch  h=$h done ---"

        # Quick progress check after each h-value completes
        if [ -f "$MLP_CSV_PATH" ]; then
            n_rows=$(( $(wc -l < "$MLP_CSV_PATH") - 1 ))
            echo "    CSV rows so far: $n_rows"
        fi
    done
    echo "=== Architecture $arch complete ==="
    echo ""
done

echo "=== MLP Control Sweep complete at $(date) ==="
echo "=== Copying results from /tmp/ to ${FINAL_CSV} ==="
cp "$MLP_CSV_PATH" "$FINAL_CSV"
echo "=== Results in: ${FINAL_CSV} ($(wc -l < "$FINAL_CSV") lines) ==="
