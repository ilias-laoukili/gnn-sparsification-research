#!/usr/bin/env bash
# Sweep: 12 homophily × 3 seeds × 3 metrics = 108 jobs
# 3 concurrent (one per seed), 2h timeout per job.
# Usage: caffeinate -s bash scripts/nb08_transfer/run_synthetic_transfer_sweep.sh
#        caffeinate -s bash scripts/nb08_transfer/run_synthetic_transfer_sweep.sh --resume

set -uo pipefail
# NOTE: -e intentionally omitted — we track failures manually so one
# failed job doesn't abort the entire sweep.
cd "$(dirname "$0")/../.."

RESUME_FLAG=""
[[ "${1:-}" == "--resume" ]] && RESUME_FLAG="--resume"

VENV=".venv/bin/python"
SCRIPT="scripts/nb08_transfer/run_synthetic_transfer.py"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR" results/hpo_transfer

H_VALUES=(0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95)
SEEDS=(0 1 2)
METRICS=(jaccard approx_er random)

TOTAL=$(( ${#H_VALUES[@]} * ${#SEEDS[@]} * ${#METRICS[@]} ))
DONE=0
FAIL=0
echo "=== Synthetic Transfer sweep started at $(date) ==="
echo "=== Resume flag: '$RESUME_FLAG' ==="
echo "=== Jobs: ${#H_VALUES[@]} h × ${#SEEDS[@]} seeds × ${#METRICS[@]} metrics = $TOTAL ==="
echo ""

for h in "${H_VALUES[@]}"; do
    echo "--- h=$h ---"
    for metric in "${METRICS[@]}"; do
        PIDS=()
        for gs in "${SEEDS[@]}"; do
            LOG="$LOG_DIR/synth_transfer_h${h}_gs${gs}_${metric}.log"
            echo "[$(date +%H:%M:%S)] START  h=$h gs=$gs [$metric]"
            gtimeout 7200 "$VENV" -u "$SCRIPT" \
                --h "$h" --graph_seed "$gs" --metric "$metric" $RESUME_FLAG \
                > "$LOG" 2>&1 &
            PIDS+=($!)
        done

        for i in "${!PIDS[@]}"; do
            gs="${SEEDS[$i]}"
            if wait "${PIDS[$i]}"; then
                echo "[$(date +%H:%M:%S)] DONE   h=$h gs=$gs [$metric]"
                DONE=$((DONE + 1))
            else
                echo "[$(date +%H:%M:%S)] FAILED h=$h gs=$gs [$metric]"
                FAIL=$((FAIL + 1))
            fi
        done
    done
    echo "--- h=$h done ($DONE/$TOTAL done, $FAIL failures) ---"
    echo ""
done

echo "=== Synthetic Transfer sweep completed at $(date) ==="
echo "=== Results: $DONE completed, $FAIL failed, $TOTAL total ==="
