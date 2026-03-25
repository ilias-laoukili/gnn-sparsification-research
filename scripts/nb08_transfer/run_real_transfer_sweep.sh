#!/usr/bin/env bash
# Real-dataset HPO Transfer sweep v2: all splits.
#
# 7 datasets × 3 metrics × {1 or 10} splits = 129 jobs total.
# Planetoid (cora, citeseer, pubmed): 1 split (fixed semi-supervised)
# WebKB (cornell, texas, wisconsin) + Actor: 10 splits each
#
# Run 3 jobs in parallel (each job is ~3-10 min).

set -euo pipefail
cd "$(dirname "$0")/../.."

METRICS="jaccard approx_er random"
PARALLEL=3

# Datasets with number of splits
declare -A SPLITS
SPLITS[cora]=1
SPLITS[citeseer]=1
SPLITS[pubmed]=1
SPLITS[actor]=10
SPLITS[cornell]=10
SPLITS[texas]=10
SPLITS[wisconsin]=10

TOTAL=0
DONE=0

# Count total jobs
for ds in "${!SPLITS[@]}"; do
    n_splits=${SPLITS[$ds]}
    for met in $METRICS; do
        for ((s=0; s<n_splits; s++)); do
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "=== Real Transfer Sweep v2 ==="
echo "Total jobs: $TOTAL  |  Parallel: $PARALLEL"
echo ""

# Run jobs
for ds in cora citeseer pubmed actor cornell texas wisconsin; do
    n_splits=${SPLITS[$ds]}
    for met in $METRICS; do
        for ((s=0; s<n_splits; s++)); do
            outfile="results/hpo_transfer/real_${ds}_s${s}_transfer_${met}.json"
            if [ -f "$outfile" ]; then
                DONE=$((DONE + 1))
                echo "[SKIP] $ds split=$s $met  ($DONE/$TOTAL)"
                continue
            fi

            # Wait if too many background jobs
            while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
                sleep 5
            done

            DONE=$((DONE + 1))
            echo "[RUN]  $ds split=$s $met  ($DONE/$TOTAL)"
            python scripts/nb08_transfer/run_real_transfer.py \
                --dataset "$ds" --metric "$met" --split_idx "$s" \
                > "results/hpo_transfer/log_${ds}_s${s}_${met}.txt" 2>&1 &
        done
    done
done

# Wait for all background jobs
wait
echo ""
echo "=== All $TOTAL jobs complete ==="
