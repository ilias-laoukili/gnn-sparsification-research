#!/usr/bin/env bash
# Synthetic HPO Robustness — controlled sweeps.
#
# PURPOSE: verify that dense yield (not h itself) is the causal driver of
#          robustness gain, and identify confounders (mu, theta).
#
# SWEEP 1a — h as the IV, GCN, all method families (12 h × 13 methods × 3 seeds = 468 jobs)
#   Fix: n=1000, c=5, d=10, mu=1.0, f=64
#   Vary: h ∈ {0.05,0.10,...,0.90,0.95}
#   Methods (threshold):      jaccard, approx_er, adamic_adar, feature_cosine, random
#   Methods (inv threshold):  jaccard_inv, approx_er_inv, adamic_adar_inv
#   Methods (metric backbone): metric_backbone_jaccard, metric_backbone_approx_er,
#                               metric_backbone_adamic_adar
#   Methods (degree-aware):   degree_aware_jaccard, degree_aware_approx_er
#   Methods (sampled):        sampled_jaccard, sampled_approx_er
#
# SWEEP 1b — h as the IV, architecture comparison (12 h × 2 arch × 3 methods × 3 seeds = 216)
#   Fix: same as 1a
#   Architectures: graphsage, gat  (GCN results come from Sweep 1a)
#   Methods: jaccard, approx_er, random
#   Expected: compare acc_ratio vs h curves across GCN/GraphSAGE/GAT
#
# SWEEP 2 — mu as confounder (5 mu-values × 5 seeds × 4 methods = 100 jobs)
#   Fix: h=0.10, n=1000, c=5, d=10, f=64
#   Vary: mu ∈ {0.25, 0.50, 1.00, 2.00, 4.00}
#   Methods: jaccard, approx_er, adamic_adar, feature_cosine, random
#   mu=1.00 excluded: same params as h=0.10 in Sweep 1a (already cached)
#
# SWEEP 3 — theta (degree heterogeneity) as confounder (3 theta × 3 seeds × 3 methods = 27)
#   Fix: h=0.10, mu=1.0, n=1000, c=5, d=10, f=64
#   Vary: theta_exp ∈ {2.0, 3.0, 5.0}  (Pareto exponent; lower = more heterogeneous)
#   Methods: jaccard, approx_er, adamic_adar, random
#
# Estimated runtime per job on Apple M4 MPS: ~4–6 min (n=1000, 50 HPs × 3 seeds × 10 rates)
# Total with max ~39 parallel: Sweep 1a ~4-6h, Sweep 1b ~2-3h, Sweeps 2+3 ~1-2h
#
# Usage:
#   caffeinate -s bash scripts/run_synthetic_sweeps.sh > results/logs/synthetic_sweeps.log 2>&1 &
#   caffeinate -s bash scripts/run_synthetic_sweeps.sh --resume > results/logs/synthetic_sweeps.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

# Use the project venv Python (has synth_graph_rs, torch-geometric, etc.)
PYTHON="$(pwd)/.venv/bin/python"

RESUME_FLAG="${1:-}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR" "results/hpo"

GRAPH_SEEDS="0 1 2"

run_synth() {
    local h=$1; local mu=$2; local metric=$3; local gs=$4
    local extra="${5:-}"   # optional extra args (e.g. --theta_exp 3.0 or --arch graphsage)
    local tag="h${h}_mu${mu}_gs${gs}${extra:+_$(echo "$extra" | tr ' =' '_' | tr -d '-')}"
    local logfile="$LOGDIR/synth_${tag}_${metric}.log"
    echo "[$(date '+%H:%M:%S')] START  h=$h mu=$mu gs=$gs $extra [$metric]"
    # 90-min timeout guards against MPS deadlocks on very sparse graphs (use gtimeout on macOS)
    gtimeout 5400 "$PYTHON" -u scripts/run_synthetic_hpo.py \
        --h "$h" --mu "$mu" --graph_seed "$gs" --metric "$metric" \
        $extra $RESUME_FLAG \
        > "$logfile" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE   h=$h mu=$mu gs=$gs $extra [$metric]"
    elif [ $status -eq 124 ]; then
        echo "[$(date '+%H:%M:%S')] TIMEOUT h=$h mu=$mu gs=$gs $extra [$metric] — see $logfile"
    else
        echo "[$(date '+%H:%M:%S')] FAILED h=$h mu=$mu gs=$gs $extra [$metric] — see $logfile"
    fi
}

echo "=== Synthetic HPO sweeps started at $(date) ==="
echo "=== Resume flag: '${RESUME_FLAG}' ==="

# ── Sweep 1a: h sweep, GCN, all method families ───────────────────────────────
echo ""
echo "--- Sweep 1a: h sweep — GCN, all method families ---"
H_VALUES="0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95"

# Standard threshold methods
S1A_THRESHOLD="jaccard approx_er adamic_adar feature_cosine random"
# Inverse threshold (keep lowest-scoring edges — heterophilous graphs)
S1A_INVERSE="jaccard_inv approx_er_inv adamic_adar_inv"
# Metric backbone (self-determining retention)
S1A_BACKBONE="metric_backbone_jaccard metric_backbone_approx_er metric_backbone_adamic_adar"
# Degree-aware threshold
S1A_DEGREE_AWARE="degree_aware_jaccard degree_aware_approx_er"
# Probabilistic sampling
S1A_SAMPLED="sampled_jaccard sampled_approx_er"

# All GCN methods
S1A_ALL_METHODS="$S1A_THRESHOLD $S1A_INVERSE $S1A_BACKBONE $S1A_DEGREE_AWARE $S1A_SAMPLED"

# Concurrency: 2 methods × 3 seeds = 6 concurrent Python processes per batch.
# Doubles throughput vs 3-serial while staying well within RAM limits (~3-4 GB).
for h in $H_VALUES; do
    echo "--- Sweep 1a: h=$h ---"
    _batch=0
    for metric in $S1A_ALL_METHODS; do
        for gs in $GRAPH_SEEDS; do
            run_synth "$h" "1.0" "$metric" "$gs" &
        done
        _batch=$((_batch + 1))
        if [ $((_batch % 2)) -eq 0 ]; then
            wait  # wait after every 2 methods (6 concurrent)
        fi
    done
    wait  # catch any remaining odd method
    echo "--- Sweep 1a: h=$h done ---"
done
echo "--- Sweep 1a complete ---"

# ── Sweep 1b: h sweep, architecture comparison (GraphSAGE and GAT) ────────────
echo ""
echo "--- Sweep 1b: h sweep — architecture comparison (graphsage, gat) ---"
# Core methods only (GCN results already from Sweep 1a)
S1B_METHODS="jaccard approx_er random"
S1B_ARCHS="graphsage gat"

for h in $H_VALUES; do
    echo "--- Sweep 1b: h=$h ---"
    _batch=0
    for arch in $S1B_ARCHS; do
        for metric in $S1B_METHODS; do
            for gs in $GRAPH_SEEDS; do
                run_synth "$h" "1.0" "$metric" "$gs" "--arch $arch" &
            done
            _batch=$((_batch + 1))
            if [ $((_batch % 2)) -eq 0 ]; then
                wait
            fi
        done
    done
    wait
    echo "--- Sweep 1b: h=$h done ---"
done
echo "--- Sweep 1b complete ---"

# ── Sweep 2: mu sweep (confounder check) ──────────────────────────────────────
echo ""
echo "--- Sweep 2: mu sweep at h=0.10 (feature signal confounder) ---"
# mu=1.00 excluded: same params as h=0.10 Sweep 1a (already cached with --resume)
MU_VALUES="0.25 0.50 2.00 4.00"
S2_METHODS="jaccard approx_er adamic_adar feature_cosine random"

for mu in $MU_VALUES; do
    echo "--- Sweep 2: mu=$mu ---"
    _batch=0
    for metric in $S2_METHODS; do
        for gs in $GRAPH_SEEDS; do
            run_synth "0.10" "$mu" "$metric" "$gs" &
        done
        _batch=$((_batch + 1))
        if [ $((_batch % 2)) -eq 0 ]; then
            wait
        fi
    done
    wait
    echo "--- Sweep 2: mu=$mu done ---"
done
echo "--- Sweep 2 complete ---"

# ── Sweep 3: theta sweep (degree heterogeneity) ───────────────────────────────
echo ""
echo "--- Sweep 3: theta sweep at h=0.10, mu=1.0 (degree heterogeneity confounder) ---"
THETA_VALUES="2.0 3.0 5.0"
S3_METHODS="jaccard approx_er adamic_adar random"

for te in $THETA_VALUES; do
    echo "--- Sweep 3: theta_exp=$te ---"
    _batch=0
    for metric in $S3_METHODS; do
        for gs in $GRAPH_SEEDS; do
            run_synth "0.10" "1.0" "$metric" "$gs" "--theta_exp $te" &
        done
        _batch=$((_batch + 1))
        if [ $((_batch % 2)) -eq 0 ]; then
            wait
        fi
    done
    wait
    echo "--- Sweep 3: theta_exp=$te done ---"
done
echo "--- Sweep 3 complete ---"

echo ""
echo "=== All synthetic sweeps complete at $(date) ==="
echo "=== Results in: results/hpo/synth_*.json ==="
