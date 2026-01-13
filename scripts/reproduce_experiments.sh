#!/bin/bash
# Reproduces all experiments from paper
# Ensures reproducibility across all datasets and methods

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "================================================"
echo "GNN Sparsification Research - Experiment Runner"
echo "================================================"
echo ""

# Set reproducibility flags
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Configuration
SEEDS=(0 1 2 3 4)
DATASETS=(cora pubmed flickr)
METHODS=(jaccard adamic_adar effective_resistance random)

# Parse command line arguments
DATASET_FILTER=""
METHOD_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        --method)
            METHOD_FILTER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET    Run only specified dataset (cora, pubmed, flickr)"
            echo "  --method METHOD      Run only specified method (jaccard, adamic_adar, etc.)"
            echo "  --help              Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Filter datasets if specified
if [[ -n "$DATASET_FILTER" ]]; then
    DATASETS=("$DATASET_FILTER")
    echo "Running only dataset: $DATASET_FILTER"
fi

# Filter methods if specified
if [[ -n "$METHOD_FILTER" ]]; then
    METHODS=("$METHOD_FILTER")
    echo "Running only method: $METHOD_FILTER"
fi

echo "Starting experiments..."
echo "Datasets: ${DATASETS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo ""

# Create results directory
RESULTS_DIR="results/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Run experiments
TOTAL_RUNS=$((${#DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]}))
CURRENT_RUN=0

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $dataset / $method / seed=$seed"
            
            # Run experiment
            python scripts/train.py \
                --dataset "$dataset" \
                --method "$method" \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR" \
                --config configs/experiment/baseline.yaml \
                2>&1 | tee -a "$RESULTS_DIR/experiment_${dataset}_${method}_seed${seed}.log"
            
            # Check if run succeeded
            if [ $? -eq 0 ]; then
                echo "  ✓ Completed successfully"
            else
                echo "  ✗ Failed (check logs)"
            fi
            echo ""
        done
    done
done

echo ""
echo "================================================"
echo "All experiments completed!"
echo "Results saved to: $RESULTS_DIR"
echo "================================================"
echo ""

# Generate summary statistics
echo "Generating summary statistics..."
python scripts/analyze_results.py \
    --input_dir "$RESULTS_DIR" \
    --output_file "$RESULTS_DIR/summary_statistics.csv"

echo ""
echo "Summary statistics saved to: $RESULTS_DIR/summary_statistics.csv"
echo ""

# Generate plots (if requested)
if command -v python &> /dev/null; then
    echo "Generating plots..."
    python scripts/plot_results.py \
        --input_file "$RESULTS_DIR/summary_statistics.csv" \
        --output_dir "$RESULTS_DIR/figures"
    
    echo "Plots saved to: $RESULTS_DIR/figures"
fi

echo ""
echo "Done! 🎉"
