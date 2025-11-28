# Architecture Documentation

## System Architecture

The GNN Sparsification Research framework follows a modular, component-based architecture that enables easy experimentation with different graph sparsification methods.

```
┌─────────────────────────────────────────────────────────┐
│                    Hydra Config Layer                    │
│  (configs/)                                              │
│  ├─ experiment/  ├─ model/  ├─ dataset/  ├─ sparsifier/ │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Training Script                       │
│  (scripts/train.py)                                      │
│  - Experiment orchestration                              │
│  - W&B integration                                       │
│  - Training loop                                         │
└────┬───────────────┬──────────────┬─────────────────────┘
     │               │              │
     ▼               ▼              ▼
┌─────────┐   ┌──────────┐   ┌─────────────┐
│  Data   │   │  Model   │   │Sparsification│
│ Loaders │   │ Classes  │   │   Methods    │
└─────────┘   └──────────┘   └─────────────┘
```

## Core Components

### 1. Sparsification Module (`src/sparsification/`)

**Base Class: `BaseSparsifier`**
- Abstract interface for all sparsification methods
- Enforces consistent API across implementations
- Provides utility methods for statistics

**Concrete Implementations:**
- `RandomSparsifier`: Baseline random edge removal
- `SpectralSparsifier`: Spectral graph theory-based methods
- `MetricSparsifier`: Similarity metric-based (Jaccard, Cosine, Euclidean)

**Key Design Principles:**
- All sparsifiers operate on `torch_geometric.data.Data` objects
- Sparsification is deterministic when seed is provided
- Edge attributes and masks are preserved when applicable

### 2. Model Module (`src/models/`)

**Planned Models:**
- `GCN`: Graph Convolutional Network
- `GAT`: Graph Attention Network
- Extensible to other PyG-compatible architectures

**Design Patterns:**
- All models inherit from `torch.nn.Module`
- Configurable via Hydra YAML files
- Support for variable depth and hidden dimensions

### 3. Data Module (`src/data/`)

**Components:**
- `loader.py`: Dataset loading and splitting utilities
- Support for Planetoid datasets (Cora, CiteSeer, PubMed)
- Extensible to custom datasets

### 4. Training Module (`src/training/`)

**Responsibilities:**
- Training loop implementation
- Optimization strategies
- Learning rate scheduling
- Early stopping logic

### 5. Utilities Module (`src/utils/`)

**Components:**
- `metrics.py`: Evaluation metrics (accuracy, F1, etc.)
- `misc.py`: Reproducibility utilities (seed setting)
- Logging and visualization helpers

## Configuration System

### Hydra Composition

The project uses Hydra for hierarchical configuration:

```yaml
defaults:
  - dataset: cora
  - model: gcn
  - sparsifier: random
  - _self_
```

**Benefits:**
- CLI overrides: `python train.py model=gat`
- Experiment reproducibility
- No hard-coded hyperparameters
- Easy A/B testing

### Configuration Structure

```
configs/
├── config.yaml           # Main entry point
├── model/
│   ├── gcn.yaml
│   └── gat.yaml
├── dataset/
│   ├── cora.yaml
│   └── pubmed.yaml
├── sparsifier/
│   ├── random.yaml
│   ├── spectral.yaml
│   ├── jaccard.yaml
│   └── cosine.yaml
└── experiment/
    ├── baseline.yaml
    └── jaccard_cora.yaml
```

## Data Flow

### Training Pipeline

1. **Configuration Loading**
   - Hydra loads and composes configs
   - CLI overrides applied

2. **Data Loading**
   - Dataset loaded from `data/raw/`
   - Cached in `data/processed/` for speed

3. **Sparsification**
   - Original graph passed to sparsifier
   - Sparsified graph created
   - Statistics logged

4. **Model Initialization**
   - Model created based on config
   - Optimizer and scheduler initialized

5. **Training Loop**
   - Forward pass
   - Loss computation
   - Backward pass
   - Metrics logged to W&B

6. **Evaluation**
   - Best model loaded
   - Test set evaluation
   - Results saved

### Sparsification Pipeline

```python
# Input: Data(x, edge_index, y)
data = load_dataset(config)

# Sparsification
sparsifier = create_sparsifier(config)
sparse_data = sparsifier.sparsify(data)

# Output: Data(x, edge_index', y)
# where edge_index' ⊂ edge_index
```

## Extensibility Points

### Adding a New Sparsifier

1. Create `src/sparsification/your_method.py`:
```python
from .base import BaseSparsifier

class YourSparsifier(BaseSparsifier):
    def sparsify(self, data: Data) -> Data:
        # Your implementation
        pass
```

2. Create `configs/sparsifier/your_method.yaml`:
```yaml
name: your_method
class: YourSparsifier
sparsification_ratio: 0.5
# Your parameters
```

3. Use it: `python train.py sparsifier=your_method`

### Adding a New Model

1. Create `src/models/your_model.py`
2. Create `configs/model/your_model.yaml`
3. Update model factory in `scripts/train.py`

### Adding a New Dataset

1. Implement loader in `src/data/loader.py`
2. Create `configs/dataset/your_dataset.yaml`
3. Update dataset factory in `scripts/train.py`

## Testing Strategy

### Unit Tests (`tests/`)

- Test individual components in isolation
- Focus on sparsification logic correctness
- Verify edge case handling

### Integration Tests (TODO)

- Test full pipeline end-to-end
- Verify config composition
- Check W&B logging

### Property-Based Tests (TODO)

- Verify sparsification properties:
  - Edge count reduction
  - Node preservation
  - Connectivity preservation (optional)

## Reproducibility

### Seed Management

Seeds are set at multiple levels:
1. Python `random` module
2. NumPy random generator
3. PyTorch random generator
4. CUDA random generator (if available)

### Deterministic Operations

When `deterministic=True`:
- CUDNN deterministic mode enabled
- Benchmark mode disabled
- Trade-off: slower but reproducible

### Experiment Tracking

- All hyperparameters logged to W&B
- Git commit hash logged (optional)
- Environment info captured

## Performance Considerations

### Caching

- Processed datasets cached in `data/processed/`
- Sparsified graphs can be cached (optional)

### GPU Utilization

- Automatic device detection
- Fallback to CPU if no CUDA
- Batched operations where possible

### Memory Management

- Sparse graph storage using COO format
- Gradient checkpointing for large models (optional)

## Future Enhancements

### Planned Features

- [ ] Multi-GPU training support
- [ ] Graph-level tasks (classification)
- [ ] Link prediction evaluation
- [ ] Adversarial sparsification
- [ ] Automated hyperparameter search
- [ ] Checkpoint averaging
- [ ] TensorBoard integration

### Research Directions

- Compare sparsification impact across model architectures
- Study spectral properties preservation
- Analyze computational cost vs. accuracy trade-offs
- Investigate optimal sparsification ratios per dataset
