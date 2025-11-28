# Quick Reference Card

## 📌 Essential Commands

### Setup & Verification
```bash
./setup.sh                           # Run complete setup
python scripts/verify_setup.py       # Verify installation
python scripts/demo_sparsification.py # Test sparsification
```

### Testing
```bash
pytest tests/                        # Run all tests
pytest tests/ -v                     # Verbose output
pytest --cov=src tests/              # With coverage
make test                            # Using Makefile
```

### Training (After model implementation)
```bash
# Basic
python scripts/train.py

# Override components
python scripts/train.py model=gat
python scripts/train.py dataset=pubmed
python scripts/train.py sparsifier=jaccard

# Override parameters
python scripts/train.py sparsifier.sparsification_ratio=0.3
python scripts/train.py training.learning_rate=0.001

# Combine multiple overrides
python scripts/train.py model=gat dataset=pubmed sparsifier=jaccard \
    sparsifier.sparsification_ratio=0.4 training.learning_rate=0.005

# Use experiment presets
python scripts/train.py experiment=baseline
python scripts/train.py experiment=jaccard_cora
```

### Code Quality
```bash
make format        # Format with black & isort
make lint          # Check with flake8
make clean         # Clean cache files
```

## 📁 Key Files & Directories

### Configuration
- `configs/config.yaml` - Main configuration
- `configs/model/` - Model architectures
- `configs/dataset/` - Dataset configurations
- `configs/sparsifier/` - Sparsification methods
- `configs/experiment/` - Experiment presets

### Source Code
- `src/sparsification/base.py` - Abstract base class
- `src/sparsification/random.py` - Random baseline
- `src/sparsification/metric.py` - Metric-based methods
- `src/sparsification/spectral.py` - Spectral methods
- `src/models/gcn.py` - GCN implementation (TODO)
- `src/models/gat.py` - GAT implementation (TODO)

### Scripts
- `scripts/train.py` - Main training script
- `scripts/demo_sparsification.py` - Demo script
- `scripts/verify_setup.py` - Setup verification

### Documentation
- `README.md` - Project overview
- `docs/QUICKSTART.md` - Getting started
- `docs/MIGRATION.md` - Migration guide
- `docs/ARCHITECTURE.md` - System design
- `REFACTORING_SUMMARY.md` - Complete refactoring details

## 🔧 Configuration Patterns

### Override Model
```yaml
# Command line
python scripts/train.py model=gat

# Or create configs/model/my_model.yaml:
name: my_model
architecture:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.5
```

### Override Dataset
```yaml
# Command line
python scripts/train.py dataset=pubmed

# Or create configs/dataset/my_dataset.yaml:
name: my_dataset
dataset_class: Planetoid
split: public
```

### Override Sparsifier
```yaml
# Command line
python scripts/train.py sparsifier=jaccard

# Or create configs/sparsifier/my_sparsifier.yaml:
name: my_method
class: MetricSparsifier
sparsification_ratio: 0.5
metric: jaccard
```

### Create Experiment Preset
```yaml
# configs/experiment/my_experiment.yaml
# @package _global_

defaults:
  - override /dataset: cora
  - override /model: gcn
  - override /sparsifier: jaccard

experiment:
  name: my_experiment_name
  description: "Description here"

sparsifier:
  sparsification_ratio: 0.4

training:
  learning_rate: 0.005
```

## 🐍 Python API Examples

### Using Sparsifiers
```python
from torch_geometric.datasets import Planetoid
from src.sparsification.random import RandomSparsifier
from src.sparsification.metric import MetricSparsifier

# Load dataset
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]

# Random sparsification
sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
sparse_data = sparsifier.sparsify(data)

# Jaccard sparsification
sparsifier = MetricSparsifier(
    sparsification_ratio=0.5,
    metric='jaccard'
)
sparse_data = sparsifier.sparsify(data)

# Get statistics
stats = sparsifier.get_sparsification_stats(data, sparse_data)
print(stats)
```

### Creating Custom Sparsifier
```python
from src.sparsification.base import BaseSparsifier
from torch_geometric.data import Data

class MySparsifier(BaseSparsifier):
    def sparsify(self, data: Data) -> Data:
        # Your implementation here
        return sparse_data
```

## 📊 Project Statistics

- **38** configuration and code files created
- **~3000** lines of production code
- **~1500** lines of documentation
- **4** sparsification methods implemented
- **2** model architectures templated
- **2** dataset configurations
- **2** experiment presets

## 🚀 Next Implementation Steps

1. **Complete GCN Model** (`src/models/gcn.py`)
   - Implement `__init__` with GCNConv layers
   - Implement `forward` with ReLU and dropout
   - Test with small dataset

2. **Complete Training Loop** (`scripts/train.py`)
   - Uncomment `create_model()`
   - Uncomment `train_epoch()`
   - Uncomment `evaluate()`
   - Test end-to-end

3. **Enable W&B Logging**
   - Uncomment `wandb.init()`
   - Uncomment `wandb.log()` calls
   - Set `wandb_entity` in config

4. **Run First Experiment**
   ```bash
   python scripts/train.py experiment=baseline
   ```

## 🆘 Troubleshooting

### Import Errors
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Package Not Found
```bash
# Install requirements
pip install -r requirements.txt

# Check virtual environment
which python  # Should show venv/bin/python
```

### Hydra Config Errors
```bash
# Check config syntax
python scripts/train.py --cfg job

# Print full config
python scripts/train.py --cfg all
```

### CUDA Errors
```bash
# Force CPU
python scripts/train.py experiment.device=cpu

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 Learning Resources

### Internal Documentation
- Start with `docs/QUICKSTART.md`
- Deep dive in `docs/ARCHITECTURE.md`
- Migration details in `docs/MIGRATION.md`

### External Resources
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Weights & Biases](https://docs.wandb.ai/)

## ✅ Verification Checklist

- [ ] Run `./setup.sh` successfully
- [ ] Run `python scripts/verify_setup.py` - all checks pass
- [ ] Run `python scripts/demo_sparsification.py` - works
- [ ] Run `pytest tests/` - all tests pass
- [ ] Implement GCN model
- [ ] Complete training loop
- [ ] Run first experiment
- [ ] Enable W&B logging
- [ ] Document results in notebooks

---

**Keep this card handy for quick reference during development!**
