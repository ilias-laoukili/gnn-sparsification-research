# Contributing to GNN Sparsification Research

Thank you for considering contributing to this research project! This document provides guidelines for contributing to the codebase.

## 🎯 Project Philosophy

This is a **research-oriented** project, not a production system. The priorities are:

1. **Scientific Validity** - Correctness of implementations
2. **Reproducibility** - Consistent results across runs
3. **Clarity** - Well-documented code and experiments
4. **Performance** - Efficient implementations for large-scale experiments

## 🚀 Getting Started

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/gnn-sparsification-research.git
cd gnn-sparsification-research

# Create conda environment
conda env create -f environment.yml
conda activate gnn-sparsification

# Install in editable mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black pylint jupyter
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_sparsification.py -v

# Run tests in parallel
pytest tests/ -n auto
```

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format all code
black src/ tests/ scripts/

# Check formatting without changes
black --check src/ tests/
```

## 📝 Contribution Guidelines

### 1. Code Style

- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Docstrings**: Use NumPy-style docstrings
- **Type hints**: Add type hints to all public functions
- **Imports**: Group into stdlib, third-party, and local imports

Example:
```python
def calculate_jaccard_scores(adj: sp.csr_matrix) -> NDArray[np.float64]:
    """Compute Jaccard similarity for all edges in a graph.
    
    Args:
        adj: Sparse adjacency matrix in CSR format.
    
    Returns:
        Array of Jaccard scores for each edge.
        
    Example:
        >>> adj = sp.csr_matrix([[0,1,1], [1,0,1], [1,1,0]])
        >>> scores = calculate_jaccard_scores(adj)
    """
    # Implementation here
    pass
```

### 2. Testing Requirements

All new code must include tests:

- **Unit tests**: Test individual functions in isolation
- **Edge cases**: Test boundary conditions (empty graphs, single nodes, etc.)
- **Numerical stability**: Test with extreme values
- **Determinism**: Verify reproducibility with fixed seeds

Example test:
```python
def test_jaccard_handles_isolated_nodes():
    """Jaccard should handle isolated nodes without errors."""
    adj = sp.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    scores = calculate_jaccard_scores(adj)
    assert not np.any(np.isnan(scores))
    assert not np.any(np.isinf(scores))
```

### 3. Documentation

- **Algorithm complexity**: Document time/space complexity
- **References**: Cite papers for algorithms
- **Examples**: Include usage examples in docstrings
- **Assumptions**: Document assumptions about input data

### 4. Commit Messages

Use conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(metrics): Add effective resistance approximation
fix(backbone): Handle graphs without triangles
docs(README): Update installation instructions
test(baselines): Add tests for DropEdge baseline
```

## 🔬 Research Workflow

### Adding a New Sparsification Method

1. **Implement the method** in `src/sparsification/`:
   ```python
   def my_new_metric(adj: sp.csr_matrix) -> NDArray[np.float64]:
       """Your implementation here."""
       pass
   ```

2. **Add unit tests** in `tests/test_sparsification.py`:
   ```python
   def test_my_new_metric():
       """Test your new metric."""
       pass
   ```

3. **Run experiments** in notebooks:
   ```python
   # notebooks/exploratory/XX_My_New_Method.ipynb
   ```

4. **Document results** with visualizations and statistics

### Running Experiments

```bash
# Run single experiment
python scripts/train.py --dataset cora --method jaccard --seed 42

# Run full experimental suite
bash scripts/reproduce_experiments.sh

# Analyze results
python scripts/analyze_results.py --input_dir results/
```

## 🐛 Reporting Issues

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Reproducible example**: Minimal code to reproduce
3. **Environment**: Python version, OS, package versions
4. **Expected vs. actual behavior**
5. **Error messages and stack traces**

## 📊 Code Review Process

All contributions go through code review:

1. **Automated checks**: Tests, linting, formatting
2. **Scientific validity**: Algorithm correctness
3. **Code quality**: Readability, documentation
4. **Performance**: No significant regressions

## 🎓 Research Ethics

- **Cite sources**: Always cite papers for algorithms
- **Reproducibility**: Document all hyperparameters and seeds
- **Negative results**: Report failures and limitations
- **Data transparency**: Document dataset preprocessing

## 📚 Resources

- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [SciPy Sparse Matrix Guide](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [Metric Backbone Paper](https://www.pnas.org/doi/10.1073/pnas.0808904106)

## 💬 Communication

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: For sensitive matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT + CC-BY for documentation).

---

Thank you for contributing to advancing GNN sparsification research! 🚀
