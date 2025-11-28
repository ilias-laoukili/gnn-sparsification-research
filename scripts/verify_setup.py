"""Verify that the GNN Sparsification Research package is correctly set up.

This script checks:
1. Python version
2. Required packages
3. Directory structure
4. Import statements
5. Basic functionality

Run after setup to ensure everything is working.

Usage:
    python scripts/verify_setup.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_packages():
    """Check if required packages are installed."""
    print("\nChecking required packages...")
    
    packages = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("numpy", "NumPy"),
        ("pytest", "pytest"),
    ]
    
    all_installed = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            all_installed = False
    
    return all_installed


def check_directories():
    """Check if required directories exist."""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "configs",
        "configs/model",
        "configs/dataset",
        "configs/sparsifier",
        "configs/experiment",
        "data/raw",
        "data/processed",
        "docs",
        "notebooks/exploratory",
        "notebooks/tutorials",
        "scripts",
        "src/data",
        "src/models",
        "src/sparsification",
        "src/training",
        "src/utils",
        "tests",
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (missing)")
            all_exist = False
    
    return all_exist


def check_files():
    """Check if key files exist."""
    print("\nChecking key files...")
    
    required_files = [
        "configs/config.yaml",
        "scripts/train.py",
        "scripts/demo_sparsification.py",
        "src/__init__.py",
        "src/sparsification/base.py",
        "src/sparsification/random.py",
        "src/sparsification/metric.py",
        "src/sparsification/spectral.py",
        "requirements.txt",
        "setup.py",
        "README.md",
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.isfile(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if package imports work."""
    print("\nChecking package imports...")
    
    imports = [
        ("from src.sparsification.base import BaseSparsifier", "BaseSparsifier"),
        ("from src.sparsification.random import RandomSparsifier", "RandomSparsifier"),
        ("from src.sparsification.metric import MetricSparsifier", "MetricSparsifier"),
        ("from src.sparsification.spectral import SpectralSparsifier", "SpectralSparsifier"),
    ]
    
    all_work = True
    for import_stmt, name in imports:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} ({str(e)})")
            all_work = False
    
    return all_work


def check_basic_functionality():
    """Test basic sparsification functionality."""
    print("\nChecking basic functionality...")
    
    try:
        import torch
        from torch_geometric.data import Data
        from src.sparsification.random import RandomSparsifier
        
        # Create simple test graph
        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=10)
        
        # Test sparsification
        sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
        sparse_data = sparsifier.sparsify(data)
        
        if sparse_data.num_edges <= data.num_edges:
            print("  ✓ Basic sparsification works")
            return True
        else:
            print("  ✗ Sparsification increased edges (unexpected)")
            return False
            
    except Exception as e:
        print(f"  ✗ Functionality test failed: {str(e)}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("GNN Sparsification Research - Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Required Packages", check_packages()))
    results.append(("Directory Structure", check_directories()))
    results.append(("Key Files", check_files()))
    results.append(("Package Imports", check_imports()))
    results.append(("Basic Functionality", check_basic_functionality()))
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! Setup is complete.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run demo: python scripts/demo_sparsification.py")
        print("2. Run tests: pytest tests/")
        print("3. Implement models in src/models/")
        print("4. Complete training script and run experiments")
        print("\nSee docs/QUICKSTART.md for detailed instructions.")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Run setup script: ./setup.sh")
        print("2. Manually install missing packages: pip install -r requirements.txt")
        print("3. Ensure you're in the virtual environment")
        print("4. Check that all files were created correctly")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
