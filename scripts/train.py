"""Main training script for GNN sparsification experiments.

This script uses Hydra for configuration management and Weights & Biases
for experiment tracking. It supports swapping models, datasets, and
sparsification methods via command-line arguments.

Usage:
    # Run with default configuration
    python scripts/train.py
    
    # Override specific components
    python scripts/train.py model=gat dataset=pubmed sparsifier=jaccard
    
    # Use experiment preset
    python scripts/train.py experiment=baseline
    
    # Override specific parameters
    python scripts/train.py model.hidden_dim=128 training.learning_rate=0.001
"""

import os
import random
from pathlib import Path
from typing import Dict, Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

# TODO: Uncomment when implementing W&B integration
# import wandb

from src.sparsification.base import BaseSparsifier
from src.sparsification.random import RandomSparsifier
from src.sparsification.spectral import SpectralSparsifier
from src.sparsification.metric import MetricSparsifier

# TODO: Import your model implementations
# from src.models.gcn import GCN
# from src.models.gat import GAT


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(cfg: DictConfig) -> Data:
    """Load and prepare the dataset.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        PyTorch Geometric Data object
    """
    print(f"Loading dataset: {cfg.dataset.name}")
    
    # Load dataset based on configuration
    if cfg.dataset.dataset_class == "Planetoid":
        dataset = Planetoid(
            root=cfg.data.root_dir,
            name=cfg.dataset.name.capitalize(),
            split=cfg.dataset.split,
        )
        data = dataset[0]
    else:
        raise NotImplementedError(
            f"Dataset class {cfg.dataset.dataset_class} not implemented"
        )
    
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def create_sparsifier(cfg: DictConfig) -> BaseSparsifier:
    """Create sparsifier instance from configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        BaseSparsifier instance
    """
    sparsifier_class = cfg.sparsifier.get("class", "RandomSparsifier")
    
    sparsifier_map = {
        "RandomSparsifier": RandomSparsifier,
        "SpectralSparsifier": SpectralSparsifier,
        "MetricSparsifier": MetricSparsifier,
    }
    
    if sparsifier_class not in sparsifier_map:
        raise ValueError(f"Unknown sparsifier class: {sparsifier_class}")
    
    # Extract sparsifier parameters
    params = OmegaConf.to_container(cfg.sparsifier, resolve=True)
    params.pop("name", None)
    params.pop("class", None)
    params.pop("description", None)
    
    sparsifier = sparsifier_map[sparsifier_class](**params)
    print(f"Created sparsifier: {sparsifier}")
    
    return sparsifier


def create_model(cfg: DictConfig, num_features: int, num_classes: int):
    """Create GNN model from configuration.
    
    TODO: Implement this function with your actual model classes.
    
    Args:
        cfg: Hydra configuration object
        num_features: Number of input features
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    # Placeholder - replace with actual model implementation
    print(f"Creating model: {cfg.model.name}")
    print(f"  Hidden dim: {cfg.model.architecture.hidden_dim}")
    print(f"  Num layers: {cfg.model.architecture.num_layers}")
    print(f"  Dropout: {cfg.model.architecture.dropout}")
    
    # TODO: Implement model creation
    # if cfg.model.name == "gcn":
    #     model = GCN(
    #         in_channels=num_features,
    #         hidden_channels=cfg.model.architecture.hidden_dim,
    #         out_channels=num_classes,
    #         num_layers=cfg.model.architecture.num_layers,
    #         dropout=cfg.model.architecture.dropout,
    #     )
    # elif cfg.model.name == "gat":
    #     model = GAT(
    #         in_channels=num_features,
    #         hidden_channels=cfg.model.architecture.hidden_dim,
    #         out_channels=num_classes,
    #         num_layers=cfg.model.architecture.num_layers,
    #         heads=cfg.model.architecture.heads,
    #         dropout=cfg.model.architecture.dropout,
    #     )
    # else:
    #     raise ValueError(f"Unknown model: {cfg.model.name}")
    
    raise NotImplementedError("Model creation not yet implemented")


def train_epoch(model, data, optimizer, device: str):
    """Train for one epoch.
    
    TODO: Implement training logic.
    
    Args:
        model: PyTorch model
        data: PyTorch Geometric Data object
        optimizer: PyTorch optimizer
        device: Device to train on
        
    Returns:
        Training loss
    """
    model.train()
    optimizer.zero_grad()
    
    # TODO: Implement training step
    # out = model(data.x, data.edge_index)
    # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # loss.backward()
    # optimizer.step()
    
    # return loss.item()
    
    raise NotImplementedError("Training loop not yet implemented")


@torch.no_grad()
def evaluate(model, data, device: str) -> Dict[str, float]:
    """Evaluate model on train/val/test splits.
    
    TODO: Implement evaluation logic.
    
    Args:
        model: PyTorch model
        data: PyTorch Geometric Data object
        device: Device to evaluate on
        
    Returns:
        Dictionary with train/val/test accuracies
    """
    model.eval()
    
    # TODO: Implement evaluation
    # out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)
    
    # results = {}
    # for split in ['train', 'val', 'test']:
    #     mask = data[f'{split}_mask']
    #     correct = pred[mask] == data.y[mask]
    #     results[f'{split}_acc'] = correct.sum().item() / mask.sum().item()
    
    # return results
    
    raise NotImplementedError("Evaluation not yet implemented")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set up reproducibility
    set_seed(cfg.seed, cfg.deterministic)
    
    # Set device
    device = cfg.experiment.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.logs_dir, exist_ok=True)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    
    # =========================================================================
    # WEIGHTS & BIASES INITIALIZATION
    # Initialize W&B run here if enabled in config
    # =========================================================================
    if cfg.logging.use_wandb:
        print("\n[W&B Integration Point]")
        print("Initialize Weights & Biases here:")
        print("  wandb.init(")
        print(f"    project='{cfg.logging.wandb_project}',")
        print(f"    entity={cfg.logging.wandb_entity},")
        print(f"    name='{cfg.experiment.name}',")
        print("    config=OmegaConf.to_container(cfg, resolve=True)")
        print("  )")
        print()
        
        # TODO: Uncomment when ready to use W&B
        # wandb.init(
        #     project=cfg.logging.wandb_project,
        #     entity=cfg.logging.wandb_entity,
        #     name=cfg.experiment.name,
        #     config=OmegaConf.to_container(cfg, resolve=True),
        # )
    
    # Load dataset
    data = load_dataset(cfg)
    
    # Apply sparsification
    print("\n" + "=" * 80)
    print("SPARSIFICATION")
    print("=" * 80)
    sparsifier = create_sparsifier(cfg)
    sparse_data = sparsifier.sparsify(data)
    
    # Print sparsification statistics
    stats = sparsifier.get_sparsification_stats(data, sparse_data)
    print("\nSparsification Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Log sparsification stats to W&B
    if cfg.logging.use_wandb:
        print("\n[W&B Logging Point]")
        print("Log sparsification statistics:")
        print(f"  wandb.log({stats})")
        # TODO: wandb.log(stats)
    
    # Move data to device
    sparse_data = sparse_data.to(device)
    
    # Create model
    print("\n" + "=" * 80)
    print("MODEL")
    print("=" * 80)
    # TODO: Uncomment when model creation is implemented
    # model = create_model(
    #     cfg,
    #     num_features=data.num_features,
    #     num_classes=data.y.max().item() + 1
    # )
    # model = model.to(device)
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    # TODO: Uncomment when model is created
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=cfg.training.learning_rate,
    #     weight_decay=cfg.training.weight_decay,
    # )
    
    # =========================================================================
    # TRAINING LOOP
    # Main training loop goes here
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print("\n[Training Loop Implementation Point]")
    print("Implement the main training loop here:")
    print("""
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, cfg.training.epochs + 1):
        # Train
        train_loss = train_epoch(model, sparse_data, optimizer, device)
        
        # Evaluate
        if epoch % cfg.logging.log_interval == 0:
            results = evaluate(model, sparse_data, device)
            
            print(f"Epoch {epoch:03d}: "
                  f"Loss: {train_loss:.4f}, "
                  f"Train: {results['train_acc']:.4f}, "
                  f"Val: {results['val_acc']:.4f}, "
                  f"Test: {results['test_acc']:.4f}")
            
            # Log to W&B
            if cfg.logging.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **results
                })
            
            # Early stopping
            if results['val_acc'] > best_val_acc:
                best_val_acc = results['val_acc']
                patience_counter = 0
                # Save checkpoint
                if cfg.logging.save_checkpoints:
                    torch.save(
                        model.state_dict(),
                        Path(cfg.logging.checkpoint_dir) / 'best_model.pt'
                    )
            else:
                patience_counter += 1
                if patience_counter >= cfg.training.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    """)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("[Final Evaluation Point]")
    print("Load best model and evaluate on test set")
    
    # TODO: Implement final evaluation
    # if cfg.logging.save_checkpoints:
    #     model.load_state_dict(torch.load(
    #         Path(cfg.logging.checkpoint_dir) / 'best_model.pt'
    #     ))
    # 
    # final_results = evaluate(model, sparse_data, device)
    # print("\nFinal Results:")
    # for key, value in final_results.items():
    #     print(f"  {key}: {value:.4f}")
    # 
    # if cfg.logging.use_wandb:
    #     wandb.log({"final_" + k: v for k, v in final_results.items()})
    #     wandb.finish()
    
    print("\n" + "=" * 80)
    print("Training script scaffold complete!")
    print("Next steps:")
    print("  1. Implement model classes in src/models/")
    print("  2. Uncomment and complete train_epoch() function")
    print("  3. Uncomment and complete evaluate() function")
    print("  4. Enable W&B logging")
    print("=" * 80)


if __name__ == "__main__":
    main()
