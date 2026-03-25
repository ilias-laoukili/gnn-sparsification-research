"""HP probe: fixed random configs for landscape flatness + rank correlation."""

import numpy as np
import torch

from src.hpo.config import (
    DO_MAX, DO_MIN, HIDDEN_CHOICES, LAYER_CHOICES,
    LR_MAX, LR_MIN, WD_MAX, WD_MIN,
    DEFAULT_EPOCHS, DEFAULT_PATIENCE,
)
from src.models.flexible import FlexibleGCN
from src.training.trainer import GNNTrainer


def generate_probe_configs(n_configs, seed=99):
    """Generate *n_configs* fixed random HP configs (deterministic)."""
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n_configs):
        configs.append({
            "hp_idx": i,
            "lr": float(np.exp(rng.uniform(np.log(LR_MIN), np.log(LR_MAX)))),
            "weight_decay": float(rng.uniform(WD_MIN, WD_MAX)),
            "dropout": float(rng.uniform(DO_MIN, DO_MAX)),
            "hidden_channels": int(rng.choice(HIDDEN_CHOICES)),
            "num_layers": int(rng.choice(LAYER_CHOICES)),
        })
    return configs


def run_probe(graph, n_feat, n_class, device, probe_configs,
              eval_graph=None, model_cls=FlexibleGCN,
              epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE):
    """Evaluate each probe config on *graph* (1 seed each).

    If *eval_graph* is provided, also evaluate the trained model on it
    (train on ``graph``, evaluate on ``eval_graph``). This gives
    ``val_acc_on_full`` — needed for measuring genuine HP-landscape
    flattening rather than trivial information-loss compression.

    Returns list of dicts with keys: hp_idx, val_acc[, val_acc_on_full].
    """
    results = []
    for cfg in probe_configs:
        hp = {k: cfg[k] for k in ["lr", "weight_decay", "dropout",
                                    "hidden_channels", "num_layers"]}
        torch.manual_seed(0)
        np.random.seed(0)
        model = model_cls(n_feat, hp["hidden_channels"], n_class,
                          hp["num_layers"], hp["dropout"]).to(device)
        model.reset_parameters()
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                               weight_decay=hp["weight_decay"])
        trainer = GNNTrainer(model=model, optimizer=opt, device=device)
        history = trainer.train(graph, epochs=epochs, patience=patience)

        entry = {"hp_idx": cfg["hp_idx"],
                 "val_acc": float(history["best_val_acc"])}

        if eval_graph is not None:
            val_acc_full = trainer._evaluate(eval_graph, eval_graph.val_mask)
            entry["val_acc_on_full"] = float(val_acc_full)

        results.append(entry)
    return results
