"""
seeds.py — reproducibility control for the stochastic classifiers.

Addresses supervisor comment 120 ("Aren't these stochastic? How do you deal
with?"). Random Forest, XGBoost, the DANN and the CNNs are all stochastic
(bootstrap sampling, column subsampling, weight init, dropout, mini-batch
order). Without a fixed seed each LOPO run yields slightly different numbers,
so reported means are not reproducible and per-patient comparisons are noisy.

Call ``set_global_seed(seed)`` once at the top of every notebook/experiment,
and pass ``random_state=seed`` to every scikit-learn / xgboost estimator.
For a fully honest report, run each stochastic model over N seeds and report
mean +/- std (see ``seed_sweep`` helper).
"""
from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_sweep(run_once: Callable[[int], float], seeds=(0, 1, 2, 3, 4)) -> dict:
    """Run ``run_once(seed) -> metric`` across seeds; return mean/std/values.

    Use to report stochastic-model performance as mean +/- std rather than a
    single unreproducible number.
    """
    vals = [float(run_once(s)) for s in seeds]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "values": vals,
        "seeds": list(seeds),
    }
