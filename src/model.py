"""
model.py
========
CNN architecture that classifies 18×18 Granger causality matrices as
preictal (1) or interictal (0) — thesis §3.4.

Architecture
------------
Input : (batch, 1, 18, 18)
  Conv2d(1 → 32,  k=3, pad=1) → BatchNorm → ReLU
  Conv2d(32 → 64, k=3, pad=1) → BatchNorm → ReLU → MaxPool(2)   → (64, 9, 9)
  Conv2d(64 → 128,k=3, pad=1) → BatchNorm → ReLU → MaxPool(2)   → (128, 4, 4)
  Flatten → 2048
  Dropout(0.5)
  Linear(2048 → 256) → ReLU
  Dropout(0.3)
  Linear(256 → 1) → Sigmoid

Loss : Weighted binary cross-entropy (class weight inversely proportional
       to class frequency) — thesis §3.4.
"""

import torch
import torch.nn as nn
from typing import Optional


class GCPredictor(nn.Module):
    """
    2-D CNN operating on 18×18 Granger causality connectivity matrices.
    """

    def __init__(self, n_channels: int = 18, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 18→9
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 9→4
        )

        # Compute flat size dynamically (handles different n_channels)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_channels)
            flat_size = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, 1, 18, 18)

        Returns
        -------
        out : torch.Tensor  shape (batch, 1)  — P(preictal)
        """
        return self.classifier(self.features(x))


# ── Loss ──────────────────────────────────────────────────────────────────────


def build_weighted_bce(
    labels: "torch.Tensor", device: torch.device, clip: float = 50.0
) -> nn.BCELoss:
    """
    Build a weighted binary cross-entropy loss where the positive class weight
    is inversely proportional to class frequency (thesis §3.4).

    w_pos = n_neg / n_pos   (clipped at `clip` to avoid extreme weights)
    """
    n_pos = float((labels == 1).sum().item())
    n_neg = float((labels == 0).sum().item())

    if n_pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = min(n_neg / n_pos, clip)

    # nn.BCELoss does not natively support per-sample weights via pos_weight;
    # we use BCEWithLogitsLoss-style manual weighting through a wrapper.
    # However, since our model outputs sigmoid, we use a manual weight approach:
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    class WeightedBCE(nn.Module):
        def forward(self, preds, targets):
            # preds: (N,1), targets: (N,)
            preds = preds.squeeze(1)
            targets = targets.float()
            bce = -(
                targets * torch.log(preds + 1e-8) * weight_tensor.squeeze()
                + (1 - targets) * torch.log(1 - preds + 1e-8)
            )
            return bce.mean()

    return WeightedBCE()


# ── Dataset ───────────────────────────────────────────────────────────────────


class GCDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping GC matrices and binary labels.

    Parameters
    ----------
    matrices : np.ndarray  shape (N, 18, 18)
    labels   : np.ndarray  shape (N,)  {0, 1}
    """

    def __init__(self, matrices: "np.ndarray", labels: "np.ndarray"):
        import numpy as np

        self.X = torch.from_numpy(matrices.astype("float32")).unsqueeze(
            1
        )  # (N,1,18,18)
        self.y = torch.from_numpy(labels.astype("float32"))  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
