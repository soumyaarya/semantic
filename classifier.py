"""
classifier.py
Nearest Subspace Classifier (NSC) for semantic communication evaluation.

Implements Eq. (13) from the paper:
    j' = argmin_j ||(I - VⱼVⱼᴴ)(ȳ' - μⱼ)||²

This is a NON-PARAMETRIC classifier applied directly in the feature space.
No additional training required — it uses the subspace structure learned
by the coding rate reduction loss.

Steps:
  1. Fit: Collect features from training set → compute class means μⱼ and
          principal components Vⱼ for each class j
  2. Predict: For test features, compute distance to each class subspace
"""

import torch
import numpy as np
from typing import Optional
import config as C


class NearestSubspaceClassifier:
    """
    Nearest Subspace Classifier — Eq. (13) in paper.

    After training the JSCC model with coding rate reduction,
    the received features Ȳ have clear subspace structures
    (one subspace per class). This classifier exploits that structure.

    Distance metric:
        d(ȳ', class j) = ||(I - VⱼVⱼᴴ)(ȳ' - μⱼ)||²
        = distance from test point to the affine subspace of class j

    Predicted class: j* = argmin_j d(ȳ', class j)
    """

    def __init__(self, num_classes: int = C.NUM_CLASSES,
                 n_components: int = C.NSC_COMPONENTS):
        """
        Args:
            num_classes:  Number of classes (100 for ImageNet-100)
            n_components: pⱼ — number of principal components per class.
                          Rule of thumb: ~floor(n_train_per_class / 10)
                          For ImageNet-100: ~1300 train images/class → pⱼ ≈ 12
        """
        self.num_classes   = num_classes
        self.n_components  = n_components
        self.class_means   = None   # List of μⱼ tensors, shape (feat_dim,)
        self.class_bases   = None   # List of Vⱼ matrices, shape (feat_dim, pⱼ)
        self.is_fitted     = False

    # ─────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Compute class subspace parameters from training features.

        Args:
            features: (N, feat_dim) tensor of extracted features
                      from the full training set
            labels:   (N,) integer class labels

        Note: This should be run AFTER training is complete, using
              the trained model to extract features from training data.
              Usually takes ~5 minutes on GPU for full ImageNet-100 train set.
        """
        print(f"[NSC] Fitting on {features.shape[0]} samples, "
              f"{self.num_classes} classes, "
              f"{self.n_components} components per class...")

        features = features.cpu().float()
        labels   = labels.cpu()

        self.class_means = []
        self.class_bases = []

        for c in range(self.num_classes):
            mask   = (labels == c)
            Y_c    = features[mask]                     # (nⱼ, feat_dim)
            n_j    = Y_c.shape[0]

            if n_j == 0:
                # No samples for this class — use zero mean and identity
                feat_dim = features.shape[1]
                self.class_means.append(torch.zeros(feat_dim))
                self.class_bases.append(
                    torch.zeros(feat_dim, self.n_components)
                )
                continue

            # Class mean μⱼ
            mu_j = Y_c.mean(dim=0)                     # (feat_dim,)
            self.class_means.append(mu_j)

            # Center the class features
            Y_c_centered = Y_c - mu_j.unsqueeze(0)     # (nⱼ, feat_dim)

            # PCA: top pⱼ principal components via SVD
            p_j  = min(self.n_components, n_j - 1, Y_c.shape[1])
            try:
                # torch.linalg.svd can be memory intensive for large matrices
                # Use economy SVD (full_matrices=False)
                _, _, Vh = torch.linalg.svd(Y_c_centered, full_matrices=False)
                V_j = Vh[:p_j].T                       # (feat_dim, pⱼ)
            except Exception:
                # Fallback to numpy SVD
                U, S, Vh = np.linalg.svd(
                    Y_c_centered.numpy(), full_matrices=False
                )
                V_j = torch.from_numpy(Vh[:p_j].T).float()

            self.class_bases.append(V_j)

        self.is_fitted = True
        print("[NSC] Fitting complete.")

    # ─────────────────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, features: torch.Tensor,
                batch_size: int = 256) -> torch.Tensor:
        """
        Predict class labels for test features using Eq. (13).

        d(ȳ', j) = ||(I - VⱼVⱼᴴ)(ȳ' - μⱼ)||²

        Args:
            features:   (N, feat_dim) test feature tensor
            batch_size: Process this many test samples at once

        Returns:
            predictions: (N,) predicted class labels
        """
        assert self.is_fitted, "Call fit() before predict()"

        features     = features.cpu().float()
        N            = features.shape[0]
        predictions  = torch.zeros(N, dtype=torch.long)

        for start in range(0, N, batch_size):
            end      = min(start + batch_size, N)
            Y_test   = features[start:end]              # (B, feat_dim)
            B        = Y_test.shape[0]
            dists    = torch.zeros(B, self.num_classes) # (B, num_classes)

            for c in range(self.num_classes):
                mu_j  = self.class_means[c]            # (feat_dim,)
                V_j   = self.class_bases[c]            # (feat_dim, pⱼ)

                # Center test features w.r.t. class mean
                diff  = Y_test - mu_j.unsqueeze(0)    # (B, feat_dim)

                # Project onto class subspace: VⱼVⱼᴴ · diff
                proj  = diff @ V_j @ V_j.T            # (B, feat_dim)

                # Orthogonal residual: (I - VⱼVⱼᴴ)(ȳ' - μⱼ)
                resid = diff - proj                    # (B, feat_dim)

                # Squared distance
                dists[:, c] = (resid ** 2).sum(dim=-1) # (B,)

            predictions[start:end] = dists.argmin(dim=-1)

        return predictions

    def score(self, features: torch.Tensor,
              labels: torch.Tensor) -> float:
        """
        Compute classification accuracy — Eq. for classification accuracy
        in Section II of the paper: J'/N' %

        Args:
            features: (N, feat_dim) test features
            labels:   (N,) ground truth class labels

        Returns:
            accuracy: float in [0, 1]
        """
        predictions = self.predict(features)
        correct     = (predictions == labels.cpu()).sum().item()
        accuracy    = correct / len(labels)
        return accuracy

    # ─────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save fitted classifier parameters."""
        torch.save({
            'class_means':  self.class_means,
            'class_bases':  self.class_bases,
            'num_classes':  self.num_classes,
            'n_components': self.n_components,
            'is_fitted':    self.is_fitted,
        }, path)
        print(f"[NSC] Saved to {path}")

    def load(self, path: str):
        """Load fitted classifier parameters."""
        ckpt = torch.load(path, map_location='cpu')
        self.class_means   = ckpt['class_means']
        self.class_bases   = ckpt['class_bases']
        self.num_classes   = ckpt['num_classes']
        self.n_components  = ckpt['n_components']
        self.is_fitted     = ckpt['is_fitted']
        print(f"[NSC] Loaded from {path}")
