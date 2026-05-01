"""
loss.py
Loss functions for Deep JSCC training.

Implements:
  1. Coding Rate Reduction Maximization  ΔR  (Eq. 6, 7, 8 in paper)
     - R(Ȳ, ε):   Total rate — volume of whole feature space (Eq. 6)
     - Rᵈ(Ȳ|Π,ε): Within-class rate — volume of each class subspace (Eq. 7)
     - ΔR = R - Rᵈ: Rate reduction to maximize (Eq. 8)

  2. MSE Minimization  (Eq. 9 in paper)

  3. Unified Loss  (Eq. 10 in paper):
     Loss = -β · ΔR(Ȳ|Π,ε) + MSE(S, Ŝ)

The coding rate reduction loss drives features toward:
  - Cross-class discrimination: features from different classes orthogonal
  - In-class compactness: features from same class tightly clustered
"""

import torch
import torch.nn as nn
import math
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# CODING RATE REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

class CodingRateReduction(nn.Module):
    """
    Coding Rate Reduction loss from Proposition 1 and Eq. (6-8) of the paper.

    Based on:
        Yu et al., "Learning Diverse and Discriminative Representations via
        the Principle of Maximal Coding Rate Reduction", NeurIPS 2020.
        (Reference [27] in the paper)
    """

    def __init__(self, epsilon_sq: float = C.EPSILON_SQ):
        super().__init__()
        self.eps_sq = epsilon_sq

    def compute_R(self, Y_bar: torch.Tensor) -> torch.Tensor:
        """
        Compute total coding rate R(Ȳ, ε) — Eq. (6).

        R(Ȳ, ε) = (1/2) log₂ det(I + (2b / Nε²) · Ȳ·ȲᵀT)

        Args:
            Y_bar: (N, feature_dim) feature matrix

        Returns:
            R: scalar tensor — total coding rate
        """
        N, feat_dim = Y_bar.shape
        b = feat_dim // 2               # Number of complex symbols

        # Compute Ȳ·Ȳᵀ / (N · ε²) scaled by 2b
        # Matrix is (feat_dim × feat_dim) but we use log det via eigenvalues
        # for numerical stability
        scale   = (2 * b) / (N * self.eps_sq)

        # For large feature_dim, use the dual form: det(I_N + scale·ȲᵀȲ/b)
        # Both give same log-det by matrix determinant lemma.
        # We use whichever matrix is smaller.
        if N < feat_dim:
            # Use (N×N) Gram matrix: scale * Y_bar @ Y_bar.T
            M = scale * (Y_bar @ Y_bar.T)                      # (N, N)
        else:
            # Use (feat_dim×feat_dim) covariance matrix
            M = scale * (Y_bar.T @ Y_bar) / N                  # (feat_dim, feat_dim)

        # log det(I + M) via log(1 + eigenvalues) for stability
        eigvals = torch.linalg.eigvalsh(M)                      # real eigenvalues
        R = 0.5 * torch.sum(torch.log2(1.0 + eigvals.clamp(min=0.0)))
        return R

    def compute_Rd(self, Y_bar: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
        """
        Compute within-class coding rate Rᵈ(Ȳ|Π,ε) — Eq. (7).

        Rᵈ(Ȳ|Π,ε) = Σⱼ (nⱼ/2N) · log₂ det(I + (2b / nⱼε²) · Ȳⱼ·ȲⱼᵀT)

        where Ȳⱼ are features belonging to class j, nⱼ = |class j samples|.

        Args:
            Y_bar:  (N, feature_dim) feature matrix
            labels: (N,) integer class labels

        Returns:
            Rd: scalar tensor — within-class coding rate
        """
        N, feat_dim = Y_bar.shape
        b           = feat_dim // 2
        classes     = labels.unique()
        Rd          = torch.tensor(0.0, device=Y_bar.device, dtype=Y_bar.dtype)

        for c in classes:
            mask   = (labels == c)
            Y_c    = Y_bar[mask]           # (nⱼ, feat_dim) — class j features
            n_j    = Y_c.shape[0]

            if n_j < 2:
                continue                   # Skip singleton classes

            scale_j = (2 * b) / (n_j * self.eps_sq)
            weight  = n_j / (2 * N)

            if n_j < feat_dim:
                M_j = scale_j * (Y_c @ Y_c.T)               # (nⱼ, nⱼ)
            else:
                M_j = scale_j * (Y_c.T @ Y_c) / n_j         # (feat_dim, feat_dim)

            eigvals_j = torch.linalg.eigvalsh(M_j)
            Rd += weight * torch.sum(
                torch.log2(1.0 + eigvals_j.clamp(min=0.0))
            )

        return Rd

    def forward(self, Y_bar: torch.Tensor,
                labels: torch.Tensor) -> tuple:
        """
        Compute ΔR(Ȳ|Π,ε) = R(Ȳ,ε) - Rᵈ(Ȳ|Π,ε) — Eq. (8).

        We MAXIMIZE ΔR, so in the loss we use -β·ΔR (with β > 0 in linear scale).

        Args:
            Y_bar:  (N, feature_dim) received features
            labels: (N,) integer class labels

        Returns:
            delta_R: scalar — rate reduction (to be maximized)
            R:       scalar — total rate (for logging)
            Rd:      scalar — within-class rate (for logging)
        """
        R       = self.compute_R(Y_bar)
        Rd      = self.compute_Rd(Y_bar, labels)
        delta_R = R - Rd
        return delta_R, R, Rd


# ─────────────────────────────────────────────────────────────────────────────
# MSE LOSS
# ─────────────────────────────────────────────────────────────────────────────

class MSELoss(nn.Module):
    """
    Per-pixel MSE between original and reconstructed images.
    Eq. (9) in paper: MSE(S, Ŝ) = (1/NB) Σₙ ||sₙ - ŝₙ||²
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, s: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        return self.mse(s, s_hat)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED LOSS (Eq. 10 in paper)
# ─────────────────────────────────────────────────────────────────────────────

class JSCCLoss(nn.Module):
    """
    Unified loss: Loss(ω,θ) = −β·ΔR(Ȳ|Π,ε) + MSE(S, Ŝ)

    β is set in dB in config.py and converted to linear scale here.
    Negative β·ΔR because we maximize ΔR (equivalent to minimizing -ΔR).
    """

    def __init__(self,
                 beta_db:    float = C.BETA,
                 epsilon_sq: float = C.EPSILON_SQ,
                 mode:       str   = "full"):
        """
        Args:
            beta_db:    β in dB scale (e.g., -30.0 dB)
            epsilon_sq: coding distortion ε²
            mode:       "full"     → -β·ΔR + MSE  (Phase 2 & 3)
                        "mse_only" → MSE only      (Phase 1)
        """
        super().__init__()
        self.beta       = 10 ** (beta_db / 10.0)    # Convert dB → linear
        self.mode       = mode
        self.crr        = CodingRateReduction(epsilon_sq)
        self.mse_loss   = MSELoss()

    def forward(self,
                s:     torch.Tensor,
                s_hat: torch.Tensor,
                y_bar: torch.Tensor,
                labels: torch.Tensor) -> dict:
        """
        Args:
            s:      (N, 3, 224, 224) original images
            s_hat:  (N, 3, 224, 224) reconstructed images
            y_bar:  (N, feature_dim) received features (after channel)
            labels: (N,) integer class labels

        Returns:
            dict with keys:
                'loss':     total loss (scalar, to call .backward() on)
                'mse':      MSE component
                'delta_R':  rate reduction component
                'R':        total rate
                'Rd':       within-class rate
        """
        mse = self.mse_loss(s, s_hat)

        if self.mode == "mse_only":
            return {
                'loss':    mse,
                'mse':     mse.detach(),
                'delta_R': torch.tensor(0.0),
                'R':       torch.tensor(0.0),
                'Rd':      torch.tensor(0.0),
            }

        # Full loss with coding rate reduction
        delta_R, R, Rd = self.crr(y_bar, labels)
        loss = -self.beta * delta_R + mse

        return {
            'loss':    loss,
            'mse':     mse.detach(),
            'delta_R': delta_R.detach(),
            'R':       R.detach(),
            'Rd':      Rd.detach(),
        }

    def set_mode(self, mode: str):
        """Switch between 'mse_only' and 'full' loss modes."""
        assert mode in ["full", "mse_only"]
        self.mode = mode
        print(f"[Loss] Mode set to: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# PSNR METRIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(s: torch.Tensor, s_hat: torch.Tensor,
                 max_val: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) — Eq. (4) in paper.

    PSNR(s, ŝ) = 10 · log₁₀(MAX² / MSE)

    Args:
        s:       (N, 3, H, W) original images in [0, max_val]
        s_hat:   (N, 3, H, W) reconstructed images in [0, max_val]
        max_val: Maximum pixel value (1.0 for normalized images)

    Returns:
        psnr: scalar PSNR in dB, averaged over the batch
    """
    mse  = F.mse_loss(s_hat, s, reduction='none').mean(dim=[1,2,3])  # (N,)
    psnr = 10.0 * torch.log10(max_val**2 / (mse + 1e-10))
    return psnr.mean()


# Make F available
import torch.nn.functional as F
