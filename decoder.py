"""
decoder.py
Decoder and Gated Network for Deep JSCC with ViT encoder.

Decoder Architecture:
  Received features ȳ ∈ R^(feature_dim)
      ↓
  Unprojection Linear: proj_dim → 768  (per token)
      ↓
  Reshape: (N, 196, 768) → (N, 768, 14, 14)   [spatial grid]
      ↓
  Stage 1: TransposedConv2d  14×14 → 28×28,   768  → 512
  Stage 2: TransposedConv2d  28×28 → 56×56,   512  → 256
  Stage 3: TransposedConv2d  56×56 → 112×112, 256  → 128
  Stage 4: TransposedConv2d 112×112 → 224×224, 128 →  64
  Final:   Conv2d 224×224,                      64  →   3
      ↓
  ŝ ∈ [0,1]^(224×224×3)  (denormalized image)

Gated Network (Section IV of paper):
  Input:  σ̂² (scalar channel noise variance)
  Output: λ(σ̂²) ∈ {0,1}^(feature_dim)  binary mask vector
  Structure: MLP with non-negative weights + ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# DECODER BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Single upsampling block: TransposedConv2d + BN + LReLU.
    Analogous to "Transposed convolutional layer+BN+ReLu" in Table II of paper.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 2, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size = 4,
                stride      = stride,
                padding     = 1,
                bias        = False
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# DECODER
# ─────────────────────────────────────────────────────────────────────────────

class ViTDecoder(nn.Module):
    """
    Decoder that reconstructs 224×224×3 images from received JSCC features.

    The core insight: ViT patch tokens form a 14×14 spatial grid
    (224/16 = 14 patches per side), so we can reshape the 1D received
    feature vector back into a 2D spatial feature map and apply
    standard transposed convolutions for upsampling.
    """

    def __init__(self, proj_dim: int = C.PROJ_DIM):
        super().__init__()

        self.proj_dim    = proj_dim
        self.num_patches = C.NUM_PATCHES    # 196
        self.grid_size   = 14              # 14×14 spatial grid
        self.feature_dim = self.num_patches * proj_dim

        # ── Unprojection: proj_dim → 768 ────────────────────────────────
        # Inverse of encoder's projection, per token
        self.unproject = nn.Sequential(
            nn.Linear(proj_dim, C.VIT_EMBED_DIM, bias=False),
            nn.LayerNorm(C.VIT_EMBED_DIM),
        )

        # ── Progressive upsampling: 14×14 → 224×224 ─────────────────────
        # 4 stages, each doubles spatial resolution
        # Channel progression: 768 → 512 → 256 → 128 → 64
        self.upsample = nn.Sequential(
            DecoderBlock(768, 512),     # 14  → 28
            DecoderBlock(512, 256),     # 28  → 56
            DecoderBlock(256, 128),     # 56  → 112
            DecoderBlock(128, 64),      # 112 → 224
        )

        # ── Final output layer ───────────────────────────────────────────
        # Conv2d (not transposed) for pixel-level refinement
        # Tanh maps to [-1,1], which matches ImageNet normalization range
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, y_bar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_bar: (N, feature_dim) received and reshaped feature vector
                   feature_dim = 196 × proj_dim

        Returns:
            s_hat: (N, 3, 224, 224) reconstructed image in normalized space
                   Values in approximately [-1, 1] due to Tanh activation
        """
        N = y_bar.shape[0]

        # ── Reshape flat vector to token sequence ─────────────────────────
        # (N, 196 × proj_dim)  →  (N, 196, proj_dim)
        tokens = y_bar.view(N, self.num_patches, self.proj_dim)

        # ── Unproject each token: proj_dim → 768 ──────────────────────────
        tokens = self.unproject(tokens)           # (N, 196, 768)

        # ── Reshape token sequence to 2D spatial feature map ──────────────
        # (N, 196, 768) → (N, 768, 14, 14)
        x = tokens.permute(0, 2, 1)              # (N, 768, 196)
        x = x.view(N, C.VIT_EMBED_DIM,
                   self.grid_size, self.grid_size)  # (N, 768, 14, 14)

        # ── Progressive upsampling ─────────────────────────────────────────
        x = self.upsample(x)                     # (N, 64, 224, 224)

        # ── Output projection ──────────────────────────────────────────────
        s_hat = self.output_conv(x)              # (N, 3, 224, 224)

        return s_hat


# ─────────────────────────────────────────────────────────────────────────────
# GATED NETWORK (Section IV-A of paper)
# ─────────────────────────────────────────────────────────────────────────────

class GatedNet(nn.Module):
    """
    Gated network g(σ̂², γ) for adaptive feature dimension selection.

    Per Remark 2 in the paper:
        - MLP with non-negative weights enforced via ReLU on weight matrices
        - Non-negative derivative activation functions (ReLU/LReLU/Tanh)
        - Monotonically increasing w.r.t. σ̂² (more noise → more features)

    Input:  σ̂² (scalar) — equivalent noise variance after channel equalization
    Output: λ(σ̂²) ∈ R^(feature_dim) — soft mask, thresholded to binary

    At threshold γ₀:
        λᵢ(σ̂²) = 1 if gᵢ(σ̂²) > γ₀  else  0
    """

    def __init__(self,
                 feature_dim: int   = C.FEATURE_DIM,
                 hidden_dim:  int   = C.GATED_HIDDEN_DIM,
                 num_layers:  int   = C.GATED_LAYERS,
                 gamma_0:     float = C.GAMMA_0):
        super().__init__()

        self.feature_dim = feature_dim
        self.gamma_0     = gamma_0

        # Build MLP: scalar σ̂² → hidden → hidden → feature_dim
        layers = []
        in_dim = 1  # scalar input: σ̂²
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())    # Non-negative derivative
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feature_dim))
        layers.append(nn.Sigmoid())    # Output in (0,1) for easy thresholding

        self.mlp = nn.Sequential(*layers)

        # Enforce non-negative weights via re-parameterization during forward
        # (Remark 2: weights must be non-negative for monotonicity guarantee)
        self._enforce_positive_weights()

    def _enforce_positive_weights(self):
        """Initialize weights as positive to start in the right regime."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 0.0, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _clamp_weights_positive(self):
        """
        Called before each forward pass to enforce non-negative weight constraint.
        This ensures the monotonicity property (Remark 2).
        """
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    m.weight.clamp_(min=0.0)

    def forward(self, sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma_sq: (N,) or scalar tensor — equivalent noise variance σ̂²

        Returns:
            soft_mask: (N, feature_dim) soft mask values in (0,1)
        """
        # Enforce non-negative weights for monotonicity
        self._clamp_weights_positive()

        # Ensure proper shape
        if sigma_sq.dim() == 0:
            sigma_sq = sigma_sq.unsqueeze(0)    # scalar → (1,)
        if sigma_sq.dim() == 1:
            sigma_sq = sigma_sq.unsqueeze(-1)   # (N,) → (N,1)

        soft_mask = self.mlp(sigma_sq)          # (N, feature_dim)
        return soft_mask

    def get_binary_mask(self, sigma_sq: torch.Tensor,
                        gamma_0: float = None) -> torch.Tensor:
        """
        Compute binary mask λ(σ̂²) ∈ {0,1}^(feature_dim).
        Uses straight-through estimator during training for gradient flow.

        Args:
            sigma_sq: noise variance
            gamma_0:  threshold (defaults to self.gamma_0)

        Returns:
            binary_mask: (N, feature_dim) in {0.0, 1.0}
        """
        gamma_0    = gamma_0 if gamma_0 is not None else self.gamma_0
        soft_mask  = self.forward(sigma_sq)

        # Straight-through estimator: hard threshold in forward,
        # but gradient flows through soft_mask in backward
        hard_mask  = (soft_mask > gamma_0).float()
        binary_mask = hard_mask - soft_mask.detach() + soft_mask
        return binary_mask

    def get_activation_ratio(self, sigma_sq: torch.Tensor) -> float:
        """
        Returns the fraction of feature dimensions activated.
        Used for reporting communication overhead reduction (Fig. 15 in paper).
        """
        with torch.no_grad():
            binary_mask = (self.forward(sigma_sq) > self.gamma_0).float()
            return binary_mask.mean().item()
