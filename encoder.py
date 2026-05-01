"""
encoder.py
ViT-B/16 Encoder with learned projection for Deep JSCC.

Architecture:
  Input image s ∈ R^(224×224×3)
      ↓
  ViT-B/16 backbone (pretrained)  →  196 patch tokens of dim 768
      ↓
  Projection layer (per-token Linear: 768 → PROJ_DIM)
      ↓
  Flatten  →  x̄ ∈ R^(196 × PROJ_DIM)  =  R^(feature_dim)
      ↓
  Power normalization (Eq. 5)
      ↓
  x̂ ∈ R^(feature_dim)  [treated as b complex symbols]

Fine-tuning control:
  - Phase 1: All ViT weights frozen
  - Phase 2: Last N transformer blocks unfrozen
  - Phase 3: All weights unfrozen
"""

import torch
import torch.nn as nn
import timm
import config as C


class ViTEncoder(nn.Module):
    """
    Pretrained ViT-B/16 encoder with learned projection layer.

    Key design decision:
        We use ALL 196 patch tokens (not the CLS token) to preserve
        spatial information needed for the reconstruction task.
        The projection maps 768 → PROJ_DIM per token, then flattens.
    """

    def __init__(self,
                 proj_dim: int = C.PROJ_DIM,
                 pretrained: bool = True):
        super().__init__()

        # ── ViT Backbone ────────────────────────────────────────────────
        # Load ViT-B/16 pretrained on ImageNet-21k via timm.
        # We remove the classification head since we want raw patch tokens.
        self.vit = timm.create_model(
            C.VIT_MODEL,
            pretrained  = pretrained,
            num_classes = 0,    # Remove classifier head
        )
        # Verify token dimension
        assert self.vit.embed_dim == C.VIT_EMBED_DIM, \
            f"Expected embed_dim {C.VIT_EMBED_DIM}, got {self.vit.embed_dim}"

        # ── Projection Layer ─────────────────────────────────────────────
        # Maps each of the 196 patch tokens: 768 → proj_dim
        # A simple linear layer applied identically across all tokens.
        self.projection = nn.Sequential(
            nn.LayerNorm(C.VIT_EMBED_DIM),       # Stabilize before projection
            nn.Linear(C.VIT_EMBED_DIM, proj_dim, bias=False),
        )

        # Store dimensions
        self.proj_dim    = proj_dim
        self.num_patches = C.NUM_PATCHES          # 196
        self.feature_dim = C.NUM_PATCHES * proj_dim  # 6272

        # Initialize projection with small values for training stability
        nn.init.normal_(self.projection[1].weight, std=0.02)

        # Start with all ViT weights frozen (Phase 1)
        self.freeze_vit()

    # ─────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (N, 3, 224, 224) input image batch

        Returns:
            x_bar: (N, feature_dim) flattened real feature vector x̄
                   Before power normalization — normalization happens in
                   the full model to keep encoder/channel separation clean.
        """
        # ── Extract patch tokens from ViT ────────────────────────────────
        # timm's ViT with num_classes=0 returns the CLS token by default.
        # We need all patch tokens, so we hook into forward_features.
        patch_tokens = self._get_patch_tokens(s)  # (N, 196, 768)

        # ── Project each token: 768 → proj_dim ───────────────────────────
        x = self.projection(patch_tokens)          # (N, 196, proj_dim)

        # ── Flatten to 1D feature vector ──────────────────────────────────
        x_bar = x.flatten(start_dim=1)             # (N, 196 × proj_dim)

        return x_bar

    def _get_patch_tokens(self, s: torch.Tensor) -> torch.Tensor:
        """
        Extract the 196 patch tokens (excluding CLS) from ViT.
        timm ViT stores tokens as (N, 1+196, 768) where index 0 is CLS.
        """
        # Run through patch embedding + positional embedding
        x = self.vit.patch_embed(s)               # (N, 196, 768)
        # Add CLS token and positional embeddings
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)      # (N, 197, 768)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # Pass through all transformer blocks
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)                      # (N, 197, 768)

        # Return only patch tokens (drop CLS token at index 0)
        return x[:, 1:, :]                        # (N, 196, 768)

    # ─────────────────────────────────────────────────────────────────────
    # Fine-tuning Control Methods
    # ─────────────────────────────────────────────────────────────────────

    def freeze_vit(self):
        """Freeze all ViT parameters. Used in Phase 1."""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("[Encoder] All ViT weights frozen.")

    def unfreeze_last_n_blocks(self, n: int = 4):
        """
        Unfreeze the last n transformer blocks + final norm layer.
        Used in Phase 2.

        Args:
            n: Number of transformer blocks to unfreeze from the top.
               ViT-B/16 has 12 blocks total. We unfreeze last 4 by default.
        """
        # First freeze everything
        self.freeze_vit()

        # Unfreeze last n blocks
        total_blocks = len(self.vit.blocks)
        for i in range(total_blocks - n, total_blocks):
            for param in self.vit.blocks[i].parameters():
                param.requires_grad = True

        # Unfreeze final layer norm
        for param in self.vit.norm.parameters():
            param.requires_grad = True

        print(f"[Encoder] Unfrozen last {n} ViT blocks + final norm.")

    def unfreeze_all(self):
        """Unfreeze all ViT parameters. Used in Phase 3."""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("[Encoder] All ViT weights unfrozen.")

    def get_vit_params(self):
        """Return ViT parameters that require grad (for optimizer param groups)."""
        return [p for p in self.vit.parameters() if p.requires_grad]

    def get_projection_params(self):
        """Return projection layer parameters."""
        return list(self.projection.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
