"""
model.py
Complete Deep JSCC model combining encoder, channel, decoder, and gated net.

Two model variants:
  1. DeepJSCC      — Standard model (fixed SNR, Fig. 1 of paper)
  2. GatedDeepJSCC — Gated model with domain randomization (Fig. 3 of paper)

Both support:
  - End-to-end forward pass for training
  - Feature extraction for nearest subspace classification
  - PSNR evaluation for reconstruction
"""

import torch
import torch.nn as nn
import config as C
from channel import power_normalize, get_channel, sample_snr_db
from encoder import ViTEncoder
from decoder import ViTDecoder, GatedNet


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD DEEP JSCC MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DeepJSCC(nn.Module):
    """
    Standard Deep JSCC model for fixed-SNR training.
    Corresponds to Fig. 1 and Algorithm 1 in the paper.

    Forward pass flow:
      s → Encoder → x̄ → Power Norm → x̂ → Channel → ŷ → Decoder → ŝ
                                                      ↓
                                                  y_bar (for CRR loss + classifier)
    """

    def __init__(self,
                 proj_dim:     int  = C.PROJ_DIM,
                 pretrained:   bool = True,
                 channel_type: str  = C.CHANNEL_TYPE):
        super().__init__()

        self.encoder     = ViTEncoder(proj_dim=proj_dim, pretrained=pretrained)
        self.decoder     = ViTDecoder(proj_dim=proj_dim)
        self.channel     = get_channel(channel_type)
        self.proj_dim    = proj_dim
        self.feature_dim = C.FEATURE_DIM

    def forward(self, s: torch.Tensor,
                snr_db: float = C.SNR_TEST_DB) -> dict:
        """
        Full forward pass for training and evaluation.

        Args:
            s:      (N, 3, 224, 224) input image batch
            snr_db: Channel SNR in dB

        Returns:
            dict with:
                's_hat':    (N, 3, 224, 224) reconstructed image
                'y_bar':    (N, feature_dim) received features for CRR + classifier
                'sigma_sq': equivalent noise variance (scalar)
        """
        # ── TRANSMITTER ──────────────────────────────────────────────────
        x_bar = self.encoder(s)                    # (N, feature_dim)
        x_hat = power_normalize(x_bar)             # Eq. (5): ||x̂||² = b

        # ── CHANNEL (non-trainable) ───────────────────────────────────────
        y_hat, sigma_sq = self.channel(x_hat, snr_db)   # (N, feature_dim)

        # ── RECEIVER ─────────────────────────────────────────────────────
        # y_bar used for both reconstruction AND classification
        y_bar = y_hat                              # already equalized in channel
        s_hat = self.decoder(y_bar)               # (N, 3, 224, 224)

        return {
            's_hat':    s_hat,
            'y_bar':    y_bar,
            'sigma_sq': sigma_sq,
        }

    def extract_features(self, s: torch.Tensor,
                         snr_db: float = C.SNR_TEST_DB) -> torch.Tensor:
        """
        Extract received features for nearest subspace classification.
        Used during evaluation — no gradient computation needed.

        Args:
            s:      (N, 3, 224, 224) input images
            snr_db: Channel SNR in dB

        Returns:
            y_bar: (N, feature_dim) received features
        """
        with torch.no_grad():
            out = self.forward(s, snr_db)
        return out['y_bar']

    def get_param_groups(self, phase: int) -> list:
        """
        Return optimizer parameter groups for each training phase.

        Args:
            phase: 1, 2, or 3

        Returns:
            list of parameter group dicts for torch.optim
        """
        if phase == 1:
            # Only projection + decoder, ViT frozen
            return [
                {'params': self.encoder.get_projection_params(),
                 'lr': C.PHASE1_LR},
                {'params': list(self.decoder.parameters()),
                 'lr': C.PHASE1_LR},
            ]
        elif phase == 2:
            # Last 4 ViT blocks + projection + decoder
            self.encoder.unfreeze_last_n_blocks(n=4)
            return [
                {'params': self.encoder.get_vit_params(),
                 'lr': C.PHASE2_LR_VIT},
                {'params': self.encoder.get_projection_params(),
                 'lr': C.PHASE2_LR_REST},
                {'params': list(self.decoder.parameters()),
                 'lr': C.PHASE2_LR_REST},
            ]
        elif phase == 3:
            # All parameters
            self.encoder.unfreeze_all()
            return [
                {'params': self.encoder.get_vit_params(),
                 'lr': C.PHASE3_LR_VIT},
                {'params': self.encoder.get_projection_params(),
                 'lr': C.PHASE3_LR_REST},
                {'params': list(self.decoder.parameters()),
                 'lr': C.PHASE3_LR_REST},
            ]
        else:
            raise ValueError(f"Phase must be 1, 2, or 3. Got {phase}")


# ─────────────────────────────────────────────────────────────────────────────
# GATED DEEP JSCC MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GatedDeepJSCC(nn.Module):
    """
    Gated Deep JSCC with domain randomization.
    Corresponds to Fig. 3 and Algorithm 2 in the paper.

    Extends DeepJSCC with a GatedNet that adaptively prunes features
    based on channel conditions to reduce communication overhead.

    Gating flow:
      σ̂² → GatedNet → λ(σ̂²) ∈ {0,1}^(feature_dim)
      x̃ = λ ∘ x̂   (element-wise mask on encoded features)
      Transmit only the active features in x̃
      Receiver: reconstruct x̂-shaped vector from active features
    """

    def __init__(self,
                 proj_dim:     int   = C.PROJ_DIM,
                 pretrained:   bool  = True,
                 channel_type: str   = C.CHANNEL_TYPE,
                 gamma_0:      float = C.GAMMA_0):
        super().__init__()

        self.encoder     = ViTEncoder(proj_dim=proj_dim, pretrained=pretrained)
        self.decoder     = ViTDecoder(proj_dim=proj_dim)
        self.gated_net   = GatedNet(feature_dim=C.FEATURE_DIM, gamma_0=gamma_0)
        self.channel     = get_channel(channel_type)
        self.proj_dim    = proj_dim
        self.feature_dim = C.FEATURE_DIM

    def forward(self, s: torch.Tensor,
                snr_db: float = None) -> dict:
        """
        Full forward pass with gating and domain randomization.

        Args:
            s:      (N, 3, 224, 224) input image batch
            snr_db: Channel SNR in dB. If None, samples randomly
                    from [SNR_MIN_DB, SNR_MAX_DB] (domain randomization)

        Returns:
            dict with:
                's_hat':          (N, 3, 224, 224) reconstructed image
                'y_tilde':        (N, feature_dim) gated received features
                'sigma_sq':       equivalent noise variance
                'snr_db':         actual SNR used
                'activation_ratio': fraction of features activated
        """
        # ── Domain Randomization ─────────────────────────────────────────
        if snr_db is None:
            snr_db = sample_snr_db()               # Algorithm 2, step 5

        # ── TRANSMITTER ──────────────────────────────────────────────────
        x_bar = self.encoder(s)                    # (N, feature_dim)
        x_hat = power_normalize(x_bar)             # Eq. (5)

        # ── Channel: get σ̂² estimate (before masking) ────────────────────
        # We run a short channel pass to get σ̂² for the gated net
        # (In practice, σ̂² can be estimated from pilot symbols)
        _, sigma_sq = self.channel(x_hat, snr_db)  # Get σ̂² estimate

        # ── Gated Net: compute binary mask ────────────────────────────────
        # Expand sigma_sq to (N,) for per-sample mask
        sigma_sq_batch = sigma_sq.expand(x_hat.shape[0])
        binary_mask    = self.gated_net.get_binary_mask(sigma_sq_batch)
                                                   # (N, feature_dim)

        # ── Apply mask: x̃ = λ(σ̂²) ∘ x̂  — Eq. (11) ─────────────────────
        x_tilde = binary_mask * x_hat              # (N, feature_dim)

        # Re-normalize after masking to maintain power constraint
        x_tilde = power_normalize(x_tilde)

        # ── Channel transmission of masked features ───────────────────────
        y_hat, sigma_sq = self.channel(x_tilde, snr_db)

        # ── RECEIVER: Reconstruct gated features ─────────────────────────
        # Apply same mask at receiver to reconstruct the feature structure
        y_tilde = binary_mask * y_hat              # (N, feature_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        s_hat = self.decoder(y_tilde)              # (N, 3, 224, 224)

        activation_ratio = binary_mask.mean().item()

        return {
            's_hat':           s_hat,
            'y_tilde':         y_tilde,
            'sigma_sq':        sigma_sq,
            'snr_db':          snr_db,
            'activation_ratio': activation_ratio,
        }

    def extract_features(self, s: torch.Tensor,
                         snr_db: float = C.SNR_TEST_DB) -> torch.Tensor:
        """Extract received (gated) features for classification."""
        with torch.no_grad():
            out = self.forward(s, snr_db=snr_db)
        return out['y_tilde']

    def get_param_groups(self, phase: int) -> list:
        """Return optimizer parameter groups for each training phase."""
        gated_params = list(self.gated_net.parameters())

        if phase == 1:
            return [
                {'params': self.encoder.get_projection_params(),
                 'lr': C.PHASE1_LR},
                {'params': list(self.decoder.parameters()),
                 'lr': C.PHASE1_LR},
                {'params': gated_params,
                 'lr': C.PHASE1_LR},
            ]
        elif phase == 2:
            self.encoder.unfreeze_last_n_blocks(n=4)
            return [
                {'params': self.encoder.get_vit_params(),
                 'lr': C.PHASE2_LR_VIT},
                {'params': self.encoder.get_projection_params(),
                 'lr': C.PHASE2_LR_REST},
                {'params': list(self.decoder.parameters()),
                 'lr': C.PHASE2_LR_REST},
                {'params': gated_params,
                 'lr': C.PHASE2_LR_REST},
            ]
        elif phase == 3:
            self.encoder.unfreeze_all()
            return [
                {'params': self.encoder.get_vit_params(),
                 'lr': C.PHASE3_LR_VIT},
                {'params': self.encoder.get_projection_params(),
                 'lr': C.PHASE3_LR_REST},
                {'params': list(self.decoder.parameters()),
                 'lr': C.PHASE3_LR_REST},
                {'params': gated_params,
                 'lr': C.PHASE3_LR_REST},
            ]
        else:
            raise ValueError(f"Phase must be 1, 2, or 3. Got {phase}")
