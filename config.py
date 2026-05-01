"""
config.py
All hyperparameters and settings for Deep JSCC with ViT on ImageNet-100.
Based on: "Semantic Communications for Image Recovery and Classification
via Deep Joint Source and Channel Coding" (Lyu et al., IEEE TWC 2024)
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_ROOT        = "/kaggle/input/imagenet100"  # Root of ImageNet-100 dataset
CHECKPOINT_DIR   = "./checkpoints"
LOG_DIR          = "./logs"
RESULTS_DIR      = "./results"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
IMAGE_SIZE       = 224          # ViT-B/16 native input resolution
NUM_CLASSES      = 100          # ImageNet-100
NUM_WORKERS      = 4

# ─────────────────────────────────────────────
# ENCODER (ViT-B/16)
# ─────────────────────────────────────────────
VIT_MODEL        = "vit_base_patch16_224"  # timm model name
VIT_EMBED_DIM    = 768          # ViT-B token dimension
NUM_PATCHES      = 196          # 14×14 patch grid (224/16 = 14)
PROJ_DIM         = 32           # Projection dim d: 768 → d per token
# b = NUM_PATCHES × PROJ_DIM / 2 complex symbols = 196×32/2 = 3136
# B = 224×224×3 = 150528 pixels
# b/B ≈ 0.0208  (compression ratio, comparable to paper's 0.042 for CIFAR-10)

# ─────────────────────────────────────────────
# CHANNEL
# ─────────────────────────────────────────────
# AWGN SNR range for domain randomization (in dB)
SNR_MIN_DB       = -3.0
SNR_MAX_DB       = 21.0
SNR_TEST_DB      = 12.0         # Fixed SNR for standard evaluation
CHANNEL_TYPE     = "awgn"       # "awgn" | "rayleigh" | "rician"
RICIAN_K         = 1.0          # Rician K-factor (only used if CHANNEL_TYPE="rician")

# Gated net: discretization segments
GATED_K_SEGMENTS = 10

# ─────────────────────────────────────────────
# LOSS FUNCTION  (Eq. 10 in paper)
# ─────────────────────────────────────────────
BETA             = -30.0        # dB — trade-off weight between ΔR and MSE
                                # sweep: -40 to -25 dB (Fig. 10 in paper)
EPSILON_SQ       = 0.5          # Coding distortion ε² (Proposition 1)

# ─────────────────────────────────────────────
# TRAINING — PHASE 1 (Freeze ViT, train decoder + projection only)
# ─────────────────────────────────────────────
PHASE1_EPOCHS    = 15
PHASE1_LR        = 1e-4         # Decoder + projection learning rate
PHASE1_BATCH     = 64           # Reduce if GPU OOM
PHASE1_LOSS      = "mse_only"   # Only MSE in phase 1

# ─────────────────────────────────────────────
# TRAINING — PHASE 2 (Unfreeze last 4 ViT blocks, full loss)
# ─────────────────────────────────────────────
PHASE2_EPOCHS    = 40
PHASE2_LR_VIT    = 1e-5         # Last 4 blocks of ViT
PHASE2_LR_REST   = 1e-4         # Decoder, projection, gated net
PHASE2_BATCH     = 64
PHASE2_LOSS      = "full"       # −β·ΔR + MSE

# ─────────────────────────────────────────────
# TRAINING — PHASE 3 (Full end-to-end fine-tuning)
# ─────────────────────────────────────────────
PHASE3_EPOCHS    = 60
PHASE3_LR_VIT    = 1e-5
PHASE3_LR_REST   = 1e-4
PHASE3_BATCH     = 64
PHASE3_LOSS      = "full"

# ─────────────────────────────────────────────
# GENERAL TRAINING SETTINGS
# ─────────────────────────────────────────────
SEED             = 42
DEVICE           = "cuda"       # "cuda" or "cpu"
AMP              = True         # Automatic mixed precision (saves GPU memory)
GRAD_CLIP        = 1.0          # Gradient clipping max norm
WEIGHT_DECAY     = 1e-4
LR_WARMUP_EPOCHS = 5            # Cosine schedule warmup
SAVE_EVERY       = 5            # Save checkpoint every N epochs

# ─────────────────────────────────────────────
# GATED NET HYPERPARAMETERS (Section IV in paper)
# ─────────────────────────────────────────────
GATED_HIDDEN_DIM = 64           # MLP hidden dimension
GATED_LAYERS     = 3            # Number of MLP layers
GAMMA_0          = 0.05         # Threshold γ₀ for binary mask (tune per dataset)

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
# For nearest subspace classifier (Eq. 13 in paper)
NSC_COMPONENTS   = 12           # pj: number of principal components per class
                                # Rule of thumb: floor(num_train_per_class / 10)
EVAL_BATCH       = 32

# ─────────────────────────────────────────────
# DERIVED CONSTANTS (do not edit)
# ─────────────────────────────────────────────
import math
FEATURE_DIM = NUM_PATCHES * PROJ_DIM   # Flattened real feature vector dim = 6272
NUM_SYMBOLS = FEATURE_DIM // 2         # Complex symbols b = 3136
B_PIXELS    = IMAGE_SIZE * IMAGE_SIZE * 3  # 150528
COMP_RATIO  = NUM_SYMBOLS / B_PIXELS   # ≈ 0.0208

def make_dirs():
    for d in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
