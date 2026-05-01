"""
train.py
Three-phase training pipeline for Deep JSCC with ViT on ImageNet-100.

Phase 1 (15 epochs):  Freeze ViT → train projection + decoder with MSE only
Phase 2 (40 epochs):  Unfreeze last 4 ViT blocks → full unified loss
Phase 3 (60 epochs):  Unfreeze all → full fine-tuning with small LR

Run: python train.py
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import config as C
from dataset import get_dataloaders, denormalize
from model import DeepJSCC, GatedDeepJSCC
from loss import JSCCLoss, compute_psnr
from classifier import NearestSubspaceClassifier

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running average of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, phase, metrics, path):
    torch.save({
        'epoch':     epoch,
        'phase':     phase,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics':   metrics,
    }, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} phase {ckpt['phase']} ← {path}")
    return ckpt['epoch'], ckpt['phase'], ckpt['metrics']


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE EPOCH TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn,
                    scaler, device, epoch, use_domain_rand=False):
    """
    Run one training epoch.

    Args:
        model:           DeepJSCC or GatedDeepJSCC
        loader:          Training DataLoader
        optimizer:       Optimizer with correct param groups
        loss_fn:         JSCCLoss instance
        scaler:          GradScaler for AMP
        device:          torch.device
        epoch:           Current epoch number (for logging)
        use_domain_rand: If True, sample random SNR each batch
                         (used for GatedDeepJSCC training)
    """
    model.train()
    meters = {k: AverageMeter() for k in ['loss', 'mse', 'delta_R', 'psnr']}
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Sample SNR ──────────────────────────────────────────────────
        if use_domain_rand:
            snr_db = None           # Model will sample internally
        else:
            snr_db = C.SNR_TEST_DB  # Fixed SNR for standard training

        # ── Forward pass (AMP) ──────────────────────────────────────────
        with autocast(enabled=C.AMP):
            if isinstance(model, GatedDeepJSCC):
                out = model(images, snr_db=snr_db)
                y_bar = out['y_tilde']
            else:
                out   = model(images, snr_db=snr_db or C.SNR_TEST_DB)
                y_bar = out['y_bar']

            s_hat  = out['s_hat']

            # Denormalize for PSNR (need pixel values in [0,1])
            s_orig = denormalize(images)
            s_rec  = denormalize(s_hat)

            loss_dict = loss_fn(
                s     = s_orig,       # Original image [0,1]
                s_hat = s_rec,        # Reconstructed image [0,1]
                y_bar = y_bar,        # Received features for CRR
                labels = labels,
            )
            loss = loss_dict['loss']

        # ── Backward ────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            C.GRAD_CLIP
        )
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics ─────────────────────────────────────────────────────
        with torch.no_grad():
            psnr = compute_psnr(s_orig, s_rec)

        N = images.shape[0]
        meters['loss'].update(loss_dict['loss'].item(), N)
        meters['mse'].update(loss_dict['mse'].item(), N)
        meters['delta_R'].update(loss_dict['delta_R'].item(), N)
        meters['psnr'].update(psnr.item(), N)

        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader)} | "
                  f"Loss {meters['loss'].avg:.4f} | "
                  f"MSE {meters['mse'].avg:.4f} | "
                  f"ΔR {meters['delta_R'].avg:.4f} | "
                  f"PSNR {meters['psnr'].avg:.2f} dB | "
                  f"Time {elapsed:.1f}s")

    return {k: v.avg for k, v in meters.items()}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device,
             snr_db: float = C.SNR_TEST_DB) -> dict:
    """
    Validate PSNR on the validation set.
    Classification accuracy is computed separately via NSC after training.
    """
    model.eval()
    psnr_meter = AverageMeter()

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        if isinstance(model, GatedDeepJSCC):
            out   = model(images, snr_db=snr_db)
            s_hat = out['s_hat']
        else:
            out   = model(images, snr_db=snr_db)
            s_hat = out['s_hat']

        s_orig = denormalize(images)
        s_rec  = denormalize(s_hat)
        psnr   = compute_psnr(s_orig, s_rec)
        psnr_meter.update(psnr.item(), images.shape[0])

    return {'psnr': psnr_meter.avg}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(phase: int, model, train_loader, val_loader,
              device, use_gated: bool = False):
    """
    Run a single training phase with its own optimizer and scheduler.

    Args:
        phase:        1, 2, or 3
        model:        DeepJSCC or GatedDeepJSCC
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        device:       torch.device
        use_gated:    If True, use domain randomization (GatedDeepJSCC)
    """
    phase_config = {
        1: {'epochs': C.PHASE1_EPOCHS, 'loss_mode': C.PHASE1_LOSS,
            'batch': C.PHASE1_BATCH},
        2: {'epochs': C.PHASE2_EPOCHS, 'loss_mode': C.PHASE2_LOSS,
            'batch': C.PHASE2_BATCH},
        3: {'epochs': C.PHASE3_EPOCHS, 'loss_mode': C.PHASE3_LOSS,
            'batch': C.PHASE3_BATCH},
    }[phase]

    print(f"\n{'='*60}")
    print(f"  PHASE {phase} — {phase_config['epochs']} epochs | "
          f"Loss: {phase_config['loss_mode']}")
    print(f"{'='*60}")

    # ── Loss function ────────────────────────────────────────────────────
    loss_fn = JSCCLoss(
        beta_db    = C.BETA,
        epsilon_sq = C.EPSILON_SQ,
        mode       = phase_config['loss_mode'],
    ).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────
    param_groups = model.get_param_groups(phase)
    optimizer    = torch.optim.AdamW(
        param_groups,
        weight_decay = C.WEIGHT_DECAY,
    )

    # ── LR Scheduler: Cosine with warmup ─────────────────────────────────
    total_epochs   = phase_config['epochs']
    warmup_epochs  = min(C.LR_WARMUP_EPOCHS, total_epochs // 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── AMP Scaler ───────────────────────────────────────────────────────
    scaler    = GradScaler(enabled=C.AMP)
    best_psnr = 0.0

    for epoch in range(1, total_epochs + 1):
        print(f"\n[Phase {phase}] Epoch {epoch}/{total_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, epoch,
            use_domain_rand = use_gated
        )
        scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, device)

        print(f"  [Val] PSNR: {val_metrics['psnr']:.2f} dB | "
              f"Train Loss: {train_metrics['loss']:.4f}")

        # Save checkpoint
        if epoch % C.SAVE_EVERY == 0:
            ckpt_path = os.path.join(
                C.CHECKPOINT_DIR,
                f"phase{phase}_epoch{epoch}.pt"
            )
            save_checkpoint(model, optimizer, epoch, phase,
                          {**train_metrics, **val_metrics}, ckpt_path)

        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            best_path = os.path.join(C.CHECKPOINT_DIR,
                                     f"phase{phase}_best.pt")
            save_checkpoint(model, optimizer, epoch, phase,
                          {**train_metrics, **val_metrics}, best_path)

    print(f"\n[Phase {phase}] Done. Best Val PSNR: {best_psnr:.2f} dB")
    return best_psnr


# ─────────────────────────────────────────────────────────────────────────────
# NSC FITTING AFTER TRAINING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def fit_classifier(model, train_loader, device,
                   snr_db: float = C.SNR_TEST_DB) -> NearestSubspaceClassifier:
    """
    After training is complete, extract features from the entire training set
    and fit the nearest subspace classifier.

    Args:
        model:       Trained DeepJSCC or GatedDeepJSCC
        train_loader: Training DataLoader
        device:      torch.device
        snr_db:      SNR for feature extraction

    Returns:
        Fitted NearestSubspaceClassifier
    """
    print("\n[NSC] Extracting training features for classifier fitting...")
    model.eval()
    all_features = []
    all_labels   = []

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)

        if isinstance(model, GatedDeepJSCC):
            out   = model(images, snr_db=snr_db)
            feats = out['y_tilde']
        else:
            feats = model.extract_features(images, snr_db=snr_db)

        all_features.append(feats.cpu())
        all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_labels   = torch.cat(all_labels,   dim=0)

    nsc = NearestSubspaceClassifier(
        num_classes  = C.NUM_CLASSES,
        n_components = C.NSC_COMPONENTS,
    )
    nsc.fit(all_features, all_labels)
    nsc.save(os.path.join(C.CHECKPOINT_DIR, "nsc_classifier.pt"))
    return nsc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(C.SEED)
    C.make_dirs()
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")
    print(f"[Main] Compression ratio b/B = {C.COMP_RATIO:.4f}")
    print(f"[Main] Feature dim = {C.FEATURE_DIM}")

    # ── Choose model variant ─────────────────────────────────────────────
    # Set use_gated=True to train the GatedDeepJSCC (Section IV of paper)
    use_gated = False

    if use_gated:
        model = GatedDeepJSCC(pretrained=True).to(device)
        print("[Main] Using GatedDeepJSCC")
    else:
        model = DeepJSCC(pretrained=True).to(device)
        print("[Main] Using DeepJSCC")

    print(f"[Main] Trainable params (Phase 1): "
          f"{model.encoder.count_trainable_params():,}")

    # ── DataLoaders (rebuilt per phase for correct batch size) ───────────

    # ── PHASE 1: Freeze ViT, MSE only ───────────────────────────────────
    train_loader, val_loader = get_dataloaders(C.PHASE1_BATCH)
    run_phase(1, model, train_loader, val_loader, device, use_gated)

    # ── PHASE 2: Unfreeze last 4 blocks, full loss ───────────────────────
    train_loader, val_loader = get_dataloaders(C.PHASE2_BATCH)
    run_phase(2, model, train_loader, val_loader, device, use_gated)

    # ── PHASE 3: Full fine-tuning ─────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(C.PHASE3_BATCH)
    run_phase(3, model, train_loader, val_loader, device, use_gated)

    # ── Fit NSC Classifier ───────────────────────────────────────────────
    train_loader, _ = get_dataloaders(C.EVAL_BATCH)
    nsc = fit_classifier(model, train_loader, device)

    # ── Final save ───────────────────────────────────────────────────────
    final_path = os.path.join(C.CHECKPOINT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[Main] Training complete. Model saved → {final_path}")


if __name__ == "__main__":
    main()
