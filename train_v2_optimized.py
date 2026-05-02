#!/usr/bin/env python3
"""
Optimized training run with better hyperparameters.

Key changes from previous run (run1):
- Learning rate: 0.0001 → 0.001 (10x higher)
- Batch size: 32 → 64 (larger, less noisy)
- Label smoothing: 0.1 → 0.05 (reduce over-smoothing)
- Hand weight: 1.3 → 2.0 (emphasize hands strongly)
- Face weight: 0.7 → 0.5 (reduce face noise)
- Warmup doubled to give model more stability
- Output to outputs/run2
"""

import os
import sys
import json
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from transformer_only.data.how2sign_loader import build_tokenizer, build_dataloaders
from transformer_only.models import SignLanguageTransformer
from transformer_only.training import WarmupCosineScheduler, train_epoch, validate
from transformer_only.evaluate import compute_bleu


def main():
    # ─────── Configuration ─────────────────────────────────
    cfg = {
        "train_csv": "dataset/how2sign_realigned_train.csv",
        "val_csv": "dataset/how2sign_realigned_val.csv",
        "train_landmarks_dir": "dataset/landmarks_normalized",
        "val_landmarks_dir": "dataset/landmarks_validation_normalized",
        "output_dir": "outputs/run2_optimized",
        "device": "cuda",
        "use_amp": True,
        
        # Model
        "feat_dim": 2108,
        "d_model": 512,
        "nhead": 8,
        "num_enc_layers": 6,
        "num_dec_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.15,
        "label_smoothing": 0.05,  # REDUCED from 0.1
        "max_src_len": 512,
        "max_tgt_len": 128,
        
        # Training - OPTIMIZED
        "epochs": 100,
        "batch_size": 32,  # Reduced from 64 due to GPU memory (10.57GB GPU)
        "lr": 0.001,  # INCREASED from 0.0001
        "weight_decay": 1e-4,
        "clip_norm": 1.0,
        "warmup_steps": 8000,  # DOUBLED from 4000
        "num_workers": 4,
        
        # Data weighting - OPTIMIZED
        "pose_weight": 1.0,
        "hand_weight": 2.0,  # INCREASED from 1.3
        "face_weight": 0.5,  # REDUCED from 0.7
        
        # Data sampling
        "train_subset_fraction": None,
        "val_subset_fraction": None,
        "train_max_samples": None,
        "val_max_samples": None,
        "subset_seed": 42,
        
        # Logging
        "log_interval": 50,
        "val_interval": 1,
        "use_wandb": False,
    }
    
    # Create output directory
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN2] Output directory: {output_dir}")
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[RUN2] Config saved to {config_path}")
    
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[RUN2] Device: {device}")
    
    # ─────── Build data ────────────────────────────────────
    print("[RUN2] Building tokenizer and dataloaders...")
    tok_path = output_dir / "tokenizer.json"
    tokenizer = build_tokenizer(cfg["train_csv"], save_path=tok_path)
    
    loaders = build_dataloaders(
        train_csv=cfg["train_csv"],
        val_csv=cfg["val_csv"],
        train_landmarks_dir=cfg["train_landmarks_dir"],
        val_landmarks_dir=cfg["val_landmarks_dir"],
        tokenizer=tokenizer,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pose_weight=cfg["pose_weight"],
        hand_weight=cfg["hand_weight"],
        face_weight=cfg["face_weight"],
        train_subset_fraction=cfg["train_subset_fraction"],
        val_subset_fraction=cfg["val_subset_fraction"],
        train_max_samples=cfg["train_max_samples"],
        val_max_samples=cfg["val_max_samples"],
        subset_seed=cfg["subset_seed"],
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"[RUN2] Train loader: {len(train_loader)} batches")
    print(f"[RUN2] Val loader: {len(val_loader)} batches")
    
    # ─────── Build model ──────────────────────────────────
    print("[RUN2] Building model...")
    model = SignLanguageTransformer(
        feat_dim=cfg["feat_dim"],
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_enc_layers=cfg["num_enc_layers"],
        num_dec_layers=cfg["num_dec_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_src_len=cfg["max_src_len"],
        max_tgt_len=cfg["max_tgt_len"],
        pad_id=tokenizer.pad_id,
        label_smoothing=cfg["label_smoothing"],
    )
    model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[RUN2] Model parameters: {n_params:,}")
    
    # ─────── Optimizer & Scheduler ────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=cfg["warmup_steps"],
        total_steps=total_steps,
        min_lr_ratio=0.05,
    )
    scaler = GradScaler(enabled=cfg["use_amp"])
    
    # ─────── Training loop ────────────────────────────────
    best_bleu = 0.0
    best_epoch = 0
    
    print("[RUN2] Starting training...")
    print(f"[RUN2] Epochs: {cfg['epochs']}, Learning rate: {cfg['lr']}, Batch size: {cfg['batch_size']}")
    print(f"[RUN2] Hand weight: {cfg['hand_weight']}, Face weight: {cfg['face_weight']}")
    print(f"[RUN2] Warmup: {cfg['warmup_steps']} steps, Label smoothing: {cfg['label_smoothing']}")
    print("")
    
    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        train_stats = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            clip_norm=cfg["clip_norm"],
            use_amp=cfg["use_amp"],
            log_interval=cfg["log_interval"],
        )
        
        # Validate
        if epoch % cfg["val_interval"] == 0:
            val_stats = validate(
                model=model,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                use_amp=cfg["use_amp"],
            )
            
            val_bleu = val_stats.get("bleu", 0.0)
            
            # Log epoch
            log_str = (
                f"[E{epoch:3d}] "
                f"train_loss={train_stats['loss']:.4f} "
                f"train_ppl={train_stats['ppl']:.2f} | "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_ppl={val_stats['ppl']:.2f} "
                f"val_bleu={val_bleu:.4f}"
            )
            print(log_str)
            
            # Save if best
            if val_bleu > best_bleu:
                best_bleu = val_bleu
                best_epoch = epoch
                
                ckpt_path = output_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_bleu": best_bleu,
                    "cfg": cfg,
                }, ckpt_path)
                print(f"  ✓ New best! Saved to {ckpt_path}")
        else:
            print(f"[E{epoch:3d}] train_loss={train_stats['loss']:.4f} train_ppl={train_stats['ppl']:.2f}")
    
    # Final save
    last_path = output_dir / "last.pt"
    torch.save({
        "epoch": cfg["epochs"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_bleu": best_bleu,
        "cfg": cfg,
    }, last_path)
    
    print("")
    print(f"[RUN2] Training complete!")
    print(f"[RUN2] Best BLEU: {best_bleu:.4f} at epoch {best_epoch}")
    print(f"[RUN2] Final checkpoint: {last_path}")


if __name__ == "__main__":
    main()
