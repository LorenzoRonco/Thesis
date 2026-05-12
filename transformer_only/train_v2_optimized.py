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

# Prevent Tkinter backend crashes in non-interactive training runs.
os.environ.setdefault("MPLBACKEND", "Agg")

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
from transformer_only.attention_visualizer import debug_attention_on_batch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SignLanguageTransformer with optimized hyperparameters")
    parser.add_argument("--tokenizer_min_freq", type=int, default=None, help="Minimum word frequency to include in vocabulary (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate (overrides config)")
    parser.add_argument("--target_word_dropout", type=float, default=None, help="Target word dropout probability (overrides config)")
    args = parser.parse_args()
    # ─────── Configuration ─────────────────────────────────
    cfg = {
        "train_csv": "dataset/how2sign_realigned_train.csv",
        "val_csv": "dataset/how2sign_realigned_val.csv",
        "train_landmarks_dir": "dataset/landmarks_face_reduced",
        "val_landmarks_dir": "dataset/landmarks_validation_face_reduced",
        "output_dir": "outputs/run2_optimized",
        "device": "cuda",
        "use_amp": True,
        
        # Model
        "feat_dim": 376,  # Updated: 17 pose + 21 left_hand + 21 right_hand + 35 compressed_face (12 mouth + 8 L_eye + 8 R_eye + 1 nose + 3 L_brow + 3 R_brow) * 4
        "d_model": 512,
        "nhead": 8,
        "num_enc_layers": 3,
        "num_dec_layers": 3,
        "dim_feedforward": 1024,
        "dropout": 0.3,
        "label_smoothing": 0.05,  # REDUCED from 0.1
        "max_src_len": 256,
        "max_tgt_len": 128,
        # Source embedding ablation: "mlp" | "temporal_cnn"
        "src_embedding_type": "temporal_cnn",
        
        # Training - OPTIMIZED
        "epochs": 100,
        "batch_size": 32,  # Reduced from 64 due to GPU memory (10.57GB GPU)
        "lr": 0.0005, 
        "weight_decay": 1e-4,
        "clip_norm": 1.0,
        "warmup_steps": 2000,  
        "num_workers": 4,
        
        # Data weighting - OPTIMIZED
        "pose_weight": 1.0,
        "hand_weight": 1.0, 
        "face_weight": 1.0,  

        # Hand-focused normalization + augmentation (anti-sink)
        "use_hand_relative_norm": True,
        "thumb_dropout_prob": 0.20,
        "hand_landmark_dropout_prob": 0.03,
        "hand_noise_std": 0.015,
        # Decoder word dropout (replace with <unk> during training)
        "target_word_dropout": 0.15,
        # Tokenizer minimum frequency (words appearing < min_freq times are mapped to <unk>)
        "tokenizer_min_freq": 5,
        # Guided Attention Loss
        "gal_weight": 10.0,
        "gal_sigma": 0.25,
        # Joint CTC-Attention auxiliary loss
        "lambda_ctc": 0.05,
        
        # Data sampling
        "train_subset_fraction": None,
        "val_subset_fraction": None,
        "train_max_samples": None,
        "val_max_samples": None,
        "subset_seed": 42,

        # Bucketing (riduce padding)
        "use_bucketing": True,
        "bucket_size": 200,
        "drop_last": False,
        
        # Logging
        "log_interval": 50,
        "val_interval_early": 10,  # Validate every N epochs for first 20 epochs
        "val_interval_late": 5,  # Validate every N epochs after epoch 20
        "early_phase_epochs": 20,  # Threshold between early and late phases
        "use_wandb": False,
        "debug_print_batch": False,
        "debug_max_items": 4,
        "debug_attention": True,  # Visualizza i pesi di cross-attention
        "debug_attention_interval": 10,  # Ogni N epoch
    }
    
    # ─────── Override config from CLI arguments ───────────
    if args.tokenizer_min_freq is not None:
        cfg["tokenizer_min_freq"] = args.tokenizer_min_freq
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.dropout is not None:
        cfg["dropout"] = args.dropout
    if args.target_word_dropout is not None:
        cfg["target_word_dropout"] = args.target_word_dropout
    
    # Log CLI overrides
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
    if cli_overrides:
        print(f"[RUN2] CLI overrides: {cli_overrides}")
    
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
    tokenizer = build_tokenizer(
        cfg["train_csv"],
        save_path=tok_path,
        min_freq=cfg.get("tokenizer_min_freq", 1),
    )
    
    loaders = build_dataloaders(
        train_csv=cfg["train_csv"],
        val_csv=cfg["val_csv"],
        train_landmarks_dir=cfg["train_landmarks_dir"],
        val_landmarks_dir=cfg["val_landmarks_dir"],
        tokenizer=tokenizer,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        max_src_len=cfg["max_src_len"],
        max_tgt_len=cfg["max_tgt_len"],
        pose_weight=cfg["pose_weight"],
        hand_weight=cfg["hand_weight"],
        face_weight=cfg["face_weight"],
        train_subset_fraction=cfg["train_subset_fraction"],
        val_subset_fraction=cfg["val_subset_fraction"],
        train_max_samples=cfg["train_max_samples"],
        val_max_samples=cfg["val_max_samples"],
        subset_seed=cfg["subset_seed"],
        use_bucketing=cfg["use_bucketing"],
        bucket_size=cfg["bucket_size"],
        drop_last=cfg["drop_last"],
        use_hand_relative_norm=cfg["use_hand_relative_norm"],
        thumb_dropout_prob=cfg["thumb_dropout_prob"],
        hand_landmark_dropout_prob=cfg["hand_landmark_dropout_prob"],
        hand_noise_std=cfg["hand_noise_std"],
        target_word_dropout=cfg.get("target_word_dropout", 0.0),
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"[RUN2] Train loader: {len(train_loader)} batches")
    print(f"[RUN2] Val loader: {len(val_loader)} batches")

    if cfg["debug_print_batch"]:
        debug_batch = next(iter(train_loader))
        names = debug_batch.get("names", [])
        sent_ids = debug_batch.get("sentence_ids", [])
        npy_paths = debug_batch.get("npy_paths", [])
        sentences = debug_batch.get("sentences", [])
        src_lens = debug_batch.get("src_lens")
        tgt_lens = debug_batch.get("tgt_lens")
        max_items = min(cfg["debug_max_items"], len(sentences))
        print("[RUN2] Debug batch (first items):")
        for i in range(max_items):
            name = names[i] if i < len(names) else "<missing>"
            sent_id = sent_ids[i] if i < len(sent_ids) else "<missing>"
            npy_path = npy_paths[i] if i < len(npy_paths) else "<missing>"
            sent = sentences[i] if i < len(sentences) else "<missing>"
            src_len = int(src_lens[i].item()) if src_lens is not None else -1
            tgt_len = int(tgt_lens[i].item()) if tgt_lens is not None else -1
            print(f"  [{i:02d}] name={name} | id={sent_id} | src_len={src_len} | tgt_len={tgt_len}")
            print(f"       npy={npy_path}")
            print(f"       tgt=\"{sent}\"")
    
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
        src_embedding_type=cfg["src_embedding_type"],
        pad_id=tokenizer.pad_id,
        label_smoothing=cfg["label_smoothing"],
        temporal_kernel_size=3,        # Passiamo esplicitamente il kernel a 3
        temporal_blocks=1,             # Passiamo esplicitamente i blocchi a 1
    )
    model.to(device)
    # Enable model-level diagnostic stats logging if requested
    model.debug_log_stats = cfg.get("debug_print_batch", False)
    model.debug_positional_encoding = True
    
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
    print(f"[RUN2] max_src_len: {cfg['max_src_len']}, bucketing: {cfg['use_bucketing']} (bucket_size={cfg['bucket_size']})")
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
            gal_weight=cfg.get("gal_weight", 10.0),
            gal_sigma=cfg.get("gal_sigma", 0.25),
            lambda_ctc=cfg.get("lambda_ctc", 0.3),
        )
        
        # Determine validation interval based on phase
        val_interval = (
            cfg["val_interval_early"] 
            if epoch <= cfg["early_phase_epochs"] 
            else cfg["val_interval_late"]
        )
        
        # Validate
        if epoch % val_interval == 0:
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
            
            # Debug attention weights
            if cfg["debug_attention"] and (epoch % cfg["debug_attention_interval"] == 0):
                print(f"  [DEBUG] Analyzing cross-attention weights...")
                debug_batch = next(iter(val_loader))
                debug_attention_on_batch(
                    model, debug_batch, tokenizer, device,
                    output_dir=output_dir / "attention_heatmaps"
                )
            
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
