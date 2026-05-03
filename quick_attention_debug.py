#!/usr/bin/env python3
"""
quick_attention_debug.py

Script per ispezionare rapidamente i pesi di Cross-Attention
senza eseguire il training completo.

Uso:
    python quick_attention_debug.py --checkpoint outputs/run2_optimized/best.pt
    
Oppure con modello random (per baseline):
    python quick_attention_debug.py --random
"""

import sys
from pathlib import Path
import argparse

import torch
import torch.nn as nn

# Add repo to path
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

from transformer_only.data.how2sign_loader import build_tokenizer, build_dataloaders
from transformer_only.models import SignLanguageTransformer
from transformer_only.attention_visualizer import debug_attention_on_batch


def main():
    parser = argparse.ArgumentParser(description="Quick Cross-Attention Debug")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--random", action="store_true",
                        help="Use random model weights (baseline)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for debug")
    parser.add_argument("--val_csv", type=str, default="dataset/how2sign_realigned_val.csv")
    parser.add_argument("--landmarks_dir", type=str, default="dataset/landmarks_validation_normalized")
    parser.add_argument("--output_dir", type=str, default="outputs/attention_debug")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ─────── Build model ──────────────────────────────────
    print("[DEBUG] Building model...")
    model = SignLanguageTransformer(
        feat_dim=2108,
        vocab_size=26976,  # How2Sign vocabulary
        d_model=512,
        nhead=8,
        num_enc_layers=6,
        num_dec_layers=6,
        dim_feedforward=2048,
        dropout=0.15,
        max_src_len=256,
        max_tgt_len=128,
        pad_id=0,
        label_smoothing=0.05,
    )
    model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and not args.random:
        print(f"[DEBUG] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("[DEBUG] Checkpoint loaded successfully")
    elif args.random:
        print("[DEBUG] Using random model weights (baseline)")
    else:
        print("[DEBUG] ⚠️  No checkpoint or --random provided. Using random initialization.")
    
    # ─────── Build tokenizer ──────────────────────────────
    print("[DEBUG] Building tokenizer...")
    
    # Try to load from checkpoint directory first
    ckpt_dir = Path(args.checkpoint).parent if args.checkpoint else Path(args.output_dir)
    tokenizer_path = ckpt_dir / "tokenizer.json"
    
    if tokenizer_path.exists():
        print(f"[DEBUG] Loading tokenizer from: {tokenizer_path}")
        from transformer_only.data.how2sign_loader import SentenceTokenizer
        tokenizer = SentenceTokenizer.load(tokenizer_path)
    else:
        print(f"[DEBUG] Building tokenizer from CSV: {args.val_csv}")
        from transformer_only.data.how2sign_loader import build_tokenizer
        tokenizer = build_tokenizer(args.val_csv)
    
    print(f"[DEBUG] Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # ─────── Load validation data ──────────────────────────
    print("[DEBUG] Loading validation data...")
    loaders_dict = build_dataloaders(
        train_csv="dataset/how2sign_realigned_train.csv",
        val_csv=args.val_csv,
        train_landmarks_dir="dataset/landmarks_normalized",
        val_landmarks_dir=args.landmarks_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=0,
        max_src_len=256,
        max_tgt_len=128,
    )
    val_loader = loaders_dict["val"]
    print(f"[DEBUG] Validation loader: {len(val_loader)} batches")
    
    # ─────── Debug attention ──────────────────────────────
    print("\n" + "="*80)
    print("CROSS-ATTENTION ANALYSIS")
    print("="*80)
    
    # Get first batch
    batch = next(iter(val_loader))
    
    # Visualize attention
    print("\n[DEBUG] Analyzing attention weights...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    debug_attention_on_batch(
        model, batch, tokenizer, device,
        output_dir=output_dir
    )
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    # ─────── Additional diagnostics ──────────────────────
    print("\n[DEBUG] Additional Model Diagnostics:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Forward pass on batch
    print("\n[DEBUG] Running forward pass...")
    with torch.no_grad():
        src = batch["src"].to(device)
        tgt_in = batch["tgt_input"].to(device)
        tgt_out = batch["tgt_output"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device) if "src_key_padding_mask" in batch else None
        tgt_in_mask = batch["tgt_in_key_padding_mask"].to(device) if "tgt_in_key_padding_mask" in batch else None
        
        logits, loss = model(
            src=src,
            tgt_input=tgt_in,
            tgt_output=tgt_out,
            src_key_padding_mask=src_mask,
            tgt_in_key_padding_mask=tgt_in_mask,
        )
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Input shape: src={src.shape}, tgt_in={tgt_in.shape}")
    
    print("\n✓ Debug complete!")


if __name__ == "__main__":
    main()
