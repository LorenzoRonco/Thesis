#!/usr/bin/env python3
"""
Standalone CTC diagnostics script for PHOENIX-2014-T checkpoints produced by train.py.

Esegui questo script per diagnosticare l'Encoder senza modificare il training loop:

    python transformer_only/ctc_diagnostics_standalone.py \\
        --checkpoint outputs/phoenix_run1/best.pt \\
        --config outputs/phoenix_run1/config.json \\
        --tokenizer_path outputs/phoenix_run1/tokenizer.json \\
        --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \\
        --landmarks_dir dataset/landmarks_dev

Questo script:
1. Carica il checkpoint PHOENIX
2. Esegue il forward pass su dati di validazione
3. Decodifica CTC greedy (Encoder)
4. Decodifica greedy Decoder
5. Confronta con ground truth
6. Stampa diagnostica
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformer_only.data.phoenix_loader import (
    PhoenixDataset,
    SentenceTokenizer,
    collate_fn,
)
from transformer_only.models import SignLanguageTransformer
from transformer_only.ctc_decode import decode_ctc_predictions, decode_ctc_batch_diagnostics
from transformer_only.evaluate import _compute_landmark_stats


def _unpack_model_outputs(outputs):
    if not isinstance(outputs, (tuple, list)):
        return outputs, None, None, None

    if len(outputs) >= 2 and torch.is_tensor(outputs[1]) and outputs[1].dim() == 0:
        ce_logits = outputs[0]
        ce_loss = outputs[1]
        attn = outputs[2] if len(outputs) > 2 else None
        return ce_logits, None, ce_loss, attn

    ce_logits = outputs[0]
    ctc_logits = outputs[1] if len(outputs) > 1 else None
    attn = outputs[2] if len(outputs) > 2 else None
    return ce_logits, ctc_logits, None, attn


def _infer_temporal_frontend_cfg(state_dict: dict) -> dict:
    """Infer temporal CNN hyperparameters from checkpoint tensor shapes."""
    kernel_size = None
    blocks = None

    in_proj_w = state_dict.get("src_embed.temporal_extractor.in_proj.weight")
    if isinstance(in_proj_w, torch.Tensor) and in_proj_w.dim() == 3:
        kernel_size = int(in_proj_w.shape[-1])

    block_indices: set[int] = set()
    prefix = "src_embed.temporal_extractor.blocks."
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        idx_str = suffix.split(".", 1)[0]
        if idx_str.isdigit():
            block_indices.add(int(idx_str))
    if block_indices:
        blocks = max(block_indices) + 1

    return {
        "temporal_kernel_size": kernel_size,
        "temporal_blocks": blocks,
    }


def main():
    parser = argparse.ArgumentParser(description="CTC Encoder Diagnostics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/phoenix_run1/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="outputs/phoenix_run1/config.json",
        help="Path to config.json from training",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="dataset/PHOENIX-2014-T.dev.corpus.csv",
        help="Validation CSV path",
    )
    parser.add_argument(
        "--landmarks_dir",
        type=str,
        default="dataset/landmarks_dev",
        help="Validation landmarks directory",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="dataset/PHOENIX-2014-T.train.corpus.csv",
        help="Training CSV path (used only for normalization stats)",
    )
    parser.add_argument(
        "--train_landmarks_dir",
        type=str,
        default="dataset/landmarks_train",
        help="Training landmarks directory (used only for normalization stats)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="outputs/phoenix_run1/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="Maximum number of examples to print",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--target_field",
        type=str,
        default=None,
        choices=["translation", "orth"],
        help="Target field to evaluate (defaults to the checkpoint cfg)",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[CTC_DIAG] Device: {device}")

    # ═══════════════════════════════════════════════════════════
    # LOAD CONFIG & CHECKPOINT
    # ═══════════════════════════════════════════════════════════
    import json

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)
    print(f"[CTC_DIAG] Config loaded from {config_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    has_ctc_head = any("ctc" in key.lower() for key in state_dict.keys())

    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    model_cfg = {
        "feat_dim": cfg.get("feat_dim", ckpt_cfg.get("feat_dim", 376)),
        "d_model": cfg.get("d_model", ckpt_cfg.get("d_model", 512)),
        "nhead": cfg.get("nhead", ckpt_cfg.get("nhead", 8)),
        "num_enc_layers": cfg.get("num_enc_layers", ckpt_cfg.get("num_enc_layers", 3)),
        "num_dec_layers": cfg.get("num_dec_layers", ckpt_cfg.get("num_dec_layers", 3)),
        "dim_feedforward": cfg.get("dim_feedforward", ckpt_cfg.get("dim_feedforward", 1024)),
        "dropout": cfg.get("dropout", ckpt_cfg.get("dropout", 0.1)),
        "max_src_len": cfg.get("max_src_len", ckpt_cfg.get("max_src_len", 256)),
        "max_tgt_len": cfg.get("max_tgt_len", ckpt_cfg.get("max_tgt_len", 128)),
        "src_embedding_type": cfg.get("src_embedding_type", ckpt_cfg.get("src_embedding_type", "temporal_cnn")),
        "label_smoothing": cfg.get("label_smoothing", ckpt_cfg.get("label_smoothing", 0.1)),
        "temporal_kernel_size": cfg.get("temporal_kernel_size", ckpt_cfg.get("temporal_kernel_size")),
        "temporal_blocks": cfg.get("temporal_blocks", ckpt_cfg.get("temporal_blocks")),
    }

    if model_cfg["src_embedding_type"] == "temporal_cnn":
        inferred = _infer_temporal_frontend_cfg(state_dict)
        if model_cfg.get("temporal_kernel_size") is None:
            model_cfg["temporal_kernel_size"] = inferred["temporal_kernel_size"] or 5
        if model_cfg.get("temporal_blocks") is None:
            model_cfg["temporal_blocks"] = inferred["temporal_blocks"] or 3

    print(f"[CTC_DIAG] Checkpoint loaded from {checkpoint_path}")

    # ═══════════════════════════════════════════════════════════
    # LOAD TOKENIZER
    # ═══════════════════════════════════════════════════════════
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = SentenceTokenizer.load(tokenizer_path)
    print(f"[CTC_DIAG] Tokenizer loaded: {tokenizer.vocab_size} tokens")

    # ═══════════════════════════════════════════════════════════
    # BUILD MODEL
    # ═══════════════════════════════════════════════════════════
    model = SignLanguageTransformer(
        feat_dim=model_cfg["feat_dim"],
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_enc_layers=model_cfg["num_enc_layers"],
        num_dec_layers=model_cfg["num_dec_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
        src_embedding_type=model_cfg["src_embedding_type"],
        pad_id=tokenizer.pad_id,
        label_smoothing=model_cfg["label_smoothing"],
        temporal_kernel_size=model_cfg["temporal_kernel_size"] or 5,
        temporal_blocks=model_cfg["temporal_blocks"] or 3,
    ).to(device)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(f"[CTC_DIAG] Warning: non-strict checkpoint load. Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}")
    model.eval()
    print(f"[CTC_DIAG] Model loaded and set to eval mode")

    if not has_ctc_head:
        print(
            "[CTC_DIAG] Warning: checkpoint predates the CTC head. "
            "The CTC projection is newly initialized, so encoder CTC predictions "
            "will not be meaningful until you retrain or fine-tune this model."
        )

    # ═══════════════════════════════════════════════════════════
    # BUILD DATASET & DATALOADER
    # ═══════════════════════════════════════════════════════════
    target_field = args.target_field or cfg.get("target_field") or ckpt_cfg.get("target_field", "orth")

    normalize_stats = None
    if args.train_csv and args.train_landmarks_dir:
        print("[CTC_DIAG] Calcolo stats normalizzazione dal training set...")
        normalize_stats = _compute_landmark_stats(args.train_csv, args.train_landmarks_dir)
        if normalize_stats is None:
            print("[CTC_DIAG] Warning: impossibile calcolare le stats; si procede senza.")

    dataset = PhoenixDataset(
        csv_path=args.val_csv,
        landmarks_dir=args.landmarks_dir,
        tokenizer=tokenizer,
        target_field=target_field,
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
        flatten_landmarks=True,
        pose_weight=cfg.get("pose_weight", ckpt_cfg.get("pose_weight", 1.0)),
        hand_weight=cfg.get("hand_weight", ckpt_cfg.get("hand_weight", 1.5)),
        face_weight=cfg.get("face_weight", ckpt_cfg.get("face_weight", 0.8)),
        normalize_stats=normalize_stats,
        use_hand_relative_norm=ckpt_cfg.get("use_hand_relative_norm", True),
    )

    # Validate feature-dim consistency between dataset and model to avoid
    # conv1d channel mismatches (happens when using original vs reduced landmarks).
    ds_feat = getattr(dataset, "_dataset_feat_dim", None)
    model_feat = int(model_cfg.get("feat_dim", 2108))
    if ds_feat is not None and ds_feat != model_feat:
        raise RuntimeError(
            f"[CTC_DIAG] Feature-dim mismatch: dataset provides feat_dim={ds_feat} "
            f"but model/checkpoint expects feat_dim={model_feat}.\n"
            "Pass --landmarks_dir pointing to the correct landmarks folder, or use a config that matches your checkpoint."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(
            b,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            unk_id=tokenizer.unk_id,
            eos_id=tokenizer.eos_id,
        ),
        pin_memory=(device.type == "cuda"),
    )

    print(f"[CTC_DIAG] Validation dataset: {len(dataset)} samples (target='{target_field}')")
    print(f"[CTC_DIAG] DataLoader: {len(loader)} batches")

    # ═══════════════════════════════════════════════════════════
    # FORWARD PASS & DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════
    all_ctc_decoded = []
    all_decoder_hyps = []
    all_refs = []
    all_ctc_stats = []

    with torch.no_grad():
        collected_examples = 0
        for batch_idx, batch in enumerate(loader):
            if collected_examples >= args.max_examples:
                break

            src = batch["src"].to(device, non_blocking=True)
            tgt_in = batch["tgt_input"].to(device, non_blocking=True)
            tgt_out = batch["tgt_output"].to(device, non_blocking=True)
            src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
            tin_mask = batch["tgt_in_key_padding_mask"].to(device, non_blocking=True)

            # Forward pass
            with torch.amp.autocast(device_type=device.type, enabled=args.use_amp and device.type == "cuda"):
                outputs = model(
                    src=src,
                    tgt_input=tgt_in,
                    tgt_output=tgt_out,
                    src_key_padding_mask=src_mask,
                    tgt_in_key_padding_mask=tin_mask,
                )

            ce_logits, ctc_logits, _, _ = _unpack_model_outputs(outputs)

            # ════════════ CTC Decoding (Encoder) ════════════
            if ctc_logits is not None:
                idx2token = {
                    idx: token
                    for token, idx in tokenizer.token2idx.items()
                }

                ctc_decoded = decode_ctc_predictions(
                    ctc_logits,
                    idx2token,
                    blank_idx=ctc_logits.size(-1) - 1,
                    pad_idx=tokenizer.pad_id,
                )
                all_ctc_decoded.extend(ctc_decoded)

                # Statistiche
                # Limit per-batch diagnostics to remaining examples requested
                per_batch_max = max(1, min(args.max_examples - collected_examples, ctc_logits.size(0)))
                ctc_diag = decode_ctc_batch_diagnostics(
                    ctc_logits,
                    idx2token,
                    tokenizer,
                    max_examples=per_batch_max,
                    blank_idx=ctc_logits.size(-1) - 1,
                    pad_idx=tokenizer.pad_id,
                )
                print(f"\n[Batch {batch_idx}] CTC Stats:")
                print(
                    f"  - Avg decoded len: {ctc_diag['stats']['avg_decoded_len']:.1f}"
                )
                print(f"  - Blanks removed: {ctc_diag['stats']['num_blanks_removed']}")
                print(f"  - Pads removed: {ctc_diag['stats']['num_pads_removed']}")
                print(
                    f"  - Avg collapses: {ctc_diag['stats']['avg_collapses_per_seq']:.1f}"
                )

            # ════════════ Decoder Greedy Decoding ════════════
            hyps = model.greedy_decode(
                src=src,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=cfg.get("max_tgt_len", 128),
                src_key_padding_mask=src_mask,
            )
            hyps_str = [tokenizer.decode(h) for h in hyps]
            all_decoder_hyps.extend(hyps_str)

            # ════════════ Ground truth ════════════
            batch_targets = batch.get("targets", [])
            # Only extend up to args.max_examples to avoid collecting unnecessary examples
            remaining = args.max_examples - collected_examples
            if remaining <= 0:
                break
            if len(batch_targets) <= remaining:
                all_refs.extend(batch_targets)
                collected_examples += len(batch_targets)
            else:
                all_refs.extend(batch_targets[:remaining])
                collected_examples += remaining

    # ═══════════════════════════════════════════════════════════
    # PRINT DIAGNOSTIC RESULTS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("CTC ENCODER DIAGNOSTICS (Greedy Decoding)")
    print("=" * 100)

    num_examples = min(args.max_examples, len(all_refs))
    for i in range(num_examples):
        print(f"\n[Example {i+1:02d}]")
        print(f"  REF (Ground Truth): {all_refs[i]}")
        if i < len(all_ctc_decoded):
            print(f"  ENC (CTC Greedy):   {all_ctc_decoded[i]}")
        if i < len(all_decoder_hyps):
            print(f"  DEC (Decoder):      {all_decoder_hyps[i]}")

    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print(
        """
    ✓ Se ENC (CTC Greedy) è SIMILE a REF (Ground Truth):
      → L'Encoder sta imparando i segni correttamente!
    → Il problema è nel Decoder (exposure bias, allucinazioni)
      → Soluzione: Aumentare beam search, beam width, o aggiungere coverage penalty
      
    ✗ Se ENC (CTC Greedy) è DIVERSO da REF (Ground Truth):
      → L'Encoder NON sta imparando i segni (CTC loss non aiuta)
      → Problema nel feature extraction o nel CTC training
      → Soluzione: Verificare feature normalization, aumentare lambda_ctc, o controllare dati
      
    ✗ Se ENC è IDENTICO o MOLTO SIMILE a DEC:
      → Il Decoder sta copiando i logit dell'Encoder senza apprendere
      → Gravissimo exposure bias, cross-attention è morta
      → Soluzione: Investigare scheduled sampling o variational dropout
    """
    )

    print("\n[CTC_DIAG] Diagnostics complete!")


if __name__ == "__main__":
    main()
