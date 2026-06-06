#!/usr/bin/env python3
"""
train.py

Script di training per SignLanguageTransformer su PHOENIX-2014-T.

Differenze rispetto al train How2Sign:
- Loader: phoenix_loader (CSV pipe-separated, colonne name/video/orth/translation)
- target_field: 'translation' (default) o 'orth' (gloss)
- Nessuna augmentation (rimossa)
- feat_dim: 376 (94 landmark × 4)
- Path dataset: dataset/PHOENIX-2014-T.train.corpus.csv / dataset/train
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import numpy as np
import math
from dataclasses import dataclass
from typing import Any
import torch.nn.functional as F

from .data.phoenix_loader import build_tokenizer, build_dataloaders, SentenceTokenizer
from .models.transformer import SignLanguageTransformer
from .bleu import compute_bleu as _phoenix_compute_bleu
from .attention_visualizer import run_attention_visualization_on_loader


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_num = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.total_steps = int(state_dict["total_steps"])
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self.base_lrs = list(state_dict["base_lrs"])
        self.step_num = int(state_dict["step_num"])

    def _scale(self, step_num: int) -> float:
        if self.warmup_steps > 0 and step_num <= self.warmup_steps:
            return step_num / max(self.warmup_steps, 1)

        progress = (step_num - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self) -> None:
        self.step_num += 1
        scale = self._scale(self.step_num)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale


@dataclass
class _LossOutputs:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    ctc_loss: torch.Tensor


def _decode_batch(token_ids: list[list[int]], tokenizer: SentenceTokenizer) -> list[str]:
    return [tokenizer.decode(ids) for ids in token_ids]


def _compute_bleu_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    ref_corpus = [[ref.strip().lower().split()] for ref in references]
    hyp_corpus = [hyp.strip().lower().split() for hyp in predictions]

    results: dict[str, float] = {}
    for order in [1, 2, 3, 4]:
        bleu_score, _, _, _, _, _ = _phoenix_compute_bleu(
            ref_corpus,
            hyp_corpus,
            max_order=order,
            smooth=False,
        )
        results[f"bleu_{order}"] = bleu_score * 100.0

    results["bleu"] = results["bleu_4"]
    return results


def _unwrap_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
    return out


def _compute_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    logits: tuple[torch.Tensor, ...],
    gal_weight: float = 0.0,
    gal_sigma: float = 0.0,
    lambda_ctc: float = 0.0,
) -> _LossOutputs:
    ce_logits = logits[0]
    ctc_logits = logits[1] if len(logits) > 1 else None

    pad_id = int(getattr(model, "pad_id", 0))
    vocab_size = ce_logits.size(-1)

    tgt_output = batch["tgt_output"]
    tgt_out_mask = batch["tgt_out_key_padding_mask"]

    ce_loss = F.cross_entropy(
        ce_logits.reshape(-1, vocab_size),
        tgt_output.reshape(-1),
        ignore_index=pad_id,
    )

    ctc_loss = ce_logits.new_tensor(0.0)
    if ctc_logits is not None and lambda_ctc > 0.0:
        blank_id = ctc_logits.size(-1) - 1
        log_probs = ctc_logits.log_softmax(-1).transpose(0, 1)
        input_lengths = (~batch["src_key_padding_mask"]).sum(dim=1).to(torch.long)
        target_lengths = (~tgt_out_mask).sum(dim=1).to(torch.long)
        targets = [seq[~mask].to(torch.long) for seq, mask in zip(tgt_output, tgt_out_mask)]
        flat_targets = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long, device=ce_logits.device)
        if flat_targets.numel() > 0:
            ctc_loss = F.ctc_loss(
                log_probs,
                flat_targets,
                input_lengths,
                target_lengths,
                blank=blank_id,
                zero_infinity=True,
            )

    total_loss = ce_loss + lambda_ctc * ctc_loss
    return _LossOutputs(loss=total_loss, ce_loss=ce_loss, ctc_loss=ctc_loss)


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler | None,
    device: torch.device | str,
    clip_norm: float = 1.0,
    use_amp: bool = True,
    log_interval: int = 50,
    gal_weight: float = 0.0,
    gal_sigma: float = 0.0,
    lambda_ctc: float = 0.0,
) -> dict[str, float]:
    device = torch.device(device)
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_ctc = 0.0
    n_batches = 0

    for step, batch in enumerate(loader, 1):
        batch = _unwrap_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(
                src=batch["src"],
                tgt_input=batch["tgt_input"],
                tgt_output=batch["tgt_output"],
                src_key_padding_mask=batch["src_key_padding_mask"],
                tgt_in_key_padding_mask=batch["tgt_in_key_padding_mask"],
            )
            if not isinstance(logits, tuple):
                logits = (logits,)
            loss_out = _compute_loss(
                model=model,
                batch=batch,
                logits=logits,
                gal_weight=gal_weight,
                gal_sigma=gal_sigma,
                lambda_ctc=lambda_ctc,
            )

        if scaler is not None and use_amp:
            scaler.scale(loss_out.loss).backward()
            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_out.loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        scheduler.step()

        total_loss += float(loss_out.loss.detach())
        total_ce += float(loss_out.ce_loss.detach())
        total_ctc += float(loss_out.ctc_loss.detach())
        n_batches += 1

        if log_interval and step % log_interval == 0:
            print(f"[train] step={step} loss={total_loss / n_batches:.4f}")

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce / denom,
        "ctc_loss": total_ctc / denom,
        "ppl": math.exp(min(total_loss / denom, 20.0)),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: torch.device | str,
    use_amp: bool = True,
) -> dict[str, float]:
    device = torch.device(device)
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_ctc = 0.0
    n_batches = 0
    all_hyps: list[str] = []
    all_refs: list[str] = []

    for batch in loader:
        batch_refs = batch["targets"]
        batch = _unwrap_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(
                src=batch["src"],
                tgt_input=batch["tgt_input"],
                tgt_output=batch["tgt_output"],
                src_key_padding_mask=batch["src_key_padding_mask"],
                tgt_in_key_padding_mask=batch["tgt_in_key_padding_mask"],
            )
            if not isinstance(logits, tuple):
                logits = (logits,)
            loss_out = _compute_loss(
                model=model,
                batch=batch,
                logits=logits,
            )

        total_loss += float(loss_out.loss)
        total_ce += float(loss_out.ce_loss)
        total_ctc += float(loss_out.ctc_loss)
        n_batches += 1

        hyps = model.greedy_decode(
            src=batch["src"],
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=128,
            src_key_padding_mask=batch["src_key_padding_mask"],
        )
        all_hyps.extend(_decode_batch(hyps, tokenizer))
        all_refs.extend(batch_refs)

    denom = max(n_batches, 1)
    bleu_scores = _compute_bleu_scores(all_hyps, all_refs) if all_hyps and all_refs else {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bleu": 0.0}
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce / denom,
        "ctc_loss": total_ctc / denom,
        "ppl": math.exp(min(total_loss / denom, 20.0)),
        **bleu_scores,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SignLanguageTransformer su PHOENIX-2014-T")
    parser.add_argument("--tokenizer_min_freq", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--target_field", type=str, default=None,
                        help="'translation' (default) o 'orth' (gloss)")
    args = parser.parse_args()

    # ─────── Configurazione ────────────────────────────────
    cfg = {
        # Dataset - PHOENIX-2014-T
        "train_csv":           "dataset/PHOENIX-2014-T.train.corpus.csv",
        "val_csv":             "dataset/PHOENIX-2014-T.dev.corpus.csv",
        "train_landmarks_dir": "dataset/landmarks_train",
        "val_landmarks_dir":   "dataset/landmarks_dev",
        "output_dir":          "outputs/phoenix_run1",
        "target_field":        "orth",   # "orth" per i gloss, "translation" per le traduzioni in tedesco

        "device":  "cuda",
        "use_amp": True,

        # Modello
        "feat_dim":        376,   # 94 landmark × 4
        "d_model":         512,
        "nhead":           8,
        "num_enc_layers":  3,
        "num_dec_layers":  3,
        "dim_feedforward": 1024,
        "dropout":         0.3,
        "label_smoothing": 0.1,
        "max_src_len":     256,
        "max_tgt_len":     128,
        "src_embedding_type": "temporal_cnn",
        "temporal_kernel_size": 3,
        "temporal_blocks":      1,

        # Training
        "epochs":       150,
        "batch_size":   32,
        "lr":           5e-4,
        "weight_decay": 1e-4,
        "clip_norm":    1.0,
        "warmup_steps": 4000,
        "num_workers":  4,

        # Pesi per gruppo di landmark
        "pose_weight": 0.8,
        "hand_weight": 1.5,   # mani più importanti per la LIS
        "face_weight": 0.5,

        # Normalizzazione relativa alle mani
        "use_hand_relative_norm": False,

        # Tokenizer
        "tokenizer_min_freq": 1,

        # Loss ausiliarie
        "gal_weight":  10.0,
        "gal_sigma":   0.25,
        "lambda_ctc":  0.3,

        # Subset (None = tutto il dataset)
        "train_subset_fraction": None,
        "val_subset_fraction":   None,
        "train_max_samples":     None,
        "val_max_samples":       None,
        "subset_seed":           42,

        # Bucketing
        "use_bucketing": True,
        "bucket_size":   200,
        "drop_last":     False,

        # Logging e validazione
        "log_interval":        50,
        "val_interval_early":  5,
        "val_interval_late":   2,
        "early_phase_epochs":  20,
        "debug_print_batch":   False,
        "debug_max_items":     4,
        # Attention visualization (integration with AttentionVisualizer)
        # Set to N to run visualization every N epochs (None to disable)
        "attention_viz_every": None,
        "attention_viz_max_batches": 2,
        "attention_viz_save": False,
    }

    # ─────── Override da CLI ───────────────────────────────
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
    if args.target_field is not None:
        cfg["target_field"] = args.target_field

    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
    if cli_overrides:
        print(f"[TRAIN] CLI overrides: {cli_overrides}")

    # ─────── Output directory ─────────────────────────────
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] Output: {output_dir}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")

    # ─────── Tokenizer ────────────────────────────────────
    print("[TRAIN] Costruzione tokenizer...")
    tok_path = output_dir / "tokenizer.json"
    tokenizer = build_tokenizer(
        train_csv=cfg["train_csv"],
        save_path=tok_path,
        min_freq=cfg["tokenizer_min_freq"],
        target_field=cfg["target_field"],
    )
    print(f"[TRAIN] Vocab size: {tokenizer.vocab_size}")

    # ─────── Dataloaders ──────────────────────────────────
    print("[TRAIN] Costruzione dataloaders...")
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
        target_field=cfg["target_field"],
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
        flatten_landmarks=True,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    print(f"[TRAIN] Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ─────── Controllo frame zero ─────────────────────────
    print("[TRAIN] Verifica frame zero nel training set...")
    zero_frames = total_frames = 0
    for sample in train_loader.dataset.samples:
        lm = np.load(sample["npy_path"])
        zero_frames += int((lm == 0).all(axis=-1).sum())
        total_frames += lm.shape[0]
    zero_pct = 100.0 * zero_frames / max(1, total_frames)
    print(f"[TRAIN] Frame completamente zero: {zero_frames}/{total_frames} ({zero_pct:.2f}%)")

    if cfg["debug_print_batch"]:
        debug_batch = next(iter(train_loader))
        names   = debug_batch.get("names", [])
        targets = debug_batch.get("targets", [])
        src_lens = debug_batch.get("src_lens")
        tgt_lens = debug_batch.get("tgt_lens")
        print("[TRAIN] Debug batch:")
        for i in range(min(cfg["debug_max_items"], len(names))):
            print(f"  [{i:02d}] name={names[i]} | "
                  f"src_len={int(src_lens[i])} | tgt_len={int(tgt_lens[i])}")
            print(f"       tgt=\"{targets[i]}\"")

    # ─────── Modello ──────────────────────────────────────
    print("[TRAIN] Costruzione modello...")
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
        temporal_kernel_size=cfg["temporal_kernel_size"],
        temporal_blocks=cfg["temporal_blocks"],
        pad_id=tokenizer.pad_id,
        label_smoothing=cfg["label_smoothing"],
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] Parametri: {n_params:,}")

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
    best_bleu  = 0.0
    best_epoch = 0

    print(f"\n[TRAIN] Inizio training su PHOENIX-2014-T")
    print(f"  target_field={cfg['target_field']} | epochs={cfg['epochs']} | "
          f"lr={cfg['lr']} | batch_size={cfg['batch_size']}")
    print(f"  feat_dim={cfg['feat_dim']} | hand_weight={cfg['hand_weight']} | "
          f"warmup={cfg['warmup_steps']} steps\n")

    for epoch in range(1, cfg["epochs"] + 1):

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
            gal_weight=cfg["gal_weight"],
            gal_sigma=cfg["gal_sigma"],
            lambda_ctc=cfg["lambda_ctc"],
        )

        # Frequenza di validazione adattiva
        val_interval = (
            cfg["val_interval_early"]
            if epoch <= cfg["early_phase_epochs"]
            else cfg["val_interval_late"]
        )

        if epoch % val_interval == 0:
            val_stats = validate(
                model=model,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                use_amp=cfg["use_amp"],
            )
            val_bleu = val_stats.get("bleu_4", val_stats.get("bleu", 0.0))

            print(
                f"[E{epoch:3d}] "
                f"train_loss={train_stats['loss']:.4f} "
                f"train_ppl={train_stats['ppl']:.2f} | "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_ppl={val_stats['ppl']:.2f} "
                f"val_bleu1={val_stats.get('bleu_1', 0.0):.4f} "
                f"val_bleu2={val_stats.get('bleu_2', 0.0):.4f} "
                f"val_bleu3={val_stats.get('bleu_3', 0.0):.4f} "
                f"val_bleu4={val_stats.get('bleu_4', 0.0):.4f}"
            )

            if val_bleu > best_bleu:
                best_bleu  = val_bleu
                best_epoch = epoch
                ckpt_path  = output_dir / "best.pt"
                torch.save({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "best_bleu": best_bleu,
                    "cfg":       cfg,
                }, ckpt_path)
                print(f"  ✓ Nuovo best! BLEU={best_bleu:.4f} → {ckpt_path}")
        # ─────── Attention visualization hook (optional) ─────────
        try:
            viz_every = cfg.get("attention_viz_every")
            if viz_every is not None and viz_every > 0 and epoch % viz_every == 0:
                print(f"[TRAIN] Running attention visualization (epoch {epoch})...")
                run_attention_visualization_on_loader(
                    model=model,
                    loader=val_loader,
                    tokenizer=tokenizer,
                    device=str(device),
                    output_dir=output_dir / "attention",
                    epoch=epoch,
                    max_batches=cfg.get("attention_viz_max_batches", 2),
                    top_k=5,
                    save_heatmaps=cfg.get("attention_viz_save", True),
                )
        except Exception as e:
            print(f"[TRAIN] Attention visualization failed: {e}")
        else:
            print(f"[E{epoch:3d}] train_loss={train_stats['loss']:.4f} "
                  f"train_ppl={train_stats['ppl']:.2f}")

    # ─────── Checkpoint finale ────────────────────────────
    last_path = output_dir / "last.pt"
    torch.save({
        "epoch":     cfg["epochs"],
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_bleu": best_bleu,
        "cfg":       cfg,
    }, last_path)

    print(f"\n[TRAIN] Completato!")
    print(f"[TRAIN] Best BLEU: {best_bleu:.4f} all'epoca {best_epoch}")
    print(f"[TRAIN] Ultimo checkpoint: {last_path}")


if __name__ == "__main__":
    main()