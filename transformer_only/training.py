"""
train.py

Training loop per SignLanguageTransformer con:
  - Warm-up + Cosine Annealing (o ReduceLROnPlateau)
  - Gradient clipping
  - Mixed precision (torch.amp)
  - Checkpoint save/resume
  - Logging su console (+ opzionale WandB)
  - Valutazione BLEU ogni N epoche
"""

import os
import time
import math
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from transformer_only.data.how2sign_loader import build_tokenizer, build_dataloaders, SentenceTokenizer
from transformer_only.models import SignLanguageTransformer
from evaluate import compute_bleu, decode_batch          # definito in evaluate.py


# ──────────────────────────────────────────────
# Scheduler: warm-up lineare + cosine decay
# ──────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
        self.warmup = warmup_steps
        self.total  = total_steps
        self.min_r  = min_lr_ratio
        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup:
            return step / max(1, self.warmup)
        progress = (step - self.warmup) / max(1, self.total - self.warmup)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_r + (1 - self.min_r) * cosine


# ──────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────
def train_epoch(
    model:     SignLanguageTransformer,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    device:    torch.device,
    clip_norm: float,
    use_amp:   bool,
) -> dict:
    model.train()
    total_loss  = 0.0
    total_tok   = 0
    total_steps = 0
    t0          = time.time()

    for batch in loader:
        src      = batch["src"].to(device, non_blocking=True)
        tgt_in   = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out  = batch["tgt_output"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        tin_mask = batch["tgt_in_key_padding_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            _, loss = model(
                src=src,
                tgt_input=tgt_in,
                tgt_output=tgt_out,
                src_key_padding_mask=src_mask,
                tgt_in_key_padding_mask=tin_mask,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        n_tok        = (tgt_out != model.pad_id).sum().item()
        total_loss  += loss.item() * n_tok
        total_tok   += n_tok
        total_steps += 1

    avg_loss = total_loss / max(1, total_tok)
    ppl      = math.exp(min(avg_loss, 20))
    elapsed  = time.time() - t0
    return {"loss": avg_loss, "ppl": ppl, "steps": total_steps, "time": elapsed}


# ──────────────────────────────────────────────
# Validation step
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(
    model:   SignLanguageTransformer,
    loader,
    device:  torch.device,
    use_amp: bool,
    tokenizer: SentenceTokenizer,
    max_decode_len: int = 128,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tok  = 0
    all_hyps   = []
    all_refs   = []

    for batch in loader:
        src      = batch["src"].to(device, non_blocking=True)
        tgt_in   = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out  = batch["tgt_output"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        tin_mask = batch["tgt_in_key_padding_mask"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            _, loss = model(
                src=src,
                tgt_input=tgt_in,
                tgt_output=tgt_out,
                src_key_padding_mask=src_mask,
                tgt_in_key_padding_mask=tin_mask,
            )

        n_tok       = (tgt_out != model.pad_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tok  += n_tok

        # Greedy per BLEU
        hyps = model.greedy_decode(
            src=src,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=max_decode_len,
            src_key_padding_mask=src_mask,
        )
        refs = batch["sentences"]
        all_hyps.extend([tokenizer.decode(h) for h in hyps])
        all_refs.extend(refs)

    avg_loss = total_loss / max(1, total_tok)
    ppl      = math.exp(min(avg_loss, 20))
    bleu     = compute_bleu(all_hyps, all_refs)
    return {"loss": avg_loss, "ppl": ppl, "bleu": bleu}


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────
def save_checkpoint(
    path: Path,
    epoch: int,
    model: SignLanguageTransformer,
    optimizer,
    scheduler,
    scaler,
    best_bleu: float,
    cfg: dict,
):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "scaler":     scaler.state_dict(),
        "best_bleu":  best_bleu,
        "cfg":        cfg,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt.get("best_bleu", 0.0)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(cfg: dict):
    # ── Setup ─────────────────────────────────
    device  = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg["use_amp"] and device.type == "cuda"
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Tokenizer ─────────────────────────────
    tok_path = out_dir / "tokenizer.json"
    if tok_path.exists():
        tokenizer = SentenceTokenizer.load(tok_path)
        print(f"Tokenizer caricato: {tokenizer.vocab_size} token")
    else:
        tokenizer = build_tokenizer(cfg["train_csv"], save_path=tok_path)

    # ── Data ──────────────────────────────────
    loaders = build_dataloaders(
        train_csv     = cfg["train_csv"],
        val_csv       = cfg["val_csv"],
        landmarks_dir = cfg["landmarks_dir"],
        tokenizer     = tokenizer,
        batch_size    = cfg["batch_size"],
        num_workers   = cfg["num_workers"],
        max_src_len   = cfg["max_src_len"],
        max_tgt_len   = cfg["max_tgt_len"],
        test_csv      = cfg.get("test_csv"),
    )

    steps_per_epoch = len(loaders["train"])
    total_steps     = steps_per_epoch * cfg["epochs"]
    warmup_steps    = cfg["warmup_steps"]

    # ── Modello ───────────────────────────────
    model = SignLanguageTransformer(
        feat_dim       = cfg["feat_dim"],
        vocab_size     = tokenizer.vocab_size,
        d_model        = cfg["d_model"],
        nhead          = cfg["nhead"],
        num_enc_layers = cfg["num_enc_layers"],
        num_dec_layers = cfg["num_dec_layers"],
        dim_feedforward= cfg["dim_feedforward"],
        dropout        = cfg["dropout"],
        max_src_len    = cfg["max_src_len"],
        max_tgt_len    = cfg["max_tgt_len"],
        pad_id         = tokenizer.pad_id,
        label_smoothing= cfg["label_smoothing"],
    ).to(device)

    print(f"Parametri trainabili: {model.num_parameters():,}")

    # ── Ottimizzatore ─────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        betas        = (0.9, 0.98),
        eps          = 1e-9,
        weight_decay = cfg["weight_decay"],
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler(enabled=use_amp)

    # ── Resume ────────────────────────────────
    start_epoch = 0
    best_bleu   = 0.0
    resume_path = out_dir / "last.pt"
    if cfg.get("resume") and resume_path.exists():
        start_epoch, best_bleu = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        print(f"Ripreso dall'epoca {start_epoch}, best BLEU: {best_bleu:.2f}")

    # ── WandB (opzionale) ─────────────────────
    use_wandb = cfg.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg.get("wandb_project", "sign2text"), config=cfg)

    # ── Training loop ─────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n{'═'*60}")
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        print(f"{'═'*60}")

        train_stats = train_epoch(
            model, loaders["train"], optimizer, scheduler,
            scaler, device, cfg["clip_norm"], use_amp,
        )
        print(
            f"  TRAIN  loss={train_stats['loss']:.4f}  "
            f"ppl={train_stats['ppl']:.2f}  "
            f"time={train_stats['time']:.1f}s  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        val_stats = validate(
            model, loaders["val"], device, use_amp, tokenizer,
            max_decode_len=cfg["max_tgt_len"],
        )
        print(
            f"  VAL    loss={val_stats['loss']:.4f}  "
            f"ppl={val_stats['ppl']:.2f}  "
            f"BLEU={val_stats['bleu']:.2f}"
        )

        # Checkpoint
        save_checkpoint(resume_path, epoch, model, optimizer, scheduler, scaler, best_bleu, cfg)

        if val_stats["bleu"] > best_bleu:
            best_bleu = val_stats["bleu"]
            best_path = out_dir / "best.pt"
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, scaler, best_bleu, cfg)
            print(f"  ★ Nuovo best BLEU: {best_bleu:.2f} → salvato in {best_path}")

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_stats["loss"],
                "train/ppl":  train_stats["ppl"],
                "val/loss":   val_stats["loss"],
                "val/ppl":    val_stats["ppl"],
                "val/bleu":   val_stats["bleu"],
                "lr":         optimizer.param_groups[0]["lr"],
            })

    print(f"\nTraining completato. Best BLEU: {best_bleu:.2f}")
    if use_wandb:
        import wandb
        wandb.finish()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Percorso a config.json (sovrascrive i default)")
    # Override singoli argomenti da CLI
    parser.add_argument("--train_csv",     type=str)
    parser.add_argument("--val_csv",       type=str)
    parser.add_argument("--test_csv",      type=str, default=None)
    parser.add_argument("--landmarks_dir", type=str)
    parser.add_argument("--output_dir",    type=str, default="outputs/run1")
    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--resume",        action="store_true")
    parser.add_argument("--use_wandb",     action="store_true")
    args = parser.parse_args()

    # Config di default
    cfg = {
        "train_csv":     args.train_csv,
        "val_csv":       args.val_csv,
        "test_csv":      args.test_csv,
        "landmarks_dir": args.landmarks_dir,
        "output_dir":    args.output_dir,
        "device":        "cuda",
        "use_amp":       True,
        # modello
        "feat_dim":       2108,
        "d_model":        512,
        "nhead":          8,
        "num_enc_layers": 6,
        "num_dec_layers": 6,
        "dim_feedforward":2048,
        "dropout":        0.1,
        "label_smoothing":0.1,
        "max_src_len":    512,
        "max_tgt_len":    128,
        # training
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "weight_decay":   1e-4,
        "clip_norm":      1.0,
        "warmup_steps":   4000,
        "num_workers":    4,
        "resume":         args.resume,
        "use_wandb":      args.use_wandb,
        "wandb_project":  "sign2text",
    }

    # Sovrascrittura da file JSON
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    main(cfg)