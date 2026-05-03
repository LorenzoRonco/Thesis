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
from torch.amp import GradScaler, autocast

from transformer_only.data.how2sign_loader import build_tokenizer, build_dataloaders, SentenceTokenizer
from transformer_only.models import SignLanguageTransformer
from transformer_only.evaluate import compute_bleu, decode_batch          # definito in evaluate.py


# ──────────────────────────────────────────────
# Scheduler: linear warmup followed by cosine annealing (SequentialLR wrapper)
# ──────────────────────────────────────────────
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


class WarmupCosineScheduler:
    """
    Wrapper that composes a Linear warmup followed by CosineAnnealingLR.

    Exposes a compatible API: step(), state_dict(), load_state_dict().
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
        self.optimizer = optimizer
        self.warmup = int(warmup_steps)
        self.total = int(total_steps)
        self.min_r = float(min_lr_ratio)

        # Protect against degenerate schedules
        main_steps = max(1, self.total - max(0, self.warmup))

        # Cosine annealing from base_lr -> base_lr * min_lr_ratio
        try:
            base_lr = optimizer.param_groups[0]["lr"]
            eta_min = base_lr * self.min_r
        except Exception:
            eta_min = 0.0

        self.cosine_sched = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=eta_min)

        # Warmup is optional; LinearLR requires start_factor > 0
        self._has_warmup = self.warmup > 0
        if self._has_warmup:
            warmup_start = 1e-8
            self.warmup_sched = LinearLR(
                optimizer,
                start_factor=warmup_start,
                end_factor=1.0,
                total_iters=self.warmup,
            )
            self.scheduler = SequentialLR(
                optimizer,
                schedulers=[self.warmup_sched, self.cosine_sched],
                milestones=[self.warmup],
            )
        else:
            self.warmup_sched = None
            self.scheduler = self.cosine_sched

    def step(self):
        return self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state):
        return self.scheduler.load_state_dict(state)


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
    log_interval: int,
) -> dict:
    model.train()
    total_loss  = 0.0
    total_tok   = 0
    total_steps = 0
    t0          = time.time()

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    last_log_time = time.time()

    for batch in loader:
        src      = batch["src"].to(device, non_blocking=True)
        tgt_in   = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out  = batch["tgt_output"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        tin_mask = batch["tgt_in_key_padding_mask"].to(device, non_blocking=True)

        if not hasattr(train_epoch, "_logged_padding"):
            src_lens = batch.get("src_lens")
            tgt_lens = batch.get("tgt_lens")
            if src_lens is not None and tgt_lens is not None:
                T_max = int(src_lens.max().item()) if src_lens.numel() else 1
                L_max = int(tgt_lens.max().item()) if tgt_lens.numel() else 1
                mean_src_len = src_lens.float().mean().item() if src_lens.numel() else 0.0
                mean_tgt_len = tgt_lens.float().mean().item() if tgt_lens.numel() else 0.0
                src_padding_ratio = (T_max - mean_src_len) / T_max if T_max > 0 else 0.0
                tgt_padding_ratio = (L_max - mean_tgt_len) / L_max if L_max > 0 else 0.0
                print("[train_epoch] Padding statistics (first batch):")
                print(f"  Source (landmarks): T_max={T_max}, mean_len={mean_src_len:.1f}, padding_ratio={src_padding_ratio:.1%}")
                print(f"  Target (tokens):    L_max={L_max}, mean_len={mean_tgt_len:.1f}, padding_ratio={tgt_padding_ratio:.1%}")
            train_epoch._logged_padding = True

        optimizer.zero_grad(set_to_none=True)

        with autocast(amp_device, enabled=use_amp):
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

        if log_interval and total_steps % log_interval == 0:
            elapsed = time.time() - last_log_time
            avg_loss = total_loss / max(1, total_tok)
            print(
                f"    step {total_steps:5d}/{len(loader):5d}  "
                f"loss={avg_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"{elapsed:.1f}s"
            )
            last_log_time = time.time()

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
    total_unk  = 0
    total_hyp_tok = 0
    total_hyp_unk = 0
    all_hyps   = []
    all_refs   = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for batch in loader:
        src      = batch["src"].to(device, non_blocking=True)
        tgt_in   = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out  = batch["tgt_output"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        tin_mask = batch["tgt_in_key_padding_mask"].to(device, non_blocking=True)

        with autocast(amp_device, enabled=use_amp):
            _, loss = model(
                src=src,
                tgt_input=tgt_in,
                tgt_output=tgt_out,
                src_key_padding_mask=src_mask,
                tgt_in_key_padding_mask=tin_mask,
            )

        n_tok       = (tgt_out != model.pad_id).sum().item()
        n_unk       = (tgt_out == tokenizer.unk_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tok  += n_tok
        total_unk  += n_unk

        # Greedy per BLEU
        hyps = model.greedy_decode(
            src=src,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=max_decode_len,
            src_key_padding_mask=src_mask,
        )
        total_hyp_tok += sum(len(h) for h in hyps)
        total_hyp_unk += sum(sum(1 for t in h if t == tokenizer.unk_id) for h in hyps)
        refs = batch["sentences"]
        all_hyps.extend([tokenizer.decode(h) for h in hyps])
        all_refs.extend(refs)

    avg_loss = total_loss / max(1, total_tok)
    ppl      = math.exp(min(avg_loss, 20))
    bleu     = compute_bleu(all_hyps, all_refs)
    tgt_unk_rate = total_unk / max(1, total_tok)
    hyp_unk_rate = total_hyp_unk / max(1, total_hyp_tok)
    avg_ref_len = total_tok / max(1, len(all_refs))
    avg_hyp_len = total_hyp_tok / max(1, len(all_hyps))
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "bleu": bleu,
        "tgt_unk_rate": tgt_unk_rate,
        "hyp_unk_rate": hyp_unk_rate,
        "avg_ref_len": avg_ref_len,
        "avg_hyp_len": avg_hyp_len,
    }


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
        train_landmarks_dir = cfg.get("train_landmarks_dir"),
        val_landmarks_dir   = cfg.get("val_landmarks_dir"),
        test_landmarks_dir  = cfg.get("test_landmarks_dir"),
        tokenizer     = tokenizer,
        batch_size    = cfg["batch_size"],
        num_workers   = cfg["num_workers"],
        max_src_len   = cfg["max_src_len"],
        max_tgt_len   = cfg["max_tgt_len"],
        pose_weight   = cfg.get("pose_weight", 1.0),
        hand_weight   = cfg.get("hand_weight", 1.6),
        face_weight   = cfg.get("face_weight", 0.7),
        test_csv      = cfg.get("test_csv"),
        train_subset_fraction = cfg.get("train_subset_fraction"),
        val_subset_fraction   = cfg.get("val_subset_fraction"),
        train_max_samples      = cfg.get("train_max_samples"),
        val_max_samples        = cfg.get("val_max_samples"),
        subset_seed           = cfg.get("subset_seed", 42),
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
    scaler    = GradScaler("cuda", enabled=use_amp)

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
            scaler, device, cfg["clip_norm"], use_amp, cfg["log_interval"],
        )
        print(
            f"  TRAIN  loss={train_stats['loss']:.4f}  "
            f"ppl={train_stats['ppl']:.2f}  "
            f"time={train_stats['time']:.1f}s  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        run_val = ((epoch + 1) % cfg["val_interval"] == 0) or (epoch + 1 == cfg["epochs"])
        val_stats = None
        if run_val:
            val_stats = validate(
                model, loaders["val"], device, use_amp, tokenizer,
                max_decode_len=cfg["max_tgt_len"],
            )
            print(
                f"  VAL    loss={val_stats['loss']:.4f}  "
                f"ppl={val_stats['ppl']:.2f}  "
                f"BLEU={val_stats['bleu']:.2f}"
            )
        else:
            print(f"  VAL    skipped (interval={cfg['val_interval']})")

        # Checkpoint
        save_checkpoint(resume_path, epoch, model, optimizer, scheduler, scaler, best_bleu, cfg)

        if val_stats and val_stats["bleu"] > best_bleu:
            best_bleu = val_stats["bleu"]
            best_path = out_dir / "best.pt"
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, scaler, best_bleu, cfg)
            print(f"  ★ Nuovo best BLEU: {best_bleu:.2f} → salvato in {best_path}")

        if use_wandb:
            import wandb
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": train_stats["loss"],
                "train/ppl":  train_stats["ppl"],
                "lr":         optimizer.param_groups[0]["lr"],
            }
            if val_stats:
                log_payload.update({
                    "val/loss":   val_stats["loss"],
                    "val/ppl":    val_stats["ppl"],
                    "val/bleu":   val_stats["bleu"],
                })
            wandb.log(log_payload)

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
    parser.add_argument("--output_dir",    type=str, default="outputs/run1")
    parser.add_argument("--lr",            type=float, default=1e-4)
    args = parser.parse_args()

    # Config di default
    cfg = {
        # Dataset paths (edit here)
        "train_csv":     args.train_csv or "dataset/how2sign_realigned_train.csv",
        "val_csv":       args.val_csv or "dataset/how2sign_realigned_val.csv",
        "test_csv":      args.test_csv,
        "train_landmarks_dir": "dataset/landmarks_normalized",
        "val_landmarks_dir":   "dataset/landmarks_validation_normalized",
        "test_landmarks_dir":  None,
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
        "pose_weight":   1.0,
        "hand_weight":   1.3,
        "face_weight":   0.7,
        # training
        "epochs":         100,
        "batch_size":     32,
        "lr":             args.lr,
        "weight_decay":   1e-4,
        "clip_norm":      1.0,
        "warmup_steps":   4000,
        "num_workers":    4,
        "resume":         False,
        "use_wandb":      False,
        "wandb_project":  "sign2text",
        "log_interval":   100,
        "val_interval":   1,
        "train_subset_fraction": None,
        "val_subset_fraction":   None,
        "train_max_samples": None,
        "val_max_samples":   None,
        "subset_seed":       42,
    }

    # Sovrascrittura da file JSON
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    main(cfg)