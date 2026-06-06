#!/usr/bin/env python3
"""
train_stage2.py

Training del Modello 2: gloss (orth) → translation (tedesco).

Il Modello 2 è un seq2seq puro (GlossToTextTransformer) che non vede
mai i landmark: legge le coppie (orth, translation) direttamente dal CSV
di PHOENIX-2014-T.

Prerequisito: il Modello 1 deve essere già addestrato (serve il suo tokenizer
per il vocabolario gloss). In alternativa si può costruire un tokenizer gloss
separato direttamente dalle colonne 'orth' del CSV.

Utilizzo:
  python train_stage2.py
  python train_stage2.py --epochs 50 --lr 1e-3

Fix applicati rispetto alla versione originale:
- [FIX 1] Seed globale per random, numpy, torch e CUDA → riproducibilità garantita
- [FIX 2] cudnn.deterministic=True e benchmark=False → no non-determinismo GPU
- [FIX 3] DataLoader worker seed tramite worker_init_fn e Generator → batch identici ad ogni run
- [FIX 4] "seed" aggiunto a cfg e overridabile da CLI (--seed)
"""

import os
import sys
import csv
import json
import random
from pathlib import Path
from functools import partial

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .data.phoenix_loader import SentenceTokenizer, build_tokenizer
from .two_stage import GlossToTextTransformer
from .train import WarmupCosineScheduler   # scheduler già esistente


# ─────────────────────────────────────────────────────────────────────────────
# [FIX 1 + 2] Seed globale e determinismo GPU
# ─────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """
    Fissa tutti i seed per garantire la riproducibilità completa.
    Chiamare PRIMA di creare qualsiasi oggetto (modello, loader, ecc.).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # [FIX 2]
    torch.backends.cudnn.benchmark = False       # [FIX 2]


# ─────────────────────────────────────────────────────────────────────────────
# [FIX 3] Worker init function per DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def _make_worker_init_fn(base_seed: int):
    """
    Restituisce una worker_init_fn che assegna un seed deterministico
    a ciascun worker process del DataLoader.
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


# ─────────────────────────────────────────────
# Dataset gloss → translation
# ─────────────────────────────────────────────
class GlossTranslationDataset(Dataset):
    """
    Dataset che legge coppie (orth, translation) dal CSV PHOENIX pipe-separated.

    Non carica nessun file .npy: opera solo su testo.

    Args:
        csv_path:        CSV pipe-separated PHOENIX-2014-T
        gloss_tokenizer: tokenizer per il campo 'orth'
        trans_tokenizer: tokenizer per il campo 'translation'
        max_gloss_len:   lunghezza massima gloss (token)
        max_trans_len:   lunghezza massima traduzione (token)
    """

    def __init__(
        self,
        csv_path:        str | Path,
        gloss_tokenizer: SentenceTokenizer,
        trans_tokenizer: SentenceTokenizer,
        max_gloss_len:   int = 64,
        max_trans_len:   int = 128,
    ):
        self.gloss_tok   = gloss_tokenizer
        self.trans_tok   = trans_tokenizer
        self.max_gloss   = max_gloss_len
        self.max_trans   = max_trans_len
        self.samples: list[dict] = []
        self._load(csv_path)

    def _load(self, csv_path: str | Path):
        skipped = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                orth  = row.get("orth", "").strip()
                trans = row.get("translation", "").strip()
                if not orth or not trans:
                    skipped += 1
                    continue
                self.samples.append({"orth": orth, "translation": trans,
                                     "name": row.get("name", "").strip()})
        if skipped:
            print(f"[GlossTranslationDataset] Saltati {skipped} campioni (campo vuoto)")
        print(f"[GlossTranslationDataset] Caricati {len(self.samples)} coppie da {Path(csv_path).name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # Gloss: fonte
        gloss_ids = self.gloss_tok.encode(s["orth"], add_special_tokens=True)
        if len(gloss_ids) > self.max_gloss:
            gloss_ids = gloss_ids[: self.max_gloss - 1] + [self.gloss_tok.eos_id]

        # Translation: target
        trans_ids = self.trans_tok.encode(s["translation"], add_special_tokens=True)
        if len(trans_ids) > self.max_trans:
            trans_ids = trans_ids[: self.max_trans - 1] + [self.trans_tok.eos_id]

        return {
            "gloss":       torch.tensor(gloss_ids,  dtype=torch.long),
            "translation": torch.tensor(trans_ids,  dtype=torch.long),
            "name":        s["name"],
            "orth":        s["orth"],
            "trans_text":  s["translation"],
        }


def collate_stage2(batch: list[dict], gloss_pad_id: int, trans_pad_id: int) -> dict:
    """Padda gloss e translation al massimo del batch."""
    glosses = [item["gloss"] for item in batch]
    transs  = [item["translation"] for item in batch]

    gloss_lens = [g.shape[0] for g in glosses]
    trans_lens = [t.shape[0] for t in transs]
    B = len(batch)

    G_max = max(gloss_lens)
    L_max = max(trans_lens)

    gloss_padded = torch.full((B, G_max), gloss_pad_id, dtype=torch.long)
    gloss_mask   = torch.ones(B, G_max, dtype=torch.bool)   # True = padding
    for i, (g, l) in enumerate(zip(glosses, gloss_lens)):
        gloss_padded[i, :l] = g
        gloss_mask[i,   :l] = False

    trans_padded = torch.full((B, L_max), trans_pad_id, dtype=torch.long)
    trans_mask   = torch.ones(B, L_max, dtype=torch.bool)
    for i, (t, l) in enumerate(zip(transs, trans_lens)):
        trans_padded[i, :l] = t
        trans_mask[i,   :l] = False

    # Shift per teacher forcing
    tgt_input  = trans_padded[:, :-1]
    tgt_output = trans_padded[:, 1:]
    tgt_in_mask  = trans_mask[:, :-1]
    tgt_out_mask = trans_mask[:, 1:]

    return {
        "gloss":                   gloss_padded,
        "gloss_key_padding_mask":  gloss_mask,
        "tgt_input":               tgt_input,
        "tgt_output":              tgt_output,
        "tgt_in_key_padding_mask": tgt_in_mask,
        "gloss_lens":              torch.tensor(gloss_lens),
        "trans_lens":              torch.tensor(trans_lens),
        "names":                   [item["name"]       for item in batch],
        "orths":                   [item["orth"]        for item in batch],
        "trans_texts":             [item["trans_text"]  for item in batch],
    }


# ─────────────────────────────────────────────
# Validazione rapida (BLEU-4)
# ─────────────────────────────────────────────
def validate_stage2(
    model:           GlossToTextTransformer,
    loader:          DataLoader,
    trans_tokenizer: SentenceTokenizer,
    device:          torch.device,
    use_amp:         bool = True,
    max_decode_len:  int = 128,
) -> dict:
    from bleu import compute_bleu as _phoenix_bleu

    model.eval()
    total_loss = total_tok = 0
    all_hyps: list[str] = []
    all_refs: list[str] = []

    with torch.no_grad():
        for batch in loader:
            gloss   = batch["gloss"].to(device)
            gloss_m = batch["gloss_key_padding_mask"].to(device)
            tgt_in  = batch["tgt_input"].to(device)
            tgt_out = batch["tgt_output"].to(device)
            tgt_in_m = batch["tgt_in_key_padding_mask"].to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                _, loss, _ = model(
                    gloss_ids=gloss,
                    tgt_input=tgt_in,
                    tgt_output=tgt_out,
                    gloss_key_padding_mask=gloss_m,
                    tgt_in_key_padding_mask=tgt_in_m,
                )

            n_tok = (tgt_out != model.trans_pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tok  += n_tok

            hyps = model.greedy_decode(
                gloss_ids=gloss,
                bos_id=trans_tokenizer.bos_id,
                eos_id=trans_tokenizer.eos_id,
                max_len=max_decode_len,
                gloss_key_padding_mask=gloss_m,
            )
            all_hyps.extend([trans_tokenizer.decode(h) for h in hyps])
            all_refs.extend(batch["trans_texts"])

    avg_loss = total_loss / max(1, total_tok)
    ppl = torch.exp(torch.tensor(min(avg_loss, 20.0))).item()

    ref_corpus = [[r.strip().split()] for r in all_refs]
    hyp_corpus = [h.strip().split()  for h in all_hyps]
    bleu4, *_ = _phoenix_bleu(ref_corpus, hyp_corpus, max_order=4, smooth=False)

    model.train()
    return {"loss": avg_loss, "ppl": ppl, "bleu": bleu4 * 100.0}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train GlossToTextTransformer (Stage 2)")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--stage1_dir", type=str,   default=None,
                        help="Output dir del Modello 1 (per caricare il tokenizer gloss)")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Seed globale per la riproducibilità (default: 42)")
    parser.add_argument("--early_stage_val_interval", type=int, default=None,
                        help="Ogni quante epoche eseguire la validazione BLEU nella early stage")
    parser.add_argument("--early_stage_end_epoch", type=int, default=None,
                        help="Ultima epoca inclusa nella early stage")
    parser.add_argument("--post_early_stage_val_interval", type=int, default=None,
                        help="Ogni quante epoche eseguire la validazione BLEU dopo la early stage")
    args = parser.parse_args()

    cfg = {
        # Dataset
        "train_csv": "dataset/PHOENIX-2014-T.train.corpus.csv",
        "val_csv":   "dataset/PHOENIX-2014-T.dev.corpus.csv",

        # Directory output Modello 1 (per tokenizer gloss)
        # Se None, si costruisce un tokenizer gloss dal CSV
        "stage1_output_dir": "outputs/phoenix_run1",

        # Output Modello 2
        "output_dir": "outputs/phoenix_stage2",

        "device":  "cuda",
        "use_amp": True,

        # [FIX 4] Seed globale — controlla TUTTA la casualità del training
        "seed": 42,

        # Modello 2 — intenzionalmente più leggero del Modello 1
        # (il task gloss→tedesco è più semplice di landmark→tedesco)
        "d_model":         256,
        "nhead":           4,
        "num_enc_layers":  2,
        "num_dec_layers":  2,
        "dim_feedforward": 512,
        "dropout":         0.2,
        "label_smoothing": 0.1,
        "max_gloss_len":   64,
        "max_trans_len":   128,

        # Training
        "epochs":       80,
        "batch_size":   64,
        "lr":           1e-3,
        "weight_decay": 1e-4,
        "clip_norm":    1.0,
        "warmup_steps": 2000,
        "num_workers":  4,

        # Validazione
        "early_stage_val_interval":      5,
        "early_stage_end_epoch":        20,
        "post_early_stage_val_interval": 1,
        "max_decode_len":               128,
    }

    # Override da CLI
    if args.epochs is not None:     cfg["epochs"]            = args.epochs
    if args.lr is not None:         cfg["lr"]                = args.lr
    if args.batch_size is not None: cfg["batch_size"]        = args.batch_size
    if args.stage1_dir is not None: cfg["stage1_output_dir"] = args.stage1_dir
    if args.seed is not None:       cfg["seed"]              = args.seed
    if args.early_stage_val_interval is not None:
        cfg["early_stage_val_interval"] = args.early_stage_val_interval
    if args.early_stage_end_epoch is not None:
        cfg["early_stage_end_epoch"] = args.early_stage_end_epoch
    if args.post_early_stage_val_interval is not None:
        cfg["post_early_stage_val_interval"] = args.post_early_stage_val_interval

    # ─────── [FIX 1 + 2] Seed globale — PRIMA DI TUTTO ───
    set_global_seed(cfg["seed"])
    print(f"[STAGE2] Seed globale fissato: {cfg['seed']}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[STAGE2] Device: {device}")

    # ── Tokenizer gloss (Modello 1) ───────────
    stage1_dir = Path(cfg["stage1_output_dir"])
    gloss_tok_path = stage1_dir / "tokenizer.json"
    if gloss_tok_path.exists():
        gloss_tokenizer = SentenceTokenizer.load(gloss_tok_path)
        print(f"[STAGE2] Tokenizer gloss caricato da {gloss_tok_path} "
              f"(vocab={gloss_tokenizer.vocab_size})")
    else:
        print(f"[STAGE2] Tokenizer gloss non trovato in {gloss_tok_path}, "
              f"costruzione dal CSV (target_field='orth')...")
        gloss_tokenizer = build_tokenizer(
            train_csv=cfg["train_csv"],
            save_path=stage1_dir / "tokenizer.json",
            min_freq=1,
            target_field="orth",
        )

    # ── Tokenizer translation (Modello 2) ─────
    trans_tok_path = output_dir / "tokenizer_trans.json"
    if trans_tok_path.exists():
        trans_tokenizer = SentenceTokenizer.load(trans_tok_path)
        print(f"[STAGE2] Tokenizer translation caricato (vocab={trans_tokenizer.vocab_size})")
    else:
        trans_tokenizer = build_tokenizer(
            train_csv=cfg["train_csv"],
            save_path=trans_tok_path,
            min_freq=1,
            target_field="translation",
        )
        print(f"[STAGE2] Tokenizer translation costruito (vocab={trans_tokenizer.vocab_size})")

    # ── Dataset e loader ──────────────────────
    _collate = partial(
        collate_stage2,
        gloss_pad_id=gloss_tokenizer.pad_id,
        trans_pad_id=trans_tokenizer.pad_id,
    )

    train_ds = GlossTranslationDataset(
        cfg["train_csv"], gloss_tokenizer, trans_tokenizer,
        max_gloss_len=cfg["max_gloss_len"], max_trans_len=cfg["max_trans_len"],
    )
    val_ds = GlossTranslationDataset(
        cfg["val_csv"], gloss_tokenizer, trans_tokenizer,
        max_gloss_len=cfg["max_gloss_len"], max_trans_len=cfg["max_trans_len"],
    )

    # [FIX 3] Generator e worker_init_fn per riproducibilità del DataLoader
    dl_generator = torch.Generator()
    dl_generator.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
        generator=dl_generator,                          # [FIX 3] shuffle deterministico
        worker_init_fn=_make_worker_init_fn(cfg["seed"]), # [FIX 3] worker seed
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
        worker_init_fn=_make_worker_init_fn(cfg["seed"]), # [FIX 3] worker seed
    )
    print(f"[STAGE2] Train: {len(train_loader)} batch | Val: {len(val_loader)} batch")

    # ── Modello ───────────────────────────────
    model = GlossToTextTransformer(
        gloss_vocab_size=gloss_tokenizer.vocab_size,
        trans_vocab_size=trans_tokenizer.vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_enc_layers=cfg["num_enc_layers"],
        num_dec_layers=cfg["num_dec_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_gloss_len=cfg["max_gloss_len"],
        max_trans_len=cfg["max_trans_len"],
        gloss_pad_id=gloss_tokenizer.pad_id,
        trans_pad_id=trans_tokenizer.pad_id,
        label_smoothing=cfg["label_smoothing"],
    ).to(device)

    print(f"[STAGE2] Parametri modello: {model.num_parameters():,}")

    # ── Optimizer & Scheduler ─────────────────
    optimizer   = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler   = WarmupCosineScheduler(
        optimizer, warmup_steps=cfg["warmup_steps"],
        total_steps=total_steps, min_lr_ratio=0.05,
    )
    scaler = GradScaler(enabled=cfg["use_amp"])

    # ── Training loop ─────────────────────────
    best_bleu  = 0.0
    best_epoch = 0

    print(f"\n[STAGE2] Inizio training gloss→translation")
    print(f"  epochs={cfg['epochs']} | lr={cfg['lr']} | batch={cfg['batch_size']} | seed={cfg['seed']}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = total_tok = 0
        for batch_idx, batch in enumerate(train_loader):
            gloss    = batch["gloss"].to(device)
            gloss_m  = batch["gloss_key_padding_mask"].to(device)
            tgt_in   = batch["tgt_input"].to(device)
            tgt_out  = batch["tgt_output"].to(device)
            tgt_in_m = batch["tgt_in_key_padding_mask"].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=cfg["use_amp"]):
                _, loss, _ = model(
                    gloss_ids=gloss,
                    tgt_input=tgt_in,
                    tgt_output=tgt_out,
                    gloss_key_padding_mask=gloss_m,
                    tgt_in_key_padding_mask=tgt_in_m,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n_tok = (tgt_out != trans_tokenizer.pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tok  += n_tok

        avg_loss = total_loss / max(1, total_tok)
        ppl      = torch.exp(torch.tensor(min(avg_loss, 20.0))).item()

        is_early_stage = epoch <= cfg["early_stage_end_epoch"]
        val_interval = (
            cfg["early_stage_val_interval"] if is_early_stage
            else cfg["post_early_stage_val_interval"]
        )

        if epoch % val_interval == 0:
            val_stats = validate_stage2(
                model, val_loader, trans_tokenizer, device,
                use_amp=cfg["use_amp"], max_decode_len=cfg["max_decode_len"],
            )
            stage_label = "early" if is_early_stage else "post-early"
            print(
                f"[E{epoch:3d}] train_loss={avg_loss:.4f} ppl={ppl:.2f} | "
                f"val_loss={val_stats['loss']:.4f} val_ppl={val_stats['ppl']:.2f} "
                f"val_bleu={val_stats['bleu']:.2f} ({stage_label}, every {val_interval} epoche)"
            )
            if val_stats["bleu"] > best_bleu:
                best_bleu  = val_stats["bleu"]
                best_epoch = epoch
                ckpt_path  = output_dir / "best.pt"
                torch.save({
                    "epoch":             epoch,
                    "model":             model.state_dict(),
                    "optimizer":         optimizer.state_dict(),
                    "best_bleu":         best_bleu,
                    "cfg":               cfg,
                    "gloss_vocab_size":  gloss_tokenizer.vocab_size,
                    "trans_vocab_size":  trans_tokenizer.vocab_size,
                }, ckpt_path)
                print(f"  ✓ Nuovo best! BLEU-4={best_bleu:.2f} → {ckpt_path}")
        else:
            print(f"[E{epoch:3d}] train_loss={avg_loss:.4f} ppl={ppl:.2f}")

    # Checkpoint finale
    torch.save({
        "epoch":            cfg["epochs"],
        "model":            model.state_dict(),
        "best_bleu":        best_bleu,
        "cfg":              cfg,
        "gloss_vocab_size": gloss_tokenizer.vocab_size,
        "trans_vocab_size": trans_tokenizer.vocab_size,
    }, output_dir / "last.pt")

    print(f"\n[STAGE2] Completato! Best BLEU-4: {best_bleu:.2f} all'epoca {best_epoch}")


if __name__ == "__main__":
    main()