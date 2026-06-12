#!/usr/bin/env python3
"""
train_stage2_hybrid.py

Training dello Stadio 2 con architettura ibrida:
        gloss (orth) → translation (tedesco)

Nuovo stage 2:
    - GlossEncoder custom per le gloss PHOENIX
    - bridge lineare verso la dimensione di mBART
    - decoder mBART-large-cc25 fine-tunato con LoRA

Il training alleggerisce il carico GPU mantenendo il decoder mBART
pre-addestrato e aggiornando solo:
    - GlossEncoder
    - bridge
    - adapter LoRA del decoder mBART

Requisiti:
    pip install transformers peft

Utilizzo:
    python train_stage2.py
    python train_stage2.py --epochs 20 --lr 3e-5 --batch_size 4
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
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import numpy as np

from transformers import MBartTokenizer

from .two_stage import HybridGlossToText
from .train import WarmupCosineScheduler    # scheduler già esistente nel progetto

# Il tokenizer gloss del Modello 1 serve solo se si vuole costruire la pipeline
# di inferenza TwoStagePipelineMbart; non è necessario per il training dello stage 2.
from .data.phoenix_loader import SentenceTokenizer, build_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Seed globale e determinismo GPU
# ─────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_worker_init_fn(base_seed: int):
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GlossTranslationDataset(Dataset):
    """
    Dataset che carica coppie (orth, translation) dal CSV PHOENIX pipe-separated.

    Restituisce testo grezzo; la tokenizzazione avviene nel collate.

    Args:
        csv_path: CSV pipe-separated PHOENIX-2014-T
    """

    def __init__(self, csv_path: str | Path):
        self.samples: list[dict] = []
        self._load(csv_path)

    def _load(self, csv_path: str | Path):
        skipped = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                orth  = row.get("orth",        "").strip()
                trans = row.get("translation", "").strip()
                if not orth or not trans:
                    skipped += 1
                    continue
                self.samples.append({
                    "orth":        orth,
                    "translation": trans,
                    "name":        row.get("name", "").strip(),
                })
        if skipped:
            print(f"[GlossTranslationDataset] Saltati {skipped} campioni (campo vuoto)")
        print(
            f"[GlossTranslationDataset] Caricati {len(self.samples)} coppie "
            f"da {Path(csv_path).name}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]   # {"orth", "translation", "name"}


# ─────────────────────────────────────────────────────────────────────────────
# Collate con tokenizer gloss + MBartTokenizer
# ─────────────────────────────────────────────────────────────────────────────

def collate_stage2_hybrid(
    batch:         list[dict],
    gloss_tokenizer: SentenceTokenizer,
    mbart_tokenizer: MBartTokenizer,
    max_gloss_len: int = 64,
    max_trans_len: int = 128,
    training:      bool = False,
) -> dict:
    """
    Tokenizza le gloss con il tokenizer dello stage 1 e le traduzioni con
    MBartTokenizer.
    """
    gloss_texts = [item["orth"]        for item in batch]
    trans_texts = [item["translation"] for item in batch]

    gloss_ids_list = [
        gloss_tokenizer.encode(text, add_special_tokens=True)
        for text in gloss_texts
    ]
    if training:
        # Gloss dropout: rimuove il 10% dei token casualmente durante il training.
        # Riduce l'overfitting forzando il modello a non memorizzare le sequenze esatte.
        # I token speciali (bos/eos) vengono preservati perché add_special_tokens=True
        # li mette sempre in posizione 0 e -1.
        pad  = gloss_tokenizer.pad_id
        bos  = getattr(gloss_tokenizer, "bos_id", None)
        eos  = getattr(gloss_tokenizer, "eos_id", None)
        special = {t for t in [pad, bos, eos] if t is not None}
        gloss_ids_list = [
            [t for t in ids if t in special or random.random() > 0.1] or ids
            for ids in gloss_ids_list
        ]
    gloss_lens = [len(ids) for ids in gloss_ids_list]
    gloss_max = max(gloss_lens)
    gloss_padded = torch.full(
        (len(batch), gloss_max),
        gloss_tokenizer.pad_id,
        dtype=torch.long,
    )
    gloss_mask = torch.ones(len(batch), gloss_max, dtype=torch.bool)
    for i, ids in enumerate(gloss_ids_list):
        gloss_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        gloss_mask[i, : len(ids)] = False

    trans_enc = mbart_tokenizer(
        text_target=trans_texts,
        max_length=max_trans_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Sostituisce il pad_token_id con -100 nei labels: mBART calcola la loss
    # solo sulle posizioni non mascherate.
    labels = trans_enc["input_ids"].clone()
    labels[labels == mbart_tokenizer.pad_token_id] = -100

    return {
        "gloss_ids":              gloss_padded,
        "gloss_key_padding_mask": gloss_mask,
        "labels":                 labels,
        "orths":          gloss_texts,
        "trans_texts":    trans_texts,
        "names":          [item["name"] for item in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validazione (loss + BLEU-4)
# ─────────────────────────────────────────────────────────────────────────────

def validate_stage2_hybrid(
    model:          HybridGlossToText,
    loader:         DataLoader,
    tokenizer:      MBartTokenizer,
    device:         torch.device,
    use_amp:        bool = True,
    max_new_tokens: int  = 128,
) -> dict:
    from .bleu import compute_bleu as _phoenix_bleu

    model.eval()
    total_loss = total_tok = 0
    all_hyps: list[str] = []
    all_refs: list[str] = []

    with torch.no_grad():
        for batch in loader:
            gloss_ids      = batch["gloss_ids"].to(device)
            gloss_mask     = batch["gloss_key_padding_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                _, loss, _ = model(
                    gloss_ids=gloss_ids,
                    gloss_key_padding_mask=gloss_mask,
                    labels=labels,
                )

            n_tok = (labels != -100).sum().item()
            total_loss += loss.item() * n_tok
            total_tok  += n_tok

            generated = model.generate(
                gloss_ids=gloss_ids,
                gloss_key_padding_mask=gloss_mask,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                length_penalty=0.6,
            )
            hyps = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_hyps.extend(hyps)
            all_refs.extend(batch["trans_texts"])

    avg_loss = total_loss / max(1, total_tok)
    ppl      = torch.exp(torch.tensor(min(avg_loss, 20.0))).item()

    # FIX 1: il modello nelle prime epoche può generare frasi vuote → brevity
    # penalty va in divisione per zero. Filtriamo le coppie con ipotesi vuota
    # e restituiamo bleu=0 se non rimane nulla da valutare.
    pairs = [(h, r) for h, r in zip(all_hyps, all_refs) if h.strip()]
    if not pairs:
        print("  [val] Tutte le ipotesi sono vuote — BLEU=0.00 (modello ancora freddo)")
        model.train()
        return {"loss": avg_loss, "ppl": ppl, "bleu": 0.0}

    hyps_f, refs_f = zip(*pairs)
    ref_corpus = [[r.strip().split()] for r in refs_f]
    hyp_corpus = [h.strip().split()  for h in hyps_f]
    bleu4, *_  = _phoenix_bleu(ref_corpus, hyp_corpus, max_order=4, smooth=False)

    model.train()
    return {"loss": avg_loss, "ppl": ppl, "bleu": bleu4 * 100.0}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train HybridGlossToText con LoRA + fp16 + gradient checkpointing"
    )
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--grad_accum",      type=int,   default=None,
                        help="Gradient accumulation steps (default 8 → batch effettivo 32)")
    parser.add_argument("--lora_r",          type=int,   default=None)
    parser.add_argument("--stage1_dir",      type=str,   default=None)
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--early_stage_val_interval",      type=int, default=None)
    parser.add_argument("--early_stage_end_epoch",         type=int, default=None)
    parser.add_argument("--post_early_stage_val_interval", type=int, default=None)
    args = parser.parse_args()

    cfg = {
        # ── Dataset ───────────────────────────────────────────────────────────
        "train_csv": "dataset/PHOENIX-2014-T.train.corpus.csv",
        "val_csv":   "dataset/PHOENIX-2014-T.dev.corpus.csv",

        # ── Output ────────────────────────────────────────────────────────────
        "stage1_output_dir": "outputs/phoenix_run1",   # per caricare gloss tokenizer
        "output_dir":        "outputs/phoenix_stage2_hybrid",

        # ── Device e precisione ───────────────────────────────────────────────
        "device":  "cuda",
        "use_amp": True,   # fp16 mixed precision

        # ── Riproducibilità ───────────────────────────────────────────────────
        "seed": 42,

        # ── LoRA ──────────────────────────────────────────────────────────────
        # lora_r=16 è il default; aumentare a 32 per più capacità (+ VRAM).
        # LoRA resta sul decoder mBART, mentre l'encoder mBART viene congelato.
        "lora_r":           16,
        "lora_alpha":       32,        # convenzionalmente 2 × lora_r
        "lora_dropout":     0.15,
        "lora_target_modules": [
            "encoder_attn.q_proj",   # cross-attention query
            "encoder_attn.k_proj",   # cross-attention key  ← il più importante
            "encoder_attn.v_proj",   # cross-attention value
            "encoder_attn.out_proj", # cross-attention output
        ], 

        "label_smoothing": 0.1,

        # ── Lunghezze sequenze ────────────────────────────────────────────────
        "max_gloss_len": 64,
        "max_trans_len": 128,

        # ── Training ──────────────────────────────────────────────────────────
        # LR molto più bassa rispetto al training from scratch: mBART è già
        # pre-addestrato, vogliamo solo adattarlo al task gloss→tedesco.
        "epochs":              150,
        "batch_size":          4,      # batch fisico per GPU da 11 GB
        "grad_accum_steps":    8,      # batch effettivo = 4 × 8 = 32
        "lr":                  3e-5,
        "weight_decay":        1e-2,
        "clip_norm":           1.0,
        "warmup_steps":        500,    
        "num_workers":         4,
        "dropout":             0.15,

        # ── Validazione ───────────────────────────────────────────────────────
        "early_stage_val_interval":      2,
        "early_stage_end_epoch":         6,
        "post_early_stage_val_interval": 1,
        "max_new_tokens":               128,
    }

    # Override da CLI
    if args.epochs     is not None: cfg["epochs"]          = args.epochs
    if args.lr         is not None: cfg["lr"]              = args.lr
    if args.batch_size is not None: cfg["batch_size"]      = args.batch_size
    if args.grad_accum is not None: cfg["grad_accum_steps"] = args.grad_accum
    if args.lora_r     is not None: cfg["lora_r"]          = args.lora_r
    if args.stage1_dir is not None: cfg["stage1_output_dir"] = args.stage1_dir
    if args.seed       is not None: cfg["seed"]            = args.seed
    if args.early_stage_val_interval is not None:
        cfg["early_stage_val_interval"] = args.early_stage_val_interval
    if args.early_stage_end_epoch is not None:
        cfg["early_stage_end_epoch"] = args.early_stage_end_epoch
    if args.post_early_stage_val_interval is not None:
        cfg["post_early_stage_val_interval"] = args.post_early_stage_val_interval

    # ── Seed globale ──────────────────────────────────────────────────────────
    set_global_seed(cfg["seed"])
    print(f"[STAGE2-HYBRID] Seed globale: {cfg['seed']}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[STAGE2-HYBRID] Device: {device}")

    # ── Tokenizer gloss + MBartTokenizer ──────────────────────────────────────
    print("[STAGE2-HYBRID] Caricamento MBartTokenizer...")
    mbart_tokenizer = MBartTokenizer.from_pretrained(
        HybridGlossToText.MBART_NAME,
        src_lang="de_DE",
        tgt_lang="de_DE",
    )
    # L'id di de_DE serve a forzare la generazione in tedesco
    forced_bos_token_id = mbart_tokenizer.lang_code_to_id["de_DE"]
    print(f"[STAGE2-HYBRID] forced_bos_token_id (de_DE): {forced_bos_token_id}")

    # ── Gloss tokenizer (Modello 1) ───────────────────────────────────────────
    # Necessario solo per salvare il riferimento nel checkpoint e per costruire
    # TwoStagePipelineMbart in inferenza. Non usato nel training dello stage 2.
    stage1_dir     = Path(cfg["stage1_output_dir"])
    gloss_tok_path = stage1_dir / "tokenizer.json"
    if gloss_tok_path.exists():
        gloss_tokenizer = SentenceTokenizer.load(gloss_tok_path)
        print(
            f"[STAGE2-HYBRID] Tokenizer gloss (Modello 1) caricato da {gloss_tok_path} "
            f"(vocab={gloss_tokenizer.vocab_size})"
        )
    else:
        print(
            f"[STAGE2-HYBRID] Tokenizer gloss non trovato in {gloss_tok_path}. "
            f"Verrà costruito dal CSV (necessario per il training)."
        )
        gloss_tokenizer = build_tokenizer(
            train_csv=cfg["train_csv"],
            save_path=gloss_tok_path,
            min_freq=1,
            target_field="orth",
        )
        print(
            f"[STAGE2-HYBRID] Tokenizer gloss costruito (vocab={gloss_tokenizer.vocab_size})"
        )

    # ── Dataset e DataLoader ──────────────────────────────────────────────────
    _collate_train = partial(
        collate_stage2_hybrid,
        gloss_tokenizer=gloss_tokenizer,
        mbart_tokenizer=mbart_tokenizer,
        max_gloss_len=cfg["max_gloss_len"],
        max_trans_len=cfg["max_trans_len"],
        training=True,
    )
    _collate_val = partial(
        collate_stage2_hybrid,
        gloss_tokenizer=gloss_tokenizer,
        mbart_tokenizer=mbart_tokenizer,
        max_gloss_len=cfg["max_gloss_len"],
        max_trans_len=cfg["max_trans_len"],
        training=False,
    )

    train_ds = GlossTranslationDataset(cfg["train_csv"])
    val_ds   = GlossTranslationDataset(cfg["val_csv"])

    dl_generator = torch.Generator()
    dl_generator.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=_collate_train,
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
        generator=dl_generator,
        worker_init_fn=_make_worker_init_fn(cfg["seed"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=_collate_val,
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
        worker_init_fn=_make_worker_init_fn(cfg["seed"]),
    )
    print(f"[STAGE2-HYBRID] Train: {len(train_loader)} batch | Val: {len(val_loader)} batch")
    print(
        f"[STAGE2-HYBRID] Batch fisico={cfg['batch_size']} × "
        f"grad_accum={cfg['grad_accum_steps']} → "
        f"batch effettivo={cfg['batch_size'] * cfg['grad_accum_steps']}"
    )

    # ── Modello ───────────────────────────────────────────────────────────────
    print("[STAGE2-HYBRID] Caricamento GlossEncoder + mBART-large-cc25 + LoRA...")
    model = HybridGlossToText(
        gloss_vocab_size=gloss_tokenizer.vocab_size,
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 4),
        num_enc_layers=cfg.get("num_enc_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
        max_gloss_len=cfg["max_gloss_len"],
        gloss_pad_id=gloss_tokenizer.pad_id,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        lora_target_modules=cfg["lora_target_modules"],
        forced_bos_token_id=forced_bos_token_id,
        gradient_checkpointing=True,
    ).to(device)

    model.init_gloss_embeddings_from_mbart(gloss_tokenizer)

    model.print_trainable_parameters()
    total_params = model.num_parameters(trainable_only=False)
    train_params = model.num_parameters(trainable_only=True)
    print(
        f"[STAGE2-HYBRID] Parametri: {train_params:,} trainable / "
        f"{total_params:,} totali "
        f"({100 * train_params / total_params:.2f}%)"
    )

    # ── Optimizer e Scheduler ─────────────────────────────────────────────────
    # AdamW solo sui parametri trainable (pesi LoRA).
    # Filtrare esplicitamente evita errori con i buffer non-trainable di mBART.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # total_steps conta gli step dell'optimizer (ogni grad_accum_steps batch)
    total_steps = (cfg["epochs"] * len(train_loader)) // cfg["grad_accum_steps"]
    scheduler   = WarmupCosineScheduler(
        optimizer,
        warmup_steps=cfg["warmup_steps"],
        total_steps=total_steps,
        min_lr_ratio=0.05,
    )
    scaler = GradScaler(enabled=cfg["use_amp"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_bleu  = 0.0
    best_epoch = 0
    grad_accum = cfg["grad_accum_steps"]

    print(
        f"\n[STAGE2-HYBRID] Inizio training gloss→translation con GlossEncoder + mBART LoRA\n"
        f"  epochs={cfg['epochs']} | lr={cfg['lr']} | "
        f"batch={cfg['batch_size']} | grad_accum={grad_accum} | seed={cfg['seed']}\n"
    )

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = total_tok = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            gloss_ids   = batch["gloss_ids"].to(device)
            gloss_mask  = batch["gloss_key_padding_mask"].to(device)
            labels      = batch["labels"].to(device)

            # ── Forward ───────────────────────────────────────────────────────
            with autocast(device_type=device.type, enabled=cfg["use_amp"]):
                _, loss, _ = model(
                    gloss_ids=gloss_ids,
                    gloss_key_padding_mask=gloss_mask,
                    labels=labels,
                )
                # Scala la loss per il gradient accumulation:
                # ogni step contribuisce 1/grad_accum del gradiente totale.
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            # ── Optimizer step ogni grad_accum batch ──────────────────────────
            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["clip_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Tracking loss (usiamo loss × grad_accum per tornare al valore reale)
            n_tok = (labels != -100).sum().item()
            total_loss += loss.item() * grad_accum * n_tok
            total_tok  += n_tok

        # Ultimo step parziale (se len(train_loader) non è multiplo di grad_accum)
        if len(train_loader) % grad_accum != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg["clip_norm"],
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, total_tok)
        ppl      = torch.exp(torch.tensor(min(avg_loss, 20.0))).item()

        # ── Validazione ───────────────────────────────────────────────────────
        is_early_stage = epoch <= cfg["early_stage_end_epoch"]
        val_interval   = (
            cfg["early_stage_val_interval"] if is_early_stage
            else cfg["post_early_stage_val_interval"]
        )

        if epoch % val_interval == 0:
            val_stats   = validate_stage2_hybrid(
                model, val_loader, mbart_tokenizer, device,
                use_amp=cfg["use_amp"],
                max_new_tokens=cfg["max_new_tokens"],
            )
            stage_label = "early" if is_early_stage else "post-early"
            print(
                f"[E{epoch:3d}] train_loss={avg_loss:.4f} ppl={ppl:.2f} | "
                f"val_loss={val_stats['loss']:.4f} val_ppl={val_stats['ppl']:.2f} "
                f"val_bleu={val_stats['bleu']:.2f} ({stage_label}, ogni {val_interval} ep)"
            )

            if val_stats["bleu"] > best_bleu:
                best_bleu  = val_stats["bleu"]
                best_epoch = epoch

                best_dir = output_dir / "best"
                model.save(best_dir)

                torch.save(
                    {
                        "epoch":               epoch,
                        "best_bleu":           best_bleu,
                        "cfg":                 cfg,
                        "forced_bos_token_id": forced_bos_token_id,
                    },
                    output_dir / "best_meta.pt",
                )
                print(f"  ✓ Nuovo best! BLEU-4={best_bleu:.2f} → {best_dir}")
        else:
            print(f"[E{epoch:3d}] train_loss={avg_loss:.4f} ppl={ppl:.2f}")

    # ── Checkpoint finale ─────────────────────────────────────────────────────
    last_dir = output_dir / "last"
    model.save(last_dir)
    torch.save(
        {
            "epoch":               cfg["epochs"],
            "best_bleu":           best_bleu,
            "cfg":                 cfg,
            "forced_bos_token_id": forced_bos_token_id,
        },
        output_dir / "last_meta.pt",
    )

    print(
        f"\n[STAGE2-HYBRID] Completato!\n"
        f"  Best BLEU-4: {best_bleu:.2f} all'epoca {best_epoch}\n"
        f"  Pesi LoRA salvati in: {output_dir}"
    )

    # ── Esempio di caricamento per inferenza ──────────────────────────────────
    print("\n[STAGE2-HYBRID] Per caricare il modello in inferenza:")
    print(
        f"  meta = torch.load('{output_dir}/best_meta.pt')\n"
        f"  model = HybridGlossToText.load(\n"
        f"      checkpoint_dir='{output_dir}/best',\n"
        f"      gloss_vocab_size=gloss_tokenizer.vocab_size,\n"
        f"      forced_bos_token_id=meta['forced_bos_token_id'],\n"
        f"      d_model=meta['cfg'].get('d_model', 256),\n"
        f"      nhead=meta['cfg'].get('nhead', 4),\n"
        f"      num_enc_layers=meta['cfg'].get('num_enc_layers', 2),\n"
        f"      dim_feedforward=meta['cfg'].get('dim_feedforward', 512),\n"
        f"      dropout=meta['cfg'].get('dropout', 0.1),\n"
        f"      max_gloss_len=meta['cfg'].get('max_gloss_len', 64),\n"
        f"      gloss_pad_id=gloss_tokenizer.pad_id,\n"
        f"      lora_r=meta['cfg'].get('lora_r', 16),\n"
        f"      lora_alpha=meta['cfg'].get('lora_alpha', 32),\n"
        f"      lora_dropout=meta['cfg'].get('lora_dropout', 0.1),\n"
        f"      lora_target_modules=meta['cfg'].get('lora_target_modules', ['q_proj', 'v_proj']),\n"
        f"  )"
    )


if __name__ == "__main__":
    main()