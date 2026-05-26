#!/usr/bin/env python3
"""
evaluate.py

Script di valutazione per SignLanguageTransformer su PHOENIX-2014-T.

Metriche:
  - BLEU-1/2/3/4 e BLEU corpus  → bleu.py (fornito da PHOENIX)
  - ROUGE-1/2/L                  → rouge.py (fornito da PHOENIX)
  - WER                          → implementazione interna
  - Diversità / similarità output → diagnostiche

Utilizzo:
  python evaluate.py \
    --checkpoint outputs/phoenix_run1/best.pt \
    --val_csv    dataset/PHOENIX-2014-T.dev.corpus.csv \
    --landmarks_dir dataset/landmarks_dev \
    --train_csv  dataset/PHOENIX-2014-T.train.corpus.csv \
    --train_landmarks_dir dataset/landmarks_train
    --target_field translation o orth 
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# Dataset e tokenizer PHOENIX
from .data.phoenix_loader import (
    PhoenixDataset,
    SentenceTokenizer,
    collate_fn,
    build_tokenizer,
)

# Modello
from .models.transformer import SignLanguageTransformer

# Metriche ufficiali PHOENIX
from .bleu import compute_bleu as _phoenix_compute_bleu
from .rouge import rouge as _phoenix_rouge


# ─────────────────────────────────────────────────────────────
# Metriche
# ─────────────────────────────────────────────────────────────

def compute_bleu_scores(predictions: list[str], references: list[str]) -> dict:
    """
    Calcola BLEU-1/2/3/4 e BLEU corpus usando bleu.py di PHOENIX.

    bleu.py si aspetta:
      reference_corpus : list[ list[list[str]] ]  # una o più ref per frase
      translation_corpus: list[ list[str] ]

    Tokenizziamo semplicemente per spazi (normalizzazione già nel loader).
    """
    ref_corpus  = [[ref.strip().lower().split()] for ref in references]
    hyp_corpus  = [hyp.strip().lower().split()  for hyp in predictions]

    results = {}
    for order in [1, 2, 3, 4]:
        bleu_score, precisions, bp, ratio, _, _ = _phoenix_compute_bleu(
            ref_corpus, hyp_corpus, max_order=order, smooth=False
        )
        results[f"bleu_{order}"] = bleu_score * 100.0

    # BLEU-4 è quello standard riportato su PHOENIX
    results["bleu"] = results["bleu_4"]
    return results


def compute_rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """
    Calcola ROUGE-1/2/L usando rouge.py di PHOENIX.

    rouge() si aspetta liste di stringhe (frasi già tokenizzate come stringa).
    """
    rouge_dict = _phoenix_rouge(predictions, references)
    # Rinomina per comodità e scala a percentuale
    return {
        "rouge_1_f": rouge_dict["rouge_1/f_score"] * 100.0,
        "rouge_1_p": rouge_dict["rouge_1/p_score"] * 100.0,
        "rouge_1_r": rouge_dict["rouge_1/r_score"] * 100.0,
        "rouge_2_f": rouge_dict["rouge_2/f_score"] * 100.0,
        "rouge_2_p": rouge_dict["rouge_2/p_score"] * 100.0,
        "rouge_2_r": rouge_dict["rouge_2/r_score"] * 100.0,
        "rouge_l_f": rouge_dict["rouge_l/f_score"] * 100.0,
        "rouge_l_p": rouge_dict["rouge_l/p_score"] * 100.0,
        "rouge_l_r": rouge_dict["rouge_l/r_score"] * 100.0,
    }


def _edit_distance(ref_words: list[str], hyp_words: list[str]) -> int:
    if not ref_words:
        return len(hyp_words)
    if not hyp_words:
        return len(ref_words)
    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, start=1):
        curr = [i] + [0] * len(hyp_words)
        for j, h in enumerate(hyp_words, start=1):
            curr[j] = prev[j - 1] if r == h else 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[-1]


def compute_wer(predictions: list[str], references: list[str]) -> float:
    total_edits = total_words = 0
    for hyp, ref in zip(predictions, references):
        ref_w = ref.strip().split()
        hyp_w = hyp.strip().split()
        total_edits += _edit_distance(ref_w, hyp_w)
        total_words += len(ref_w)
    return total_edits / max(1, total_words)


# ─────────────────────────────────────────────────────────────
# Diagnostiche di output
# ─────────────────────────────────────────────────────────────

def _compute_output_diversity(predictions: list[str]) -> dict:
    from collections import Counter
    import math
    all_words = [w for pred in predictions for w in pred.strip().split()]
    if not all_words:
        return {"unique_words": 0, "total_words": 0, "vocab_coverage": 0.0,
                "entropy": 0.0, "top_10_word_ratio": 0.0, "most_common_word": ("N/A", 0)}
    wc = Counter(all_words)
    total = len(all_words)
    probs = [c / total for c in wc.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    top10_freq = sum(c for _, c in wc.most_common(10))
    return {
        "unique_words":    len(wc),
        "total_words":     total,
        "vocab_coverage":  len(wc) / total,
        "entropy":         entropy,
        "top_10_word_ratio": top10_freq / total,
        "most_common_word":  wc.most_common(1)[0],
    }


def _compute_output_similarities(predictions: list[str]) -> dict:
    from difflib import SequenceMatcher
    import random
    if len(predictions) < 2:
        return {"avg_similarity": 0.0, "max_similarity": 0.0, "identical_or_near_count": 0}
    sims, identical = [], 0
    sample_size = min(100, len(predictions) // 2)
    for _ in range(sample_size):
        i, j = random.sample(range(len(predictions)), 2)
        s = SequenceMatcher(None, predictions[i], predictions[j]).ratio()
        sims.append(s)
        if s > 0.95:
            identical += 1
    return {
        "avg_similarity": sum(sims) / max(1, len(sims)),
        "max_similarity": max(sims) if sims else 0.0,
        "identical_or_near_count": identical,
    }


# ─────────────────────────────────────────────────────────────
# Helpers checkpoint / config
# ─────────────────────────────────────────────────────────────

def _load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def _resolve_checkpoint_path(cfg: dict) -> Path:
    if cfg.get("checkpoint"):
        p = Path(cfg["checkpoint"])
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint non trovato: {p}")
    best = Path(cfg.get("output_dir", "outputs/phoenix_run1")) / "best.pt"
    if best.exists():
        return best
    raise FileNotFoundError(
        f"Checkpoint di default non trovato: {best}. Usa --checkpoint."
    )


def _load_tokenizer(cfg: dict) -> SentenceTokenizer:
    if cfg.get("tokenizer_path"):
        p = Path(cfg["tokenizer_path"])
        if p.exists():
            return SentenceTokenizer.load(p)
    default = Path(cfg.get("output_dir", "outputs/phoenix_run1")) / "tokenizer.json"
    if default.exists():
        return SentenceTokenizer.load(default)
    if cfg.get("train_csv"):
        return build_tokenizer(
            cfg["train_csv"],
            save_path=default,
            target_field=cfg.get("target_field", "translation"),
        )
    raise ValueError("Tokenizer non trovato. Fornisci --tokenizer_path o --train_csv.")


def _resolve_model_cfg(cli_cfg: dict, ckpt_cfg: dict) -> dict:
    def _pick(key, default):
        return cli_cfg[key] if cli_cfg.get(key) is not None else ckpt_cfg.get(key, default)
    return {
        "feat_dim":            _pick("feat_dim", 376),
        "d_model":             _pick("d_model", 512),
        "nhead":               _pick("nhead", 8),
        "num_enc_layers":      _pick("num_enc_layers", 3),
        "num_dec_layers":      _pick("num_dec_layers", 3),
        "dim_feedforward":     _pick("dim_feedforward", 1024),
        "dropout":             _pick("dropout", 0.1),
        "max_src_len":         _pick("max_src_len", 256),
        "max_tgt_len":         _pick("max_tgt_len", 128),
        "label_smoothing":     _pick("label_smoothing", 0.1),
        "src_embedding_type":  _pick("src_embedding_type", "temporal_cnn"),
        "temporal_kernel_size":_pick("temporal_kernel_size", None),
        "temporal_blocks":     _pick("temporal_blocks", None),
    }


def _infer_temporal_cfg(state_dict: dict) -> dict:
    kernel_size = blocks = None
    w = state_dict.get("src_embed.temporal_extractor.in_proj.weight")
    if isinstance(w, torch.Tensor) and w.dim() == 3:
        kernel_size = int(w.shape[-1])
    prefix = "src_embed.temporal_extractor.blocks."
    idxs = {
        int(k[len(prefix):].split(".", 1)[0])
        for k in state_dict
        if k.startswith(prefix) and k[len(prefix):].split(".", 1)[0].isdigit()
    }
    if idxs:
        blocks = max(idxs) + 1
    return {"temporal_kernel_size": kernel_size, "temporal_blocks": blocks}


def _resolve_data_weights(cli_cfg: dict, ckpt_cfg: dict) -> dict:
    def _pick(key, default):
        return cli_cfg[key] if cli_cfg.get(key) is not None else ckpt_cfg.get(key, default)
    return {
        "pose_weight": _pick("pose_weight", 1.0),
        "hand_weight": _pick("hand_weight", 1.5),
        "face_weight": _pick("face_weight", 0.8),
    }


def _compute_landmark_stats(train_csv: str | Path, landmarks_dir: str | Path) -> dict | None:
    """Ricalcola mean/std del training set per la normalizzazione."""
    train_csv     = Path(train_csv)
    landmarks_dir = Path(landmarks_dir)
    if not train_csv.exists() or not landmarks_dir.exists():
        return None

    ssum = ssq = None
    count = 0
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            name = row["name"].strip()
            npy  = landmarks_dir / f"{name}_landmarks.npy"
            if not npy.exists():
                continue
            try:
                arr = np.load(npy, mmap_mode="r").astype(np.float64)
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0], -1)
            except Exception:
                continue
            ssum  = arr.sum(0) if ssum is None else ssum + arr.sum(0)
            ssq   = (arr ** 2).sum(0) if ssq is None else ssq + (arr ** 2).sum(0)
            count += arr.shape[0]

    if count == 0 or ssum is None:
        return None
    mean = (ssum / count).astype(np.float32)
    std  = np.sqrt(np.maximum((ssq / count) - (ssum / count) ** 2, 1e-12)).astype(np.float32)
    return {"mean": mean, "std": std}


def decode_batch(token_ids: list[list[int]], tokenizer: SentenceTokenizer) -> list[str]:
    return [tokenizer.decode(ids) for ids in token_ids]


def _unpack_model_outputs(outputs):
    if not isinstance(outputs, (tuple, list)):
        return outputs, None, None, None
    ce_logits  = outputs[0]
    ctc_logits = outputs[1] if len(outputs) > 1 else None
    attn       = outputs[2] if len(outputs) > 2 else None
    return ce_logits, ctc_logits, None, attn


# ─────────────────────────────────────────────────────────────
# Funzione principale di valutazione
# ─────────────────────────────────────────────────────────────

def evaluate(cfg: dict) -> dict:
    device  = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", False) and device.type == "cuda"

    # ── Checkpoint ───────────────────────────────────────
    checkpoint_path = _resolve_checkpoint_path(cfg)
    print(f"[EVAL] Checkpoint: {checkpoint_path}")
    ckpt     = _load_checkpoint(checkpoint_path, device)
    ckpt_cfg = ckpt.get("cfg", {})

    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt

    # ── Tokenizer ────────────────────────────────────────
    tokenizer  = _load_tokenizer(cfg)

    # ── Config modello ───────────────────────────────────
    model_cfg    = _resolve_model_cfg(cfg, ckpt_cfg)
    data_weights = _resolve_data_weights(cfg, ckpt_cfg)

    if model_cfg["src_embedding_type"] == "temporal_cnn":
        inferred = _infer_temporal_cfg(state_dict)
        if model_cfg["temporal_kernel_size"] is None:
            model_cfg["temporal_kernel_size"] = inferred["temporal_kernel_size"] or 3
        if model_cfg["temporal_blocks"] is None:
            model_cfg["temporal_blocks"] = inferred["temporal_blocks"] or 1

    # ── Stats normalizzazione (dal training set) ──────────
    normalize_stats = None
    train_csv  = cfg.get("train_csv")  or ckpt_cfg.get("train_csv")
    train_lm_dir = cfg.get("train_landmarks_dir") or ckpt_cfg.get("train_landmarks_dir")
    if train_csv and train_lm_dir:
        print("[EVAL] Calcolo stats normalizzazione dal training set...")
        normalize_stats = _compute_landmark_stats(train_csv, train_lm_dir)
        if normalize_stats is None:
            print("[EVAL] Warning: impossibile calcolare le stats; si procede senza.")
    else:
        print("[EVAL] Warning: train_csv/train_landmarks_dir non disponibili; nessuna normalizzazione.")

    # ── Dataset e loader ─────────────────────────────────
    target_field = cfg.get("target_field") or ckpt_cfg.get("target_field", "translation")
    dataset = PhoenixDataset(
        csv_path=cfg["val_csv"],
        landmarks_dir=cfg["landmarks_dir"],
        tokenizer=tokenizer,
        target_field=target_field,
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
        flatten_landmarks=True,
        pose_weight=data_weights["pose_weight"],
        hand_weight=data_weights["hand_weight"],
        face_weight=data_weights["face_weight"],
        normalize_stats=normalize_stats,
        use_hand_relative_norm=ckpt_cfg.get("use_hand_relative_norm", True),
    )
    print(f"[EVAL] Dataset: {len(dataset)} campioni (target='{target_field}')")

    # Verifica feat_dim
    ds_feat    = getattr(dataset, "_dataset_feat_dim", None)
    model_feat = int(model_cfg["feat_dim"])
    if ds_feat is not None and ds_feat != model_feat:
        raise RuntimeError(
            f"[EVAL] feat_dim mismatch: dataset={ds_feat}, modello={model_feat}."
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=lambda b: collate_fn(
            b,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            unk_id=tokenizer.unk_id,
            eos_id=tokenizer.eos_id,
        ),
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
    )

    # ── Modello ──────────────────────────────────────────
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
        temporal_kernel_size=model_cfg["temporal_kernel_size"] or 3,
        temporal_blocks=model_cfg["temporal_blocks"] or 1,
        pad_id=tokenizer.pad_id,
        label_smoothing=model_cfg["label_smoothing"],
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[EVAL] Warning non-strict load — missing: {missing} | unexpected: {unexpected}")
    model.eval()

    # ── Inferenza ─────────────────────────────────────────
    total_loss = total_tok = total_unk = 0
    total_hyp_tok = total_hyp_unk = 0
    all_hyps: list[str] = []
    all_refs: list[str] = []

    for batch_idx, batch in enumerate(loader):
        src     = batch["src"].to(device, non_blocking=True)
        tgt_in  = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out = batch["tgt_output"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        tin_mask = batch["tgt_in_key_padding_mask"].to(device, non_blocking=True)

        with torch.no_grad(), autocast(enabled=use_amp):
            outputs = model(
                src=src,
                tgt_input=tgt_in,
                tgt_output=tgt_out,
                src_key_padding_mask=src_mask,
                tgt_in_key_padding_mask=tin_mask,
            )
            ce_logits, _, returned_loss, _ = _unpack_model_outputs(outputs)
            loss = returned_loss if returned_loss is not None else \
                torch.nn.functional.cross_entropy(
                    ce_logits.reshape(-1, ce_logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=model.pad_id,
                )

        n_tok = (tgt_out != model.pad_id).sum().item()
        n_unk = (tgt_out == tokenizer.unk_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tok  += n_tok
        total_unk  += n_unk

        hyps = model.greedy_decode(
            src=src,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=cfg["max_decode_len"],
            src_key_padding_mask=src_mask,
        )
        total_hyp_tok += sum(len(h) for h in hyps)
        total_hyp_unk += sum(sum(t == tokenizer.unk_id for t in h) for h in hyps)
        all_hyps.extend(decode_batch(hyps, tokenizer))
        # "targets" è il campo del collate di PhoenixDataset
        all_refs.extend(batch["targets"])

        if (batch_idx + 1) % 20 == 0:
            print(f"[EVAL] Batch {batch_idx + 1}/{len(loader)} ...")

    # ── Metriche ──────────────────────────────────────────
    avg_loss = total_loss / max(1, total_tok)
    ppl      = torch.exp(torch.tensor(min(avg_loss, 20.0))).item()
    wer      = compute_wer(all_hyps, all_refs)

    # BLEU (bleu.py PHOENIX)
    bleu_scores  = compute_bleu_scores(all_hyps, all_refs)

    # ROUGE (rouge.py PHOENIX)
    rouge_scores = compute_rouge_scores(all_hyps, all_refs)

    # Diagnostiche
    diversity_stats   = _compute_output_diversity(all_hyps)
    similarity_stats  = _compute_output_similarities(all_hyps)

    return {
        "loss":         avg_loss,
        "ppl":          ppl,
        "wer":          wer,
        "num_samples":  len(dataset),
        # BLEU (per ordine e corpus)
        **bleu_scores,
        # ROUGE
        **rouge_scores,
        # Token stats
        "tgt_unk_rate": total_unk / max(1, total_tok),
        "hyp_unk_rate": total_hyp_unk / max(1, total_hyp_tok),
        "avg_ref_len":  total_tok / max(1, len(all_refs)),
        "avg_hyp_len":  total_hyp_tok / max(1, len(all_hyps)),
        # Diagnostiche
        "diversity":    diversity_stats,
        "similarity":   similarity_stats,
        # Testo grezzo
        "hyps": all_hyps,
        "refs": all_refs,
    }


def _cleanup(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valuta SignLanguageTransformer su PHOENIX-2014-T"
    )
    parser.add_argument("--checkpoint",           type=str, default=None)
    parser.add_argument("--output_dir",           type=str, default="outputs/phoenix_run1")
    parser.add_argument("--tokenizer_path",       type=str, default=None)
    parser.add_argument("--train_csv",            type=str, default=None,
                        help="CSV training (per stats normalizzazione e tokenizer)")
    parser.add_argument("--val_csv",              type=str,
                        default="dataset/PHOENIX-2014-T.dev.corpus.csv")
    parser.add_argument("--landmarks_dir",        type=str,
                        default="dataset/landmarks_dev")
    parser.add_argument("--train_landmarks_dir",  type=str,
                        default="dataset/landmarks_train")
    parser.add_argument("--target_field",         type=str, default=None,
                        choices=["translation", "orth"],
                        help="Campo target nel CSV (default: valore dal checkpoint, altrimenti translation)")
    parser.add_argument("--batch_size",           type=int, default=8)
    parser.add_argument("--num_workers",          type=int, default=4)
    parser.add_argument("--max_decode_len",       type=int, default=128)
    parser.add_argument("--num_examples",         type=int, default=5)
    parser.add_argument("--device",               type=str, default="cuda")
    parser.add_argument("--use_amp",              action="store_true")
    parser.add_argument("--pose_weight",          type=float, default=None)
    parser.add_argument("--hand_weight",          type=float, default=None)
    parser.add_argument("--face_weight",          type=float, default=None)
    parser.add_argument("--save_results",         type=str, default=None,
                        help="Salva i risultati in un file JSON")
    args = parser.parse_args()
    cfg  = vars(args)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    try:
        results = evaluate(cfg)

        print("\n" + "=" * 55)
        print("RISULTATI VALUTAZIONE — PHOENIX-2014-T")
        print("=" * 55)
        print(f"  Campioni:   {results['num_samples']}")
        print(f"  Loss:       {results['loss']:.4f}")
        print(f"  Perplexity: {results['ppl']:.2f}")
        print(f"  WER:        {results['wer']:.2%}")
        print()
        print("  BLEU (bleu.py PHOENIX):")
        print(f"    BLEU-1:  {results['bleu_1']:.2f}")
        print(f"    BLEU-2:  {results['bleu_2']:.2f}")
        print(f"    BLEU-3:  {results['bleu_3']:.2f}")
        print(f"    BLEU-4:  {results['bleu_4']:.2f}  ← metrica principale")
        print()
        print("  ROUGE (rouge.py PHOENIX):")
        print(f"    ROUGE-1 F1: {results['rouge_1_f']:.2f}  "
              f"P={results['rouge_1_p']:.2f}  R={results['rouge_1_r']:.2f}")
        print(f"    ROUGE-2 F1: {results['rouge_2_f']:.2f}  "
              f"P={results['rouge_2_p']:.2f}  R={results['rouge_2_r']:.2f}")
        print(f"    ROUGE-L F1: {results['rouge_l_f']:.2f}  "
              f"P={results['rouge_l_p']:.2f}  R={results['rouge_l_r']:.2f}")
        print()
        print(f"  tgt UNK:    {results['tgt_unk_rate']:.2%}")
        print(f"  hyp UNK:    {results['hyp_unk_rate']:.2%}")
        print(f"  avg ref len:{results['avg_ref_len']:.1f} tok")
        print(f"  avg hyp len:{results['avg_hyp_len']:.1f} tok")

        div = results["diversity"]
        sim = results["similarity"]
        print(f"\n  Output Diversity:")
        print(f"    parole uniche: {div['unique_words']} / {div['total_words']} "
              f"({div['vocab_coverage']:.1%})")
        print(f"    entropia: {div['entropy']:.3f}")
        print(f"    top-10 word ratio: {div['top_10_word_ratio']:.1%}")
        print(f"    più frequente: '{div['most_common_word'][0]}' "
              f"({div['most_common_word'][1]}x)")
        print(f"\n  Output Similarity:")
        print(f"    avg similarity: {sim['avg_similarity']:.3f}")
        print(f"    coppie identiche/quasi (>0.95): {sim['identical_or_near_count']}")
        if sim["avg_similarity"] > 0.8:
            print("    ⚠️  ALTO: possibile collasso del modello")
        elif sim["avg_similarity"] > 0.6:
            print("    ⚠️  MEDIO: output moderatamente simili")
        else:
            print("    ✓ BASSO: buona diversità")

        num_ex = cfg.get("num_examples", 0) or 0
        if num_ex > 0:
            print(f"\n  Esempi (reference → hypothesis):")
            for i in range(min(num_ex, len(results["refs"]))):
                print(f"  [{i+1:02d}] REF: {results['refs'][i]}")
                print(f"       HYP: {results['hyps'][i]}")

        # Salvataggio opzionale
        if cfg.get("save_results"):
            save_path = Path(cfg["save_results"])
            save_data = {k: v for k, v in results.items()
                         if k not in ("hyps", "refs", "diversity", "similarity")}
            save_data["diversity"]  = results["diversity"]
            save_data["similarity"] = results["similarity"]
            # Le liste di testo possono essere grandi; le salviamo a parte
            save_data["num_hyps"] = len(results["hyps"])
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"\n[EVAL] Risultati salvati in {save_path}")

    finally:
        _cleanup(device)


if __name__ == "__main__":
    main()