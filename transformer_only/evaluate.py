#!/usr/bin/env python3
"""
evaluate.py

Script di valutazione per PHOENIX-2014-T. Supporta tre modalità:

  --mode stage1     : Modello 1 solo (landmark → gloss)
                      Metrica principale: WER sui gloss
  --mode stage2     : Modello 2 solo con gloss GOLD (upper bound)
                      Metrica principale: BLEU-4 sulla traduzione
  --mode pipeline   : Pipeline completa (landmark → gloss → translation)
                      Metrica principale: BLEU-4 sulla traduzione

Utilizzo:
  # Valuta solo il Modello 1
  python evaluate.py --mode stage1 \
    --checkpoint    outputs/phoenix_run1/best.pt \
    --val_csv       dataset/PHOENIX-2014-T.dev.corpus.csv \
    --landmarks_dir dataset/landmarks_dev \
    --train_csv     dataset/PHOENIX-2014-T.train.corpus.csv \
    --train_landmarks_dir dataset/landmarks_train

  # Valuta solo il Modello 2 (con gloss gold)
  python evaluate.py --mode stage2 \
    --stage2_checkpoint outputs/phoenix_stage2/best.pt \
    --val_csv           dataset/PHOENIX-2014-T.dev.corpus.csv \
    --stage1_dir        outputs/phoenix_run1

  # Valuta la pipeline completa
  python evaluate.py --mode pipeline \
    --checkpoint        outputs/phoenix_run1/best.pt \
    --stage2_checkpoint outputs/phoenix_stage2/best.pt \
    --val_csv           dataset/PHOENIX-2014-T.dev.corpus.csv \
    --landmarks_dir     dataset/landmarks_dev \
    --train_csv         dataset/PHOENIX-2014-T.train.corpus.csv \
    --train_landmarks_dir dataset/landmarks_train \
    --stage1_dir        outputs/phoenix_run1
"""

import argparse
import csv
import gc
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from .data.phoenix_loader import (
    PhoenixDataset,
    SentenceTokenizer,
    collate_fn,
    build_tokenizer,
)
from .models.transformer import SignLanguageTransformer
from .two_stage import GlossToTextTransformer
from .bleu import compute_bleu as _phoenix_compute_bleu
from .rouge import rouge as _phoenix_rouge


# ─────────────────────────────────────────────────────────────
# Metriche
# ─────────────────────────────────────────────────────────────

def compute_bleu_scores(predictions: list[str], references: list[str]) -> dict:
    ref_corpus = [[ref.strip().lower().split()] for ref in references]
    hyp_corpus = [hyp.strip().lower().split()  for hyp in predictions]
    results = {}
    for order in [1, 2, 3, 4]:
        bleu_score, _, _, _, _, _ = _phoenix_compute_bleu(
            ref_corpus, hyp_corpus, max_order=order, smooth=False
        )
        results[f"bleu_{order}"] = bleu_score * 100.0
    results["bleu"] = results["bleu_4"]
    return results


def compute_rouge_scores(predictions: list[str], references: list[str]) -> dict:
    rouge_dict = _phoenix_rouge(predictions, references)
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
            curr[j] = prev[j-1] if r == h else 1 + min(prev[j], curr[j-1], prev[j-1])
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
        "unique_words":      len(wc),
        "total_words":       total,
        "vocab_coverage":    len(wc) / total,
        "entropy":           entropy,
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
        "avg_similarity":        sum(sims) / max(1, len(sims)),
        "max_similarity":        max(sims) if sims else 0.0,
        "identical_or_near_count": identical,
    }


def _print_metrics(results: dict, mode: str, num_examples: int = 5):
    print("\n" + "=" * 60)
    print(f"RISULTATI VALUTAZIONE — PHOENIX-2014-T  [{mode.upper()}]")
    print("=" * 60)
    print(f"  Campioni: {results['num_samples']}")

    if mode == "stage1":
        print(f"\n  WER gloss:  {results['wer_gloss']:.2%}  ← metrica principale Stage 1")
        print(f"\n  BLEU gloss (indicativo):")
        print(f"    BLEU-1: {results['bleu_1']:.2f}")
        print(f"    BLEU-4: {results['bleu_4']:.2f}")

    else:
        if mode == "stage2":
            print("  (gloss gold → translation: upper bound del sistema)")
        else:
            print("  (pipeline completa: landmark → gloss → translation)")
            print(f"\n  WER gloss intermedio: {results.get('wer_gloss', float('nan')):.2%}")

        print(f"\n  BLEU (bleu.py PHOENIX):")
        print(f"    BLEU-1: {results['bleu_1']:.2f}")
        print(f"    BLEU-2: {results['bleu_2']:.2f}")
        print(f"    BLEU-3: {results['bleu_3']:.2f}")
        print(f"    BLEU-4: {results['bleu_4']:.2f}  ← metrica principale")
        print(f"\n  ROUGE (rouge.py PHOENIX):")
        print(f"    ROUGE-1 F1: {results['rouge_1_f']:.2f}  "
              f"P={results['rouge_1_p']:.2f}  R={results['rouge_1_r']:.2f}")
        print(f"    ROUGE-2 F1: {results['rouge_2_f']:.2f}  "
              f"P={results['rouge_2_p']:.2f}  R={results['rouge_2_r']:.2f}")
        print(f"    ROUGE-L F1: {results['rouge_l_f']:.2f}  "
              f"P={results['rouge_l_p']:.2f}  R={results['rouge_l_r']:.2f}")

    div = results.get("diversity", {})
    sim = results.get("similarity", {})
    if div:
        print(f"\n  Output Diversity:")
        print(f"    parole uniche: {div['unique_words']} / {div['total_words']} "
              f"({div['vocab_coverage']:.1%})")
        print(f"    entropia: {div['entropy']:.3f}")
        print(f"    top-10 word ratio: {div['top_10_word_ratio']:.1%}")
        print(f"    più frequente: '{div['most_common_word'][0]}' ({div['most_common_word'][1]}x)")
    if sim:
        print(f"\n  Output Similarity:")
        print(f"    avg similarity: {sim['avg_similarity']:.3f}")
        print(f"    coppie identiche/quasi (>0.95): {sim['identical_or_near_count']}")
        if sim["avg_similarity"] > 0.8:
            print("    ⚠️  ALTO: possibile collasso del modello")
        elif sim["avg_similarity"] > 0.6:
            print("    ⚠️  MEDIO: output moderatamente simili")
        else:
            print("    ✓ BASSO: buona diversità")

    if num_examples > 0 and "hyps" in results:
        label = "gloss" if mode == "stage1" else "translation"
        ref_key = "gloss_refs" if mode == "stage1" else "trans_refs"
        print(f"\n  Esempi ({label}):")
        refs = results.get(ref_key, results.get("refs", []))
        for i in range(min(num_examples, len(results["hyps"]))):
            print(f"  [{i+1:02d}] REF: {refs[i] if i < len(refs) else '—'}")
            print(f"       HYP: {results['hyps'][i]}")
            if mode == "pipeline" and "gloss_hyps" in results:
                print(f"       GLO: {results['gloss_hyps'][i]}")


# ─────────────────────────────────────────────────────────────
# Helpers checkpoint / config  (invariati dal file originale)
# ─────────────────────────────────────────────────────────────

def _load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def _resolve_checkpoint_path(cfg: dict, key: str = "checkpoint",
                              default_dir_key: str = "output_dir",
                              default_dir: str = "outputs/phoenix_run1") -> Path:
    if cfg.get(key):
        p = Path(cfg[key])
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint non trovato: {p}")
    best = Path(cfg.get(default_dir_key, default_dir)) / "best.pt"
    if best.exists():
        return best
    raise FileNotFoundError(f"Checkpoint non trovato: {best}. Usa --{key}.")


def _load_tokenizer(cfg: dict, tok_path: str | Path | None = None,
                    target_field: str = "translation") -> SentenceTokenizer:
    if tok_path and Path(tok_path).exists():
        return SentenceTokenizer.load(tok_path)
    if cfg.get("tokenizer_path") and Path(cfg["tokenizer_path"]).exists():
        return SentenceTokenizer.load(cfg["tokenizer_path"])
    default = Path(cfg.get("output_dir", "outputs/phoenix_run1")) / "tokenizer.json"
    if default.exists():
        return SentenceTokenizer.load(default)
    if cfg.get("train_csv"):
        return build_tokenizer(cfg["train_csv"], save_path=default,
                               target_field=target_field)
    raise ValueError("Tokenizer non trovato.")


def _resolve_model_cfg(cli_cfg: dict, ckpt_cfg: dict) -> dict:
    def _pick(key, default):
        return cli_cfg[key] if cli_cfg.get(key) is not None else ckpt_cfg.get(key, default)
    return {
        "feat_dim":             _pick("feat_dim", 376),
        "d_model":              _pick("d_model", 512),
        "nhead":                _pick("nhead", 8),
        "num_enc_layers":       _pick("num_enc_layers", 3),
        "num_dec_layers":       _pick("num_dec_layers", 3),
        "dim_feedforward":      _pick("dim_feedforward", 1024),
        "dropout":              _pick("dropout", 0.1),
        "max_src_len":          _pick("max_src_len", 256),
        "max_tgt_len":          _pick("max_tgt_len", 128),
        "label_smoothing":      _pick("label_smoothing", 0.1),
        "src_embedding_type":   _pick("src_embedding_type", "temporal_cnn"),
        "temporal_kernel_size": _pick("temporal_kernel_size", None),
        "temporal_blocks":      _pick("temporal_blocks", None),
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
            ssum  = arr.sum(0)      if ssum is None else ssum + arr.sum(0)
            ssq   = (arr**2).sum(0) if ssq  is None else ssq  + (arr**2).sum(0)
            count += arr.shape[0]
    if count == 0 or ssum is None:
        return None
    mean = (ssum / count).astype(np.float32)
    std  = np.sqrt(np.maximum((ssq / count) - (ssum / count)**2, 1e-12)).astype(np.float32)
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
# Costruzione Modello 1
# ─────────────────────────────────────────────────────────────

def _build_model1(cfg: dict, device: torch.device, gloss_tokenizer: SentenceTokenizer):
    ckpt_path  = _resolve_checkpoint_path(cfg, key="checkpoint",
                                          default_dir_key="output_dir",
                                          default_dir="outputs/phoenix_run1")
    ckpt       = _load_checkpoint(ckpt_path, device)
    ckpt_cfg   = ckpt.get("cfg", {})
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt

    model_cfg = _resolve_model_cfg(cfg, ckpt_cfg)
    if model_cfg["src_embedding_type"] == "temporal_cnn":
        inferred = _infer_temporal_cfg(state_dict)
        model_cfg["temporal_kernel_size"] = model_cfg["temporal_kernel_size"] or inferred.get("temporal_kernel_size") or 3
        model_cfg["temporal_blocks"]      = model_cfg["temporal_blocks"]      or inferred.get("temporal_blocks")      or 1

    model = SignLanguageTransformer(
        feat_dim=model_cfg["feat_dim"],
        vocab_size=gloss_tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_enc_layers=model_cfg["num_enc_layers"],
        num_dec_layers=model_cfg["num_dec_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
        src_embedding_type=model_cfg["src_embedding_type"],
        temporal_kernel_size=model_cfg["temporal_kernel_size"],
        temporal_blocks=model_cfg["temporal_blocks"],
        pad_id=gloss_tokenizer.pad_id,
        label_smoothing=model_cfg["label_smoothing"],
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[EVAL] Stage1 non-strict load — missing: {missing} | unexpected: {unexpected}")
    model.eval()
    return model, model_cfg, ckpt_cfg


# ─────────────────────────────────────────────────────────────
# Costruzione Modello 2
# ─────────────────────────────────────────────────────────────

def _build_model2(cfg: dict, device: torch.device,
                  gloss_tokenizer: SentenceTokenizer,
                  trans_tokenizer: SentenceTokenizer):
    ckpt_path  = _resolve_checkpoint_path(cfg, key="stage2_checkpoint",
                                          default_dir_key="stage2_dir",
                                          default_dir="outputs/phoenix_stage2")
    ckpt       = _load_checkpoint(ckpt_path, device)
    ckpt_cfg   = ckpt.get("cfg", {})
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt

    def _pick(key, default):
        return cfg.get(key) or ckpt_cfg.get(key, default)

    model = GlossToTextTransformer(
        gloss_vocab_size=gloss_tokenizer.vocab_size,
        trans_vocab_size=trans_tokenizer.vocab_size,
        d_model=       _pick("s2_d_model",        256),
        nhead=         _pick("s2_nhead",           4),
        num_enc_layers=_pick("s2_num_enc_layers",  2),
        num_dec_layers=_pick("s2_num_dec_layers",  2),
        dim_feedforward=_pick("s2_dim_feedforward",512),
        dropout=       _pick("s2_dropout",         0.1),
        max_gloss_len= _pick("max_gloss_len",      64),
        max_trans_len= _pick("max_trans_len",       128),
        gloss_pad_id=  gloss_tokenizer.pad_id,
        trans_pad_id=  trans_tokenizer.pad_id,
        label_smoothing=_pick("label_smoothing",   0.1),
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[EVAL] Stage2 non-strict load — missing: {missing} | unexpected: {unexpected}")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Modalità Stage 1: landmark → gloss
# ─────────────────────────────────────────────────────────────

def evaluate_stage1(cfg: dict) -> dict:
    """Valuta il Modello 1 (landmark → gloss). Metrica: WER sui gloss."""
    device  = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", False) and device.type == "cuda"

    # Tokenizer gloss (target_field='orth')
    stage1_dir    = Path(cfg.get("stage1_dir", "outputs/phoenix_run1"))
    gloss_tok_path = stage1_dir / "tokenizer.json"
    gloss_tokenizer = _load_tokenizer(cfg, tok_path=gloss_tok_path, target_field="orth")
    print(f"[EVAL/S1] Tokenizer gloss: {gloss_tokenizer.vocab_size} token")

    model, model_cfg, ckpt_cfg = _build_model1(cfg, device, gloss_tokenizer)
    data_weights = _resolve_data_weights(cfg, ckpt_cfg)

    # Stats normalizzazione
    normalize_stats = None
    train_csv    = cfg.get("train_csv")    or ckpt_cfg.get("train_csv")
    train_lm_dir = cfg.get("train_landmarks_dir") or ckpt_cfg.get("train_landmarks_dir")
    if train_csv and train_lm_dir:
        print("[EVAL/S1] Calcolo stats normalizzazione...")
        normalize_stats = _compute_landmark_stats(train_csv, train_lm_dir)

    # Dataset: target = gloss (orth)
    dataset = PhoenixDataset(
        csv_path=cfg["val_csv"],
        landmarks_dir=cfg["landmarks_dir"],
        tokenizer=gloss_tokenizer,
        target_field="orth",
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
        flatten_landmarks=True,
        pose_weight=data_weights["pose_weight"],
        hand_weight=data_weights["hand_weight"],
        face_weight=data_weights["face_weight"],
        normalize_stats=normalize_stats,
        use_hand_relative_norm=ckpt_cfg.get("use_hand_relative_norm", True),
    )
    print(f"[EVAL/S1] Dataset: {len(dataset)} campioni")

    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=partial(collate_fn,
                           pad_id=gloss_tokenizer.pad_id,
                           bos_id=gloss_tokenizer.bos_id,
                           unk_id=gloss_tokenizer.unk_id,
                           eos_id=gloss_tokenizer.eos_id),
        pin_memory=True, persistent_workers=(cfg["num_workers"] > 0),
    )

    all_hyps: list[str] = []
    all_refs: list[str] = []

    for batch_idx, batch in enumerate(loader):
        src      = batch["src"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)

        hyps = model.greedy_decode(
            src=src,
            bos_id=gloss_tokenizer.bos_id,
            eos_id=gloss_tokenizer.eos_id,
            max_len=cfg["max_decode_len"],
            src_key_padding_mask=src_mask,
        )
        all_hyps.extend(decode_batch(hyps, gloss_tokenizer))
        all_refs.extend(batch["targets"])

        if (batch_idx + 1) % 20 == 0:
            print(f"[EVAL/S1] Batch {batch_idx+1}/{len(loader)} ...")

    wer        = compute_wer(all_hyps, all_refs)
    bleu_scores = compute_bleu_scores(all_hyps, all_refs)   # indicativo sui gloss

    return {
        "num_samples": len(dataset),
        "wer_gloss":   wer,
        **bleu_scores,
        "diversity":   _compute_output_diversity(all_hyps),
        "similarity":  _compute_output_similarities(all_hyps),
        "hyps":        all_hyps,
        "gloss_refs":  all_refs,
        "refs":        all_refs,
    }


# ─────────────────────────────────────────────────────────────
# Modalità Stage 2: gloss gold → translation  (upper bound)
# ─────────────────────────────────────────────────────────────

def evaluate_stage2(cfg: dict) -> dict:
    """
    Valuta il Modello 2 con gloss GOLD come input.
    Rappresenta l'upper bound del sistema: quanto bene può tradurre
    il Modello 2 se il Modello 1 fosse perfetto.
    """
    import csv as _csv
    from functools import partial as _partial

    device  = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", False) and device.type == "cuda"

    # Tokenizer gloss
    stage1_dir     = Path(cfg.get("stage1_dir", "outputs/phoenix_run1"))
    gloss_tok_path = stage1_dir / "tokenizer.json"
    gloss_tokenizer = _load_tokenizer(cfg, tok_path=gloss_tok_path, target_field="orth")

    # Tokenizer traduzione
    stage2_dir     = Path(cfg.get("stage2_dir", "outputs/phoenix_stage2"))
    trans_tok_path = stage2_dir / "tokenizer_trans.json"
    trans_tokenizer = _load_tokenizer(cfg, tok_path=trans_tok_path, target_field="translation")

    print(f"[EVAL/S2] Tokenizer gloss: {gloss_tokenizer.vocab_size} | "
          f"trans: {trans_tokenizer.vocab_size}")

    model2 = _build_model2(cfg, device, gloss_tokenizer, trans_tokenizer)

    # Legge le coppie (orth, translation) direttamente dal CSV
    pairs: list[dict] = []
    with open(cfg["val_csv"], newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f, delimiter="|")
        for row in reader:
            orth  = row.get("orth",        "").strip()
            trans = row.get("translation", "").strip()
            if orth and trans:
                pairs.append({"orth": orth, "translation": trans})
    print(f"[EVAL/S2] Coppie gloss-translation: {len(pairs)}")

    all_hyps: list[str] = []
    all_refs: list[str] = []

    batch_size = cfg["batch_size"]
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        B = len(batch_pairs)

        # Tokenizza gloss
        gloss_encoded = [
            gloss_tokenizer.encode(p["orth"], add_special_tokens=True)
            for p in batch_pairs
        ]
        max_g = max(len(g) for g in gloss_encoded)
        gloss_padded = torch.full((B, max_g), gloss_tokenizer.pad_id, dtype=torch.long, device=device)
        gloss_mask   = torch.ones(B, max_g, dtype=torch.bool, device=device)
        for i, g in enumerate(gloss_encoded):
            gloss_padded[i, :len(g)] = torch.tensor(g, dtype=torch.long)
            gloss_mask[i, :len(g)] = False

        with torch.no_grad():
            hyps = model2.greedy_decode(
                gloss_ids=gloss_padded,
                bos_id=trans_tokenizer.bos_id,
                eos_id=trans_tokenizer.eos_id,
                max_len=cfg["max_decode_len"],
                gloss_key_padding_mask=gloss_mask,
            )

        all_hyps.extend(decode_batch(hyps, trans_tokenizer))
        all_refs.extend(p["translation"] for p in batch_pairs)

        if (start // batch_size + 1) % 20 == 0:
            print(f"[EVAL/S2] Batch {start//batch_size+1}/{(len(pairs)+batch_size-1)//batch_size} ...")

    bleu_scores  = compute_bleu_scores(all_hyps, all_refs)
    rouge_scores = compute_rouge_scores(all_hyps, all_refs)
    wer          = compute_wer(all_hyps, all_refs)

    return {
        "num_samples": len(pairs),
        "wer":         wer,
        **bleu_scores,
        **rouge_scores,
        "diversity":   _compute_output_diversity(all_hyps),
        "similarity":  _compute_output_similarities(all_hyps),
        "hyps":        all_hyps,
        "trans_refs":  all_refs,
        "refs":        all_refs,
    }


# ─────────────────────────────────────────────────────────────
# Modalità Pipeline: landmark → gloss → translation
# ─────────────────────────────────────────────────────────────

def evaluate_pipeline(cfg: dict) -> dict:
    """
    Valuta la pipeline completa:
      landmark → (Modello 1) → gloss → (Modello 2) → translation

    Riporta:
      - WER sui gloss intermedi (qualità Modello 1)
      - BLEU/ROUGE sulla traduzione finale (qualità complessiva)
    """
    device  = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", False) and device.type == "cuda"

    # Tokenizer
    stage1_dir      = Path(cfg.get("stage1_dir", "outputs/phoenix_run1"))
    stage2_dir      = Path(cfg.get("stage2_dir", "outputs/phoenix_stage2"))
    gloss_tokenizer = _load_tokenizer(cfg, tok_path=stage1_dir / "tokenizer.json",
                                      target_field="orth")
    trans_tokenizer = _load_tokenizer(cfg, tok_path=stage2_dir / "tokenizer_trans.json",
                                      target_field="translation")
    print(f"[EVAL/PL] Tokenizer gloss: {gloss_tokenizer.vocab_size} | "
          f"trans: {trans_tokenizer.vocab_size}")

    # Costruzione modelli
    model1, model_cfg, ckpt_cfg = _build_model1(cfg, device, gloss_tokenizer)
    model2 = _build_model2(cfg, device, gloss_tokenizer, trans_tokenizer)
    data_weights = _resolve_data_weights(cfg, ckpt_cfg)

    # Stats normalizzazione per il Modello 1
    normalize_stats = None
    train_csv    = cfg.get("train_csv")           or ckpt_cfg.get("train_csv")
    train_lm_dir = cfg.get("train_landmarks_dir") or ckpt_cfg.get("train_landmarks_dir")
    if train_csv and train_lm_dir:
        print("[EVAL/PL] Calcolo stats normalizzazione...")
        normalize_stats = _compute_landmark_stats(train_csv, train_lm_dir)

    # Dataset: usiamo target_field='orth' per avere i gloss gold come riferimento
    # e leggiamo le translation gold separatamente dal CSV
    dataset = PhoenixDataset(
        csv_path=cfg["val_csv"],
        landmarks_dir=cfg["landmarks_dir"],
        tokenizer=gloss_tokenizer,
        target_field="orth",
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
        flatten_landmarks=True,
        pose_weight=data_weights["pose_weight"],
        hand_weight=data_weights["hand_weight"],
        face_weight=data_weights["face_weight"],
        normalize_stats=normalize_stats,
        use_hand_relative_norm=ckpt_cfg.get("use_hand_relative_norm", True),
    )

    # Legge anche le translation gold per la valutazione finale
    trans_gold: dict[str, str] = {}
    with open(cfg["val_csv"], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            trans_gold[row["name"].strip()] = row.get("translation", "").strip()

    print(f"[EVAL/PL] Dataset: {len(dataset)} campioni")

    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=partial(collate_fn,
                           pad_id=gloss_tokenizer.pad_id,
                           bos_id=gloss_tokenizer.bos_id,
                           unk_id=gloss_tokenizer.unk_id,
                           eos_id=gloss_tokenizer.eos_id),
        pin_memory=True, persistent_workers=(cfg["num_workers"] > 0),
    )

    all_gloss_hyps: list[str] = []   # gloss predetti dal Modello 1
    all_gloss_refs: list[str] = []   # gloss gold
    all_trans_hyps: list[str] = []   # traduzioni finali
    all_trans_refs: list[str] = []   # traduzioni gold

    for batch_idx, batch in enumerate(loader):
        src      = batch["src"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)
        names    = batch["names"]
        B        = src.size(0)

        # ── Stadio 1: landmark → gloss ────────
        with torch.no_grad():
            gloss_ids_list = model1.greedy_decode(
                src=src,
                bos_id=gloss_tokenizer.bos_id,
                eos_id=gloss_tokenizer.eos_id,
                max_len=cfg["max_decode_len"],
                src_key_padding_mask=src_mask,
            )

        gloss_texts = decode_batch(gloss_ids_list, gloss_tokenizer)
        all_gloss_hyps.extend(gloss_texts)
        all_gloss_refs.extend(batch["targets"])   # gloss gold

        # ── Padda gloss per Modello 2 ─────────
        gloss_with_special = [
            [gloss_tokenizer.bos_id] + ids + [gloss_tokenizer.eos_id]
            for ids in gloss_ids_list
        ]
        max_g = max(len(g) for g in gloss_with_special)
        gloss_padded = torch.full((B, max_g), gloss_tokenizer.pad_id,
                                  dtype=torch.long, device=device)
        gloss_mask2  = torch.ones(B, max_g, dtype=torch.bool, device=device)
        for i, g in enumerate(gloss_with_special):
            gloss_padded[i, :len(g)] = torch.tensor(g, dtype=torch.long, device=device)
            gloss_mask2[i, :len(g)] = False

        # ── Stadio 2: gloss → translation ─────
        with torch.no_grad():
            trans_ids_list = model2.greedy_decode(
                gloss_ids=gloss_padded,
                bos_id=trans_tokenizer.bos_id,
                eos_id=trans_tokenizer.eos_id,
                max_len=cfg["max_decode_len"],
                gloss_key_padding_mask=gloss_mask2,
            )

        all_trans_hyps.extend(decode_batch(trans_ids_list, trans_tokenizer))
        all_trans_refs.extend(
            trans_gold.get(name, "") for name in names
        )

        if (batch_idx + 1) % 20 == 0:
            print(f"[EVAL/PL] Batch {batch_idx+1}/{len(loader)} ...")

    wer_gloss    = compute_wer(all_gloss_hyps, all_gloss_refs)
    bleu_scores  = compute_bleu_scores(all_trans_hyps, all_trans_refs)
    rouge_scores = compute_rouge_scores(all_trans_hyps, all_trans_refs)

    return {
        "num_samples": len(dataset),
        "wer_gloss":   wer_gloss,
        **bleu_scores,
        **rouge_scores,
        "diversity":   _compute_output_diversity(all_trans_hyps),
        "similarity":  _compute_output_similarities(all_trans_hyps),
        "hyps":        all_trans_hyps,
        "gloss_hyps":  all_gloss_hyps,
        "trans_refs":  all_trans_refs,
        "gloss_refs":  all_gloss_refs,
        "refs":        all_trans_refs,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _cleanup(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Valuta PHOENIX-2014-T (stage1 / stage2 / pipeline)")

    parser.add_argument("--mode", type=str, default="pipeline",
                        choices=["stage1", "stage2", "pipeline"],
                        help="Modalità di valutazione (default: pipeline)")

    # Modello 1
    parser.add_argument("--checkpoint",           type=str, default=None,
                        help="Checkpoint Modello 1 (landmark→gloss)")
    parser.add_argument("--output_dir",           type=str, default="outputs/phoenix_run1")
    parser.add_argument("--stage1_dir",           type=str, default="outputs/phoenix_run1",
                        help="Dir output Stage 1 (per tokenizer gloss)")

    # Modello 2
    parser.add_argument("--stage2_checkpoint",    type=str, default=None,
                        help="Checkpoint Modello 2 (gloss→translation)")
    parser.add_argument("--stage2_dir",           type=str, default="outputs/phoenix_stage2",
                        help="Dir output Stage 2 (per tokenizer translation)")

    # Dataset
    parser.add_argument("--val_csv",              type=str,
                        default="dataset/PHOENIX-2014-T.dev.corpus.csv")
    parser.add_argument("--landmarks_dir",        type=str,
                        default="dataset/landmarks_dev",
                        help="Richiesto per stage1 e pipeline")
    parser.add_argument("--train_csv",            type=str, default=None,
                        help="CSV training (per stats normalizzazione)")
    parser.add_argument("--train_landmarks_dir",  type=str,
                        default="dataset/landmarks_train")

    # Inferenza
    parser.add_argument("--batch_size",           type=int,   default=8)
    parser.add_argument("--num_workers",          type=int,   default=4)
    parser.add_argument("--max_decode_len",       type=int,   default=128)
    parser.add_argument("--num_examples",         type=int,   default=5)
    parser.add_argument("--device",               type=str,   default="cuda")
    parser.add_argument("--use_amp",              action="store_true")

    # Pesi landmark (override)
    parser.add_argument("--pose_weight",          type=float, default=None)
    parser.add_argument("--hand_weight",          type=float, default=None)
    parser.add_argument("--face_weight",          type=float, default=None)

    # Output
    parser.add_argument("--save_results",         type=str,   default=None)

    args = parser.parse_args()
    cfg  = vars(args)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    try:
        if cfg["mode"] == "stage1":
            results = evaluate_stage1(cfg)
        elif cfg["mode"] == "stage2":
            results = evaluate_stage2(cfg)
        else:
            results = evaluate_pipeline(cfg)

        _print_metrics(results, mode=cfg["mode"], num_examples=cfg["num_examples"])

        if cfg.get("save_results"):
            save_path = Path(cfg["save_results"])
            save_data = {k: v for k, v in results.items()
                         if k not in ("hyps", "refs", "gloss_hyps", "trans_refs",
                                      "gloss_refs", "diversity", "similarity")}
            save_data["diversity"]  = results.get("diversity", {})
            save_data["similarity"] = results.get("similarity", {})
            save_data["num_hyps"]   = len(results.get("hyps", []))
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"\n[EVAL] Risultati salvati in {save_path}")

    finally:
        _cleanup(device)


if __name__ == "__main__":
    main()
