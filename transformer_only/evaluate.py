"""
Evaluation script for SignLanguageTransformer (transformer_only).

Defaults:
- CSV: dataset/how2sign_realigned_validation.csv
- Landmarks: dataset/landmarks_validation_normalized
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from transformer_only.data.how2sign_loader import (
    How2SignDataset,
    SentenceTokenizer,
    collate_fn,
    build_tokenizer,
)
from transformer_only.models import SignLanguageTransformer

try:
    from sacrebleu.metrics import BLEU
except ImportError as exc:
    raise ImportError(
        "sacrebleu is required for BLEU. Install it with: pip install sacrebleu"
    ) from exc


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(predictions, [references])
    return result.score


def decode_batch(token_ids: list[list[int]], tokenizer: SentenceTokenizer) -> list[str]:
    return [tokenizer.decode(ids) for ids in token_ids]


def _edit_distance(ref_words: list[str], hyp_words: list[str]) -> int:
    if not ref_words:
        return len(hyp_words)
    if not hyp_words:
        return len(ref_words)

    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, start=1):
        curr = [i] + [0] * len(hyp_words)
        for j, h in enumerate(hyp_words, start=1):
            if r == h:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[-1]


def compute_wer(predictions: list[str], references: list[str]) -> float:
    total_edits = 0
    total_words = 0
    for hyp, ref in zip(predictions, references):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_edits += _edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
    return total_edits / max(1, total_words)


def _compute_output_diversity(predictions: list[str]) -> dict:
    """Analizza la diversità degli output per rilevare collasso su frasi ripetute."""
    from collections import Counter
    import math
    
    all_words = []
    for pred in predictions:
        words = pred.strip().split()
        all_words.extend(words)
    
    if not all_words:
        return {
            "unique_words": 0,
            "total_words": 0,
            "vocab_coverage": 0,
            "entropy": 0,
            "top_10_word_ratio": 0,
            "most_common_word": ("N/A", 0),
        }
    
    word_counts = Counter(all_words)
    unique_words = len(word_counts)
    total_words = len(all_words)
    
    # Entropia di Shannon
    probs = [c / total_words for c in word_counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    
    top_10_freq = sum(count for word, count in word_counts.most_common(10))
    top_10_ratio = top_10_freq / total_words
    
    return {
        "unique_words": unique_words,
        "total_words": total_words,
        "vocab_coverage": unique_words / max(1, total_words),
        "entropy": entropy,
        "top_10_word_ratio": top_10_ratio,
        "most_common_word": word_counts.most_common(1)[0] if word_counts else ("N/A", 0),
    }


def _compute_output_similarities(predictions: list[str]) -> dict:
    """Rileva se il modello produce output troppo simili (segno di memorizzazione)."""
    from difflib import SequenceMatcher
    import random
    
    if len(predictions) < 2:
        return {
            "avg_similarity": 0,
            "max_similarity": 0,
            "identical_or_near_count": 0,
        }
    
    similarities = []
    identical = 0
    
    # Campiona coppie casuali (per velocità)
    sample_size = min(100, len(predictions) // 2)
    
    for _ in range(sample_size):
        i, j = random.sample(range(len(predictions)), 2)
        sim = SequenceMatcher(None, predictions[i], predictions[j]).ratio()
        similarities.append(sim)
        if sim > 0.95:
            identical += 1
    
    return {
        "avg_similarity": sum(similarities) / max(1, len(similarities)),
        "max_similarity": max(similarities) if similarities else 0,
        "identical_or_near_count": identical,
    }


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device)


def _resolve_checkpoint_path(cfg: dict) -> Path:
    checkpoint_value = cfg.get("checkpoint")
    output_dir = Path(cfg.get("output_dir") or "outputs/run2_optimized")

    candidates = []
    if checkpoint_value:
        candidates.append(Path(checkpoint_value))
    candidates.extend([
        output_dir / "best.pt",
        output_dir / "last.pt",
    ])

    for candidate in candidates:
        if candidate.exists():
            if checkpoint_value and candidate != Path(checkpoint_value):
                print(f"[EVAL] Checkpoint not found at {checkpoint_value}, using {candidate} instead.")
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched: {searched}")


def _load_tokenizer(cfg: dict) -> SentenceTokenizer:
    tok_path = Path(cfg["tokenizer_path"]) if cfg.get("tokenizer_path") else None
    if tok_path and tok_path.exists():
        return SentenceTokenizer.load(tok_path)

    output_dir = Path(cfg["output_dir"])
    default_tok = output_dir / "tokenizer.json"
    if default_tok.exists():
        return SentenceTokenizer.load(default_tok)

    train_csv = cfg.get("train_csv")
    if not train_csv:
        raise ValueError(
            "Tokenizer not found. Provide --tokenizer_path or --train_csv to build it."
        )
    return build_tokenizer(train_csv, save_path=default_tok)


def _resolve_model_cfg(cli_cfg: dict, ckpt_cfg: dict) -> dict:
    def _pick(key: str, default):
        return cli_cfg.get(key) if cli_cfg.get(key) is not None else ckpt_cfg.get(key, default)

    return {
        "feat_dim": _pick("feat_dim", 2108),
        "d_model": _pick("d_model", 512),
        "nhead": _pick("nhead", 8),
        "num_enc_layers": _pick("num_enc_layers", 6),
        "num_dec_layers": _pick("num_dec_layers", 6),
        "dim_feedforward": _pick("dim_feedforward", 2048),
        "dropout": _pick("dropout", 0.1),
        "max_src_len": _pick("max_src_len", 512),
        "max_tgt_len": _pick("max_tgt_len", 128),
        "label_smoothing": _pick("label_smoothing", 0.1),
        "src_embedding_type": _pick("src_embedding_type", "mlp"),
        "temporal_kernel_size": _pick("temporal_kernel_size", None),
        "temporal_blocks": _pick("temporal_blocks", None),
    }


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


def _resolve_data_weights(cli_cfg: dict, ckpt_cfg: dict) -> dict:
    def _pick(key: str, default):
        return cli_cfg.get(key) if cli_cfg.get(key) is not None else ckpt_cfg.get(key, default)

    return {
        "pose_weight": _pick("pose_weight", 1.0),
        "hand_weight": _pick("hand_weight", 1.3),
        "face_weight": _pick("face_weight", 0.7),
    }


def evaluate(cfg: dict) -> dict:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"

    checkpoint_path = _resolve_checkpoint_path(cfg)

    ckpt = _load_checkpoint(checkpoint_path, device)
    ckpt_cfg = ckpt.get("cfg", {})

    state_dict = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
    if state_dict is None:
        state_dict = ckpt

    tokenizer = _load_tokenizer(cfg)
    model_cfg = _resolve_model_cfg(cfg, ckpt_cfg)
    data_weights = _resolve_data_weights(cfg, ckpt_cfg)

    # Backward-compatible loading: older checkpoints may not store temporal CNN
    # hyperparameters in cfg, so infer them from checkpoint shapes.
    if model_cfg["src_embedding_type"] == "temporal_cnn":
        inferred = _infer_temporal_frontend_cfg(state_dict)
        if model_cfg.get("temporal_kernel_size") is None:
            model_cfg["temporal_kernel_size"] = inferred["temporal_kernel_size"] or 5
        if model_cfg.get("temporal_blocks") is None:
            model_cfg["temporal_blocks"] = inferred["temporal_blocks"] or 3

    dataset = How2SignDataset(
        csv_path=cfg["val_csv"],
        landmarks_dir=cfg["landmarks_dir"],
        tokenizer=tokenizer,
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
        pose_weight=data_weights["pose_weight"],
        hand_weight=data_weights["hand_weight"],
        face_weight=data_weights["face_weight"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
    )

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
        temporal_kernel_size=model_cfg["temporal_kernel_size"] or 5,
        temporal_blocks=model_cfg["temporal_blocks"] or 3,
        pad_id=tokenizer.pad_id,
        label_smoothing=model_cfg["label_smoothing"],
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    total_loss = 0.0
    total_tok = 0
    total_unk = 0
    total_hyp_tok = 0
    total_hyp_unk = 0
    all_hyps: list[str] = []
    all_refs: list[str] = []

    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out = batch["tgt_output"].to(device, non_blocking=True)
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

        n_tok = (tgt_out != model.pad_id).sum().item()
        n_unk = (tgt_out == tokenizer.unk_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tok += n_tok
        total_unk += n_unk

        hyps = model.greedy_decode(
            src=src,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=cfg["max_decode_len"],
            src_key_padding_mask=src_mask,
        )
        total_hyp_tok += sum(len(h) for h in hyps)
        total_hyp_unk += sum(sum(1 for t in h if t == tokenizer.unk_id) for h in hyps)
        all_hyps.extend(decode_batch(hyps, tokenizer))
        all_refs.extend(batch["sentences"])

    avg_loss = total_loss / max(1, total_tok)
    ppl = torch.exp(torch.tensor(min(avg_loss, 20.0))).item()
    bleu = compute_bleu(all_hyps, all_refs)
    wer = compute_wer(all_hyps, all_refs)

    tgt_unk_rate = total_unk / max(1, total_tok)
    hyp_unk_rate = total_hyp_unk / max(1, total_hyp_tok)
    avg_ref_len = total_tok / max(1, len(all_refs))
    avg_hyp_len = total_hyp_tok / max(1, len(all_hyps))
    
    # Diagnosi: il modello sta imparando token-level o intere frasi?
    diversity_stats = _compute_output_diversity(all_hyps)
    similarity_stats = _compute_output_similarities(all_hyps)

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "bleu": bleu,
        "wer": wer,
        "num_samples": len(dataset),
        "hyps": all_hyps,
        "refs": all_refs,
        "tgt_unk_rate": tgt_unk_rate,
        "hyp_unk_rate": hyp_unk_rate,
        "avg_ref_len": avg_ref_len,
        "avg_hyp_len": avg_hyp_len,
        "diversity": diversity_stats,
        "similarity": similarity_stats,
    }


def _cleanup(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    batch_size_default = 8
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/run2_optimized")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument(
        "--val_csv", type=str, default="dataset/how2sign_realigned_val.csv"
    )
    parser.add_argument(
        "--landmarks_dir", type=str, default="dataset/landmarks_validation_normalized"
    )
    parser.add_argument("--batch_size", type=int, default=batch_size_default)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_decode_len", type=int, default=128)
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pose_weight", type=float, default=None)
    parser.add_argument("--hand_weight", type=float, default=None)
    parser.add_argument("--face_weight", type=float, default=None)

    args = parser.parse_args()
    cfg = vars(args)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    try:
        results = evaluate(cfg)
        print("\nValidation results")
        print(f"  samples: {results['num_samples']}")
        print(f"  loss:    {results['loss']:.4f}")
        print(f"  ppl:     {results['ppl']:.2f}")
        print(f"  BLEU:    {results['bleu']:.2f}")
        print(f"  WER:     {results['wer']:.2%}")
        print(f"  tgt UNK: {results['tgt_unk_rate']:.2%}")
        print(f"  hyp UNK: {results['hyp_unk_rate']:.2%}")
        print(f"  avg ref len: {results['avg_ref_len']:.1f}")
        print(f"  avg hyp len: {results['avg_hyp_len']:.1f}")
        
        # Diagnosi diversità output
        div = results['diversity']
        sim = results['similarity']
        print(f"\n  Output Diversity (token-level learning?)")
        print(f"    unique words: {div['unique_words']} / {div['total_words']} ({div['vocab_coverage']:.1%})")
        print(f"    entropy: {div['entropy']:.3f}")
        print(f"    top-10 word ratio: {div['top_10_word_ratio']:.1%}")
        print(f"    most common: '{div['most_common_word'][0]}' ({div['most_common_word'][1]}x)")
        print(f"\n  Output Similarity (memorization check)")
        print(f"    avg similarity: {sim['avg_similarity']:.3f}")
        print(f"    identical/near-identical pairs (>0.95): {sim['identical_or_near_count']}")
        if sim['avg_similarity'] > 0.8:
            print(f"    ⚠️  ALTO: il modello genera output molto simili (possibile collasso)")
        elif sim['avg_similarity'] > 0.6:
            print(f"    ⚠️  MEDIO: output moderatamente simili")
        else:
            print(f"    ✓ BASSO: buona diversità")

        num_examples = cfg.get("num_examples", 0) or 0
        if num_examples > 0:
            print("\nExamples (reference -> prediction)")
            for i in range(min(num_examples, len(results["refs"]))):
                print(f"[{i+1:02d}] REF:  {results['refs'][i]}")
                print(f"     HYP:  {results['hyps'][i]}")
    finally:
        device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        _cleanup(device)


if __name__ == "__main__":
    main()
