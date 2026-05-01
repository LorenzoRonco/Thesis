"""
Evaluation script for SignLanguageTransformer (transformer_only).

Defaults:
- CSV: dataset/how2sign_realigned_validation.csv
- Landmarks: dataset/landmarks_validation_normalized
"""

import argparse
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


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device)


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
    }


def evaluate(cfg: dict) -> dict:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"

    checkpoint_path = Path(cfg["checkpoint"]) 
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = _load_checkpoint(checkpoint_path, device)
    ckpt_cfg = ckpt.get("cfg", {})

    tokenizer = _load_tokenizer(cfg)
    model_cfg = _resolve_model_cfg(cfg, ckpt_cfg)

    dataset = How2SignDataset(
        csv_path=cfg["val_csv"],
        landmarks_dir=cfg["landmarks_dir"],
        tokenizer=tokenizer,
        max_src_len=model_cfg["max_src_len"],
        max_tgt_len=model_cfg["max_tgt_len"],
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
        pad_id=tokenizer.pad_id,
        label_smoothing=model_cfg["label_smoothing"],
    ).to(device)

    state_dict = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
    if state_dict is None:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    total_loss = 0.0
    total_tok = 0
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
        total_loss += loss.item() * n_tok
        total_tok += n_tok

        hyps = model.greedy_decode(
            src=src,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=cfg["max_decode_len"],
            src_key_padding_mask=src_mask,
        )
        all_hyps.extend(decode_batch(hyps, tokenizer))
        all_refs.extend(batch["sentences"])

    avg_loss = total_loss / max(1, total_tok)
    ppl = torch.exp(torch.tensor(min(avg_loss, 20.0))).item()
    bleu = compute_bleu(all_hyps, all_refs)

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "bleu": bleu,
        "num_samples": len(dataset),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/run1/best.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/run1")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument(
        "--val_csv", type=str, default="dataset/how2sign_realigned_validation.csv"
    )
    parser.add_argument(
        "--landmarks_dir", type=str, default="dataset/landmarks_validation_normalized"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_decode_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    cfg = vars(args)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    results = evaluate(cfg)
    print("\nValidation results")
    print(f"  samples: {results['num_samples']}")
    print(f"  loss:    {results['loss']:.4f}")
    print(f"  ppl:     {results['ppl']:.2f}")
    print(f"  BLEU:    {results['bleu']:.2f}")


if __name__ == "__main__":
    main()
