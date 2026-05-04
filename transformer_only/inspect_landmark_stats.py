#!/usr/bin/env python3
"""Inspect landmark scale/statistics from the How2Sign dataloader.

This script helps verify whether source landmarks are:
- centered too tightly around zero,
- dominated by extreme outliers,
- polluted by padding or NaN/Inf values.

It can report:
- per-sample stats before padding
- batch stats after collate_fn padding
- approximate zero ratios and percentiles

Example:
    python -m transformer_only.inspect_landmark_stats \
        --csv dataset/how2sign_realigned_train.csv \
        --landmarks_dir dataset/landmarks_normalized \
        --num_samples 8
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from transformer_only.data.how2sign_loader import (
    How2SignDataset,
    build_tokenizer,
    collate_fn,
)


def _percentiles(x: torch.Tensor, probs: Iterable[float]) -> dict[float, float]:
    flat = x.detach().reshape(-1).float().cpu()
    if flat.numel() == 0:
        return {p: float("nan") for p in probs}
    qs = torch.tensor(list(probs), dtype=torch.float32)
    values = torch.quantile(flat, qs).tolist()
    return {float(p): float(v) for p, v in zip(probs, values)}


def _tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().float()
    finite = torch.isfinite(x)
    finite_x = x[finite]
    total = x.numel()
    finite_count = int(finite.sum().item())
    nonfinite_count = total - finite_count

    if finite_count == 0:
        return {
            "shape": tuple(x.shape),
            "numel": total,
            "finite": 0,
            "nonfinite": nonfinite_count,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "abs_max": float("nan"),
            "zero_ratio": float("nan"),
            "p01": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
        }

    abs_x = finite_x.abs()
    percentiles = _percentiles(finite_x, [0.01, 0.50, 0.99])
    zero_ratio = float((finite_x == 0).float().mean().item())

    return {
        "shape": tuple(x.shape),
        "numel": total,
        "finite": finite_count,
        "nonfinite": nonfinite_count,
        "mean": float(finite_x.mean().item()),
        "std": float(finite_x.std(unbiased=False).item()),
        "min": float(finite_x.min().item()),
        "max": float(finite_x.max().item()),
        "abs_max": float(abs_x.max().item()),
        "zero_ratio": zero_ratio,
        "p01": percentiles[0.01],
        "p50": percentiles[0.50],
        "p99": percentiles[0.99],
    }


def _print_stats(title: str, stats: dict) -> None:
    print(f"\n{title}")
    print(f"  shape:       {stats['shape']}")
    print(f"  numel:       {stats['numel']}")
    print(f"  finite:      {stats['finite']}  nonfinite: {stats['nonfinite']}")
    print(f"  mean:        {stats['mean']:.6f}")
    print(f"  std:         {stats['std']:.6f}")
    print(f"  min / max:   {stats['min']:.6f} / {stats['max']:.6f}")
    print(f"  abs max:     {stats['abs_max']:.6f}")
    print(f"  zero ratio:   {stats['zero_ratio']:.2%}")
    print(f"  p01 / p50 / p99: {stats['p01']:.6f} / {stats['p50']:.6f} / {stats['p99']:.6f}")


def _framewise_stats(src: torch.Tensor) -> dict:
    """Summarize frame-level norms to detect spikes or collapsed sequences."""
    x = src.detach().float()
    if x.ndim != 2:
        raise ValueError(f"Expected (T, F) tensor, got shape {tuple(x.shape)}")
    frame_norms = torch.linalg.vector_norm(x, dim=-1)
    return _tensor_stats(frame_norms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect landmark statistics from the dataloader.")
    parser.add_argument("--csv", type=str, default="dataset/how2sign_realigned_train.csv")
    parser.add_argument("--landmarks_dir", type=str, default="dataset/landmarks_normalized")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--flatten_landmarks", action="store_true", default=True)
    parser.add_argument("--no_flatten_landmarks", action="store_false", dest="flatten_landmarks")
    parser.add_argument("--use_hand_relative_norm", action="store_true", default=True)
    parser.add_argument("--no_hand_relative_norm", action="store_false", dest="use_hand_relative_norm")
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--pose_weight", type=float, default=1.0)
    parser.add_argument("--hand_weight", type=float, default=1.0)
    parser.add_argument("--face_weight", type=float, default=1.0)
    parser.add_argument("--show_per_sample", action="store_true", default=True)
    parser.add_argument("--no_show_per_sample", action="store_false", dest="show_per_sample")
    parser.add_argument("--show_batch", action="store_true", default=True)
    parser.add_argument("--no_show_batch", action="store_false", dest="show_batch")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    landmarks_dir = Path(args.landmarks_dir)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else None

    if tokenizer_path and tokenizer_path.exists():
        tokenizer = build_tokenizer(csv_path, save_path=None)
        from transformer_only.data.how2sign_loader import SentenceTokenizer

        tokenizer = SentenceTokenizer.load(tokenizer_path)
    else:
        tokenizer = build_tokenizer(csv_path, save_path=None)

    dataset = How2SignDataset(
        csv_path=csv_path,
        landmarks_dir=landmarks_dir,
        tokenizer=tokenizer,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        flatten_landmarks=args.flatten_landmarks,
        pose_weight=args.pose_weight,
        hand_weight=args.hand_weight,
        face_weight=args.face_weight,
        normalize_stats=None,
        use_hand_relative_norm=args.use_hand_relative_norm,
        augment=args.augment,
    )

    print("[Inspect] Dataset loaded")
    print(f"  samples: {len(dataset)}")
    print(f"  csv:     {csv_path}")
    print(f"  dir:     {landmarks_dir}")
    print(f"  flatten: {args.flatten_landmarks}")
    print(f"  hand-relative-norm: {args.use_hand_relative_norm}")
    print(f"  augment: {args.augment}")
    print(f"  weights: pose={args.pose_weight} hand={args.hand_weight} face={args.face_weight}")

    sample_count = min(args.num_samples, len(dataset))
    per_sample_src = []
    per_sample_frames = []

    if args.show_per_sample:
        print("\n[Inspect] Per-sample source stats (before padding)")

    for idx in range(sample_count):
        item = dataset[idx]
        src = item["src"]
        if src.ndim == 3:
            flat_src = src.view(src.shape[0], -1)
        else:
            flat_src = src

        per_sample_src.append(flat_src)
        per_sample_frames.append(flat_src.shape[0])

        if args.show_per_sample:
            stats = _tensor_stats(flat_src)
            frame_stats = _framewise_stats(flat_src)
            print(f"\n  sample[{idx}] {item['sentence_id']} | {item['sentence_name']}")
            print(f"    file: {item['npy_path']}")
            _print_stats("    src tensor", stats)
            _print_stats("    frame norms", frame_stats)

    if per_sample_src:
        all_concat = torch.cat(per_sample_src, dim=0)
        _print_stats("\n[Inspect] Aggregate over sampled frames (pre-padding)", _tensor_stats(all_concat))
        _print_stats("[Inspect] Aggregate frame norms (pre-padding)", _framewise_stats(all_concat))
        print(f"[Inspect] sampled frame lengths: min={min(per_sample_frames)} max={max(per_sample_frames)} mean={sum(per_sample_frames)/len(per_sample_frames):.2f}")

    if args.show_batch:
        loader = DataLoader(
            dataset,
            batch_size=min(args.batch_size, sample_count),
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(
                b,
                pad_id=tokenizer.pad_id,
                bos_id=tokenizer.bos_id,
                unk_id=tokenizer.unk_id,
                eos_id=tokenizer.eos_id,
                word_dropout_prob=0.0,
            ),
            pin_memory=False,
            persistent_workers=False,
        )
        batch = next(iter(loader))
        src = batch["src"]
        src_mask = batch["src_key_padding_mask"]
        valid = ~src_mask
        valid_src = src[valid]
        print("\n[Inspect] Batch stats (after collate/padding)")
        _print_stats("  padded batch src", _tensor_stats(src))
        _print_stats("  valid-only src", _tensor_stats(valid_src))
        print(f"  padding ratio: {float(src_mask.float().mean().item()):.2%}")

        # Per-frame norm distribution for valid positions only.
        valid_frames = src[valid].view(-1, src.shape[-1]) if src.ndim == 3 else valid_src
        if valid_frames.numel() > 0:
            _print_stats("  valid-only frame norms", _framewise_stats(valid_frames))

    print("\n[Inspect] Done.")


if __name__ == "__main__":
    main()
