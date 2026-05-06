#!/usr/bin/env python3
"""
Reduce MediaPipe face landmarks to selected subsets while keeping pose and hands.

Output layout keeps the same file names and stores arrays as (T, N_selected, 4), where:
N_selected = 17 pose + 21 left hand + 21 right hand + selected face points.

Also exports a diagnostic plot for one frame to visually verify selected indices.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _ordered_unique(values: Iterable[int]) -> list[int]:
    seen = set()
    out = []
    for v in values:
        iv = int(v)
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    return out


# FaceMesh local indices provided by user
MOUTH_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
]
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_IDX = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NOSE_TIP_IDX = [1]  # MediaPipe FaceMesh nose tip

POSE = 17
LEFT_HAND = 21
RIGHT_HAND = 21
FACE_OFFSET = POSE + LEFT_HAND + RIGHT_HAND  # 59
N_DIMS = 4


def build_face_subset_indices() -> dict[str, list[int]]:
    return {
        "mouth": _ordered_unique(MOUTH_IDX),
        "left_eye": _ordered_unique(LEFT_EYE_IDX),
        "right_eye": _ordered_unique(RIGHT_EYE_IDX),
        "left_brow": _ordered_unique(LEFT_BROW_IDX),
        "right_brow": _ordered_unique(RIGHT_BROW_IDX),
        "nose_tip": _ordered_unique(NOSE_TIP_IDX),
    }


def build_global_keep_indices(face_groups: dict[str, list[int]]) -> list[int]:
    pose_and_hands = list(range(FACE_OFFSET))
    face_local = []
    for group in ("mouth", "left_eye", "right_eye", "left_brow", "right_brow", "nose_tip"):
        face_local.extend(face_groups[group])
    face_local = _ordered_unique(face_local)
    face_global = [FACE_OFFSET + i for i in face_local]
    return pose_and_hands + face_global


def ensure_3d_landmarks(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[2] != N_DIMS:
            raise ValueError(f"Expected last dim {N_DIMS}, got {arr.shape}")
        return arr
    if arr.ndim == 2:
        if arr.shape[1] % N_DIMS != 0:
            raise ValueError(f"Flattened features not divisible by {N_DIMS}: {arr.shape}")
        n_landmarks = arr.shape[1] // N_DIMS
        return arr.reshape(arr.shape[0], n_landmarks, N_DIMS)
    raise ValueError(f"Unsupported shape: {arr.shape}")


def reduce_landmarks_array(arr: np.ndarray, keep_global: list[int]) -> np.ndarray:
    lm = ensure_3d_landmarks(arr)
    if lm.shape[1] < max(keep_global) + 1:
        raise ValueError(
            f"Input has {lm.shape[1]} landmarks, but max required index is {max(keep_global)}"
        )
    reduced = lm[:, keep_global, :]
    return reduced.astype(np.float32, copy=False)


def process_directory(input_dir: Path, output_dir: Path, keep_global: list[int]) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_files = sorted(input_dir.glob("*_landmarks.npy"))
    ok = 0
    skipped = 0

    for i, npy_path in enumerate(npy_files, start=1):
        out_path = output_dir / npy_path.name
        try:
            arr = np.load(npy_path)
            reduced = reduce_landmarks_array(arr, keep_global)
            np.save(out_path, reduced)
            ok += 1
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping {npy_path.name}: {exc}")

        if i % 200 == 0 or i == len(npy_files):
            print(f"[{input_dir.name}] {i}/{len(npy_files)} processed")

    return ok, skipped


def plot_diagnostic_frame(
    source_npy: Path,
    keep_global: list[int],
    face_groups: dict[str, list[int]],
    out_png: Path,
    frame_idx: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    arr = np.load(source_npy)
    lm = ensure_3d_landmarks(arr)
    frame_idx = max(0, min(frame_idx, lm.shape[0] - 1))
    frame = lm[frame_idx]

    face_all = frame[FACE_OFFSET:, :2]
    selected_global = np.array(keep_global[FACE_OFFSET:], dtype=np.int64)
    selected_face_local = selected_global - FACE_OFFSET

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(face_all[:, 0], face_all[:, 1], s=8, c="#bdbdbd", alpha=0.45, label="face all")

    colors = {
        "mouth": "#e53935",
        "left_eye": "#1e88e5",
        "right_eye": "#1e88e5",
        "left_brow": "#43a047",
        "right_brow": "#43a047",
        "nose_tip": "#fb8c00",
    }

    for group_name, local_idx in face_groups.items():
        idx = np.array(local_idx, dtype=np.int64)
        pts = frame[FACE_OFFSET + idx, :2]
        ax.scatter(pts[:, 0], pts[:, 1], s=28, c=colors[group_name], label=group_name)

    ax.set_title("Reduced face landmarks diagnostic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    print(f"[PLOT] Saved diagnostic plot: {out_png}")
    print(f"[PLOT] Source file: {source_npy.name}, frame={frame_idx}, selected_face_points={len(selected_face_local)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce face landmarks to selected indices")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plot_path", type=str, default="outputs/landmark_debug/reduced_face_frame.png")
    parser.add_argument("--sample_npy", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    face_groups = build_face_subset_indices()
    keep_global = build_global_keep_indices(face_groups)

    print("[INFO] Face groups counts:")
    for k, v in face_groups.items():
        print(f"  - {k}: {len(v)}")
    print(f"[INFO] Total selected face points: {len(keep_global) - FACE_OFFSET}")
    print(f"[INFO] New landmarks per frame: {len(keep_global)}")
    print(f"[INFO] New feat_dim: {len(keep_global) * N_DIMS}")

    ok, skipped = process_directory(in_dir, out_dir, keep_global)
    print(f"[DONE] Converted files: {ok}, skipped: {skipped}")

    sample_npy = Path(args.sample_npy) if args.sample_npy else None
    if sample_npy is None:
        candidates = sorted(in_dir.glob("*_landmarks.npy"))
        if not candidates:
            raise RuntimeError(f"No landmark files found in {in_dir}")
        sample_npy = candidates[0]

    plot_diagnostic_frame(
        source_npy=sample_npy,
        keep_global=keep_global,
        face_groups=face_groups,
        out_png=Path(args.plot_path),
        frame_idx=args.frame_idx,
    )


if __name__ == "__main__":
    main()
