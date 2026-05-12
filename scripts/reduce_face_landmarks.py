#!/usr/bin/env python3
"""
Reduce MediaPipe face landmarks to selected subsets while keeping pose and hands.

Output layout keeps the same file names and stores arrays as (T, N_selected, 4), where:
N_selected = 17 pose + 21 left hand + 21 right hand + compressed face points.

Face compression strategy:
- mouth: explicitly sample 12 key points to perfectly preserve shape
- left/right eye: sample every 2nd point to perfectly preserve the loop shape (8 points)
- left/right eyebrow: calculate exact average between 3 upper/lower pairs to form a central line
- nose tip: keep 1 point
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# -------------------------------------------------------------------------
# EXPLICIT INDICES SELECTION
# -------------------------------------------------------------------------

# Selezioniamo 12 punti chiave (8 esterni, 4 interni), ordinati in senso orario
REDUCED_MOUTH_IDX = [
    # Labbro Esterno (partendo da sinistra, senso orario)
    61, 39, 0, 269, 291, 375, 17, 181,
    # Labbro Interno (partendo da sinistra, senso orario)
    78, 13, 308, 14
]

# Per gli occhi, prendiamo un punto su due dall'anello originale di 16 punti (8 punti totali)
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
REDUCED_LEFT_EYE_IDX = LEFT_EYE_IDX[::2]

RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
REDUCED_RIGHT_EYE_IDX = RIGHT_EYE_IDX[::2]

NOSE_TIP_IDX = [1]

# Per le sopracciglia, definiamo le COPPIE (superiore, inferiore)
# da mediare per ottenere una linea centrale di 3 punti per lato.
# Formato: (Interno, Centrale, Esterno)
LEFT_BROW_PAIRS = [(107, 55), (105, 52), (70, 46)]
RIGHT_BROW_PAIRS = [(336, 285), (334, 282), (300, 276)]

POSE = 17
LEFT_HAND = 21
RIGHT_HAND = 21
FACE_OFFSET = POSE + LEFT_HAND + RIGHT_HAND  # 59
N_DIMS = 4

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


def average_pairs(lm: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Calcola la media esatta per un elenco di coppie di indici per creare la linea centrale."""
    pts = []
    for p1, p2 in pairs:
        pt = (lm[:, FACE_OFFSET + p1, :] + lm[:, FACE_OFFSET + p2, :]) / 2.0
        pts.append(pt)
    return np.stack(pts, axis=1)


def reduce_landmarks_array(arr: np.ndarray) -> np.ndarray:
    lm = ensure_3d_landmarks(arr)

    if lm.shape[1] <= FACE_OFFSET:
        raise ValueError(
            f"Input has {lm.shape[1]} landmarks, but face landmarks start at index {FACE_OFFSET}"
        )

    pose_and_hands = lm[:, :FACE_OFFSET, :]
    reduced_groups = [pose_and_hands]

    # 1. Punti diretti (Bocca, Occhi, Naso)
    direct_groups = [
        REDUCED_MOUTH_IDX,
        REDUCED_LEFT_EYE_IDX,
        REDUCED_RIGHT_EYE_IDX,
        NOSE_TIP_IDX
    ]
    for idx_list in direct_groups:
        local_idx = np.asarray(idx_list, dtype=np.int64)
        reduced_groups.append(lm[:, FACE_OFFSET + local_idx, :])

    # 2. Punti mediati (Sopracciglia)
    reduced_groups.append(average_pairs(lm, LEFT_BROW_PAIRS))
    reduced_groups.append(average_pairs(lm, RIGHT_BROW_PAIRS))

    # Concatena tutto lungo l'asse dei landmarks
    reduced = np.concatenate(reduced_groups, axis=1)
    return reduced.astype(np.float32, copy=False)


def process_directory(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_files = sorted(input_dir.glob("*_landmarks.npy"))
    ok = 0
    skipped = 0

    for i, npy_path in enumerate(npy_files, start=1):
        out_path = output_dir / npy_path.name
        try:
            arr = np.load(npy_path)
            reduced = reduce_landmarks_array(arr)
            np.save(out_path, reduced)
            ok += 1
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping {npy_path.name}: {exc}")

        if i % 200 == 0 or i == len(npy_files):
            print(f"[{input_dir.name}] {i}/{len(npy_files)} processed")

    return ok, skipped


def plot_diagnostic_frame(source_npy: Path, out_png: Path, frame_idx: int = 0) -> None:
    import matplotlib.pyplot as plt

    # Carichiamo ed eseguiamo la riduzione sul frame di test per plottare direttamente il risultato
    arr = np.load(source_npy)
    reduced_arr = reduce_landmarks_array(arr)
    
    frame_idx = max(0, min(frame_idx, reduced_arr.shape[0] - 1))
    frame = reduced_arr[frame_idx]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Mappiamo gli offset del nuovo array ridotto per il plot
    # L'array 'frame' ora contiene: [0:59 Pose/Mani] + [59:71 Bocca] + [71:79 Occhio Sx] + [79:87 Occhio Dx] + [87 Naso] + [88:91 Sopracciglia Sx] + [91:94 Sopracciglia Dx]
    plot_sections = {
        "mouth":      {"slice": slice(59, 71), "color": "#e53935"},
        "left_eye":   {"slice": slice(71, 79), "color": "#1e88e5"},
        "right_eye":  {"slice": slice(79, 87), "color": "#1e88e5"},
        "nose_tip":   {"slice": slice(87, 88), "color": "#fb8c00"},
        "left_brow":  {"slice": slice(88, 91), "color": "#43a047"},
        "right_brow": {"slice": slice(91, 94), "color": "#43a047"},
    }

    for group_name, info in plot_sections.items():
        pts = frame[info["slice"], :]
        
        # Opzionale: per occhio e bocca chiudiamo il poligono visivo unendo l'ultimo punto al primo nel plot
        if group_name in ["left_eye", "right_eye"]:
            pts = np.concatenate([pts, pts[0:1]], axis=0)

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            marker="o",
            markersize=5,
            linewidth=1.5 if group_name != "nose_tip" else 0,
            c=info["color"],
            label=group_name if group_name not in ["right_eye", "right_brow"] else None, # Evita duplicati in legenda
        )

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

    total_face_points = len(REDUCED_MOUTH_IDX) + len(REDUCED_LEFT_EYE_IDX) + len(REDUCED_RIGHT_EYE_IDX) + len(NOSE_TIP_IDX) + 6
    total_landmarks = FACE_OFFSET + total_face_points

    print(f"[INFO] Total selected face points: {total_face_points}")
    print(f"[INFO] New landmarks per frame: {total_landmarks} (59 body + {total_face_points} face)")
    print(f"[INFO] New feat_dim: {total_landmarks * N_DIMS}")

    ok, skipped = process_directory(in_dir, out_dir)
    print(f"[DONE] Converted files: {ok}, skipped: {skipped}")

    sample_npy = Path(args.sample_npy) if args.sample_npy else None
    if sample_npy is None:
        candidates = sorted(in_dir.glob("*_landmarks.npy"))
        if candidates:
            sample_npy = candidates[0]

    if sample_npy:
        try:
            plot_diagnostic_frame(
                source_npy=sample_npy,
                out_png=Path(args.plot_path),
                frame_idx=args.frame_idx,
            )
        except ModuleNotFoundError as e:
            print(f"[WARN] Skipping plot generation: {e}")
            print(f"[INFO] Install matplotlib if you want diagnostic plots: pip install matplotlib")

if __name__ == "__main__":
    main()