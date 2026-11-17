#!/usr/bin/env python3
"""Visualize 94 holistic landmarks over a single dataset image.

Usage examples:
  python scripts/visualize_landmarks.py --image dataset/dev/<sequence>/<frame>.png
  python scripts/visualize_landmarks.py --split dev
"""
from pathlib import Path
import argparse
import sys
import cv2
import numpy as np
import importlib.util
import mediapipe as mp


def load_landmark_extractor_module(repo_root: Path):
    """Dynamically load the existing extract_landmarks.py module.

    Returns the LandmarkExtractor class.
    """
    src = repo_root / "scripts" / "extract_landmarks.py"
    spec = importlib.util.spec_from_file_location("extract_landmarks", str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LandmarkExtractor


def pick_example_image(root: Path, split: str = "dev") -> Path | None:
    data_dir = root / "dataset" / split
    if not data_dir.exists():
        return None
    # Find first sequence folder then first image inside
    for seq in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        imgs = sorted(list(seq.glob("*.png")) + list(seq.glob("*.jpg")))
        if imgs:
            return imgs[0]
    return None


def draw_landmarks_overlay(image_path: Path, landmarks: np.ndarray, out_path: Path, draw_ids: bool = True, draw_points: bool = True):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    # Colors for groups: pose, left hand, right hand, face
    colors = {
        'pose': (0, 255, 0),        # green
        'left_hand': (255, 0, 0),   # blue
        'right_hand': (0, 0, 255),  # red
        'face': (0, 255, 255),      # yellow
    }

    # Landmark ranges from extractor
    groups = [
        (0, 16, 'pose'),
        (17, 37, 'left_hand'),
        (38, 58, 'right_hand'),
        (59, 93, 'face'),
    ]

    overlay = img.copy()
    for start, end, gname in groups:
        for i in range(start, end + 1):
            if i >= landmarks.shape[0]:
                continue
            x, y, z, conf = landmarks[i]
            if conf <= 0.0:
                continue
            # Some face landmarks may be slightly outside [0,1]; clamp
            px = int(np.clip(x, 0.0, 1.0) * (w - 1))
            py = int(np.clip(y, 0.0, 1.0) * (h - 1))
            if draw_points:
                cv2.circle(overlay, (px, py), radius=3, color=colors[gname], thickness=-1)
            if draw_ids:
                cv2.putText(overlay, str(i), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend overlay for semi-transparent points
    out = cv2.addWeighted(overlay, 0.9, img, 0.1, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def draw_skeleton_overlay(image_path: Path, landmarks: np.ndarray, out_path: Path):
    """Draw pose (upper-body) and hand skeletons using hardcoded connection sets.

    This version keeps the original image as background. Use draw_full_skeleton_overlay
    for transparent or face-including skeletons.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    overlay = img.copy()

    # Simple upper-body pose connections (use only indices 0..16)
    pose_connections = [
        (11, 13), (13, 15),  # left shoulder->elbow->wrist
        (12, 14), (14, 16),  # right shoulder->elbow->wrist
        (11, 12),            # shoulders
        (0, 11), (0, 12),    # nose to shoulders (approx neck)
    ]

    for a, b in pose_connections:
        if a < landmarks.shape[0] and b < landmarks.shape[0]:
            xa, ya, za, ca = landmarks[a]
            xb, yb, zb, cb = landmarks[b]
            if ca > 0.0 and cb > 0.0:
                pa = (int(np.clip(xa, 0.0, 1.0) * (w - 1)), int(np.clip(ya, 0.0, 1.0) * (h - 1)))
                pb = (int(np.clip(xb, 0.0, 1.0) * (w - 1)), int(np.clip(yb, 0.0, 1.0) * (h - 1)))
                cv2.line(overlay, pa, pb, (0, 255, 0), 2)

    # Hand connections for 21-landmark hands
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    left_offset = 17
    for a, b in hand_connections:
        ia, ib = a + left_offset, b + left_offset
        if ia < landmarks.shape[0] and ib < landmarks.shape[0]:
            xa, ya, za, ca = landmarks[ia]
            xb, yb, zb, cb = landmarks[ib]
            if ca > 0.0 and cb > 0.0:
                pa = (int(np.clip(xa, 0.0, 1.0) * (w - 1)), int(np.clip(ya, 0.0, 1.0) * (h - 1)))
                pb = (int(np.clip(xb, 0.0, 1.0) * (w - 1)), int(np.clip(yb, 0.0, 1.0) * (h - 1)))
                cv2.line(overlay, pa, pb, (255, 0, 0), 2)

    right_offset = 38
    for a, b in hand_connections:
        ia, ib = a + right_offset, b + right_offset
        if ia < landmarks.shape[0] and ib < landmarks.shape[0]:
            xa, ya, za, ca = landmarks[ia]
            xb, yb, zb, cb = landmarks[ib]
            if ca > 0.0 and cb > 0.0:
                pa = (int(np.clip(xa, 0.0, 1.0) * (w - 1)), int(np.clip(ya, 0.0, 1.0) * (h - 1)))
                pb = (int(np.clip(xb, 0.0, 1.0) * (w - 1)), int(np.clip(yb, 0.0, 1.0) * (h - 1)))
                cv2.line(overlay, pa, pb, (0, 0, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def draw_full_skeleton_overlay(image_path: Path, landmarks: np.ndarray, out_path: Path, include_face: bool = False, transparent: bool = False, scale: float = 1.0):
    """Draw skeleton including pose, hands, and optional face connections.

    If `transparent` is True, the output is a RGBA PNG with transparent background.
    `scale` scales the output image resolution (and implicitly the drawing thickness).
    """
    # Determine base size from image
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h0, w0 = img.shape[:2]
    w, h = int(w0 * scale), int(h0 * scale)

    # Create target canvas
    if transparent:
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        bg_is_rgba = True
    else:
        canvas_bgr = cv2.resize(img, (w, h)) if scale != 1.0 else img.copy()
        canvas = canvas_bgr.copy()
        bg_is_rgba = False

    def to_px(x, y):
        return (int(np.clip(x, 0.0, 1.0) * (w - 1)), int(np.clip(y, 0.0, 1.0) * (h - 1)))

    # Pose connections (same as before)
    pose_connections = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12),
        (0, 11), (0, 12),
    ]

    # Line thickness scales with image size
    thickness = max(1, int(round(2 * scale)))

    # Draw pose
    for a, b in pose_connections:
        if a < landmarks.shape[0] and b < landmarks.shape[0]:
            xa, ya, za, ca = landmarks[a]
            xb, yb, zb, cb = landmarks[b]
            if ca > 0.0 and cb > 0.0:
                pa = to_px(xa, ya)
                pb = to_px(xb, yb)
                if bg_is_rgba:
                    cv2.line(canvas, pa, pb, (0, 255, 0, 255), thickness)
                else:
                    cv2.line(canvas, pa, pb, (0, 255, 0), thickness)

    # Hand connections
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    left_offset = 17
    right_offset = 38
    for a, b in hand_connections:
        for offset, color in [(left_offset, (255, 0, 0, 255)), (right_offset, (0, 0, 255, 255))]:
            ia, ib = a + offset, b + offset
            if ia < landmarks.shape[0] and ib < landmarks.shape[0]:
                xa, ya, za, ca = landmarks[ia]
                xb, yb, zb, cb = landmarks[ib]
                if ca > 0.0 and cb > 0.0:
                    pa = to_px(xa, ya)
                    pb = to_px(xb, yb)
                    if bg_is_rgba:
                        cv2.line(canvas, pa, pb, color, thickness)
                    else:
                        cv2.line(canvas, pa, pb, color[:3], thickness)

    # Face connections (connect groups: mouth, left eye, right eye, eyebrows)
    if include_face:
        # mouth 59-70 (12 pts) -> connect in loop
        mouth_start = 59
        mouth_len = 12
        mouth_idxs = list(range(mouth_start, mouth_start + mouth_len))
        for i in range(len(mouth_idxs)):
            a = mouth_idxs[i]
            b = mouth_idxs[(i + 1) % len(mouth_idxs)]
            if a < landmarks.shape[0] and b < landmarks.shape[0]:
                xa, ya, za, ca = landmarks[a]
                xb, yb, zb, cb = landmarks[b]
                if ca > 0.0 and cb > 0.0:
                    pa = to_px(xa, ya)
                    pb = to_px(xb, yb)
                    color = (0, 255, 255, 255) if bg_is_rgba else (0, 255, 255)
                    if bg_is_rgba:
                        cv2.line(canvas, pa, pb, color, thickness)
                    else:
                        cv2.line(canvas, pa, pb, color)

        # left eye 71-78 (8 pts)
        left_eye = list(range(71, 71 + 8))
        for i in range(len(left_eye)):
            a = left_eye[i]
            b = left_eye[(i + 1) % len(left_eye)]
            if a < landmarks.shape[0] and b < landmarks.shape[0]:
                xa, ya, za, ca = landmarks[a]
                xb, yb, zb, cb = landmarks[b]
                if ca > 0.0 and cb > 0.0:
                    pa = to_px(xa, ya)
                    pb = to_px(xb, yb)
                    color = (0, 200, 200, 255) if bg_is_rgba else (0, 200, 200)
                    if bg_is_rgba:
                        cv2.line(canvas, pa, pb, color, thickness)
                    else:
                        cv2.line(canvas, pa, pb, color)

        # right eye 79-86 (8 pts)
        right_eye = list(range(79, 79 + 8))
        for i in range(len(right_eye)):
            a = right_eye[i]
            b = right_eye[(i + 1) % len(right_eye)]
            if a < landmarks.shape[0] and b < landmarks.shape[0]:
                xa, ya, za, ca = landmarks[a]
                xb, yb, zb, cb = landmarks[b]
                if ca > 0.0 and cb > 0.0:
                    pa = to_px(xa, ya)
                    pb = to_px(xb, yb)
                    color = (0, 200, 200, 255) if bg_is_rgba else (0, 200, 200)
                    if bg_is_rgba:
                        cv2.line(canvas, pa, pb, color, thickness)
                    else:
                        cv2.line(canvas, pa, pb, color)

        # eyebrows: left 88-90 (3 pts), right 91-93 (3 pts) -> connect sequentially
        left_eyebrow = list(range(88, 88 + 3))
        right_eyebrow = list(range(91, 91 + 3))
        for group in (left_eyebrow, right_eyebrow):
            for i in range(len(group) - 1):
                a = group[i]
                b = group[i + 1]
                if a < landmarks.shape[0] and b < landmarks.shape[0]:
                    xa, ya, za, ca = landmarks[a]
                    xb, yb, zb, cb = landmarks[b]
                    if ca > 0.0 and cb > 0.0:
                        pa = to_px(xa, ya)
                        pb = to_px(xb, yb)
                        color = (200, 200, 0, 255) if bg_is_rgba else (200, 200, 0)
                        if bg_is_rgba:
                            cv2.line(canvas, pa, pb, color, thickness)
                        else:
                            cv2.line(canvas, pa, pb, color)

        # nose tip (87) as a small circle
        if 87 < landmarks.shape[0]:
            xn, yn, zn, cn = landmarks[87]
            if cn > 0.0:
                pn = to_px(xn, yn)
                if bg_is_rgba:
                    cv2.circle(canvas, pn, max(1, thickness), (0, 255, 255, 255), -1)
                else:
                    cv2.circle(canvas, pn, max(1, thickness), (0, 255, 255), -1)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure PNG if transparent requested
    if transparent and bg_is_rgba:
        cv2.imwrite(str(out_path), canvas)
    else:
        # If we produced RGBA but transparent=False, convert to BGR
        if canvas.shape[2] == 4:
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(str(out_path), canvas_bgr)
        else:
            cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Visualize holistic landmarks on one image")
    parser.add_argument("--image", type=str, help="Path to an image file (png/jpg)")
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], default="dev", help="Dataset split to pick an example from")
    parser.add_argument("--model-path", type=str, default=None, help="Path to holistic_landmarker.task model file")
    parser.add_argument("--output", type=str, default=None, help="Output overlay image path")
    parser.add_argument("--no-ids", action="store_true", help="Do not draw landmark id labels on overlay")
    parser.add_argument("--skeleton", action="store_true", help="Also save an additional skeleton-only image (pose+hands)")
    parser.add_argument("--skeleton-output", type=str, default=None, help="Path for skeleton-only output image")
    parser.add_argument("--full-skeleton", action="store_true", help="Include face landmarks in the skeleton output")
    parser.add_argument("--transparent", action="store_true", help="When creating skeleton-only output, produce a transparent PNG")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for output images (e.g., 2.0 for double resolution)")
    parser.add_argument("--transparent-output", type=str, default=None, help="Path for transparent skeleton output image (PNG)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = pick_example_image(repo_root, args.split)
        if image_path is None:
            print("No example image found in dataset; provide --image")
            sys.exit(1)

    LandmarkExtractor = load_landmark_extractor_module(repo_root)
    extractor = LandmarkExtractor(model_path=args.model_path)
    try:
        lm = extractor.extract_landmarks_from_image(image_path)
    finally:
        extractor.close()

    if lm is None:
        print(f"No landmarks extracted for {image_path}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else repo_root / "outputs" / "visualized_landmarks" / (image_path.stem + "_landmarks.png")
    draw_landmarks_overlay(image_path, lm, out_path, draw_ids=(not args.no_ids), draw_points=True)
    print(f"Saved overlay to: {out_path}")

    if args.skeleton:
        sk_out = Path(args.skeleton_output) if args.skeleton_output else repo_root / "outputs" / "visualized_landmarks" / (image_path.stem + "_skeleton.png")
        if args.scale != 1.0 or args.full_skeleton or args.transparent:
            # use full renderer for scaling/face/transparent options
            draw_full_skeleton_overlay(image_path, lm, sk_out, include_face=args.full_skeleton, transparent=args.transparent, scale=args.scale)
        else:
            draw_skeleton_overlay(image_path, lm, sk_out)
        print(f"Saved skeleton to: {sk_out}")

    if args.transparent and not args.skeleton:
        # If user asked for transparent but not skeleton, still allow generating a transparent full-skeleton image
        t_out = Path(args.transparent_output) if args.transparent_output else repo_root / "outputs" / "visualized_landmarks" / (image_path.stem + "_skeleton_transparent.png")
        draw_full_skeleton_overlay(image_path, lm, t_out, include_face=args.full_skeleton, transparent=True, scale=args.scale)
        print(f"Saved transparent skeleton to: {t_out}")


if __name__ == "__main__":
    main()
