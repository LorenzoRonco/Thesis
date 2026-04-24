#!/usr/bin/env python3
"""
Video Segmentation Script for How2Sign validation set.

Usage:
    python scripts/video_segmentation_validation.py [sample|full]

Input:  dataset/how2sign_realigned_val.csv
        dataset/validation/raw_videos/
Output: dataset/segmented_validation/
"""

import sys
from pathlib import Path
import argparse

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from preprocessing.video_segmentation import VideoSegmenter


def main():
    """Segment validation videos based on the validation CSV."""
    parser = argparse.ArgumentParser(
        description="Segmenta i video di validation in base al CSV realigned."
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='sample',
        choices=['sample', 'test', 'full'],
        help="sample/test per debug veloce, full per tutto il dataset",
    )
    parser.add_argument(
        '--target-fps',
        type=float,
        default=None,
        help="FPS output desiderati (es. 30 o 25). Se omesso, mantiene stream originale.",
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=None,
        help="Altezza output video (es. 480). Larghezza adattata mantenendo aspect ratio.",
    )
    parser.add_argument(
        '--num-videos',
        type=int,
        default=None,
        help="Numero massimo di video distinti da preprocessare.",
    )
    args = parser.parse_args()

    csv_path = root_dir / "dataset" / "how2sign_realigned_val.csv"
    video_dir = root_dir / "dataset" / "validation" / "raw_videos"
    output_dir = root_dir / "dataset" / "segmented_validation"

    if not csv_path.exists():
        print(f"ERROR: CSV non trovato: {csv_path}")
        sys.exit(1)

    if not video_dir.exists():
        print(f"ERROR: Cartella video non trovata: {video_dir}")
        sys.exit(1)

    segmenter = VideoSegmenter(
        str(csv_path),
        str(video_dir),
        str(output_dir),
        target_fps=args.target_fps,
        target_height=args.target_height,
    )
    segmenter.get_statistics()

    mode = args.mode
    if mode in ('test', 'sample'):
        sample_videos = args.num_videos if args.num_videos is not None else 2
        print(f"\n🔍 Avviando in MODALITA TEST (primi {sample_videos} video)")
        success, errors = segmenter.process_sample(num_videos=sample_videos)
    else:
        print("\n▶️ Avviando processamento COMPLETO")
        success, errors = segmenter.process_all(num_videos=args.num_videos)

    print("\n" + "=" * 60)
    print("RISULTATI")
    print("=" * 60)
    print(f"✓ Segmentazioni completate: {success}")
    print(f"✗ Errori: {errors}")
    print(f"Cartella output: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
