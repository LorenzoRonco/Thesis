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

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from preprocessing.video_segmentation import VideoSegmenter


def main():
    """Segment validation videos based on the validation CSV."""
    csv_path = root_dir / "dataset" / "how2sign_realigned_val.csv"
    video_dir = root_dir / "dataset" / "validation" / "raw_videos"
    output_dir = root_dir / "dataset" / "segmented_validation"

    if not csv_path.exists():
        print(f"ERROR: CSV non trovato: {csv_path}")
        sys.exit(1)

    if not video_dir.exists():
        print(f"ERROR: Cartella video non trovata: {video_dir}")
        sys.exit(1)

    segmenter = VideoSegmenter(str(csv_path), str(video_dir), str(output_dir))
    segmenter.get_statistics()

    mode = 'sample' if len(sys.argv) < 2 else sys.argv[1]
    if mode in ('test', 'sample'):
        print("\n🔍 Avviando in MODALITA TEST (primi 2 video)")
        success, errors = segmenter.process_sample(num_videos=2)
    else:
        print("\n▶️ Avviando processamento COMPLETO")
        success, errors = segmenter.process_all()

    print("\n" + "=" * 60)
    print("RISULTATI")
    print("=" * 60)
    print(f"✓ Segmentazioni completate: {success}")
    print(f"✗ Errori: {errors}")
    print(f"Cartella output: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
