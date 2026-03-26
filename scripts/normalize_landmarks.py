#!/usr/bin/env python3
"""
Normalize landmarks using per-frame bounding box normalization.
Preprocessing step: landmarks grezzi → landmarks normalizzati.

Usage:
    python scripts/normalize_landmarks.py
    
Output:
    dataset/landmarks_normalized/ (30k file normalized)
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from preprocessing.landmark_utils import LandmarkLoader, LandmarkNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_landmarks_batch(landmarks_dir, output_dir):
    """
    Normalize all landmark files in a directory.
    
    Args:
        landmarks_dir: Input directory with raw .npy files
        output_dir: Output directory for normalized files
    """
    landmarks_dir = Path(landmarks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("NORMALIZING LANDMARKS")
    print("="*70)
    print(f"\nInput:  {landmarks_dir}")
    print(f"Output: {output_dir}")
    
    # Get all landmark files
    npy_files = sorted(landmarks_dir.glob("*_landmarks.npy"))
    print(f"\nFound {len(npy_files)} files to normalize")
    
    if not npy_files:
        print(f"ERROR: No landmark files found in {landmarks_dir}")
        return
    
    # Initialize normalizer
    normalizer = LandmarkNormalizer()
    
    # Process each file
    errors = []
    successfully_normalized = 0
    
    print(f"\nProcessing...")
    for idx, npy_file in enumerate(npy_files, 1):
        video_id = npy_file.stem.replace("_landmarks", "")
        
        try:
            # Load original
            landmarks = LandmarkLoader.load_landmarks(npy_file)
            
            # Normalize using root-relative + anatomical scaling
            normalized = normalizer.normalize_by_root_relative_anatomical(landmarks)
            
            # Save normalized
            output_file = output_dir / npy_file.name
            np.save(output_file, normalized)
            
            successfully_normalized += 1
            
            if idx % max(1, len(npy_files) // 10) == 0:
                print(f"  [{idx}/{len(npy_files)}] {video_id}... ✓")
        
        except Exception as e:
            errors.append((video_id, str(e)))
            print(f"  ERROR in {video_id}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("NORMALIZATION COMPLETE")
    print("="*70)
    print(f"\nSuccessfully normalized: {successfully_normalized}/{len(npy_files)}")
    
    if errors:
        print(f"Errors ({len(errors)}):")
        for vid_id, err in errors[:5]:
            print(f"  - {vid_id}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    print(f"\n✓ Output saved to: {output_dir}")
    
    # Disk space info
    total_files = list(output_dir.glob("*.npy"))
    total_size_mb = sum(f.stat().st_size for f in total_files) / (1024**2)
    print(f"Total disk space used: {total_size_mb:.1f} MB")


def main():
    """Main entry point."""
    root_dir = Path(__file__).parent.parent
    landmarks_dir = root_dir / "dataset" / "landmarks"
    output_dir = root_dir / "dataset" / "landmarks_normalized"
    
    # Check if input exists
    if not landmarks_dir.exists():
        print(f"ERROR: Input directory not found: {landmarks_dir}")
        return
    
    # Run normalization
    normalize_landmarks_batch(
        landmarks_dir=landmarks_dir,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
