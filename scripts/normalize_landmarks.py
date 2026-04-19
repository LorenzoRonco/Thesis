#!/usr/bin/env python3
"""
Wrapper script for normalizing landmarks.

Usage:
    python scripts/normalize_landmarks.py
    
Input:  dataset/landmarks/
Output: dataset/landmarks_normalized/

Normalization method: Shoulder-centric + Global Scale
(Root-relative + Anatomical scaling)
"""

import sys
from pathlib import Path

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from preprocessing.normalize_landmarks import normalize_landmarks_batch

def main():
    """Main entry point."""
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
