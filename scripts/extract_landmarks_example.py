#!/usr/bin/env python3
"""
Example: Extract landmarks and prepare data for neural network.
"""

import sys
from pathlib import Path
import logging

# Add src to path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

from preprocessing.landmark_extraction import process_all_videos
from preprocessing.landmark_utils import (
    LandmarkLoader, LandmarkPreprocessor, LandmarkStatistics, prepare_for_network
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example workflow."""
    
    # Paths
    root_dir = Path(__file__).parent.parent.parent
    segmented_dir = root_dir / "dataset" / "segmented"
    landmarks_dir = root_dir / "dataset" / "landmarks"
    
    print("\n" + "="*70)
    print("LANDMARK EXTRACTION EXAMPLE")
    print("="*70)
    
    # Step 1: Extract landmarks from all videos
    print("\n[1] Extracting landmarks from segmented videos...")
    print(f"    Input:  {segmented_dir}")
    print(f"    Output: {landmarks_dir}")
    process_all_videos(segmented_dir, landmarks_dir)
    
    # Step 2: Load and analyze landmarks
    print("\n[2] Loading and analyzing landmarks...")
    landmark_files = list(landmarks_dir.glob("*_landmarks.npy"))
    
    if not landmark_files:
        logger.error("No landmark files found!")
        return
    
    # Load first file as example
    first_file = landmark_files[0]
    logger.info(f"Loading example: {first_file.name}")
    
    landmarks = LandmarkLoader.load_landmarks(first_file)
    logger.info(f"Landmarks shape: {landmarks.shape}")
    logger.info(f"  - Frames: {landmarks.shape[0]}")
    logger.info(f"  - Landmarks per frame: {landmarks.shape[1]}")
    logger.info(f"  - Coordinates per landmark: {landmarks.shape[2]}")
    
    # Step 3: Print statistics
    print("\n[3] Landmark statistics:")
    LandmarkStatistics.print_stats(landmarks, first_file.stem)
    
    # Step 4: Demonstrate preprocessing
    print("\n[4] Preprocessing options:")
    
    # Extract body parts
    print("  a) Extracting body parts...")
    body_parts = LandmarkPreprocessor.extract_body_parts(landmarks)
    for name, data in body_parts.items():
        print(f"     {name}: {data.shape}")
    
    # Normalize
    print("  b) Normalizing landmarks...")
    normalized = LandmarkPreprocessor.normalize_landmarks(landmarks, norm_type='minmax')
    print(f"     Normalized shape: {normalized.shape}")
    print(f"     Value range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Fill missing
    print("  c) Filling missing landmarks...")
    filled = LandmarkPreprocessor.fill_missing_landmarks(normalized, method='forward')
    print(f"     Filled shape: {filled.shape}")
    
    # Flatten for network
    print("  d) Flattening for neural network...")
    flattened_3d = LandmarkPreprocessor.flatten_for_network(
        filled, include_confidence=True, drop_z=False
    )
    flattened_2d = LandmarkPreprocessor.flatten_for_network(
        filled, include_confidence=True, drop_z=True
    )
    print(f"     Flattened (3D + confidence): {flattened_3d.shape}")
    print(f"       - Input dimension: {flattened_3d.shape[1]}")
    print(f"     Flattened (2D + confidence): {flattened_2d.shape}")
    print(f"       - Input dimension: {flattened_2d.shape[1]}")
    
    # Step 5: Complete pipeline example
    print("\n[5] Complete preprocessing pipeline:")
    print("    prepare_for_network() - one function to do everything")
    network_input = prepare_for_network(
        first_file,
        normalize=True,
        fill_missing=True,
        method='forward',
        drop_z=False,
        include_confidence=True
    )
    print(f"    Ready for network input: {network_input.shape}")
    print(f"    Example first frame: {network_input[0, :8]}...")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Extracted {len(landmark_files)} landmark files")
    print(f"✓ Each file contains a video's landmarks")
    print(f"✓ Format: (frames, 543 landmarks, 4 coordinates) as .npy arrays")
    print(f"\nFor neural network input:")
    print(f"  - Use LandmarkLoader.load_landmarks() to load .npy files")
    print(f"  - Use prepare_for_network() for complete preprocessing")
    print(f"  - Output shape: (frames, {network_input.shape[1]})")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
