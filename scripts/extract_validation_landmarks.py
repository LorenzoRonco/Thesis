#!/usr/bin/env python3
"""
Extract MediaPipe landmarks from validation videos.

Usage:
    python scripts/extract_validation_landmarks.py
    
Input:  dataset/validation/raw_videos/
Output: dataset/landmarks_validation/
"""

import sys
from pathlib import Path
import argparse

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from preprocessing.landmark_extraction import LandmarkExtractor
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_validation_landmarks(num_videos: int | None = None):
    """Extract landmarks from all validation videos."""
    
    validation_videos_dir = root_dir / "dataset" / "validation" / "raw_videos"
    output_dir = root_dir / "dataset" / "landmarks_validation"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Check if input exists
    if not validation_videos_dir.exists():
        logger.error(f"Validation videos directory not found: {validation_videos_dir}")
        return
    
    # Get all video files
    video_files = sorted(validation_videos_dir.glob("*.mp4"))
    if num_videos is not None:
        video_files = video_files[:num_videos]
    
    if not video_files:
        logger.warning(f"No .mp4 files found in {validation_videos_dir}")
        return
    
    logger.info(f"Found {len(video_files)} validation videos")
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        try:
            # Create a fresh extractor per video to reset MediaPipe VIDEO timestamps
            extractor = LandmarkExtractor(confidence_threshold=0.5, use_gpu=True)
            # Extract landmarks
            landmarks = extractor.extract_landmarks_from_video(video_path)
            
            if landmarks is None:
                logger.warning(f"  ✗ Failed to extract landmarks from {video_path.name}")
                failed += 1
                continue
            
            # Generate output filename (keep original name, add _landmarks.npy)
            output_name = video_path.stem + "_landmarks.npy"
            output_path = output_dir / output_name
            
            # Save landmarks
            np.save(str(output_path), landmarks)
            logger.info(f"  ✓ Saved: {output_path.name} (shape: {landmarks.shape})")
            successful += 1
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {video_path.name}: {str(e)}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {len(video_files)}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract landmarks from validation videos")
    parser.add_argument(
        '--num-videos',
        type=int,
        default=None,
        help='Numero massimo di video da preprocessare.',
    )
    args = parser.parse_args()
    extract_validation_landmarks(num_videos=args.num_videos)
