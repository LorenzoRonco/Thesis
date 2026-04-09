#!/usr/bin/env python3
"""
Batch processing script for cropping all videos with their corresponding landmarks.

Processa automaticamente TUTTI i video in dataset/segmented/ con i landmarks
corrispondenti da dataset/landmarks_normalized/ e salva i risultati in
dataset/cropped/

Usage:
    python scripts/batch_crop_videos.py                          # Processa tutti
    python scripts/batch_crop_videos.py --output-dir ./my_crops  # Percorso personalizzato
    python scripts/batch_crop_videos.py --padding 0.2            # Parametri personalizzati
    python scripts/batch_crop_videos.py --skip-existing          # Salta video già processati
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import time
from datetime import timedelta

from crop_video_with_landmarks import crop_video_with_landmarks


def find_landmark_file(video_path: str, landmarks_dir: Path) -> Path:
    """Trova il file landmarks corrispondente al video."""
    video_name = Path(video_path).stem
    landmarks_file = landmarks_dir / f"{video_name}_landmarks.npy"
    return landmarks_file if landmarks_file.exists() else None


def process_video(
    video_path: str,
    landmarks_path: str,
    output_path: str,
    padding: float = 0.15,
    smoothing: str = "moving"
) -> tuple:
    """
    Processa un singolo video.
    
    Returns:
        (success: bool, output_shape: tuple, error_msg: str)
    """
    try:
        # Carica landmarks
        landmarks = np.load(landmarks_path)
        
        # Estrai solo (x, y) se necessario
        if landmarks.shape[2] > 2:
            landmarks = landmarks[:, :, :2]
        
        # Esegui cropping
        output = crop_video_with_landmarks(
            video_path=video_path,
            landmarks=landmarks,
            output_path=output_path,
            padding=padding,
            target_size=(224, 224),
            smoothing=smoothing,
            return_format="array"
        )
        
        return True, output.shape, None
        
    except Exception as e:
        return False, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Batch crop all videos with their landmarks"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/cropped",
        help="Output directory for cropped videos (default: dataset/cropped)"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Padding percentage (default: 0.15)"
    )
    parser.add_argument(
        "--smoothing",
        choices=["global", "moving"],
        default="moving",
        help="Temporal smoothing method (default: moving)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have a cropped version"
    )
    
    args = parser.parse_args()
    
    # Definisci directory
    video_dir = Path("dataset/segmented")
    landmarks_dir = Path("dataset/landmarks_normalized")
    output_dir = Path(args.output_dir)
    
    # Verifica che directory input esistano
    if not video_dir.exists():
        print(f"✗ Error: {video_dir} not found")
        return
    
    if not landmarks_dir.exists():
        print(f"✗ Error: {landmarks_dir} not found")
        return
    
    # Crea output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cerca tutti i video
    video_files = sorted([f for f in video_dir.glob("*-rgb_front.mp4")])
    
    if not video_files:
        print(f"✗ No videos found in {video_dir}")
        return
    
    print("=" * 70)
    print("BATCH VIDEO CROPPING")
    print("=" * 70)
    print(f"Input videos: {video_dir}")
    print(f"Landmarks: {landmarks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total videos found: {len(video_files)}")
    print(f"Padding: {args.padding * 100:.0f}%")
    print(f"Smoothing: {args.smoothing}")
    print("=" * 70 + "\n")
    
    # Statistiche
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    failed_videos = []
    
    # Processa ogni video
    for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
        video_name = video_file.stem
        video_path = str(video_file)
        
        # Cerca landmarks
        landmarks_path = find_landmark_file(video_path, landmarks_dir)
        if not landmarks_path:
            print(f"\n  ⚠️  {video_name}: landmarks not found, skipping")
            skipped += 1
            continue
        
        # Output path
        output_path = output_dir / f"{video_name}_cropped.mp4"
        
        # Skip se già esiste
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        
        # Processa
        success, output_shape, error = process_video(
            video_path,
            str(landmarks_path),
            str(output_path),
            padding=args.padding,
            smoothing=args.smoothing
        )
        
        if success:
            processed += 1
            tqdm.write(f"  ✓ {video_name}: {output_shape}")
        else:
            failed += 1
            failed_videos.append((video_name, error))
            tqdm.write(f"  ✗ {video_name}: {error}")
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total videos: {len(video_files)}")
    print(f"Processed: {processed} ✓")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed} ✗")
    print(f"Time elapsed: {timedelta(seconds=int(elapsed))}")
    
    if processed > 0:
        avg_time = elapsed / processed
        print(f"Average time per video: {avg_time:.1f}s")
    
    if failed_videos:
        print(f"\nFailed videos:")
        for name, error in failed_videos:
            print(f"  - {name}: {error}")
    
    print("=" * 70)
    
    if processed > 0:
        print(f"\n✓ Cropped videos saved to: {output_dir}")
        print(f"  Command to inspect: python scripts/visualize_cropped_video.py {output_dir}/[video]_cropped.mp4")
    else:
        print("\n✗ No videos were processed")


if __name__ == "__main__":
    main()
