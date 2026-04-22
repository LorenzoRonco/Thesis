#!/usr/bin/env python3
"""
Script per identificare video corrotti nel dataset How2Sign.

Uso:
    python scripts/identify_corrupted_videos.py --dataset-dir dataset/cropped
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    print("❌ OpenCV non installato. Installa con: pip install opencv-python")
    sys.exit(1)


def check_video_integrity(video_path: Path, timeout_ms: int = 5000) -> Tuple[bool, str]:
    """
    Verifica integrità di un video file.
    
    Args:
        video_path: Path al file video
        timeout_ms: Timeout in millisecondi (non usato in OpenCV ma documentato)
    
    Returns:
        (is_valid, message)
    """
    if not video_path.exists():
        return False, f"File non esiste"
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            cap.release()
            return False, "Impossibile aprire il video (VideoCapture.isOpened() = False)"
        
        # Check frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return False, f"Frame count = {frame_count} (invalido)"
        
        # Check FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return False, f"FPS = {fps} (invalido)"
        
        # Try to read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, "Impossibile leggere il primo frame"
        
        # Try to read middle frame
        mid_frame_idx = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, f"Impossibile leggere frame a metà ({mid_frame_idx}/{frame_count})"
        
        # Try to read last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, f"Impossibile leggere ultimo frame ({frame_count-1})"
        
        cap.release()
        return True, f"✓ {frame_count} frames @ {fps:.1f}fps"
        
    except Exception as e:
        return False, f"Eccezione: {str(e)}"


def scan_video_directory(video_dir: Path, extension: str = "*.mp4") -> Tuple[List[Path], List[Path]]:
    """
    Scansiona directory di video e identifica quelli corrotti.
    
    Returns:
        (valid_videos, corrupted_videos)
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"❌ Directory non esiste: {video_dir}")
        return [], []
    
    all_videos = list(video_dir.glob(extension))
    print(f"\n📹 Trovati {len(all_videos)} video con estensione '{extension}'")
    print(f"   Directory: {video_dir}")
    print("\n🔍 Scansionando...\n")
    
    valid_videos = []
    corrupted_videos = []
    
    for i, video_path in enumerate(all_videos, 1):
        is_valid, message = check_video_integrity(video_path)
        
        status = "✓" if is_valid else "✗"
        pct = (i / len(all_videos)) * 100
        print(f"[{i:4d}/{len(all_videos)}] {pct:5.1f}% | {status} {video_path.name:80s} | {message}")
        
        if is_valid:
            valid_videos.append(video_path)
        else:
            corrupted_videos.append(video_path)
    
    return valid_videos, corrupted_videos


def main():
    parser = argparse.ArgumentParser(
        description="Identifica video corrotti nel dataset How2Sign"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/cropped",
        help="Directory con video MP4 (default: dataset/cropped)"
    )
    parser.add_argument(
        "--export-corrupted",
        type=str,
        default=None,
        help="Esporta lista video corrotti in file (default: None)"
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="*.mp4",
        help="Pattern estensione file (default: *.mp4)"
    )
    
    args = parser.parse_args()
    
    # Scan
    valid_videos, corrupted_videos = scan_video_directory(
        Path(args.dataset_dir),
        args.extension
    )
    
    # Summary
    print("\n" + "="*100)
    print(f"📊 RISULTATO SCANSIONE")
    print("="*100)
    print(f"Video validi:   {len(valid_videos)}")
    print(f"Video corrotti: {len(corrupted_videos)}")
    print(f"Tasso corruzione: {(len(corrupted_videos) / (len(valid_videos) + len(corrupted_videos)) * 100):.1f}%")
    
    if corrupted_videos:
        print(f"\n❌ VIDEO CORROTTI TROVATI:")
        print("-" * 100)
        for video_path in sorted(corrupted_videos):
            is_valid, message = check_video_integrity(video_path)
            print(f"  {video_path.name:80s} | {message}")
        
        # Export
        if args.export_corrupted:
            export_path = Path(args.export_corrupted)
            with open(export_path, 'w') as f:
                for video_path in corrupted_videos:
                    f.write(f"{video_path.name}\n")
            print(f"\n✓ Lista esportata in: {export_path}")
    else:
        print("\n✓ Nessun video corrotto trovato!")
    
    print()


if __name__ == "__main__":
    main()
