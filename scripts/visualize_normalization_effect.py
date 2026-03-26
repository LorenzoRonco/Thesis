"""
Visualize the effect of normalization on skeleton position.
Shows before/after to verify that normalization works correctly.
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.preprocessing.landmark_utils import LandmarkLoader

# MediaPipe Holistic skeleton connections (only pose)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),      # Head
    (0, 4), (4, 5), (5, 6), (6, 8),      # Left arm
    (0, 9), (9, 10), (10, 11), (11, 13), (9, 12), (12, 14), # Shoulders
    (11, 23), (12, 24),                  # Torso
    (23, 25), (25, 27), (24, 26), (26, 28), # Legs
]

LEFT_HAND_START = 33
RIGHT_HAND_START = 33 + 21


def plot_skeleton(ax, landmarks, title, connections=POSE_CONNECTIONS, draw_hands=False):
    """Plot skeleton on matplotlib axis."""
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert Y for image coordinates
    ax.grid(True, alpha=0.3)
    
    pose_landmarks = landmarks[:33, :]
    
    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
            start = pose_landmarks[start_idx, :2]
            end = pose_landmarks[end_idx, :2]
            
            if pose_landmarks[start_idx, 3] > 0 and pose_landmarks[end_idx, 3] > 0:
                ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=1.5, alpha=0.6)
    
    # Draw landmarks
    for i in range(33):
        if pose_landmarks[i, 3] > 0:
            ax.plot(pose_landmarks[i, 0], pose_landmarks[i, 1], 'bo', markersize=4, alpha=0.7)
    
    # Draw hands if requested
    if draw_hands and landmarks.shape[0] >= 33 + 21 + 21:
        left_hand = landmarks[LEFT_HAND_START:LEFT_HAND_START+21, :]
        right_hand = landmarks[RIGHT_HAND_START:RIGHT_HAND_START+21, :]
        
        for i in range(21):
            if left_hand[i, 3] > 0:
                ax.plot(left_hand[i, 0], left_hand[i, 1], 'r.', markersize=3, alpha=0.5)
            if right_hand[i, 3] > 0:
                ax.plot(right_hand[i, 0], right_hand[i, 1], 'g.', markersize=3, alpha=0.5)
    
    # Reference crosshair at (0.5, 0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.plot(0.5, 0.5, 'kx', markersize=8, markeredgewidth=2, label='Shoulder center')
    
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)


def main():
    landmarks_dir = root_dir / "dataset" / "landmarks"
    normalized_dir = root_dir / "dataset" / "landmarks_normalized"
    
    raw_files = sorted(landmarks_dir.glob("*.npy"))
    norm_files = sorted(normalized_dir.glob("*.npy"))
    
    if not raw_files or not norm_files:
        print("❌ No landmark files found!")
        return
    
    # Find matching files
    for raw_file in raw_files:
        if "viz" in raw_file.name:
            continue
        
        norm_file = None
        for nf in norm_files:
            if raw_file.stem in nf.stem:
                norm_file = nf
                break
        
        if norm_file is None:
            continue
        
        print(f"\n{'='*70}")
        print(f"FILE: {raw_file.name}")
        print(f"{'='*70}")
        
        raw_landmarks = LandmarkLoader.load_landmarks(raw_file)
        norm_landmarks = LandmarkLoader.load_landmarks(norm_file)
        
        num_frames = raw_landmarks.shape[0]
        print(f"Total frames: {num_frames}")
        
        # Select frames: first, quarters, last
        frame_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
        frame_indices = [f for f in frame_indices if f < num_frames]
        
        print(f"Visualizing frames: {frame_indices}")
        
        for frame_idx in frame_indices:
            raw_frame = raw_landmarks[frame_idx]
            norm_frame = norm_landmarks[frame_idx]
            
            # Skip if shoulders not detected
            if raw_frame[11, 3] == 0 or raw_frame[12, 3] == 0:
                print(f"  Frame {frame_idx}: Shoulders not detected, skipping")
                continue
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            left_shoulder_raw = raw_frame[11, :2]
            right_shoulder_raw = raw_frame[12, :2]
            shoulder_center_raw = (left_shoulder_raw + right_shoulder_raw) / 2
            
            left_shoulder_norm = norm_frame[11, :2]
            right_shoulder_norm = norm_frame[12, :2]
            shoulder_center_norm = (left_shoulder_norm + right_shoulder_norm) / 2
            
            plot_skeleton(ax1, raw_frame, 
                         f"RAW (Frame {frame_idx})\nShoulder center: ({shoulder_center_raw[0]:.3f}, {shoulder_center_raw[1]:.3f})",
                         draw_hands=True)
            
            plot_skeleton(ax2, norm_frame,
                         f"NORMALIZED (Frame {frame_idx})\nShoulder center: ({shoulder_center_norm[0]:.3f}, {shoulder_center_norm[1]:.3f})",
                         draw_hands=True)
            
            ax1.plot(*left_shoulder_raw, 'C0s', markersize=8, label='Left shoulder')
            ax1.plot(*right_shoulder_raw, 'C1s', markersize=8, label='Right shoulder')
            ax1.legend(fontsize=8)
            
            ax2.plot(*left_shoulder_norm, 'C0s', markersize=8, label='Left shoulder')
            ax2.plot(*right_shoulder_norm, 'C1s', markersize=8, label='Right shoulder')
            ax2.legend(fontsize=8)
            
            plt.tight_layout()
            
            output_dir = root_dir / "visualization_output" / "normalization_check"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            safe_name = raw_file.stem.replace('_viz_landmarks', '')
            fig_path = output_dir / f"{safe_name}_frame{frame_idx:03d}.png"
            
            plt.savefig(fig_path, dpi=100, bbox_inches='tight')
            print(f"  ✓ Frame {frame_idx} saved to {fig_path.relative_to(root_dir)}")
            
            plt.close()
        
        break
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"Output: visualization_output/normalization_check/")
    print(f"\nHow to interpret:")
    print(f"  LEFT (RAW): skeleton positioned wherever the person was in original video")
    print(f"  RIGHT (NORMALIZED): skeleton should be centered at (0.5, 0.5)")
    print(f"  ✓ Crosshair = expected shoulder center position")
    print(f"  ✓ Body proportions identical, just shifted/scaled")


if __name__ == "__main__":
    main()
