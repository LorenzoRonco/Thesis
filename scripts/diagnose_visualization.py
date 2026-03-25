#!/usr/bin/env python3
"""
Diagnostic script: Check landmarks visualization issue.
Verifies that landmarks are drawn correctly on video.
"""

import numpy as np
import cv2
from pathlib import Path

def diagnose_landmarks():
    """Check landmark extraction and visualization."""
    
    print("\n" + "="*70)
    print("LANDMARK VISUALIZATION DIAGNOSTIC")
    print("="*70)
    
    # Find files
    landmarks_dir = Path("dataset/landmarks")
    segmented_dir = Path("dataset/segmented")
    
    lm_files = sorted(landmarks_dir.glob("*_landmarks.npy"))
    vid_files = sorted(segmented_dir.glob("*.mp4"))
    
    if not lm_files or not vid_files:
        print("❌ No landmark or video files found")
        return
    
    # Pick first pair
    lm_file = lm_files[0]
    vid_file = vid_files[0]
    
    print(f"\n📁 Files to analyze:")
    print(f"   Landmarks: {lm_file.name}")
    print(f"   Video: {vid_file.name}")
    
    # Load data
    landmarks = np.load(lm_file)
    cap = cv2.VideoCapture(str(vid_file))
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📏 Dimensions: {w}x{h}")
    print(f"📊 Landmarks shape: {landmarks.shape}")
    
    # Landmark indices
    POSE_LANDMARKS = 17  # Now includes shoulders, elbows, wrists
    LEFT_HAND_END = POSE_LANDMARKS + 21
    RIGHT_HAND_END = LEFT_HAND_END + 21
    
    # Analyze first few frames
    print(f"\n🔍 Analyzing first 5 frames:\n")
    
    for frame_idx in range(min(5, len(landmarks))):
        lm_data = landmarks[frame_idx]
        
        pose = lm_data[:POSE_LANDMARKS]
        left_hand = lm_data[POSE_LANDMARKS:LEFT_HAND_END]
        right_hand = lm_data[LEFT_HAND_END:RIGHT_HAND_END]
        face = lm_data[RIGHT_HAND_END:]
        
        # Count non-zero confidence
        pose_count = np.sum(pose[:, 3] > 0)
        lh_count = np.sum(left_hand[:, 3] > 0)
        rh_count = np.sum(right_hand[:, 3] > 0)
        face_count = np.sum(face[:, 3] > 0)
        
        print(f"Frame {frame_idx}:")
        print(f"  Pose:       {pose_count:2d}/11 | ", end="")
        
        # Show pixel coordinates for first pose point
        if pose_count > 0:
            px = int(pose[0, 0] * w)
            py = int(pose[0, 1] * h)
            print(f"pt0=(x={px}, y={py})", end="")
        print()
        
        print(f"  Left Hand:  {lh_count:2d}/21 | ", end="")
        if lh_count > 0:
            px = int(left_hand[0, 0] * w)
            py = int(left_hand[0, 1] * h)
            print(f"pt0=(x={px}, y={py})", end="")
        print()
        
        print(f"  Right Hand: {rh_count:2d}/21 | ", end="")
        if rh_count > 0:
            px = int(right_hand[0, 0] * w)
            py = int(right_hand[0, 1] * h)
            print(f"pt0=(x={px}, y={py})", end="")
        print()
        
        print(f"  Face:       {face_count:3d}/468")
        print()
    
    # Check for common issues
    print("🚨 Common issues to check:\n")
    
    issues = []
    
    # Issue 1: All hands are zero
    lh_all_zero = np.all(landmarks[:, POSE_LANDMARKS:LEFT_HAND_END, 3] == 0)
    rh_all_zero = np.all(landmarks[:, LEFT_HAND_END:RIGHT_HAND_END, 3] == 0)
    
    if lh_all_zero:
        issues.append("❌ LEFT HAND: Never detected in any frame")
    if rh_all_zero:
        issues.append("❌ RIGHT HAND: Never detected in any frame")
    
    # Issue 2: Pose looks wrong
    pose_data = landmarks[:, :POSE_LANDMARKS, :2]
    if np.any(pose_data > 1.0) or np.any(pose_data < 0):
        issues.append("⚠️  POSE: Some coordinates outside [0, 1] range")
    
    # Issue 3: Hand coordinates out of range
    lh_data = landmarks[:, POSE_LANDMARKS:LEFT_HAND_END, :2]
    if np.any(lh_data > 1.1):  # Allow slight overshoot
        issues.append("⚠️  LEFT HAND: Some coordinates > 1.1 (outside frame)")
    
    if not issues:
        issues.append("✅ No obvious data issues detected")
    
    for issue in issues:
        print(f"   {issue}")
    
    # Suggest solutions
    print("\n💡 Possible causes if hands are missing from visualization:\n")
    print("   1. MediaPipe can't detect hands in the specific video angle/quality")
    print("   2. Hands are occluded or outside frame during recording")
    print("   3. Visualization script has a bug (check draw_points logic)")
    print("   4. Hand confidence threshold is too high in extraction")
    
    cap.release()
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    diagnose_landmarks()
