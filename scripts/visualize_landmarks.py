"""
Visualize extracted landmarks from .npy files on video.
Draws skeleton overlay using pre-extracted MediaPipe landmarks to verify extraction quality.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# Landmark connection indices for MediaPipe upper body (0-10)
POSE_CONNECTIONS = frozenset([
    (0, 1), (0, 2), (0, 3), (0, 4),  # Head connections (nose to eyes/ears)
    (5, 7), (7, 9),  # Left arm: shoulder -> elbow -> wrist
    (6, 8), (8, 10),  # Right arm: shoulder -> elbow -> wrist
    (5, 6),  # Shoulders connection
])

HAND_CONNECTIONS = frozenset([
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
])

# Color scheme (BGR)
COLORS = {
    'pose': (0, 255, 0),        # Green
    'left_hand': (255, 0, 0),   # Blue
    'right_hand': (0, 0, 255),  # Red
    'face': (255, 255, 0),      # Cyan
}

LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2

# Landmark structure
POSE_LANDMARKS = 11
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
FACE_LANDMARKS = 468
POSE_END = POSE_LANDMARKS
LEFT_HAND_END = POSE_END + LEFT_HAND_LANDMARKS
RIGHT_HAND_END = LEFT_HAND_END + RIGHT_HAND_LANDMARKS


def find_landmarks_file(video_path):
    """Find corresponding .npy landmarks file for a video."""
    video_path = Path(video_path)
    video_name = video_path.stem
    
    # Search in dataset/landmarks/
    root_dir = Path(__file__).parent.parent
    landmarks_dir = root_dir / "dataset" / "landmarks"
    
    # Try exact match
    npy_file = landmarks_dir / f"{video_name}_landmarks.npy"
    if npy_file.exists():
        return npy_file
    
    # Try to find file with similar name
    if landmarks_dir.exists():
        for npy in landmarks_dir.glob(f"{video_name}*.npy"):
            return npy
    
    return None


def draw_connections(frame, points, connections, color):
    """Draw lines connecting landmarks."""
    for start, end in connections:
        if start < len(points) and end < len(points):
            pt1 = points[start]
            pt2 = points[end]
            # Only draw if both points have non-zero confidence
            if pt1[2] > 0 and pt2[2] > 0:
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), color, CONNECTION_THICKNESS)


def draw_points(frame, points, color, radius=LANDMARK_RADIUS):
    """Draw circles at landmark positions."""
    for pt in points:
        if pt[2] > 0:  # Only draw if confidence > 0
            cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)


def visualize_from_npy(video_path, landmarks_path, output_path=None):
    """Visualize landmarks from .npy file on video."""
    
    video_path = Path(video_path)
    landmarks_path = Path(landmarks_path)
    
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS ({total_frames} frames)")
    
    # Load landmarks from .npy
    print(f"Loading landmarks: {landmarks_path}")
    landmarks = np.load(str(landmarks_path), allow_pickle=False)
    print(f"Landmarks shape: {landmarks.shape}")
    
    if landmarks.shape[0] != total_frames:
        print(f"WARNING: Landmark frames ({landmarks.shape[0]}) != video frames ({total_frames})")
    
    # Set up output video writer if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    else:
        out = None
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get landmarks for this frame
            if frame_count < len(landmarks):
                frame_landmarks = landmarks[frame_count]
                
                # Extract landmark groups
                pose_pts = frame_landmarks[:POSE_END]
                left_hand_pts = frame_landmarks[POSE_END:LEFT_HAND_END]
                right_hand_pts = frame_landmarks[LEFT_HAND_END:RIGHT_HAND_END]
                face_pts = frame_landmarks[RIGHT_HAND_END:]
                
                # Denormalize landmarks: convert from [0, 1] to pixel coordinates
                h, w = frame.shape[:2]
                
                # Draw pose skeleton
                pose_px = []
                for lm in pose_pts:
                    x = lm[0] * w
                    y = lm[1] * h
                    conf = lm[3] if len(lm) > 3 else 1.0  # confidence
                    pose_px.append((x, y, conf))
                
                draw_connections(frame, pose_px, POSE_CONNECTIONS, COLORS['pose'])
                draw_points(frame, pose_px, COLORS['pose'])
                
                # Draw left hand
                left_hand_px = []
                for lm in left_hand_pts:
                    x = lm[0] * w
                    y = lm[1] * h
                    conf = lm[3] if len(lm) > 3 else 1.0
                    left_hand_px.append((x, y, conf))
                
                draw_connections(frame, left_hand_px, HAND_CONNECTIONS, COLORS['left_hand'])
                draw_points(frame, left_hand_px, COLORS['left_hand'])
                
                # Draw right hand
                right_hand_px = []
                for lm in right_hand_pts:
                    x = lm[0] * w
                    y = lm[1] * h
                    conf = lm[3] if len(lm) > 3 else 1.0
                    right_hand_px.append((x, y, conf))
                
                draw_connections(frame, right_hand_px, HAND_CONNECTIONS, COLORS['right_hand'])
                draw_points(frame, right_hand_px, COLORS['right_hand'])
                
                # Draw face points (no connections, just points)
                for lm in face_pts:
                    x = lm[0] * w
                    y = lm[1] * h
                    conf = lm[3] if len(lm) > 3 else 1.0
                    if conf > 0:
                        cv2.circle(frame, (int(x), int(y)), 1, COLORS['face'], -1)
            
            # Add frame counter
            cv2.putText(
                frame,
                f"Frame: {frame_count}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            
            # Write frame to output
            if out:
                out.write(frame)
            
            # Display in window
            try:
                cv2.imshow("Landmarks Visualization", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            except cv2.error:
                pass
            
            frame_count += 1
            if frame_count % max(1, total_frames // 10) == 0:
                print(f"  Processed {frame_count}/{total_frames} frames")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if output_path and Path(output_path).exists():
            print(f"✓ Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize landmarks from .npy files on video"
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Video file path (optional, uses first video in dataset/segmented/ if not provided)"
    )
    parser.add_argument(
        "--landmarks",
        help="Landmarks .npy file path (auto-detected if not provided)"
    )
    parser.add_argument(
        "--output",
        help="Output video path (optional, displays in window if not provided)"
    )
    
    args = parser.parse_args()
    
    # Find video to process
    if args.video:
        video_path = Path(args.video)
    else:
        # Use first video in dataset/segmented/
        root_dir = Path(__file__).parent.parent
        video_dir = root_dir / "dataset" / "segmented"
        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            print("ERROR: No videos found in dataset/segmented/")
            return
        video_path = videos[0]
        print(f"Using first video: {video_path.name}")
    
    # Find landmarks file
    if args.landmarks:
        landmarks_path = Path(args.landmarks)
    else:
        landmarks_path = find_landmarks_file(video_path)
        if not landmarks_path:
            print(f"ERROR: Cannot find landmarks file for {video_path.name}")
            return
    
    print(f"Using landmarks: {landmarks_path.name}")
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = video_path.parent / f"{video_path.stem}_landmarks_viz.mp4"
    
    # Visualize
    visualize_from_npy(video_path, landmarks_path, output_path)


if __name__ == "__main__":
    main()
