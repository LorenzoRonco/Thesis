"""Extract MediaPipe holistic landmarks from segmented videos.

This implementation uses MediaPipe Tasks API (HolisticLandmarker), which is the
API exposed by recent mediapipe versions (e.g. 0.10.33 on Python 3.13).
"""

from collections import defaultdict
import logging
import argparse
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pose landmarks: upper body with arms (0-16), excluding legs (17-32)
# 0: nose
# 1-6: eyes and ears
# 7-8: mouth
# 9-10: left shoulder/elbow/wrist
# 11-16: right shoulder/elbow/wrist + middle area
# This includes shoulders, elbows, and wrists (indices 12-16)
POSE_LANDMARKS = 17
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
FACE_LANDMARKS = 468

# Default official model location from MediaPipe model zoo.
DEFAULT_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)


class LandmarkExtractor:
    """Extract holistic landmarks from video frames using MediaPipe Tasks."""

    def __init__(self, confidence_threshold=0.5, model_path=None, use_gpu=True):
        """Initialize the holistic landmarker.

        Args:
            confidence_threshold: Minimum confidence used by detector heads.
            model_path: Optional path to `holistic_landmarker.task`.
            use_gpu: Whether to use GPU for inference (default: True).
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = self._resolve_model_path(model_path)
        self.use_gpu = use_gpu

        # Note: MediaPipe will automatically use GPU acceleration if available.
        # The use_gpu parameter is stored for logging purposes.
        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=confidence_threshold,
            min_face_landmarks_confidence=confidence_threshold,
            min_pose_detection_confidence=confidence_threshold,
            min_pose_landmarks_confidence=confidence_threshold,
            min_hand_landmarks_confidence=confidence_threshold,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

    def _resolve_model_path(self, model_path):
        """Return a valid model path, downloading the default model if needed."""
        if model_path:
            path = Path(model_path)
        else:
            root_dir = Path(__file__).parent.parent.parent
            path = root_dir / "models" / "holistic_landmarker.task"

        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Model file not found. Downloading: %s", path)
        try:
            urlretrieve(DEFAULT_HOLISTIC_MODEL_URL, str(path))
        except Exception as exc:
            raise RuntimeError(
                "Unable to download holistic model. "
                "Download it manually and pass --model-path."
            ) from exc

        return path
    
    def extract_landmarks_from_video(self, video_path):
        """
        Extract landmarks from all frames in a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            numpy array of shape (num_frames, num_landmarks, 4)
            where 4 = [x, y, z, confidence]
            None if video cannot be processed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Processing video: {video_path.name} ({total_frames} frames)")
            
            # List to store landmarks for all frames.
            all_landmarks = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB and run task in VIDEO mode.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int((frame_count / fps) * 1000)
                results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                # Extract landmarks
                frame_landmarks = self._extract_frame_landmarks(results)
                all_landmarks.append(frame_landmarks)
                
                frame_count += 1
                if frame_count % max(1, total_frames // 10) == 0:
                    logger.info(f"  Processed {frame_count}/{total_frames} frames")
            
            if not all_landmarks:
                logger.warning(f"No landmarks extracted from {video_path.name}")
                return None
            
            # Convert to numpy array
            landmarks_array = np.array(all_landmarks, dtype=np.float32)
            logger.info(f"Extracted shape: {landmarks_array.shape}")
            
            return landmarks_array
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            return None
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()

    @staticmethod
    def _as_landmark_list(maybe_landmarks):
        """Normalize task output to a flat landmark list."""
        if not maybe_landmarks:
            return []
        first = maybe_landmarks[0]
        # Some APIs return a list of lists; for holistic we keep the first target.
        if isinstance(first, (list, tuple)):
            return list(first)
        return list(maybe_landmarks)

    @staticmethod
    def _extract_points(landmarks, expected_count, confidence_attr=None):
        """Convert landmark objects to [x, y, z, confidence] with zero-padding."""
        out = []
        for lm in landmarks[:expected_count]:
            conf = getattr(lm, confidence_attr, None) if confidence_attr else None
            if conf is None:
                conf = 1.0
            out.append([lm.x, lm.y, lm.z, conf])

        if len(out) < expected_count:
            out.extend([[0.0, 0.0, 0.0, 0.0]] * (expected_count - len(out)))
        return out
    
    def _extract_frame_landmarks(self, results):
        """
        Extract all landmarks from MediaPipe results for a single frame.
        
        Returns:
            numpy array of shape (num_landmarks, 4) with [x, y, z, confidence]
            Total landmarks: 11 (pose upper body only) + 21 (left_hand) + 21 (right_hand) + 468 (face) = 521
        """
        landmarks = []
        
        # Pose landmarks - upper body only (11 points: indices 0-10)
        # Excludes legs and lower body (indices 11-32)
        pose = self._as_landmark_list(results.pose_landmarks)
        pose_upper = pose[:POSE_LANDMARKS]  # Take only first 11 (upper body)
        landmarks.extend(self._extract_points(pose_upper, POSE_LANDMARKS, "visibility"))

        # Left hand landmarks (21 points)
        left_hand = self._as_landmark_list(results.left_hand_landmarks)
        landmarks.extend(self._extract_points(left_hand, LEFT_HAND_LANDMARKS, "presence"))

        # Right hand landmarks (21 points)
        right_hand = self._as_landmark_list(results.right_hand_landmarks)
        landmarks.extend(self._extract_points(right_hand, RIGHT_HAND_LANDMARKS, "presence"))

        # Face landmarks (468 points)
        face = self._as_landmark_list(results.face_landmarks)
        landmarks.extend(self._extract_points(face, FACE_LANDMARKS, None))
        
        return np.array(landmarks, dtype=np.float32)
    
    def close(self):
        """Close the holistic model."""
        if self.landmarker:
            self.landmarker.close()


def process_all_videos(input_dir, output_dir, model_path=None, use_gpu=True):
    """
    Process all videos in input directory and save landmarks.
    
    Args:
        input_dir: Directory containing segmented videos
        output_dir: Directory where to save landmark arrays
        model_path: Optional path to holistic_landmarker.task
        use_gpu: Whether to use GPU for inference (default: True)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = sorted(input_dir.glob("*.mp4"))
    
    if not video_files:
        logger.error(f"No MP4 files found in {input_dir}")
        return
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Track statistics
    stats = {
        'total': len(video_files),
        'successful': 0,
        'failed': 0,
        'landmark_shapes': defaultdict(int)
    }
    
    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")

        # In VIDEO mode MediaPipe requires strictly monotonic timestamps.
        # Recreate the landmarker per video so timestamps can safely restart.
        try:
            extractor = LandmarkExtractor(model_path=model_path, use_gpu=use_gpu)
        except Exception as exc:
            logger.error("Failed to initialize landmark extractor for %s: %s", video_path.name, exc)
            stats['failed'] += 1
            continue

        try:
            landmarks = extractor.extract_landmarks_from_video(video_path)
        finally:
            extractor.close()

        if landmarks is not None:
            # Save as numpy array
            output_path = output_dir / f"{video_path.stem}_landmarks.npy"
            np.save(output_path, landmarks)
            logger.info(f"Saved to: {output_path.name}")

            stats['successful'] += 1
            stats['landmark_shapes'][landmarks.shape] += 1
        else:
            stats['failed'] += 1
            logger.warning(f"Failed to extract landmarks from {video_path.name}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total videos: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("\nLandmark shapes (frames, 543 landmarks, 4 coords):")
    for shape, count in sorted(stats['landmark_shapes'].items()):
        logger.info(f"  {shape}: {count} video(s)")
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract holistic landmarks from segmented videos")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to holistic_landmarker.task",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU for inference (default: use GPU)",
    )
    args = parser.parse_args()

    # Determine paths relative to this file
    root_dir = Path(__file__).parent.parent.parent
    segmented_dir = root_dir / "dataset" / "segmented"
    landmarks_dir = root_dir / "dataset" / "landmarks"
    
    logger.info(f"Input directory: {segmented_dir}")
    logger.info(f"Output directory: {landmarks_dir}")
    
    if not segmented_dir.exists():
        logger.error(f"Input directory not found: {segmented_dir}")
        return
    
    # Log device information
    use_gpu = not args.cpu
    device_info = "GPU" if use_gpu else "CPU"
    logger.info(f"Using device: {device_info}")
    
    # Process all videos
    process_all_videos(segmented_dir, landmarks_dir, model_path=args.model_path, use_gpu=use_gpu)


if __name__ == "__main__":
    main()
