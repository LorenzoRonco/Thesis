"""
Utility functions for working with extracted landmarks.
Provides loading, preprocessing, and normalization utilities for neural network input.
"""

import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Landmark structure
# Pose: only upper body (0-10), excluding legs (11-32)
POSE_LANDMARKS = 17  # Now includes shoulders, elbows, wrists (indices 11-16)
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
FACE_LANDMARKS = 468
TOTAL_LANDMARKS = POSE_LANDMARKS + LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS + FACE_LANDMARKS

# Coordinate dimensions
COORDS_PER_LANDMARK = 4  # [x, y, z, confidence]


class LandmarkLoader:
    """Load and manage extracted landmark data."""
    
    @staticmethod
    def load_landmarks(landmarks_path):
        """
        Load landmark array from .npy file.
        
        Args:
            landmarks_path: Path to .npy file
            
        Returns:
            numpy array of shape (frames, 543, 4)
        """
        landmarks = np.load(landmarks_path, allow_pickle=False)
        return landmarks
    
    @staticmethod
    def load_landmarks_batch(landmarks_dir, video_ids=None):
        """
        Load multiple landmark files from directory.
        
        Args:
            landmarks_dir: Directory containing .npy files
            video_ids: Optional list of video IDs to load. If None, loads all.
            
        Returns:
            Dictionary mapping video_id -> landmark array
        """
        landmarks_dir = Path(landmarks_dir)
        
        if video_ids is None:
            npy_files = sorted(landmarks_dir.glob("*_landmarks.npy"))
        else:
            npy_files = [landmarks_dir / f"{vid}_landmarks.npy" for vid in video_ids]
            npy_files = [f for f in npy_files if f.exists()]
        
        landmarks_dict = {}
        for npy_file in npy_files:
            video_id = npy_file.stem.replace("_landmarks", "")
            try:
                landmarks_dict[video_id] = np.load(npy_file, allow_pickle=False)
            except Exception as e:
                logger.warning(f"Failed to load {npy_file}: {e}")
        
        return landmarks_dict


class LandmarkPreprocessor:
    """Preprocess landmarks for neural network input."""
    
    @staticmethod
    def normalize_landmarks(landmarks, norm_type='minmax'):
        """
        Normalize landmarks to [-1, 1] or [0, 1] range.
        
        Args:
            landmarks: array of shape (frames, landmarks, 4) or (landmarks, 4)
            norm_type: 'minmax' for [0, 1] or 'standard' for standardization
            
        Returns:
            Normalized landmarks array
        """
        if landmarks.ndim == 2:
            # Single frame: (landmarks, 4)
            frames_dim = False
            landmarks = np.expand_dims(landmarks, 0)
        else:
            frames_dim = True
        
        # Flatten to (total_coords,) for normalization
        original_shape = landmarks.shape
        flat = landmarks.reshape(-1)
        
        # Remove zero padding (missing detections) for normalization
        non_zero_mask = flat != 0
        non_zero_values = flat[non_zero_mask]
        
        normalized = landmarks.copy()
        
        if norm_type == 'minmax':
            if len(non_zero_values) > 0:
                min_val = non_zero_values.min()
                max_val = non_zero_values.max()
                # Avoid division by zero
                if max_val > min_val:
                    normalized = (landmarks - min_val) / (max_val - min_val)
        
        elif norm_type == 'standard':
            if len(non_zero_values) > 0:
                mean = non_zero_values.mean()
                std = non_zero_values.std()
                if std > 0:
                    normalized = (landmarks - mean) / std
        
        if not frames_dim:
            normalized = normalized.squeeze(0)
        
        return normalized.astype(np.float32)
    
    @staticmethod
    def fill_missing_landmarks(landmarks, method='forward'):
        """
        Fill missing landmarks (where confidence is 0).
        
        Args:
            landmarks: array of shape (frames, landmarks, 4)
            method: 'forward' (forward fill), 'backward', 'interpolate', or 'zero'
            
        Returns:
            Filled landmarks array
        """
        filled = landmarks.copy()
        
        if method == 'forward':
            for frame_idx in range(1, filled.shape[0]):
                # Find missing landmarks (confidence = 0)
                missing_mask = filled[frame_idx, :, 3] == 0
                # Forward fill from previous frame
                filled[frame_idx, missing_mask] = filled[frame_idx - 1, missing_mask]
        
        elif method == 'interpolate':
            for lm_idx in range(filled.shape[1]):
                for coord_idx in range(3):  # Only x, y, z
                    col = filled[:, lm_idx, coord_idx]
                    # Find missing values
                    missing_mask = filled[:, lm_idx, 3] == 0
                    if missing_mask.any():
                        # Linear interpolation
                        valid_indices = np.where(~missing_mask)[0]
                        if len(valid_indices) > 1:
                            valid_values = col[valid_indices]
                            filled[missing_mask, lm_idx, coord_idx] = np.interp(
                                np.where(missing_mask)[0],
                                valid_indices,
                                valid_values
                            )
        
        elif method == 'zero':
            # Keep as is (already zero)
            pass
        
        return filled.astype(np.float32)
    
    @staticmethod
    def extract_body_parts(landmarks):
        """
        Extract individual body parts from full landmark array.
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            
        Returns:
            Dictionary with 'pose', 'left_hand', 'right_hand', 'face' arrays
        """
        pose_end = POSE_LANDMARKS
        left_hand_end = pose_end + LEFT_HAND_LANDMARKS
        right_hand_end = left_hand_end + RIGHT_HAND_LANDMARKS
        
        return {
            'pose': landmarks[:, :pose_end, :],
            'left_hand': landmarks[:, pose_end:left_hand_end, :],
            'right_hand': landmarks[:, left_hand_end:right_hand_end, :],
            'face': landmarks[:, right_hand_end:, :]
        }
    
    @staticmethod
    def flatten_for_network(landmarks, include_confidence=True, drop_z=False):
        """
        Flatten landmarks into vector format for neural network.
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            include_confidence: Whether to include confidence values
            drop_z: Whether to drop z coordinate (for 2D networks)
            
        Returns:
            Flattened array of shape (frames, num_features)
        """
        if drop_z:
            # Use only x, y
            features = landmarks[:, :, :2].reshape(landmarks.shape[0], -1)
        elif include_confidence:
            # Use all: x, y, z, confidence
            features = landmarks.reshape(landmarks.shape[0], -1)
        else:
            # Use x, y, z
            features = landmarks[:, :, :3].reshape(landmarks.shape[0], -1)
        
        return features.astype(np.float32)


class LandmarkStatistics:
    """Compute statistics over landmarks."""
    
    @staticmethod
    def compute_stats(landmarks):
        """
        Compute statistics for landmarks.
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            
        Returns:
            Dictionary with statistics
        """
        # Filter out zero padding
        non_zero = landmarks[landmarks != 0]
        
        stats = {
            'num_frames': landmarks.shape[0],
            'total_landmarks': landmarks.shape[1],
            'mean': landmarks.mean(),
            'std': landmarks.std(),
            'min': non_zero.min() if len(non_zero) > 0 else 0,
            'max': non_zero.max() if len(non_zero) > 0 else 0,
            'missing_landmark_frames': (landmarks[:, :, 3] == 0).any(axis=1).sum()
        }
        
        return stats
    
    @staticmethod
    def print_stats(landmarks, video_name=""):
        """Print formatted statistics."""
        stats = LandmarkStatistics.compute_stats(landmarks)
        
        print(f"\nLandmark Statistics {f'for {video_name}' if video_name else ''}")
        print("=" * 50)
        print(f"Frames: {stats['num_frames']}")
        print(f"Total Landmarks per Frame: {stats['total_landmarks']}")
        print(f"Mean value: {stats['mean']:.4f}")
        print(f"Std deviation: {stats['std']:.4f}")
        print(f"Min value: {stats['min']:.4f}")
        print(f"Max value: {stats['max']:.4f}")
        print(f"Frames with missing landmarks: {stats['missing_landmark_frames']}")
        print("=" * 50)


def prepare_for_network(landmarks_path, normalize=True, fill_missing=True, 
                       method='forward', drop_z=False, include_confidence=True):
    """
    Complete pipeline to prepare landmarks for neural network.
    
    Args:
        landmarks_path: Path to .npy file
        normalize: Whether to normalize values
        fill_missing: Whether to fill missing landmarks
        method: 'forward', 'interpolate', or 'zero'
        drop_z: Drop z coordinate for 2D networks
        include_confidence: Include confidence values
        
    Returns:
        Preprocessed flattened array ready for network input
    """
    # Load
    landmarks = LandmarkLoader.load_landmarks(landmarks_path)
    
    logger.info(f"Loaded landmarks shape: {landmarks.shape}")
    
    # Preprocess
    if fill_missing:
        landmarks = LandmarkPreprocessor.fill_missing_landmarks(landmarks, method=method)
        logger.info(f"Filled missing landmarks using '{method}' method")
    
    if normalize:
        landmarks = LandmarkPreprocessor.normalize_landmarks(landmarks)
        logger.info("Normalized landmarks")
    
    # Flatten
    flat = LandmarkPreprocessor.flatten_for_network(
        landmarks,
        include_confidence=include_confidence,
        drop_z=drop_z
    )
    
    logger.info(f"Flattened to network input shape: {flat.shape}")
    
    return flat


class LandmarkNormalizer:
    """Normalize landmarks using root-relative and anatomical scaling."""
    
    @staticmethod
    def normalize_by_root_relative_anatomical(landmarks):
        """
        Normalize landmarks using shoulder-centric root-relative positioning and anatomical scaling.
        
        Optimized for sign language recognition (focusing on face, hands, arms):
        1. Root point: Shoulder center (midpoint of landmarks 11, 12) - PER FRAME
        2. Scale: GLOBAL shoulder width (mean distance across all frames)
        
        Why global scale?
        - Hands/arms move and expand the body outline, creating apparent size changes
        - Using per-frame shoulder_width causes jittering in torso size
        - Global scale is invariant to limb motion, preventing artificial size pulsing
        - Neural network sees stable body proportions regardless of arm position
        
        Optimizations for sign language:
        - Shoulders are always visible in frontal webcam video
        - Root-relative centering per-frame keeps pose centered
        - Global anatomical scale prevents animation artifacts
        - Face and hands move relative to stable torso
        
        Args:
            landmarks: array of shape (frames, 543, 4) with [x, y, z, confidence]
                      MediaPipe format: 33 pose + 21 left_hand + 21 right_hand + 468 face
        
        Returns:
            Normalized landmarks: shoulder-centered (per-frame), anatomically scaled (global)
        """
        if landmarks.ndim != 3:
            raise ValueError(f"Expected 3D landmarks, got {landmarks.ndim}D")
        
        normalized = landmarks.copy().astype(np.float32)
        
        # STEP 0: Compute global shoulder width (mean across all frames)
        shoulder_widths = []
        for frame_idx in range(normalized.shape[0]):
            frame = normalized[frame_idx]
            
            # Skip frames with missing shoulders
            if frame[11, 3] == 0 or frame[12, 3] == 0:
                continue
            
            left_shoulder = frame[11, :3]
            right_shoulder = frame[12, :3]
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            
            if shoulder_width > 1e-6:  # Only valid shoulder widths
                shoulder_widths.append(shoulder_width)
        
        if not shoulder_widths:
            logger.error("No valid shoulder width found in video!")
            return landmarks
        
        # Use median instead of mean (more robust to outliers)
        global_shoulder_width = np.median(shoulder_widths)
        logger.info(f"Global shoulder width: {global_shoulder_width:.6f} (median of {len(shoulder_widths)} frames)")
        
        # STEP 1-5: Apply normalization with GLOBAL scale
        for frame_idx in range(normalized.shape[0]):
            frame = normalized[frame_idx]
            
            # MediaPipe pose indices: 11 = left shoulder, 12 = right shoulder
            left_shoulder = frame[11, :3]
            right_shoulder = frame[12, :3]
            
            # Skip frames with missing shoulders
            if frame[11, 3] == 0 or frame[12, 3] == 0:
                logger.warning(f"Frame {frame_idx}: Shoulders not detected, skipping normalization")
                continue
            
            # Step 1: Compute shoulder center (root point) - PER FRAME
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # Step 2: Root-relative centering: subtract shoulder center (x, y only)
            normalized[frame_idx, :, 0] = frame[:, 0] - shoulder_center[0]
            normalized[frame_idx, :, 1] = frame[:, 1] - shoulder_center[1]
            # Keep z unchanged
            
            # Step 3: Anatomical scaling: divide by GLOBAL shoulder width (same for all frames)
            normalized[frame_idx, :, 0] /= global_shoulder_width
            normalized[frame_idx, :, 1] /= global_shoulder_width
            normalized[frame_idx, :, 2] /= global_shoulder_width
            
            # Step 4: Shift to center at (0.5, 0.5) for neural network expectations
            normalized[frame_idx, :, 0] += 0.5
            normalized[frame_idx, :, 1] += 0.5
        
        return normalized
