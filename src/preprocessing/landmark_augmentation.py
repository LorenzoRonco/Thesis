"""
Data augmentation for landmark sequences.
Provides spatial and temporal augmentations for training invariance.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Landmark indices for different body parts
POSE_LANDMARKS = 17
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
FACE_LANDMARKS = 468

POSE_END = POSE_LANDMARKS
LEFT_HAND_END = POSE_END + LEFT_HAND_LANDMARKS
RIGHT_HAND_END = LEFT_HAND_END + RIGHT_HAND_LANDMARKS
FACE_END = RIGHT_HAND_END + FACE_LANDMARKS

# Hand indices for mirroring
LEFT_HAND_INDICES = list(range(POSE_END, LEFT_HAND_END))
RIGHT_HAND_INDICES = list(range(LEFT_HAND_END, RIGHT_HAND_END))


class LandmarkAugmenter:
    """Augmentation operations for landmark sequences."""
    
    @staticmethod
    def add_gaussian_noise(landmarks: np.ndarray, 
                          noise_std: float = 0.01,
                          apply_to_confidence: bool = False) -> np.ndarray:
        """
        Add Gaussian noise to landmark coordinates.
        
        Args:
            landmarks: array of shape (frames, 543, 4) or (frames, num_features)
            noise_std: Standard deviation of Gaussian noise
            apply_to_confidence: Whether to add noise to confidence values
            
        Returns:
            Augmented landmarks with noise added
        """
        augmented = landmarks.copy()
        
        if landmarks.ndim == 3:
            # Shape: (frames, landmarks, 4)
            num_coords = 4 if apply_to_confidence else 3
            noise = np.random.normal(0, noise_std, 
                                     (landmarks.shape[0], landmarks.shape[1], num_coords))
            augmented[:, :, :num_coords] += noise
        else:
            # Flattened shape: (frames, num_features)
            noise = np.random.normal(0, noise_std, landmarks.shape)
            augmented += noise
        
        return augmented.astype(np.float32)
    
    @staticmethod
    def random_scale(landmarks: np.ndarray,
                     scale_range: Tuple[float, float] = (0.8, 1.2),
                     center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Randomly scale landmarks around a center point.
        Simulates person at different distances from camera.
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            scale_range: (min_scale, max_scale)
            center: (x, y) center point. If None, uses bounding box center.
            
        Returns:
            Scaled landmarks
        """
        if landmarks.ndim != 3:
            raise ValueError("random_scale requires 3D landmarks array")
        
        augmented = landmarks.copy()
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        # Get center if not provided
        if center is None:
            # Find center of hands and face (skip pose for stability)
            valid_mask_x = landmarks[:, POSE_END:, 0] != 0
            valid_mask_y = landmarks[:, POSE_END:, 1] != 0
            if valid_mask_x.any() and valid_mask_y.any():
                center_x = landmarks[:, POSE_END:, 0][valid_mask_x].mean()
                center_y = landmarks[:, POSE_END:, 1][valid_mask_y].mean()
                center = (center_x, center_y)
            else:
                center = (0.5, 0.5)
        
        # Scale x, y coordinates
        augmented[:, :, 0] = center[0] + (landmarks[:, :, 0] - center[0]) * scale
        augmented[:, :, 1] = center[1] + (landmarks[:, :, 1] - center[1]) * scale
        
        return augmented.astype(np.float32)
    
    @staticmethod
    def random_rotation(landmarks: np.ndarray,
                       angle_range: Tuple[float, float] = (-15, 15),
                       center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Randomly rotate landmarks (in-plane rotation).
        Simulates person at different angles.
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            angle_range: (min_angle_deg, max_angle_deg)
            center: (x, y) rotation center. If None, uses bounding box center.
            
        Returns:
            Rotated landmarks
        """
        if landmarks.ndim != 3:
            raise ValueError("random_rotation requires 3D landmarks array")
        
        augmented = landmarks.copy()
        angle_deg = np.random.uniform(angle_range[0], angle_range[1])
        angle_rad = np.radians(angle_deg)
        
        # Get center if not provided
        if center is None:
            # Find center of hands and face (skip pose for stability)
            valid_mask_x = landmarks[:, POSE_END:, 0] != 0
            valid_mask_y = landmarks[:, POSE_END:, 1] != 0
            if valid_mask_x.any() and valid_mask_y.any():
                center_x = landmarks[:, POSE_END:, 0][valid_mask_x].mean()
                center_y = landmarks[:, POSE_END:, 1][valid_mask_y].mean()
                center = (center_x, center_y)
            else:
                center = (0.5, 0.5)
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate x, y coordinates
        x = landmarks[:, :, 0] - center[0]
        y = landmarks[:, :, 1] - center[1]
        
        augmented[:, :, 0] = center[0] + x * cos_a - y * sin_a
        augmented[:, :, 1] = center[1] + x * sin_a + y * cos_a
        
        return augmented.astype(np.float32)
    
    @staticmethod
    def random_translation(landmarks: np.ndarray,
                          translation_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
        """
        Randomly translate landmarks.
        Simulates person moving left/right or up/down.
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            translation_range: (min_trans, max_trans) in normalized coordinates
            
        Returns:
            Translated landmarks
        """
        if landmarks.ndim != 3:
            raise ValueError("random_translation requires 3D landmarks array")
        
        augmented = landmarks.copy()
        
        # Random translation for x and y
        tx = np.random.uniform(translation_range[0], translation_range[1])
        ty = np.random.uniform(translation_range[0], translation_range[1])
        
        augmented[:, :, 0] += tx
        augmented[:, :, 1] += ty
        
        return augmented.astype(np.float32)
    
    @staticmethod
    def mirror_landmarks(landmarks: np.ndarray,
                        probability: float = 0.5) -> np.ndarray:
        """
        Horizontally flip landmarks (mirror left/right hands).
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            probability: Probability of applying mirroring
            
        Returns:
            Possibly mirrored landmarks
        """
        if landmarks.ndim != 3:
            raise ValueError("mirror_landmarks requires 3D landmarks array")
        
        if np.random.random() > probability:
            return landmarks.copy()
        
        augmented = landmarks.copy()
        
        # Mirror x coordinates (flip left-right)
        augmented[:, :, 0] = 1.0 - landmarks[:, :, 0]
        
        # Swap left and right hands
        left_hand = augmented[:, LEFT_HAND_INDICES, :].copy()
        right_hand = augmented[:, RIGHT_HAND_INDICES, :].copy()
        
        augmented[:, LEFT_HAND_INDICES, :] = right_hand
        augmented[:, RIGHT_HAND_INDICES, :] = left_hand
        
        return augmented.astype(np.float32)
    
    @staticmethod
    def shear_landmarks(landmarks: np.ndarray,
                       shear_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
        """
        Apply random shear transformation.
        Simulates 3D perspective changes.
        
        Args:
            landmarks: array of shape (frames, 543, 4)
            shear_range: (min_shear, max_shear)
            
        Returns:
            Sheared landmarks
        """
        if landmarks.ndim != 3:
            raise ValueError("shear_landmarks requires 3D landmarks array")
        
        augmented = landmarks.copy()
        
        # Random shear in x and y directions
        shear_x = np.random.uniform(shear_range[0], shear_range[1])
        shear_y = np.random.uniform(shear_range[0], shear_range[1])
        
        # Shear transformation (simplified 2D shear)
        y = augmented[:, :, 1]
        x = augmented[:, :, 0]
        
        augmented[:, :, 0] = x + shear_x * y
        augmented[:, :, 1] = y + shear_y * x
        
        return augmented.astype(np.float32)
    
    @staticmethod
    def temporal_crop(landmarks: np.ndarray,
                     min_length: int = 10,
                     max_length: Optional[int] = None) -> np.ndarray:
        """
        Randomly crop temporal sequence.
        
        Args:
            landmarks: array of shape (frames, ...)
            min_length: Minimum frames to keep
            max_length: Maximum frames to keep. If None, uses full length.
            
        Returns:
            Cropped landmarks
        """
        num_frames = landmarks.shape[0]
        
        if max_length is None:
            max_length = num_frames
        
        max_length = min(max_length, num_frames)
        
        if min_length > max_length:
            logger.warning(f"min_length ({min_length}) > max_length ({max_length})")
            return landmarks.copy()
        
        # Random crop length
        crop_length = np.random.randint(min_length, max_length + 1)
        
        # Random start position
        max_start = num_frames - crop_length
        if max_start <= 0:
            return landmarks.copy()
        
        start_idx = np.random.randint(0, max_start + 1)
        end_idx = start_idx + crop_length
        
        return landmarks[start_idx:end_idx].copy()
    
    @staticmethod
    def frame_dropout(landmarks: np.ndarray,
                     dropout_rate: float = 0.1,
                     method: str = 'interpolate') -> np.ndarray:
        """
        Randomly drop frames and interpolate or forward-fill.
        
        Args:
            landmarks: array of shape (frames, ...)
            dropout_rate: Fraction of frames to drop
            method: 'interpolate' or 'forward_fill'
            
        Returns:
            Augmented landmarks with some frames removed/filled
        """
        num_frames = landmarks.shape[0]
        num_drop = max(1, int(num_frames * dropout_rate))
        
        # Check if we have enough frames
        if num_drop >= num_frames - 1:
            logger.warning(f"dropout_rate too high, returning original")
            return landmarks.copy()
        
        # Randomly select frames to drop
        drop_indices = np.random.choice(num_frames, num_drop, replace=False)
        keep_indices = np.delete(np.arange(num_frames), drop_indices)
        
        if method == 'interpolate':
            # Interpolate landmarks at dropped positions
            augmented = landmarks[keep_indices]
            
            # Simple linear interpolation of kept frames
            if len(keep_indices) < 2:
                return augmented.copy()
            
            # Resample to original length using interpolation
            x_old = np.arange(len(keep_indices))
            x_new = np.linspace(0, len(keep_indices) - 1, num_frames)
            
            # Interpolate along first axis
            augmented_interp = np.zeros_like(landmarks)
            for dim_idx in range(landmarks.shape[-1] if landmarks.ndim > 1 else 1):
                if landmarks.ndim == 3:
                    for lm_idx in range(landmarks.shape[1]):
                        values = landmarks[keep_indices, lm_idx, dim_idx]
                        augmented_interp[:, lm_idx, dim_idx] = np.interp(x_new, x_old, values)
                else:
                    values = landmarks[keep_indices, dim_idx]
                    augmented_interp[:, dim_idx] = np.interp(x_new, x_old, values)
            
            return augmented_interp.astype(np.float32)
        
        else:  # forward fill
            augmented = landmarks.copy()
            for drop_idx in sorted(drop_indices):
                if drop_idx > 0:
                    augmented[drop_idx] = augmented[drop_idx - 1]
            
            return augmented.astype(np.float32)
    
    @staticmethod
    def temporal_scaling(landmarks: np.ndarray,
                        speed_range: Tuple[float, float] = (0.8, 1.2),
                        method: str = 'interpolate') -> np.ndarray:
        """
        Speed up or slow down the sequence (temporal scaling).
        
        Args:
            landmarks: array of shape (frames, ...)
            speed_range: (min_speed, max_speed) ratio
            method: 'interpolate', 'repeat', or 'drop'
            
        Returns:
            Temporally scaled landmarks
        """
        num_frames = landmarks.shape[0]
        speed = np.random.uniform(speed_range[0], speed_range[1])
        
        if method == 'interpolate':
            # Linear interpolation to new frame count
            new_frame_count = max(2, int(num_frames / speed))
            
            x_old = np.arange(num_frames)
            x_new = np.linspace(0, num_frames - 1, new_frame_count)
            
            augmented = np.zeros((new_frame_count,) + landmarks.shape[1:], 
                               dtype=landmarks.dtype)
            
            if landmarks.ndim == 3:
                for lm_idx in range(landmarks.shape[1]):
                    for coord_idx in range(landmarks.shape[2]):
                        values = landmarks[:, lm_idx, coord_idx]
                        augmented[:, lm_idx, coord_idx] = np.interp(x_new, x_old, values)
            else:
                for dim_idx in range(landmarks.shape[1]):
                    values = landmarks[:, dim_idx]
                    augmented[:, dim_idx] = np.interp(x_new, x_old, values)
            
            return augmented.astype(np.float32)
        
        elif method == 'repeat':
            # Repeat or drop frames
            if speed < 1.0:  # Slow down - repeat frames
                repeat_factor = int(np.ceil(1 / speed))
                augmented = np.repeat(landmarks, repeat_factor, axis=0)
            else:  # Speed up - drop frames
                step = int(np.ceil(speed))
                augmented = landmarks[::step]
            
            return augmented
        
        else:
            logger.warning(f"Unknown temporal_scaling method: {method}")
            return landmarks.copy()
    
    @staticmethod
    def mixup_frames(landmarks1: np.ndarray,
                     landmarks2: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
        """
        Blend two landmark sequences (mixup augmentation).
        Useful for mixing different sign variations.
        
        Args:
            landmarks1: First landmark sequence
            landmarks2: Second landmark sequence
            alpha: Blend ratio (0.5 = equal blend)
            
        Returns:
            Blended landmarks
        """
        # Ensure same length (pad shorter if needed)
        min_len = min(landmarks1.shape[0], landmarks2.shape[0])
        l1 = landmarks1[:min_len]
        l2 = landmarks2[:min_len]
        
        # Blend
        alpha = np.random.uniform(0, 1)  # Random alpha for each sample
        blended = alpha * l1 + (1 - alpha) * l2
        
        return blended.astype(np.float32)


class AugmentationPipeline:
    """Compose multiple augmentations into a pipeline."""
    
    def __init__(self, augmentations: Optional[List[Tuple[str, Dict]]] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of (function_name, kwargs) tuples
                Example: [('add_gaussian_noise', {'noise_std': 0.01}),
                         ('random_scale', {'scale_range': (0.9, 1.1)})]
        """
        self.augmentations = augmentations or []
    
    def add_augmentation(self, name: str, **kwargs):
        """Add an augmentation to the pipeline."""
        self.augmentations.append((name, kwargs))
        return self
    
    def apply(self, landmarks: np.ndarray, 
              probability: float = 1.0) -> np.ndarray:
        """
        Apply all augmentations in sequence.
        
        Args:
            landmarks: Input landmarks
            probability: Overall probability of applying augmentation
            
        Returns:
            Augmented landmarks
        """
        if np.random.random() > probability:
            return landmarks.copy()
        
        augmented = landmarks
        
        for aug_name, kwargs in self.augmentations:
            if hasattr(LandmarkAugmenter, aug_name):
                method = getattr(LandmarkAugmenter, aug_name)
                augmented = method(augmented, **kwargs)
            else:
                logger.warning(f"Unknown augmentation: {aug_name}")
        
        return augmented
    
    def __repr__(self):
        """String representation of pipeline."""
        aug_str = "\n  - ".join([f"{name} ({kwargs})" 
                                 for name, kwargs in self.augmentations])
        return f"AugmentationPipeline:\n  - {aug_str}"


def create_default_augmentation_pipeline(include_temporal: bool = True,
                                         include_spatial: bool = True) -> AugmentationPipeline:
    """
    Create a default augmentation pipeline with reasonable defaults.
    
    Args:
        include_temporal: Include temporal augmentations
        include_spatial: Include spatial augmentations
        
    Returns:
        Configured AugmentationPipeline
    """
    pipeline = AugmentationPipeline()
    
    if include_spatial:
        pipeline.add_augmentation('add_gaussian_noise', 
                                 noise_std=0.01, 
                                 apply_to_confidence=False)
        pipeline.add_augmentation('random_scale', 
                                 scale_range=(0.9, 1.1))
        pipeline.add_augmentation('random_rotation', 
                                 angle_range=(-10, 10))
        pipeline.add_augmentation('random_translation', 
                                 translation_range=(-0.05, 0.05))
    
    if include_temporal:
        # Note: temporal augmentations may change sequence length
        # Use with caution in fixed-length input models
        pipeline.add_augmentation('temporal_scaling',
                                 speed_range=(0.95, 1.05),
                                 method='interpolate')
    
    return pipeline


# Convenience functions
def augment_batch(landmarks_batch: np.ndarray,
                 pipeline: Optional[AugmentationPipeline] = None,
                 num_augmentations: int = 1) -> List[np.ndarray]:
    """
    Apply augmentation pipeline to a batch of landmarks.
    
    Args:
        landmarks_batch: List of landmark arrays
        pipeline: AugmentationPipeline. If None, uses default.
        num_augmentations: Number of augmented copies per video
        
    Returns:
        List of augmented landmark arrays
    """
    if pipeline is None:
        pipeline = create_default_augmentation_pipeline()
    
    augmented_batch = []
    
    for landmarks in landmarks_batch:
        for _ in range(num_augmentations):
            augmented = pipeline.apply(landmarks)
            augmented_batch.append(augmented)
    
    return augmented_batch
