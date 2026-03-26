#!/usr/bin/env python3
"""
On-the-fly data augmentation for training.
Loads PRE-NORMALIZED landmarks and applies augmentations during loading.

Pipeline:
    1. normalize_landmarks.py: raw landmarks → normalized landmarks
    2. train_with_augmentation.py: load normalized + apply augmentation on-the-fly

Usage:
    from train_with_augmentation import AugmentedLandmarkDataset, SimpleTrainingIterator
    
    # Expects normalized landmarks from dataset/landmarks_normalized/
    dataset = AugmentedLandmarkDataset('dataset/landmarks_normalized', pipeline, apply_augmentation=True)
    iterator = SimpleTrainingIterator(dataset, batch_size=32, shuffle=True)
    
    for batch in iterator:
        predictions = model(batch)
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Optional
import logging

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from preprocessing.landmark_utils import LandmarkLoader
from preprocessing.landmark_augmentation import AugmentationPipeline

logger = logging.getLogger(__name__)


class AugmentedLandmarkDataset:
    """
    Dataset wrapper that applies augmentations on-the-fly.
    No extra disk space needed!
    """
    
    def __init__(self, 
                 landmarks_dir: Path,
                 augmentation_pipeline: Optional[AugmentationPipeline] = None,
                 apply_augmentation: bool = True):
        """
        Initialize dataset.
        
        Args:
            landmarks_dir: Directory with original .npy files
            augmentation_pipeline: Pipeline to apply. If None, no augmentation.
            apply_augmentation: Whether to apply augmentation
        """
        self.landmarks_dir = Path(landmarks_dir)
        self.pipeline = augmentation_pipeline
        self.apply_augmentation = apply_augmentation
        
        # Load list of all landmark files
        self.landmark_files = sorted(self.landmarks_dir.glob("*_landmarks.npy"))
        
        if not self.landmark_files:
            raise ValueError(f"No landmark files found in {self.landmarks_dir}")
        
        print(f"✓ Loaded dataset: {len(self.landmark_files)} files from {self.landmarks_dir}")
    
    def __len__(self):
        """Number of videos."""
        return len(self.landmark_files)
    
    def __getitem__(self, idx):
        """
        Load and optionally augment a single video.
        
        Args:
            idx: Index of video to load
            
        Returns:
            Numpy array of landmarks (augmented or original, pre-normalized)
        """
        npy_file = self.landmark_files[idx]
        
        # Load normalized landmarks (from normalize_landmarks.py)
        landmarks = LandmarkLoader.load_landmarks(npy_file)
        
        # Apply augmentation on pre-normalized landmarks
        if self.apply_augmentation and self.pipeline is not None:
            landmarks = self.pipeline.apply(landmarks, probability=1.0)
            # Clip to safe range (may go slightly outside [0,1] after augmentation)
            landmarks = np.clip(landmarks, -0.5, 1.5)
        
        return landmarks
    
    def get_video_id(self, idx):
        """Get video ID for a given index."""
        return self.landmark_files[idx].stem.replace("_landmarks", "")
    
    def load_batch(self, indices):
        """
        Load a batch of videos.
        
        Args:
            indices: List of indices to load
            
        Returns:
            List of landmark arrays
        """
        batch = []
        for idx in indices:
            landmarks = self[idx]
            batch.append(landmarks)
        return batch


class SimpleTrainingIterator:
    """
    Simple training iterator (no PyTorch dependency).
    Useful if you're not using PyTorch DataLoader.
    """
    
    def __init__(self,
                 dataset: AugmentedLandmarkDataset,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize iterator.
        
        Args:
            dataset: AugmentedLandmarkDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle indices
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = len(dataset) // batch_size
        
        if len(dataset) % batch_size != 0:
            self.num_batches += 1
    
    def __iter__(self):
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for batch_idx in range(self.num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            batch_indices = indices[start:end]
            
            # Load batch
            batch_landmarks = self.dataset.load_batch(batch_indices)
            yield batch_landmarks
    
    def __len__(self):
        """Number of batches."""
        return self.num_batches


def create_webcam_augmentation_pipeline():
    """
    Create lightweight augmentation pipeline for frontal webcam video.
    
    Returns:
        AugmentationPipeline with realistic transformations
    """
    pipeline = AugmentationPipeline()
    
    # Scale: person at different distances
    pipeline.add_augmentation('random_scale', scale_range=(0.85, 1.15))
    
    # Rotation: natural head tilt
    pipeline.add_augmentation('random_rotation', angle_range=(-8, 8))
    
    # Translation: movement in frame
    pipeline.add_augmentation('random_translation', translation_range=(-0.05, 0.05))
    
    # Temporal: slight speed variation
    pipeline.add_augmentation('temporal_scaling', speed_range=(0.98, 1.02), method='interpolate')
    
    return pipeline

