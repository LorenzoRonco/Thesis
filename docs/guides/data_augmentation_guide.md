# Data Augmentation for Landmark Sequences

## Overview

The `landmark_augmentation.py` module provides comprehensive data augmentation techniques for MediaPipe-extracted landmark sequences. These augmentations apply spatial and temporal transformations to create variations in your training data, improving model generalization and robustness.

## Why Data Augmentation Matters

For sign language recognition:

- **Position invariance**: Signs performed at different distances/positions
- **Speed variations**: Different signing speeds
- **Viewing angles**: Different camera angles
- **Environmental variations**: Different lighting, backgrounds
- **Hand mirror**: Same gesture with dominant hand switched

## Available Augmentations

### Spatial Augmentations

#### 1. Gaussian Noise

Adds random noise to landmark coordinates (simulates sensor noise).

```python
LandmarkAugmenter.add_gaussian_noise(
    landmarks,
    noise_std=0.01,           # Standard deviation
    apply_to_confidence=False  # Only x,y,z or also confidence?
)
```

**Use case**: Robustness to detection noise
**Effect**: ±0.01 random perturbation to each coordinate

---

#### 2. Random Scale

Scales all landmarks around a center point (simulates distance from camera).

```python
LandmarkAugmenter.random_scale(
    landmarks,
    scale_range=(0.8, 1.2),  # Scale factor [min, max]
    center=None               # Auto-compute from hand/face
)
```

**Use case**: Scale invariance (person at different distances)
**Effect**: Expands/contracts landmarks by 0.8-1.2x

---

#### 3. Random Rotation

Rotates landmarks in-plane (simulates head tilt/body rotation).

```python
LandmarkAugmenter.random_rotation(
    landmarks,
    angle_range=(-15, 15),  # Degrees
    center=None             # Auto-compute
)
```

**Use case**: Viewpoint invariance
**Effect**: Rotates ±15° in-plane

---

#### 4. Random Translation

Translates landmarks left/right and up/down.

```python
LandmarkAugmenter.random_translation(
    landmarks,
    translation_range=(-0.1, 0.1)  # Normalized coordinates
)
```

**Use case**: Position invariance
**Effect**: Shifts by ±0.1 in x,y directions

---

#### 5. Mirror (Horizontal Flip)

Flips landmarks left-right and swaps hands.

```python
LandmarkAugmenter.mirror_landmarks(
    landmarks,
    probability=0.5  # 50% chance of flipping
)
```

**Use case**: Mirror invariance (left-handed signers)
**Effect**: Flips x coordinates, swaps left/right hands

---

#### 6. Shear Transformation

Applies shear transformation (simulates 3D perspective).

```python
LandmarkAugmenter.shear_landmarks(
    landmarks,
    shear_range=(-0.1, 0.1)
)
```

**Use case**: Viewing angle variations
**Effect**: Skews x,y coordinates

---

### Temporal Augmentations

#### 1. Temporal Crop

Randomly extracts a subsequence (shorter video).

```python
LandmarkAugmenter.temporal_crop(
    landmarks,
    min_length=10,      # Min frames to keep
    max_length=None     # Max frames (None = full length)
)
```

**Use case**: Simulate partial video capture
**Effect**: Returns random [10, num_frames] length sequence

---

#### 2. Frame Dropout

Randomly removes frames and interpolates/fills.

```python
LandmarkAugmenter.frame_dropout(
    landmarks,
    dropout_rate=0.1,           # Drop 10% of frames
    method='interpolate'        # Or 'forward_fill'
)
```

**Use case**: Robustness to missed detections
**Effect**: Removes 10% of frames, interpolates gap

---

#### 3. Temporal Scaling

Speed up or slow down the sequence.

```python
LandmarkAugmenter.temporal_scaling(
    landmarks,
    speed_range=(0.8, 1.2),    # 0.8-1.2x speed
    method='interpolate'         # 'interpolate', 'repeat', 'drop'
)
```

**Use case**: Speed invariance (slower/faster signers)
**Effect**: Resamples to 0.8-1.2x original speed

---

## Using Augmentation Pipelines

Instead of applying augmentations individually, use `AugmentationPipeline` to compose multiple augmentations:

### Default Pipeline (Recommended)

```python
from preprocessing.landmark_augmentation import create_default_augmentation_pipeline

pipeline = create_default_augmentation_pipeline(
    include_spatial=True,   # Add spatial augmentations
    include_temporal=True   # Add temporal augmentations
)

# Apply to landmarks
augmented = pipeline.apply(landmarks, probability=1.0)
```

**What it includes**:

1. Gaussian noise (std=0.01)
2. Random scale (0.9-1.1x)
3. Random rotation (±10°)
4. Random translation (±0.05)
5. Temporal scaling (0.95-1.05x, interpolated)

---

### Custom Pipelines

```python
from preprocessing.landmark_augmentation import AugmentationPipeline

# Light augmentation
light = AugmentationPipeline()
light.add_augmentation('add_gaussian_noise', noise_std=0.005)
light.add_augmentation('random_scale', scale_range=(0.95, 1.05))

# Aggressive augmentation
aggressive = AugmentationPipeline()
aggressive.add_augmentation('add_gaussian_noise', noise_std=0.02)
aggressive.add_augmentation('random_scale', scale_range=(0.7, 1.3))
aggressive.add_augmentation('random_rotation', angle_range=(-20, 20))
aggressive.add_augmentation('random_translation', translation_range=(-0.1, 0.1))
aggressive.add_augmentation('shear_landmarks', shear_range=(-0.15, 0.15))

# Apply
augmented_light = light.apply(landmarks)
augmented_aggressive = aggressive.apply(landmarks)
```

---

## Practical Usage Examples

### During Training

```python
from preprocessing.landmark_augmentation import create_default_augmentation_pipeline

pipeline = create_default_augmentation_pipeline()

# In training loop
for batch in training_data:
    # Load original landmarks
    landmarks = load_landmarks(batch)

    # Apply augmentation
    augmented = pipeline.apply(landmarks, probability=1.0)

    # Feed to model
    predictions = model(augmented)
    loss.backward()
```

### Creating an Augmented Dataset

```python
from preprocessing.landmark_augmentation import create_augmented_dataset

# Create 5 augmented versions of each video
create_augmented_dataset(
    landmarks_dir='dataset/landmarks',
    output_dir='dataset/landmarks_augmented',
    num_augmentations=5
)
```

Output structure:

```
landmarks_augmented/
├── video1_aug_0.npy    (original)
├── video1_aug_1.npy    (augmented v1)
├── video1_aug_2.npy    (augmented v2)
├── video1_aug_3.npy    (augmented v3)
├── video1_aug_4.npy    (augmented v4)
├── video1_aug_5.npy    (augmented v5)
├── video2_aug_0.npy    (original)
│ ...
```

This creates a 6x larger dataset (1 original + 5 augmented per video).

---

### Batch Augmentation

```python
from preprocessing.landmark_augmentation import augment_batch, create_default_augmentation_pipeline

pipeline = create_default_augmentation_pipeline()

# Load multiple landmarks
landmarks_list = [load_landmarks(f) for f in landmark_files]

# Create 3 augmented versions per video
augmented = augment_batch(
    landmarks_list,
    pipeline=pipeline,
    num_augmentations=3
)
# Result: 4 videos * 4 versions each = 16 total
```

---

## Best Practices

### 1. **Training vs Validation**

```python
# Training: Use augmentation
train_augmented = pipeline.apply(train_landmarks, probability=1.0)

# Validation: Use original data
val_landmarks_original = val_landmarks  # No augmentation
```

**Why**: Validation should test generalization to real (non-augmented) data.

---

### 2. **Augmentation Strength Over Time**

```python
# Early training: Stronger augmentation
early_pipeline = create_default_augmentation_pipeline()
augmented_early = early_pipeline.apply(landmarks, probability=1.0)

# Late training: Weaker augmentation
late_pipeline = AugmentationPipeline()
late_pipeline.add_augmentation('add_gaussian_noise', noise_std=0.005)
augmented_late = late_pipeline.apply(landmarks, probability=0.5)
```

**Why**: Stronger augmentation early helps with generalization; weaker augmentation later for fine-tuning.

---

### 3. **Avoid Breaking the Problem**

⚠️ **Don't do this**:

```python
# All landmarks outside [0, 1] - breaks normalized representation
LandmarkAugmenter.random_scale(landmarks, scale_range=(0.1, 10.0))

# Can't even recognize as hand gesture anymore
LandmarkAugmenter.random_rotation(landmarks, angle_range=(-90, 90))
```

✅ **Do this instead**:

```python
# Reasonable scale variation
LandmarkAugmenter.random_scale(landmarks, scale_range=(0.8, 1.2))

# Reasonable rotation
LandmarkAugmenter.random_rotation(landmarks, angle_range=(-15, 15))
```

---

### 4. **Monitor Augmentation Effects**

```python
import numpy as np

original = landmarks
augmented = pipeline.apply(landmarks)

# Check that augmentation is reasonable
print(f"Original range: [{original.min():.3f}, {original.max():.3f}]")
print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")

# Check that sequence length is preserved (if using temporal scaling)
print(f"Original frames: {original.shape[0]}")
print(f"Augmented frames: {augmented.shape[0]}")
```

---

### 5. **Probability-Based Augmentation**

```python
# 80% chance of augmentations
augmented = pipeline.apply(landmarks, probability=0.8)

# In training loop
for landmarks in batch:
    if random() < augmentation_prob:
        augmented = pipeline.apply(landmarks)
    else:
        augmented = landmarks
```

---

## Augmentation Configuration Examples

### Minimal (for overfitted models)

```python
minimal = AugmentationPipeline()
minimal.add_augmentation('add_gaussian_noise', noise_std=0.005)
minimal.add_augmentation('random_scale', scale_range=(0.95, 1.05))
```

### Moderate (recommended default)

```python
moderate = create_default_augmentation_pipeline()
```

### Aggressive (for small datasets)

```python
aggressive = AugmentationPipeline()
aggressive.add_augmentation('add_gaussian_noise', noise_std=0.02)
aggressive.add_augmentation('random_scale', scale_range=(0.7, 1.3))
aggressive.add_augmentation('random_rotation', angle_range=(-20, 20))
aggressive.add_augmentation('random_translation', translation_range=(-0.1, 0.1))
aggressive.add_augmentation('shear_landmarks', shear_range=(-0.15, 0.15))
aggressive.add_augmentation('mirror_landmarks', probability=0.3)
aggressive.add_augmentation('temporal_scaling', speed_range=(0.8, 1.2))
```

---

## Advanced: Custom Augmentations

To add your own augmentation:

```python
from preprocessing.landmark_augmentation import LandmarkAugmenter

# Add a static method to LandmarkAugmenter
@staticmethod
def custom_augmentation(landmarks, param1=value1):
    """Your custom augmentation."""
    augmented = landmarks.copy()
    # Apply your transformation
    return augmented

LandmarkAugmenter.custom_augmentation = custom_augmentation

# Use in pipeline
pipeline = AugmentationPipeline()
pipeline.add_augmentation('custom_augmentation', param1=value)
```

---

## Troubleshooting

### Augmentation changes sequence length

**Problem**: Model expects fixed input length
**Solution**: Use `include_temporal=False` or post-process to fixed length

```python
# Exclude temporal augmentations
pipeline = create_default_augmentation_pipeline(
    include_temporal=False,
    include_spatial=True
)

# Or manually pad/crop after augmentation
if augmented.shape[0] != target_length:
    augmented = augmented[:target_length]  # Crop
    # or
    augmented = np.pad(augmented, ((0, max(0, target_length - augmented.shape[0])), (0,0), (0,0)))
```

---

### Values going out of [0, 1] range

**Problem**: After augmentation, coordinates are outside expected range
**Solution**: Clamp or use weaker augmentation

```python
# Clamp to valid range
augmented = np.clip(augmented, 0, 1)

# Or use weaker augmentation
pipeline = AugmentationPipeline()
pipeline.add_augmentation('random_scale', scale_range=(0.95, 1.05))  # Smaller range
```

---

### Not seeing augmentation benefit

**Problem**: Validation loss not improving with augmentation
**Possible causes**:

1. Augmentation too strong (breaks the problem)
2. Dataset already diverse enough
3. Model underfitting (not overfitting)

**Solutions**:

```python
# 1. Reduce augmentation strength
light_pipeline = AugmentationPipeline()
light_pipeline.add_augmentation('add_gaussian_noise', noise_std=0.005)

# 2. Check validation on original data
val_loss_augmented = validate_on_augmented_data()
val_loss_original = validate_on_original_data()

# 3. Check if model is overfitting
if train_loss << val_loss:
    # Use stronger augmentation
    use_aggressive_augmentation = True
```

---

## Performance Considerations

All augmentations are efficient NumPy operations:

| Augmentation                | Speed     | Memory         |
| --------------------------- | --------- | -------------- |
| Gaussian noise              | Very fast | In-place       |
| Scale/rotation              | Fast      | In-place       |
| Temporal crop               | Very fast | Reduces memory |
| Frame dropout (interpolate) | Moderate  | In-place       |
| Temporal scaling            | Moderate  | ×1-1.2x memory |

For training, apply augmentations on-the-fly in the data loader:

```python
class AugmentedDataLoader:
    def __init__(self, landmarks, pipeline):
        self.landmarks = landmarks
        self.pipeline = pipeline

    def __getitem__(self, idx):
        original = self.landmarks[idx]
        augmented = self.pipeline.apply(original)
        return augmented
```

---

## References

- **Regularization via augmentation**: [AutoAugment](https://arxiv.org/abs/1805.09501)
- **Temporal augmentation**: [TimeMix](https://arxiv.org/abs/2302.02836)
- **Sign language recognition**: [How2Sign Dataset](https://www.microsoft.com/en-us/research/publication/how2sign-a-large-scale-multimodal-dataset-for-continuous-american-sign-language/)

---

## Quick Start

Run the example:

```bash
python scripts/augment_landmarks_example.py
```

In your code:

```python
from preprocessing.landmark_augmentation import create_default_augmentation_pipeline

pipeline = create_default_augmentation_pipeline()
augmented_landmarks = pipeline.apply(landmarks)
```

That's it! You're ready to use augmented landmarks for training.

---

## For Large Datasets (30,000+ files)

If you have many landmarks, **do NOT pre-compute augmented files** - use on-the-fly augmentation instead:

```python
from scripts.train_with_augmentation import AugmentedLandmarkDataset, SimpleTrainingIterator

# Create dataset (applies augmentation during loading)
dataset = AugmentedLandmarkDataset(
    landmarks_dir='dataset/landmarks',
    augmentation_pipeline=pipeline,
    apply_augmentation=True  # Apply on-the-fly!
)

# Training loop
iterator = SimpleTrainingIterator(dataset, batch_size=32, shuffle=True)

for epoch in range(epochs):
    for batch in iterator:
        predictions = model(batch)
        loss.backward()

# Validation - NO augmentation!
val_dataset = AugmentedLandmarkDataset(
    landmarks_dir='dataset/landmarks',
    apply_augmentation=False  # Use original only
)
```

**Benefits**:

- ✓ Zero extra disk space (use original 30k files only)
- ✓ Different augmentations each epoch
- ✓ Faster I/O (no need to copy 120k files)
- ✓ Industry standard practice
