# Landmark Extraction with MediaPipe

## Overview

This module extracts pose, hand, and face landmarks from segmented sign language videos using MediaPipe Holistic. The landmarks are saved as NumPy arrays (.npy) and can be directly fed to neural networks for sign language recognition or translation tasks.

## Architecture

### Data Flow

```
Segmented Videos (.mp4)
        ↓
[Landmark Extraction with MediaPipe Holistic]
        ↓
NumPy Arrays (.npy) - (frames, 543 landmarks, 4 coordinates)
        ↓
[Preprocessing & Normalization]
        ↓
Network Input - (frames, flattened_features)
        ↓
Neural Network Model
```

### Landmark Structure

Each frame contains 543 total landmarks with 4 coordinates each [x, y, z, confidence]:

- **Pose Landmarks**: 33 points (body skeleton)
- **Left Hand Landmarks**: 21 points (left hand keypoints)
- **Right Hand Landmarks**: 21 points (right hand keypoints)
- **Face Landmarks**: 468 points (facial features)

**Total per frame**: 543 × 4 = 2,172 coordinates

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key packages:

- `mediapipe>=0.8.9.1` - Landmark detection
- `opencv-python>=4.5.0` - Video processing
- `numpy>=1.20.0` - Array operations

## Usage

### Step 1: Extract Landmarks

Extract landmarks from all segmented videos:

```bash
python src/preprocessing/landmark_extraction.py
```

**Parameters** (edit in code if needed):

- `confidence_threshold`: Minimum confidence to include a landmark (default: 0.5)
- `model_complexity`: MediaPipe model complexity (default: 1)

**Output**: `.npy` files in `dataset/landmarks/`

- One file per video: `{video_id}_landmarks.npy`
- Shape: `(num_frames, 543, 4)`

### Step 2: Load and Preprocess

```python
from src.preprocessing.landmark_utils import LandmarkLoader, LandmarkPreprocessor

# Load landmarks
landmarks = LandmarkLoader.load_landmarks('dataset/landmarks/video_id_landmarks.npy')
# Shape: (frames, 543, 4)

# Normalize values to [0, 1]
normalized = LandmarkPreprocessor.normalize_landmarks(landmarks)

# Fill missing landmarks (forward fill)
filled = LandmarkPreprocessor.fill_missing_landmarks(normalized, method='forward')

# Flatten for network input
network_input = LandmarkPreprocessor.flatten_for_network(filled)
# Shape: (frames, 2172)
```

### Step 3: Complete Pipeline

For a one-step preprocessing:

```python
from src.preprocessing.landmark_utils import prepare_for_network

network_input = prepare_for_network(
    'dataset/landmarks/video_id_landmarks.npy',
    normalize=True,
    fill_missing=True,
    method='forward',
    drop_z=False,           # Keep 3D coordinates
    include_confidence=True # Include confidence values
)
# Shape: (frames, 2172)
```

### Step 4: Use with Neural Network

```python
import numpy as np
from src.preprocessing.landmark_utils import prepare_for_network

# Load all video landmarks
all_videos = []
for npy_file in Path('dataset/landmarks').glob('*_landmarks.npy'):
    X = prepare_for_network(str(npy_file))
    all_videos.append(X)

# Pad to same length (if needed)
max_frames = max(len(v) for v in all_videos)
X = np.array([np.pad(v, ((0, max_frames - len(v)), (0, 0))) for v in all_videos])

# Now ready for model
model.fit(X, y)  # Your neural network
```

## Advanced Options

### Extract Specific Body Parts

```python
body_parts = LandmarkPreprocessor.extract_body_parts(landmarks)

# body_parts contains:
# - 'pose': (frames, 33, 4) - body skeleton only
# - 'left_hand': (frames, 21, 4) - left hand only
# - 'right_hand': (frames, 21, 4) - right hand only
# - 'face': (frames, 468, 4) - facial landmarks only
```

### Different Preprocessing Options

```python
# Option 1: 2D coordinates only (drop z)
network_input_2d = prepare_for_network(
    path, drop_z=True
)  # Shape: (frames, 1452)

# Option 2: 3D without confidence
network_input_3d = prepare_for_network(
    path,
    include_confidence=False
)  # Shape: (frames, 1629)

# Option 3: Interpolation for missing frames
network_input_interp = prepare_for_network(
    path,
    fill_missing=True,
    method='interpolate'
)
```

### Load Multiple Videos

```python
from src.preprocessing.landmark_utils import LandmarkLoader

# Load all landmarks in a batch
all_landmarks = LandmarkLoader.load_landmarks_batch('dataset/landmarks')
# Returns: {video_id: array, ...}

# Or specific videos
videos = ['video_id_1', 'video_id_2']
some_landmarks = LandmarkLoader.load_landmarks_batch('dataset/landmarks', video_ids=videos)
```

### Compute Statistics

```python
from src.preprocessing.landmark_utils import LandmarkStatistics

stats = LandmarkStatistics.compute_stats(landmarks)
# Returns: {
#   'num_frames': int,
#   'total_landmarks': 543,
#   'mean': float,
#   'std': float,
#   'min': float,
#   'max': float,
#   'missing_landmark_frames': int
# }

# Print formatted statistics
LandmarkStatistics.print_stats(landmarks, 'video_id')
```

## Example Workflow

See `scripts/extract_landmarks_example.py` for a complete example:

```bash
python scripts/extract_landmarks_example.py
```

This demonstrates:

1. Extracting landmarks from all videos
2. Loading and analyzing the data
3. Various preprocessing options
4. Flattening for network input

## Output Formats

### For Different Network Architectures

**LSTM/RNN** (temporal models):

```python
# Keep temporal dimension
network_input.shape  # (frames, 2172)
```

**CNN** (temporal convolution):

```python
# Reshape with channels
sequence = network_input.reshape(frames, 1, 2172)
```

**Transformer** (attention models):

```python
# Reshape as sequence of landmarks
sequence = landmarks.reshape(frames, 543, 4)  # (frames, tokens, features)
```

**Dense/MLP**:

```python
# Flatten entire sequence
flat_sequence = network_input.reshape(frames * 2172)
```

## Troubleshooting

### Missing Landmarks in Video

If some frames have no detected landmarks (confidence = 0):

- Use `fill_missing_landmarks()` with `method='forward'` or `'interpolate'`
- Or filter videos with too many missing frames

### Inconsistent Sequence Lengths

Videos have different numbers of frames:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Pad all to same length
max_len = max(len(v) for v in all_videos)
padded = pad_sequences(all_videos, maxlen=max_len)
```

### Memory Issues with Large Datasets

Process in batches:

```python
batch_size = 100
for i in range(0, len(videos), batch_size):
    batch = videos[i:i+batch_size]
    # Process batch
```

## Performance Notes

- **Extraction speed**: ~10-30 FPS depending on video resolution and pose complexity
- **File size**: ~100KB per minute of video (compressed)
- **Memory**: ~1.5-2GB for 100 videos at 30 FPS
- **Preprocessing**: < 1 second per video

## MediaPipe Holistic Details

- **Input**: RGB frames (OpenCV reads BGR, automatically converted)
- **Output**: 3D landmarks (x, y, z coordinates normalized to frame dimensions)
- **Confidence**: 0-1 range (0 = not detected, 1 = high confidence)
- **Model complexity**: 0 (lite/mobile), 1 (full, recommended for sign language)

## References

- [MediaPipe Documentation](https://mediapipe.dev/)
- [Holistic Solution](https://developers.google.com/mediapipe/solutions/vision/holistic)
- [Landmark Index Legend](https://mediapipe.dev/images/mobile/holistic_tracking_full_body_landmarks.png)
