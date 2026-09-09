"""Extract MediaPipe holistic landmarks from image sequences (PHOENIX-2014-T dataset).

Each sentence is stored as a folder of PNG frames listed in the corpus CSV.
The CSV gives paths like  tagesschau-2/1/*.png  but the subfolder '1' does not
exist: images live directly under  dataset/train/<sequence_folder>/.

Output landmark layout (94 points total):
  Indices  0-16  → pose upper body (17 pts, MediaPipe pose indices 0-16)
  Indices 17-37  → left hand (21 pts)
  Indices 38-58  → right hand (21 pts)
  Indices 59-70  → mouth (12 pts, averaged/sampled from MediaPipe lip contour)
  Indices 71-78  → left eye (8 pts, averaged/sampled from MediaPipe eye contour)
  Indices 79-86  → right eye (8 pts, averaged/sampled from MediaPipe eye contour)
  Index   87     → nose tip (1 pt, MediaPipe face landmark 1)
  Indices 88-90  → left eyebrow (3 pts, averaged from MediaPipe landmarks)
  Indices 91-93  → right eyebrow (3 pts, averaged from MediaPipe landmarks)

Each landmark is stored as [x, y, z, confidence] → shape per frame: (94, 4).
Final output per sentence: numpy array of shape (num_frames, 94, 4).
"""

from collections import defaultdict
import logging
import argparse
import csv
import glob
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_TRAIN_DIR = Path(
    "/mnt/de44ee77-b697-4fa6-acff-096979f6cd2d/Ronco/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
)
DEFAULT_DEV_DIR = Path(
    "/mnt/de44ee77-b697-4fa6-acff-096979f6cd2d/Ronco/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark counts
# ---------------------------------------------------------------------------
POSE_LANDMARKS = 17          # upper body (indices 0-16)
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
MOUTH_LANDMARKS = 12         # reduced / averaged from lip contour
LEFT_EYE_LANDMARKS = 8       # reduced / averaged from eye contour
RIGHT_EYE_LANDMARKS = 8
NOSE_LANDMARKS = 1           # just the tip
LEFT_EYEBROW_LANDMARKS = 3   # averaged from eyebrow points
RIGHT_EYEBROW_LANDMARKS = 3

TOTAL_LANDMARKS = (
    POSE_LANDMARKS + LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS +
    MOUTH_LANDMARKS + LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS +
    NOSE_LANDMARKS + LEFT_EYEBROW_LANDMARKS + RIGHT_EYEBROW_LANDMARKS
)  # = 94

# ---------------------------------------------------------------------------
# MediaPipe face mesh index groups
# (based on the 468-point canonical face mesh)
# ---------------------------------------------------------------------------

# Outer + inner lip contour (a representative 20-point ring → averaged to 12)
# MediaPipe canonical lip landmark indices (outer + inner contour)
_MOUTH_RAW_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # upper outer lip
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  # lower outer lip
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,   # upper inner lip
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,  # lower inner lip
]
# We'll average these down to MOUTH_LANDMARKS=12 evenly spaced buckets.

# Left eye contour (16 raw indices → averaged to 8)
_LEFT_EYE_RAW_INDICES = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
]

# Right eye contour (16 raw indices → averaged to 8)
_RIGHT_EYE_RAW_INDICES = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382,
]

# Nose tip
_NOSE_TIP_INDEX = 1  # face landmark index for nose tip

# Left eyebrow (10 raw indices → averaged to 3)
_LEFT_EYEBROW_RAW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# Right eyebrow (10 raw indices → averaged to 3)
_RIGHT_EYEBROW_RAW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# ---------------------------------------------------------------------------
# Default model URL
# ---------------------------------------------------------------------------
DEFAULT_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _average_buckets(points: list[list[float]], n_out: int) -> list[list[float]]:
    """Average a list of [x,y,z,conf] points into n_out evenly-spaced buckets."""
    if not points:
        return [[0.0, 0.0, 0.0, 0.0]] * n_out
    arr = np.array(points, dtype=np.float32)
    indices = np.round(np.linspace(0, len(arr) - 1, n_out)).astype(int)
    # Group into consecutive buckets
    bucket_size = len(arr) / n_out
    out = []
    for i in range(n_out):
        lo = int(round(i * bucket_size))
        hi = int(round((i + 1) * bucket_size))
        hi = max(hi, lo + 1)
        bucket = arr[lo:hi]
        out.append(bucket.mean(axis=0).tolist())
    return out


def _pick_face_points(
    face_landmarks: list,
    raw_indices: list[int],
    n_out: int,
) -> list[list[float]]:
    """
    Extract a subset of face landmarks by index, then average to n_out points.
    Returns list of [x, y, z, 1.0] (face landmarks have no per-point confidence).
    """
    if not face_landmarks:
        return [[0.0, 0.0, 0.0, 0.0]] * n_out

    n_available = len(face_landmarks)
    raw = []
    for idx in raw_indices:
        if idx < n_available:
            lm = face_landmarks[idx]
            raw.append([lm.x, lm.y, lm.z, 1.0])
        else:
            raw.append([0.0, 0.0, 0.0, 0.0])

    return _average_buckets(raw, n_out)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """Extract 94 holistic landmarks from a single image using MediaPipe Tasks."""

    def __init__(self, confidence_threshold: float = 0.5, model_path=None):
        self.confidence_threshold = confidence_threshold
        self.model_path = self._resolve_model_path(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,   # IMAGE mode for stills
            min_face_detection_confidence=confidence_threshold,
            min_face_landmarks_confidence=confidence_threshold,
            min_pose_detection_confidence=confidence_threshold,
            min_pose_landmarks_confidence=confidence_threshold,
            min_hand_landmarks_confidence=confidence_threshold,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def _resolve_model_path(self, model_path) -> Path:
        if model_path:
            path = Path(model_path)
        else:
            root_dir = Path(__file__).resolve().parent.parent
            path = root_dir / "models" / "holistic_landmarker.task"

        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Model not found, downloading: %s", path)
        try:
            urlretrieve(DEFAULT_HOLISTIC_MODEL_URL, str(path))
        except Exception as exc:
            raise RuntimeError(
                "Cannot download holistic model. "
                "Download it manually and pass --model-path."
            ) from exc
        return path

    # ------------------------------------------------------------------
    def extract_landmarks_from_image(self, image_path: Path) -> np.ndarray | None:
        """
        Run holistic landmark detection on a single image.

        Returns:
            numpy array of shape (94, 4)  [x, y, z, confidence]
            None on error.
        """
        try:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning("Cannot read image: %s", image_path)
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.landmarker.detect(mp_image)
            return self._extract_frame_landmarks(results)

        except Exception as exc:
            logger.error("Error processing %s: %s", image_path, exc)
            return None

    # ------------------------------------------------------------------
    def extract_landmarks_from_sequence(
        self, image_paths: list[Path]
    ) -> np.ndarray | None:
        """
        Extract landmarks from an ordered list of image paths.

        Returns:
            numpy array of shape (num_frames, 94, 4)
            None if no frame could be processed.
        """
        all_landmarks = []
        for img_path in image_paths:
            frame_lm = self.extract_landmarks_from_image(img_path)
            if frame_lm is not None:
                all_landmarks.append(frame_lm)
            else:
                # Keep a zero frame so temporal alignment is preserved
                all_landmarks.append(np.zeros((TOTAL_LANDMARKS, 4), dtype=np.float32))

        if not all_landmarks:
            return None
        return np.array(all_landmarks, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _as_landmark_list(maybe_landmarks):
        if not maybe_landmarks:
            return []
        first = maybe_landmarks[0]
        if isinstance(first, (list, tuple)):
            return list(first)
        return list(maybe_landmarks)

    @staticmethod
    def _extract_points(landmarks, expected_count, confidence_attr=None):
        out = []
        for lm in landmarks[:expected_count]:
            conf = getattr(lm, confidence_attr, None) if confidence_attr else None
            if conf is None:
                conf = 1.0
            out.append([lm.x, lm.y, lm.z, float(conf)])
        if len(out) < expected_count:
            out.extend([[0.0, 0.0, 0.0, 0.0]] * (expected_count - len(out)))
        return out

    # ------------------------------------------------------------------
    def _extract_frame_landmarks(self, results) -> np.ndarray:
        """
        Build the 94-point landmark vector for one frame.

        Layout:
          [0-16]   pose upper body
          [17-37]  left hand
          [38-58]  right hand
          [59-70]  mouth  (12 pts)
          [71-78]  left eye (8 pts)
          [79-86]  right eye (8 pts)
          [87]     nose tip (1 pt)
          [88-90]  left eyebrow (3 pts)
          [91-93]  right eyebrow (3 pts)
        """
        landmarks = []

        # ── Pose (upper body, indices 0-16) ──────────────────────────
        pose = self._as_landmark_list(results.pose_landmarks)
        pose_upper = pose[:POSE_LANDMARKS]
        landmarks.extend(self._extract_points(pose_upper, POSE_LANDMARKS, "visibility"))

        # ── Left hand ────────────────────────────────────────────────
        left_hand = self._as_landmark_list(results.left_hand_landmarks)
        landmarks.extend(self._extract_points(left_hand, LEFT_HAND_LANDMARKS, "presence"))

        # ── Right hand ───────────────────────────────────────────────
        right_hand = self._as_landmark_list(results.right_hand_landmarks)
        landmarks.extend(self._extract_points(right_hand, RIGHT_HAND_LANDMARKS, "presence"))

        # ── Face landmarks ───────────────────────────────────────────
        face = self._as_landmark_list(results.face_landmarks)

        # Mouth (12 pts)
        landmarks.extend(
            _pick_face_points(face, _MOUTH_RAW_INDICES, MOUTH_LANDMARKS)
        )

        # Left eye (8 pts)
        landmarks.extend(
            _pick_face_points(face, _LEFT_EYE_RAW_INDICES, LEFT_EYE_LANDMARKS)
        )

        # Right eye (8 pts)
        landmarks.extend(
            _pick_face_points(face, _RIGHT_EYE_RAW_INDICES, RIGHT_EYE_LANDMARKS)
        )

        # Nose tip (1 pt)
        if face and _NOSE_TIP_INDEX < len(face):
            lm = face[_NOSE_TIP_INDEX]
            landmarks.append([lm.x, lm.y, lm.z, 1.0])
        else:
            landmarks.append([0.0, 0.0, 0.0, 0.0])

        # Left eyebrow (3 pts)
        landmarks.extend(
            _pick_face_points(face, _LEFT_EYEBROW_RAW_INDICES, LEFT_EYEBROW_LANDMARKS)
        )

        # Right eyebrow (3 pts)
        landmarks.extend(
            _pick_face_points(face, _RIGHT_EYEBROW_RAW_INDICES, RIGHT_EYEBROW_LANDMARKS)
        )

        arr = np.array(landmarks, dtype=np.float32)
        assert arr.shape == (TOTAL_LANDMARKS, 4), (
            f"Expected ({TOTAL_LANDMARKS}, 4), got {arr.shape}"
        )
        return arr

    # ------------------------------------------------------------------
    def close(self):
        if self.landmarker:
            self.landmarker.close()


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _resolve_image_paths(video_field: str, train_dir: Path) -> list[Path]:
    """
    Convert a CSV 'video' field like  '25October_2010.../1/*.png'
    to a sorted list of actual image paths.

    The subfolder component (e.g. '1') is IGNORED because PHOENIX-2014-T
    stores frames directly under the sequence folder:
        dataset/train/<sequence_name>/<frame>.png
    """
    # The field is  <sequence_name>/<subdir>/*.png
    # We only need <sequence_name>.
    parts = Path(video_field).parts  # e.g. ('25October...tagesschau-17', '1', '*.png')
    sequence_name = parts[0]

    sequence_dir = train_dir / sequence_name
    if not sequence_dir.exists():
        logger.warning("Sequence directory not found: %s", sequence_dir)
        return []

    images = sorted(sequence_dir.glob("*.png"))
    if not images:
        # Fallback: also try jpg
        images = sorted(sequence_dir.glob("*.jpg"))
    return images


def read_corpus_csv(csv_path: Path) -> list[dict]:
    """Read the PHOENIX-2014-T corpus CSV (pipe-separated)."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)
    logger.info("Loaded %d sentences from %s", len(rows), csv_path.name)
    return rows


def resolve_data_dir(root_dir: Path, dir_arg: str | None, split: str = "train") -> Path:
    """Resolve the PHOENIX frames directory for `split` (train|dev).

    Precedence:
      1. explicit `dir_arg`
      2. DEFAULT_{SPLIT}_DIR if it exists on the mount
      3. repo dataset/<split>
    """
    if dir_arg:
        return Path(dir_arg)

    if split == "dev" and DEFAULT_DEV_DIR.exists():
        return DEFAULT_DEV_DIR
    if split == "train" and DEFAULT_TRAIN_DIR.exists():
        return DEFAULT_TRAIN_DIR

    return root_dir / "dataset" / split


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_corpus(
    csv_path: Path,
    train_dir: Path,
    output_dir: Path,
    model_path=None,
    num_sentences: int | None = None,
    skip_existing: bool = True,
):
    """
    Process every sentence in the corpus CSV.

    For each sentence a .npy file is saved:
        <output_dir>/<sentence_name>_landmarks.npy
    with shape (num_frames, 94, 4).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_corpus_csv(csv_path)
    if num_sentences is not None:
        rows = rows[:num_sentences]

    stats: dict = defaultdict(int)
    stats["total"] = len(rows)

    for idx, row in enumerate(rows, 1):
        name = row["name"]
        video_field = row["video"]

        output_path = output_dir / f"{name}_landmarks.npy"
        if skip_existing and output_path.exists():
            logger.info("[%d/%d] Skipping (already exists): %s", idx, len(rows), name)
            stats["skipped"] += 1
            continue

        logger.info("[%d/%d] Processing sentence: %s", idx, len(rows), name)

        image_paths = _resolve_image_paths(video_field, train_dir)
        if not image_paths:
            logger.warning("No images found for %s — skipping.", name)
            stats["failed"] += 1
            continue

        logger.info("  Found %d frames in %s", len(image_paths), image_paths[0].parent)

        # Recreate extractor per sentence to avoid any internal state issues
        try:
            extractor = LandmarkExtractor(model_path=model_path)
        except Exception as exc:
            logger.error("Failed to init extractor for %s: %s", name, exc)
            stats["failed"] += 1
            continue

        try:
            landmarks = extractor.extract_landmarks_from_sequence(image_paths)
        finally:
            extractor.close()

        if landmarks is None:
            logger.warning("No landmarks extracted for %s.", name)
            stats["failed"] += 1
            continue

        np.save(output_path, landmarks)
        logger.info(
            "  Saved %s  shape=%s", output_path.name, landmarks.shape
        )
        stats["successful"] += 1
        stats[f"shape_{landmarks.shape}"] += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info("Total sentences : %d", stats["total"])
    logger.info("Successful      : %d", stats["successful"])
    logger.info("Skipped         : %d", stats["skipped"])
    logger.info("Failed          : %d", stats["failed"])
    logger.info("Landmark layout : (num_frames, %d, 4)", TOTAL_LANDMARKS)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract 94 holistic landmarks from PHOENIX-2014-T image sequences"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to holistic_landmarker.task (downloaded automatically if omitted)",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=None,
        help="Maximum number of sentences to process (default: all)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process sentences even if output already exists",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to corpus CSV (default: dataset/PHOENIX-2014-T.train.corpus.csv)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        default="train",
        help="Which corpus split to process (train, dev, or test).",
    )
    parser.add_argument(
        "--data-dir",
        "--train-dir",
        type=str,
        default=None,
        help="Path to the image directory (overrides default for selected split)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save .npy files (default: dataset/landmarks_<split>)",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent

    csv_path = (
        Path(args.csv)
        if args.csv
        else root_dir / "dataset" / f"PHOENIX-2014-T.{args.split}.corpus.csv"
    )
    data_dir = resolve_data_dir(root_dir, args.data_dir, args.split)
    output_dir = (
        Path(args.output_dir) if args.output_dir else root_dir / "dataset" / f"landmarks_{args.split}"
    )

    logger.info("CSV       : %s", csv_path)
    logger.info("Split     : %s", args.split)
    logger.info("Data dir  : %s", data_dir)
    logger.info("Output dir: %s", output_dir)

    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return

    process_corpus(
        csv_path=csv_path,
        train_dir=data_dir,
        output_dir=output_dir,
        model_path=args.model_path,
        num_sentences=args.num_sentences,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()