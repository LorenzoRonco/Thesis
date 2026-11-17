# Sign Language Translation from MediaPipe Holistic Landmarks

## Abstract

This project investigates a landmark-based pipeline for sign language translation from video sequences. Rather than feeding raw RGB frames into a sequence model, the approach first extracts body, hand, and face landmarks using MediaPipe Holistic Landmarker, then models the temporal dynamics of these landmarks with a Transformer encoder-decoder architecture. The resulting system is designed to map a sequence of visual sign articulations to a textual sentence in the target language.

The current implementation is built around the PHOENIX-2014-T dataset and adopts a feature representation in which each frame is encoded as a compact landmark vector. This design reduces the dimensionality of the input while preserving the geometric and temporal structure essential for sign recognition and translation.

---

## 1. Problem Definition

Sign language translation is a sequence-to-sequence problem in which a continuous stream of visual signals must be translated into a textual output. In the present system, the visual signal is represented by landmark coordinates extracted from video frames. This formulation transforms the task into learning a mapping:

$$
X = (x_1, x_2, \dots, x_T) \rightarrow Y = (y_1, y_2, \dots, y_L)
$$

where:

- $X$ is the temporal sequence of landmarks for a sign sentence;
- $T$ is the number of video frames;
- $Y$ is the target textual sequence;
- $L$ is the sentence length in tokens.

The model learns to align the temporal structure of the sign stream with the tokenized textual target.

---

## 2. Method Overview

The architecture is composed of four main stages:

1. acquisition of image frames from the PHOENIX-2014-T corpus;
2. extraction of holistic landmarks using MediaPipe;
3. transformation of landmark sequences into a dataset suitable for temporal modeling;
4. learning a Transformer encoder-decoder for sign-to-text generation.

This pipeline is designed to be modular: feature extraction is separated from sequence modeling, which allows reuse, visualization, and controlled preprocessing.

---

## 3. System Architecture

### 3.1 Overall Pipeline

```text
PHOENIX-2014-T image sequences
        ↓
MediaPipe Holistic Landmarker
        ↓
per sample: (num_frames, 94, 4)
        ↓
.npy landmark serialization
        ↓
Phoenix dataset loader + normalization
        ↓
SentenceTokenizer
        ↓
Transformer encoder-decoder
        ↓
output text generation / BLEU / ROUGE evaluation
```

### 3.2 Main Components

- [scripts/extract_landmarks.py](scripts/extract_landmarks.py): extracts per-frame landmark vectors from image sequences.
- [transformer_only/data/phoenix_loader.py](transformer_only/data/phoenix_loader.py): loads the precomputed landmarks, applies normalization, and builds batches for training.
- [transformer_only/models/transformer.py](transformer_only/models/transformer.py): implements the Transformer-based sequence model.
- [transformer_only/train.py](transformer_only/train.py): manages training, validation, scheduling, checkpointing, and evaluation.
- [transformer_only/evaluate.py](transformer_only/evaluate.py): measures translation quality with standard metrics.
- [scripts/visualize_landmarks.py](scripts/visualize_landmarks.py): overlays landmark projections onto the original images for visual inspection.

---

## 4. Dataset and Corpus Structure

The system uses the PHOENIX-2014-T corpus, whose metadata is stored in CSV files containing sentence-level annotations. Each record includes identifiers for the sentence and the associated video sequence, as well as textual reference labels.

The relevant fields are:

- `name`: unique sentence identifier;
- `video`: path to the associated frame sequence;
- `orth`: gloss or lexical annotation;
- `translation`: target textual sentence in German.

The loader reads the corpus files and resolves each sample to its precomputed landmark representation stored in a separate `.npy` file.

---

## 5. Landmark Extraction

### 5.1 MediaPipe Holistic Landmarking

The feature extraction stage is implemented in [scripts/extract_landmarks.py](scripts/extract_landmarks.py). For each image in a sentence sequence, the system invokes MediaPipe Holistic Landmarker in IMAGE mode and extracts pose, hand, and face landmarks.

The resulting per-frame representation is a compact subset of the full MediaPipe body model, designed to preserve the most informative components for sign articulation while limiting redundancy.

### 5.2 Landmark Layout

Each frame is represented by a tensor of shape:

$$
(94, 4)
$$

where each landmark is described by:

- $x$ coordinate;
- $y$ coordinate;
- $z$ coordinate;
- confidence value.

The total number of landmarks is 94, organized as follows:

- pose / upper body: 17
- left hand: 21
- right hand: 21
- mouth: 12
- left eye: 8
- right eye: 8
- nose tip: 1
- left eyebrow: 3
- right eyebrow: 3

Hence:

$$
17 + 21 + 21 + 12 + 8 + 8 + 1 + 3 + 3 = 94
$$

### 5.3 Temporal Representation

For an entire sentence sampled over $T$ frames, the resulting feature tensor is:

$$
(T, 94, 4)
$$

This structure preserves the temporal axis required for sequence modeling while maintaining a compact feature footprint. When flattened for the Transformer input, the feature dimension becomes:

$$
94 \times 4 = 376
$$

Thus the model operates on sequences of shape:

$$
(B, T, 376)
$$

where $B$ is the batch size.

### 5.4 Face Landmark Reduction Strategy

The implementation does not use the full 468-point face mesh directly. Instead, it aggregates and subsamples face regions into a reduced set of landmarks for mouth, eyes, nose, and eyebrows. This is intentional and serves two key purposes:

1. lower redundancy in the visual representation;
2. better compatibility with temporal sequence models, which benefit from compact and robust features.

The mouth is represented using a reduced and averaged subset of lip contour points, while the eye and eyebrow regions are similarly compressed. This yields a smaller but discriminative set of facial landmarks for sign articulation analysis.

---

## 6. Serialization and Storage

Each sentence is saved as a `.npy` file, with one file per example. The default storage convention is:

```text
<name>_landmarks.npy
```

For example:

```text
dataset/landmarks_train/
    01April_2010_Thursday_heute-6697_landmarks.npy
```

Internally, the file contents are a NumPy array with shape:

```python
(num_frames, 94, 4)
```

This format is efficient for training because it decouples the expensive landmark extraction step from the learning stage. It also allows reproducible experimentation and reuse across model variants.

When a frame cannot be reliably processed, the extractor inserts a zero-filled landmark vector:

```python
np.zeros((94, 4), dtype=np.float32)
```

This preserves alignment across the sequence and prevents temporal misalignment from missing detections.

---

## 7. Preprocessing and Dataset Loading

The loader in [transformer_only/data/phoenix_loader.py](transformer_only/data/phoenix_loader.py) is responsible for converting raw landmark arrays into training-ready tensors.

The preprocessing pipeline includes:

- loading the PHOENIX CSV annotations;
- matching each sample to its `.npy` landmark sequence;
- clipping coordinate values to a valid range;
- normalizing x/y/z values using statistics computed on the training split;
- preserving the confidence channel, which remains in the range $[0,1]$;
- padding sequences within each batch;
- tokenizing the target text into a word-level vocabulary.

The configuration also supports task-specific weighting of landmark groups:

```python
pose_weight = 0.8
hand_weight = 1.5
face_weight = 0.5
```

This weighting reflects the empirical importance of manual articulation in sign language, where hand configuration and motion are often more discriminative than body pose alone.

---

## 8. Text Tokenization

The target sentence is not used as raw text during training. Instead, it is converted into a discrete token sequence by `SentenceTokenizer` implemented in [transformer_only/data/phoenix_loader.py](transformer_only/data/phoenix_loader.py).

The tokenizer is word-based and includes special tokens:

- `<pad>`
- `<bos>`
- `<eos>`
- `<unk>`

This matches the autoregressive decoding setup used by the Transformer, where the model learns to generate a target token sequence conditioned on the source landmark sequence.

---

## 9. Model Architecture

The sequence model is implemented in [transformer_only/models/transformer.py](transformer_only/models/transformer.py) and follows an encoder-decoder Transformer design.

### 9.1 Source Encoding

The source input is a temporal sequence of landmark features:

$$
(B, T, 376)
$$

where:

- $B$ is the batch size,
- $T$ is the sequence length in frames,
- $376$ is the per-frame feature dimension.

The source embedding step projects these landmark vectors into the Transformer hidden space while injecting temporal positional information.

### 9.2 Encoder-Decoder Structure

The model uses standard self-attention and cross-attention operations:

- the encoder processes the landmark sequence and builds a contextual representation;
- the decoder generates the textual target token-by-token;
- cross-attention links source frames to target tokens during generation.

The design is intended to capture long-range temporal dependencies across sign articulation while also learning textual correspondences.

### 9.3 Attention and Interpretability

The implementation also exposes attention-related mechanisms for visualization and analysis. This is valuable for examining which temporal regions of the sign sequence are most influential in producing a particular token or phrase.

---

## 10. Training Procedure

Training is handled by [transformer_only/train.py](transformer_only/train.py). The training loop includes:

- dataset construction;
- batching and padding;
- source-target alignment;
- loss computation;
- learning-rate scheduling;
- validation monitoring;
- checkpoint saving;
- evaluation over validation data.

The script also applies deterministic settings for reproducibility, including fixed random seeds and deterministic behavior for PyTorch and CUDA-related operations.

### 10.1 Losses

The model supports a combination of sequence modeling objectives, including cross-entropy over target tokens and optional CTC-based supervision in selected configurations.

### 10.2 Checkpointing

Training artifacts are stored in the output directories, including:

```text
outputs/phoenix_run1/model_last.pt
outputs/phoenix_run1/best.pt
```

These checkpoints store model weights, optimizer state, scheduler state, and associated metadata.

---

## 11. Evaluation Metrics

Evaluation is performed with standard sign language translation metrics.

- [transformer_only/bleu.py](transformer_only/bleu.py): BLEU score computation;
- [transformer_only/rouge.py](transformer_only/rouge.py): ROUGE evaluation;
- [transformer_only/evaluate.py](transformer_only/evaluate.py): end-to-end evaluation pipeline.

These metrics provide quantitative assessments of the generated textual translations against the reference sentences.

---

## 12. Execution Pipeline

### 12.1 Landmark Extraction

```bash
python scripts/extract_landmarks.py --split train --output-dir dataset/landmarks_train
python scripts/extract_landmarks.py --split dev --output-dir dataset/landmarks_dev
python scripts/extract_landmarks.py --split test --output-dir dataset/landmarks_test
```

### 12.2 Model Training

```bash
python transformer_only/train.py
```

### 12.3 Evaluation

```bash
python transformer_only/evaluate.py --checkpoint outputs/phoenix_run1/best.pt
```

---

## 13. Generated Artifacts

The project produces several persistent artifacts:

- precomputed landmark files in `.npy` format;
- tokenizer metadata (`tokenizer.json` in some training runs);
- training configuration files (`config.json`);
- model checkpoints (`.pt` files);
- validation and test metric reports;
- landmark visualizations for inspection.

---

## 14. Discussion

This approach is grounded in a hybrid representation: geometric landmark sequences replace raw video frames as the primary source representation, while Transformer-based sequence modeling provides the mechanism for mapping spatiotemporal motion into language. Compared with direct image-based modeling, this design offers a more compact and interpretable representation, while preserving the most relevant information for sign articulation.

The landmark representation intentionally prioritizes body pose, hand motion, and facial cues, which are central to sign language communication. The resulting pipeline is therefore well suited to sequence-to-sequence translation tasks where motion continuity and temporal alignment are essential.

---

## 15. Key Files

- [scripts/extract_landmarks.py](scripts/extract_landmarks.py)
- [transformer_only/data/phoenix_loader.py](transformer_only/data/phoenix_loader.py)
- [transformer_only/models/transformer.py](transformer_only/models/transformer.py)
- [transformer_only/train.py](transformer_only/train.py)
- [transformer_only/evaluate.py](transformer_only/evaluate.py)
- [scripts/visualize_landmarks.py](scripts/visualize_landmarks.py)

---

## 15.5 Training Configuration and Default Values

The main training configuration is defined in [transformer_only/train.py](transformer_only/train.py). The current values used by the project are listed below.

### 15.5.1 Dataset and runtime configuration

- `train_csv`: `dataset/PHOENIX-2014-T.train.corpus.csv`
- `val_csv`: `dataset/PHOENIX-2014-T.dev.corpus.csv`
- `train_landmarks_dir`: `dataset/landmarks_train`
- `val_landmarks_dir`: `dataset/landmarks_dev`
- `test_csv`: `None`
- `test_landmarks_dir`: `None`
- `run_test`: `False`
- `output_dir`: `outputs/phoenix_run1`
- `target_field`: `orth` (gloss) by default; can be switched to `translation`
- `device`: `cuda` if available, otherwise `cpu`
- `use_amp`: `True`
- `seed`: `42`

### 15.5.2 Model architecture

- `feat_dim`: `376` (94 landmarks × 4 features)
- `d_model`: `512`
- `nhead`: `8`
- `num_enc_layers`: `6`
- `num_dec_layers`: `3`
- `dim_feedforward`: `2048`
- `dropout`: `0.3`
- `label_smoothing`: `0.1`
- `max_src_len`: `256`
- `max_tgt_len`: `128`
- `src_embedding_type`: `temporal_cnn`
- `temporal_kernel_size`: `5`
- `temporal_blocks`: `3`

### 15.5.3 Training optimization

- `epochs`: `275`
- `batch_size`: `32`
- `lr`: `5e-4`
- `weight_decay`: `1e-4`
- `clip_norm`: `1.0`
- `warmup_steps`: `4000`
- `num_workers`: `4`

### 15.5.4 Data augmentation

- `aug_noise_std`: `0.005`
- `aug_speed_min`: `0.8`
- `aug_speed_max`: `1.2`
- `aug_frame_drop_prob`: `0.05`
- `aug_hflip_prob`: `0.5`

These augmentation parameters simulate small landmark jitter, temporal speed variation, random frame dropping, and horizontal mirroring of the sign sequence.

### 15.5.5 Landmark weighting

- `pose_weight`: `0.8`
- `hand_weight`: `1.5`
- `face_weight`: `0.5`
- `use_hand_relative_norm`: `False`

These weights adjust the relative contribution of body pose, hands, and face during preprocessing and training. The hands are intentionally weighted more strongly because they usually carry the most informative lexical content in sign language.

### 15.5.6 Tokenizer and vocabulary

- `tokenizer_min_freq`: `1`

This sets the minimum frequency required for a token to remain in the vocabulary.

### 15.5.7 Auxiliary losses

- `gal_weight`: `10.0`
- `gal_sigma`: `0.25`
- `lambda_ctc`: `0.7`

These values control the contribution of auxiliary objectives used to stabilize training and improve sequence alignment.

### 15.5.8 Dataset subsampling

- `train_subset_fraction`: `None`
- `val_subset_fraction`: `None`
- `train_max_samples`: `None`
- `val_max_samples`: `None`
- `subset_seed`: `42`

When these values are left as `None`, the full dataset is used.

### 15.5.9 Bucketing and batching

- `use_bucketing`: `True`
- `bucket_size`: `200`
- `drop_last`: `False`

Bucketization groups similar-length sequences together to reduce padding waste.

### 15.5.10 Logging and validation

- `log_interval`: `50`
- `val_interval_early`: `5`
- `val_interval_late`: `1`
- `early_phase_epochs`: `100`
- `debug_print_batch`: `False`
- `debug_max_items`: `4`
- `attention_viz_every`: `None`
- `attention_viz_max_batches`: `2`
- `attention_viz_save`: `False`

### 15.5.11 CLI override parameters

The following command-line arguments can override the default configuration:

- `--tokenizer_min_freq`
- `--epochs`
- `--lr`
- `--batch_size`
- `--dropout`
- `--target_field`
- `--train_csv`
- `--val_csv`
- `--train_landmarks_dir`
- `--val_landmarks_dir`
- `--test_csv`
- `--test_landmarks_dir`
- `--run_test`
- `--seed`

This makes the training pipeline fully configurable without editing the source code directly.

---

## 16. Conclusion

The implemented system constitutes a compact yet expressive sign language translation pipeline built on MediaPipe landmark extraction and Transformer sequence modeling. It converts raw visual motion into a structured, low-dimensional representation and learns a mapping from spatiotemporal sign dynamics to tokenized textual output. This design is both computationally tractable and conceptually aligned with the structured nature of sign language communication.
