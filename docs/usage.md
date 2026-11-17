# Usage: Training and Evaluation Commands

All commands are run from the repository root. The examples use the default PHOENIX-2014-T paths.

The system has two stages:

- **Stage 1** – `SignLanguageTransformer`: landmarks → glosses
- **Stage 2** – `GlossToTextTransformer`: glosses → German translation

---

## 1. Landmark extraction

```bash
python scripts/extract_landmarks.py --split train --output-dir dataset/landmarks_train
python scripts/extract_landmarks.py --split dev   --output-dir dataset/landmarks_dev
python scripts/extract_landmarks.py --split test  --output-dir dataset/landmarks_test
```

Optional checks:

```bash
python scripts/inspect_landmarks.py --dir dataset/landmarks_dev   # statistics / validation of the .npy files
python scripts/visualize_landmarks.py --split dev
```

## 2. Training

```bash
# Stage 1 (landmarks → glosses), output in outputs/phoenix_run1
python -m transformer_only.train

# Stage 2 (glosses → text), output in outputs/phoenix_stage2
python -m transformer_only.train_stage2 --stage1_dir outputs/phoenix_run1
```

Both scripts accept CLI overrides such as `--epochs`, `--lr`, `--batch_size` and `--seed` (see `--help`).

## 3. Evaluation

[`transformer_only/evaluate.py`](../transformer_only/evaluate.py) supports three modes.

### Stage 1 only (landmarks → glosses, metric: WER)

```bash
python -m transformer_only.evaluate --mode stage1 \
  --checkpoint outputs/phoenix_run1/best.pt \
  --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \
  --landmarks_dir dataset/landmarks_dev \
  --train_csv dataset/PHOENIX-2014-T.train.corpus.csv \
  --train_landmarks_dir dataset/landmarks_train \
  --stage1_dir outputs/phoenix_run1
```

### Stage 2 only with gold glosses (upper bound, metric: BLEU-4)

```bash
python -m transformer_only.evaluate --mode stage2 \
  --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \
  --stage1_dir outputs/phoenix_run1 \
  --stage2_dir outputs/phoenix_stage2
```

### Full pipeline (landmarks → glosses → text, metric: BLEU-4)

This is the main end-to-end result.

```bash
python -m transformer_only.evaluate --mode pipeline \
  --checkpoint outputs/phoenix_run1/best.pt \
  --stage2_checkpoint outputs/phoenix_stage2/best.pt \
  --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \
  --landmarks_dir dataset/landmarks_dev \
  --train_csv dataset/PHOENIX-2014-T.train.corpus.csv \
  --train_landmarks_dir dataset/landmarks_train \
  --stage1_dir outputs/phoenix_run1 \
  --stage2_dir outputs/phoenix_stage2
```

## 4. Diagnostics

```bash
# CTC vs. decoder greedy decoding on a stage-1 checkpoint
python transformer_only/ctc_diagnostics_standalone.py \
  --checkpoint outputs/phoenix_run1/best.pt \
  --config outputs/phoenix_run1/config.json \
  --tokenizer_path outputs/phoenix_run1/tokenizer.json \
  --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \
  --landmarks_dir dataset/landmarks_dev
```

For cross-attention heatmaps see [attention_debugging.md](attention_debugging.md).
