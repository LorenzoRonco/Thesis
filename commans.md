# Solo Modello 1 (landmark → gloss)
python -m transformer_only.evaluate --mode stage1 \
  --checkpoint outputs/phoenix_run1/best.pt \
  --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \
  --landmarks_dir dataset/landmarks_dev \
  --train_csv dataset/PHOENIX-2014-T.train.corpus.csv \
  --train_landmarks_dir dataset/landmarks_train \
  --stage1_dir outputs/phoenix_run1

# Solo Modello 2 con gloss gold (upper bound)
python -m transformer_only.evaluate --mode stage2 \
  --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \
  --stage1_dir outputs/phoenix_run1 \
  --stage2_dir outputs/phoenix_stage2

# Pipeline completa (quello che ti interessa di più)
python -m transformer_only.evaluate --mode pipeline \
  --checkpoint outputs/phoenix_run1/best.pt \
  --stage2_checkpoint outputs/phoenix_stage2/best.pt \
  --val_csv dataset/PHOENIX-2014-T.dev.corpus.csv \
  --landmarks_dir dataset/landmarks_dev \
  --train_csv dataset/PHOENIX-2014-T.train.corpus.csv \
  --train_landmarks_dir dataset/landmarks_train \
  --stage1_dir outputs/phoenix_run1 \
  --stage2_dir outputs/phoenix_stage2