# Optimization and Training Techniques

This document summarises the optimization, stabilisation and reproducibility techniques used in the project, with pointers to the code and to the configuration keys that control them.

Main reference files: [`transformer_only/train.py`](../transformer_only/train.py) (stage 1) and [`transformer_only/train_stage2.py`](../transformer_only/train_stage2.py) (stage 2).

---

## Optimizer: `AdamW`

Adam with decoupled weight decay (`torch.optim.AdamW`).

- **Why:** adaptive per-parameter learning rates plus weight decay that is decoupled from the gradient moments, a standard choice for Transformers.
- **Where:** `train.py` and `train_stage2.py`.
- **Config:** `lr`, `weight_decay`.

## Weight decay (decoupled)

Regularisation term passed to `AdamW` to limit overfitting.

- **Config:** `weight_decay`.

## Warmup + cosine learning-rate schedule

Custom `WarmupCosineScheduler`: linear warmup for `warmup_steps`, then cosine decay down to `min_lr_ratio * base_lr`.

- **Where:** class `WarmupCosineScheduler` in `train.py` (re-used by `train_stage2.py`).
- **Config:** `warmup_steps`, `epochs` (used to compute `total_steps`), `min_lr_ratio`.

## Mixed precision (AMP)

`torch.cuda.amp.autocast` together with `GradScaler`.

- **Why:** faster training and lower GPU memory use, while `GradScaler` keeps the numerics stable.
- **Where:** `train.py` and `train_stage2.py`.
- **Config:** `use_amp`.

## Gradient clipping

`torch.nn.utils.clip_grad_norm_`, applied before each optimizer step to prevent exploding gradients.

- **Config:** `clip_norm`.

## Label smoothing

Smoothed cross-entropy (default `0.1`).

- **Why:** better generalisation and less over-confident predictions.
- **Config:** `label_smoothing`.

## Composite loss: CE + CTC

Cross-entropy on the decoder output combined with an auxiliary CTC loss on the encoder output, weighted by `lambda_ctc`.

- **Why:** the CTC term improves alignment between frames and tokens and reduces insertions.
- **Where:** `_compute_loss` in `train.py`.
- **Config:** `lambda_ctc`.

## Landmark data augmentation

Gaussian noise, frame dropout, speed perturbation and horizontal flip applied to landmark sequences for robustness.

- **Where:** `augment_landmarks` in `train.py`.
- **Config:** `aug_noise_std`, `aug_frame_drop_prob`, `aug_speed_min` / `aug_speed_max`, `aug_hflip_prob`.

## Reproducibility

- Global seed for `random`, `numpy`, `torch` and `torch.cuda`.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
- DataLoader workers seeded through a `torch.Generator` and a `worker_init_fn`.
- **Where:** `set_global_seed` and `_make_worker_init_fn` in `train.py` and `train_stage2.py`.
- **Config / CLI:** `seed` (`--seed`).

## Batching and bucketing

Length bucketing groups sequences of similar length to reduce padding and stabilise updates.

- **Config:** `use_bucketing`, `bucket_size`.

## Selective checkpointing

- `best.pt` is saved whenever validation BLEU-4 improves; `last.pt` is saved at the end of training.
- Checkpoints include `optimizer.state_dict()`, `scheduler.state_dict()` and `scaler.state_dict()`, so training can be resumed exactly.

---

## Quick reference

| Technique | Config keys |
|---|---|
| AdamW + weight decay | `lr`, `weight_decay` |
| Warmup + cosine | `warmup_steps`, `epochs`, `min_lr_ratio` |
| AMP | `use_amp` |
| Gradient clipping | `clip_norm` |
| Label smoothing | `label_smoothing` |
| CE + CTC | `lambda_ctc` |
| Augmentation | `aug_noise_std`, `aug_frame_drop_prob`, `aug_speed_min`, `aug_speed_max`, `aug_hflip_prob` |
| Reproducibility | `seed` |
| Bucketing | `use_bucketing`, `bucket_size` |

The current default values are listed in the main [README](../README.md#155-training-configuration-and-default-values).
