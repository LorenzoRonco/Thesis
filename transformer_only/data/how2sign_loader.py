"""
how2sign_loader.py

Dataset loader per How2Sign con landmarks MediaPipe pre-estratti.
Landmarks: 527 punti × 4 dimensioni (x, y, z, visibility/presence)

Struttura attesa su disco:
  landmarks_dir/
      <SENTENCE_NAME>.npy        # shape: (T, 527, 4)

  annotations (CSV how2sign_realigned_*.csv):
      VIDEO_ID, VIDEO_NAME, SENTENCE_ID, SENTENCE_NAME,
      START_REALIGNED, END_REALIGNED, SENTENCE
"""

import os
import csv
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────
N_LANDMARKS  = 527
N_DIMS       = 4          # x, y, z, visibility/presence
FEAT_DIM     = N_LANDMARKS * N_DIMS   # 2108

# Landmark group sizes (MediaPipe Holistic ordering)
POSE_LANDMARKS = 17
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21
FACE_LANDMARKS = 468


# ─────────────────────────────────────────────
# Tokenizer minimale (character-level o word-level)
# ─────────────────────────────────────────────
class SentenceTokenizer:
    """
    Tokenizer word-level semplice.
    Costruisce il vocabolario dai dati di training; i token speciali
    sono inseriti agli indici 0-3 per compatibilità con il decoder.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}
        self._build_specials()

    def _build_specials(self):
        for tok in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            idx = len(self.token2idx)
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

    # ── proprietà utili ──────────────────────
    @property
    def pad_id(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token2idx[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)

    # ── costruzione vocabolario ───────────────
    def build_from_sentences(self, sentences: list[str]):
        """Costruisce il vocabolario da una lista di frasi."""
        for sent in sentences:
            for word in self._split(sent):
                if word not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[word] = idx
                    self.idx2token[idx] = word

    @staticmethod
    def _split(sentence: str) -> list[str]:
        return sentence.lower().split()

    # ── encode / decode ───────────────────────
    def encode(self, sentence: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token2idx.get(w, self.unk_id) for w in self._split(sentence)]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        tokens = []
        for i in ids:
            if skip_special_tokens and i in specials:
                continue
            tokens.append(self.idx2token.get(i, self.UNK_TOKEN))
        return " ".join(tokens)

    # ── salvataggio / caricamento ─────────────
    def save(self, path: str | Path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SentenceTokenizer":
        import json
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            tok.token2idx = json.load(f)
        tok.idx2token = {v: k for k, v in tok.token2idx.items()}
        return tok



# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class How2SignDataset(Dataset):
    """
    Dataset How2Sign con landmarks MediaPipe pre-estratti.

    Args:
        csv_path:        percorso al CSV how2sign_realigned_*.csv
        landmarks_dir:   cartella con i file <SENTENCE_NAME>.npy
        tokenizer:       istanza di SentenceTokenizer (già costruita)
        max_src_len:     lunghezza massima sequenza sorgente in frame
                         (None = nessun limite)
        max_tgt_len:     lunghezza massima sequenza target in token
                         (None = nessun limite)
        flatten_landmarks: se True restituisce (T, 2108), altrimenti (T, 527, 4)
    """

    def __init__(
        self,
        csv_path: str | Path,
        landmarks_dir: str | Path,
        tokenizer: SentenceTokenizer,
        max_src_len: Optional[int] = 512,
        max_tgt_len: Optional[int] = 128,
        flatten_landmarks: bool = True,
        pose_weight: float = 1.0,
        hand_weight: float = 1.0,
        face_weight: float = 1.0,
        subset_fraction: Optional[float] = None,
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
        normalize_stats: Optional[dict] = None,
        use_hand_relative_norm: bool = True,
        augment: bool = False,
        thumb_dropout_prob: float = 0.0,
        hand_landmark_dropout_prob: float = 0.0,
        hand_noise_std: float = 0.0,
    ):
        self.landmarks_dir  = Path(landmarks_dir)
        self.tokenizer      = tokenizer
        self.max_src_len    = max_src_len
        self.max_tgt_len    = max_tgt_len
        self.flatten        = flatten_landmarks
        self.pose_weight    = pose_weight
        self.hand_weight    = hand_weight
        self.face_weight    = face_weight
        self.subset_fraction = subset_fraction
        self.max_samples     = max_samples
        self.sample_seed     = sample_seed
        self.use_hand_relative_norm = use_hand_relative_norm
        self.augment = augment
        self.thumb_dropout_prob = thumb_dropout_prob
        self.hand_landmark_dropout_prob = hand_landmark_dropout_prob
        self.hand_noise_std = hand_noise_std
        # Normalization stats (optional): dict with 'mean' and 'std' numpy arrays
        self.normalize_stats = normalize_stats
        if normalize_stats is not None:
            mean = normalize_stats.get("mean")
            std = normalize_stats.get("std")
            if mean is None or std is None:
                raise ValueError("normalize_stats must contain 'mean' and 'std'")
            # store as numpy arrays (will convert to tensor per-sample)
            self._norm_mean = np.asarray(mean, dtype=np.float32)
            self._norm_std = np.asarray(std, dtype=np.float32)
        else:
            self._norm_mean = None
            self._norm_std = None

        self.samples: list[dict] = []
        self._load_csv(csv_path)
        self._apply_subset()

    def _apply_hand_relative_normalization(self, lm: torch.Tensor) -> torch.Tensor:
        """
        Normalize hand landmarks relative to their wrist for each frame.
        Keeps confidence channel unchanged.
        lm shape: (T, 527, 4)
        """
        # Global indices in this project layout:
        # pose [0:17), left hand [17:38), right hand [38:59), face [59:527)
        left_start = POSE_LANDMARKS
        right_start = POSE_LANDMARKS + LEFT_HAND_LANDMARKS

        # Wrist is local index 0 in MediaPipe hand landmarks.
        left_wrist = lm[:, left_start:left_start + 1, :3]    # (T, 1, 3)
        right_wrist = lm[:, right_start:right_start + 1, :3] # (T, 1, 3)

        lm[:, left_start:left_start + LEFT_HAND_LANDMARKS, :3] -= left_wrist
        lm[:, right_start:right_start + RIGHT_HAND_LANDMARKS, :3] -= right_wrist
        return lm

    def _apply_hand_augmentation(self, lm: torch.Tensor) -> torch.Tensor:
        """
        Data augmentation for hands: thumb dropout, random hand-point dropout, gaussian noise.
        Applied only during training dataset.
        lm shape: (T, 527, 4)
        """
        if not self.augment:
            return lm

        left_start = POSE_LANDMARKS
        right_start = POSE_LANDMARKS + LEFT_HAND_LANDMARKS
        hand_ranges = [
            (left_start, left_start + LEFT_HAND_LANDMARKS),
            (right_start, right_start + RIGHT_HAND_LANDMARKS),
        ]

        # Thumb local indices in MediaPipe Hands: 1..4
        thumb_local = [1, 2, 3, 4]

        for hs, he in hand_ranges:
            if self.thumb_dropout_prob > 0.0 and random.random() < self.thumb_dropout_prob:
                for t_idx in thumb_local:
                    gi = hs + t_idx
                    lm[:, gi, :] = 0.0

            if self.hand_landmark_dropout_prob > 0.0:
                keep_mask = torch.rand((he - hs,), device=lm.device) > self.hand_landmark_dropout_prob
                for i, keep in enumerate(keep_mask.tolist()):
                    if not keep:
                        lm[:, hs + i, :] = 0.0

            if self.hand_noise_std > 0.0:
                noise = torch.randn_like(lm[:, hs:he, :3]) * self.hand_noise_std
                lm[:, hs:he, :3] += noise

        return lm

    def _apply_subset(self):
        if not self.samples:
            return
        total = len(self.samples)
        target = total

        if self.subset_fraction is not None and self.subset_fraction < 1.0:
            target = max(1, int(total * self.subset_fraction))

        if self.max_samples is not None:
            target = min(target, self.max_samples)

        if target < total:
            rng = random.Random(self.sample_seed)
            self.samples = rng.sample(self.samples, k=target)
            print(f"[How2SignDataset] Subset: {target}/{total} campioni")

    # ── caricamento CSV ───────────────────────
    def _load_csv(self, csv_path: str | Path):
        csv_path = Path(csv_path)
        skipped = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sentence_name = row["SENTENCE_NAME"].strip()
                sentence_id = row["SENTENCE_ID"].strip()
                npy_path = self.landmarks_dir / f"{sentence_id}_{sentence_name}_landmarks.npy"
                if not npy_path.exists():
                    skipped += 1
                    continue
                sentence = row["SENTENCE"].strip()
                if not sentence:
                    skipped += 1
                    continue
                self.samples.append({
                    "sentence_name": sentence_name,
                    "npy_path":      npy_path,
                    "sentence":      sentence,
                    "video_id":      row.get("VIDEO_ID", "").strip(),
                    "sentence_id":   sentence_id,
                })
        if skipped:
            print(f"[How2SignDataset] Saltati {skipped} campioni "
                  f"(file .npy mancante o frase vuota)")
        print(f"[How2SignDataset] Caricati {len(self.samples)} campioni "
              f"da {csv_path.name}")

    # ── len / getitem ─────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── Carica landmarks ──────────────────
        lm = np.load(sample["npy_path"])        # (T, 527, 4) oppure (T, 2108)
        lm = torch.from_numpy(lm).float()

        # Normalizza shape a (T, 527, 4)
        if lm.ndim == 2:
            # già flatten: (T, 2108) → (T, 527, 4)
            T = lm.shape[0]
            lm = lm.view(T, N_LANDMARKS, N_DIMS)

        # Tronca temporalmente
        if self.max_src_len and lm.shape[0] > self.max_src_len:
            lm = lm[: self.max_src_len]

        # Hand-centric normalization (relative to wrist) to reduce absolute-position dominance.
        if self.use_hand_relative_norm:
            lm = self._apply_hand_relative_normalization(lm)

        # Standardizzazione per-feature (se fornita)
        if self._norm_mean is not None and self._norm_std is not None:
            # _norm_mean/std have shape (FEAT_DIM,) for flattened representation
            # Apply per-frame standardization
            T = lm.shape[0]
            flat = lm.view(T, FEAT_DIM).numpy()
            flat = (flat - self._norm_mean) / (self._norm_std + 1e-6)
            lm = torch.from_numpy(flat).float().view(T, N_LANDMARKS, N_DIMS)

        # Training-only hand augmentation (finger dropout + noise)
        lm = self._apply_hand_augmentation(lm)

        # Pesi per gruppi di landmark (pose / hands / face) -- applicati DOPO standardizzazione
        pose_end = POSE_LANDMARKS
        left_end = pose_end + LEFT_HAND_LANDMARKS
        right_end = left_end + RIGHT_HAND_LANDMARKS
        if self.pose_weight != 1.0:
            lm[:, :pose_end, :] *= self.pose_weight
        if self.hand_weight != 1.0:
            lm[:, pose_end:left_end, :] *= self.hand_weight
            lm[:, left_end:right_end, :] *= self.hand_weight
        if self.face_weight != 1.0:
            lm[:, right_end:, :] *= self.face_weight

        # Output shape sorgente
        if self.flatten:
            T = lm.shape[0]
            lm = lm.view(T, FEAT_DIM)           # (T, 2108)

        # ── Tokenizza target ──────────────────
        tgt_ids = self.tokenizer.encode(sample["sentence"], add_special_tokens=True)
        if self.max_tgt_len:
            # Tronca mantenendo <eos> finale
            if len(tgt_ids) > self.max_tgt_len:
                tgt_ids = tgt_ids[: self.max_tgt_len - 1] + [self.tokenizer.eos_id]
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

        return {
            "src":            lm,           # (T, 2108) o (T, 527, 4)
            "tgt":            tgt_ids,      # (L,)
            "sentence_name":  sample["sentence_name"],
            "sentence":       sample["sentence"],
            "sentence_id":    sample["sentence_id"],
            "npy_path":       str(sample["npy_path"]),
        }


# ─────────────────────────────────────────────
# Collate function (padding)
# ─────────────────────────────────────────────
def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    """
    Padda sorgente e target alla lunghezza massima del batch.
    Restituisce:
        src:          (B, T_max, 2108)
        src_key_padding_mask:  (B, T_max)  True dove è padding
        tgt:          (B, L_max)
        tgt_key_padding_mask:  (B, L_max)
        tgt_input:    (B, L_max-1)   — input al decoder (senza <eos>)
        tgt_output:   (B, L_max-1)   — target per la loss (senza <bos>)
    """
    srcs      = [item["src"] for item in batch]
    tgts      = [item["tgt"] for item in batch]
    sentences = [item["sentence"] for item in batch]
    names     = [item["sentence_name"] for item in batch]
    sent_ids  = [item.get("sentence_id", "") for item in batch]
    npy_paths = [item.get("npy_path", "") for item in batch]

    src_lens = [s.shape[0] for s in srcs]
    tgt_lens = [t.shape[0] for t in tgts]

    T_max = max(src_lens)
    L_max = max(tgt_lens)
    B     = len(batch)
    D     = srcs[0].shape[-1]

    # Sorgente
    src_padded = torch.zeros(B, T_max, D)
    src_mask   = torch.ones(B, T_max, dtype=torch.bool)   # True = padding
    for i, (s, l) in enumerate(zip(srcs, src_lens)):
        src_padded[i, :l] = s
        src_mask[i, :l]   = False

    # Target
    tgt_padded = torch.full((B, L_max), fill_value=pad_id, dtype=torch.long)
    tgt_mask   = torch.ones(B, L_max, dtype=torch.bool)
    for i, (t, l) in enumerate(zip(tgts, tgt_lens)):
        tgt_padded[i, :l] = t
        tgt_mask[i, :l]   = False

    # Shift per teacher forcing
    tgt_input  = tgt_padded[:, :-1]   # <bos> ... (senza ultimo token)
    tgt_output = tgt_padded[:, 1:]    # ... <eos> (senza <bos>)
    tgt_in_mask  = tgt_mask[:, :-1]
    tgt_out_mask = tgt_mask[:, 1:]

    return {
        "src":                    src_padded,    # (B, T, D)
        "src_key_padding_mask":   src_mask,      # (B, T)
        "tgt":                    tgt_padded,    # (B, L)
        "tgt_key_padding_mask":   tgt_mask,      # (B, L)
        "tgt_input":              tgt_input,     # (B, L-1)
        "tgt_output":             tgt_output,    # (B, L-1)
        "tgt_in_key_padding_mask":  tgt_in_mask,
        "tgt_out_key_padding_mask": tgt_out_mask,
        "sentences":  sentences,
        "names":      names,
        "sentence_ids": sent_ids,
        "npy_paths":  npy_paths,
        "src_lens":   torch.tensor(src_lens),
        "tgt_lens":   torch.tensor(tgt_lens),
    }


# ─────────────────────────────────────────────
# Bucketing sampler (riduce padding)
# ─────────────────────────────────────────────
class BucketingBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_size: int = 200,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, bucket_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda i: self.lengths[i])

        buckets = [indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)

        batch: list[int] = []
        for bucket in buckets:
            for idx in bucket:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

        self._epoch += 1

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


# ─────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────
def build_tokenizer(
    train_csv: str | Path,
    save_path: Optional[str | Path] = None,
) -> SentenceTokenizer:
    """
    Costruisce e (opzionalmente) salva il tokenizer dal CSV di training.
    """
    tokenizer = SentenceTokenizer()
    sentences = []
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s = row.get("SENTENCE", "").strip()
            if s:
                sentences.append(s)
    tokenizer.build_from_sentences(sentences)
    print(f"[build_tokenizer] Vocabolario: {tokenizer.vocab_size} token "
          f"da {len(sentences)} frasi")
    if save_path:
        tokenizer.save(save_path)
        print(f"[build_tokenizer] Salvato in {save_path}")
    return tokenizer


def build_dataloaders(
    train_csv:     str | Path,
    val_csv:       str | Path,
    train_landmarks_dir: str | Path,
    val_landmarks_dir: str | Path,
    tokenizer:     SentenceTokenizer,
    batch_size:    int = 32,
    num_workers:   int = 4,
    max_src_len:   int = 512,
    max_tgt_len:   int = 128,
    pose_weight:   float = 1.0,
    hand_weight:   float = 1.0,
    face_weight:   float = 1.0,
    pin_memory:    bool = True,
    test_csv:      Optional[str | Path] = None,
    test_landmarks_dir: Optional[str | Path] = None,
    train_subset_fraction: Optional[float] = None,
    val_subset_fraction: Optional[float] = None,
    train_max_samples: Optional[int] = None,
    val_max_samples: Optional[int] = None,
    subset_seed: int = 42,
    use_bucketing: bool = False,
    bucket_size: int = 200,
    drop_last: bool = False,
    use_hand_relative_norm: bool = True,
    thumb_dropout_prob: float = 0.0,
    hand_landmark_dropout_prob: float = 0.0,
    hand_noise_std: float = 0.0,
) -> dict[str, DataLoader]:
    """
    Restituisce un dizionario {"train": ..., "val": ..., "test": ...}.
    """
    from functools import partial
    _collate = partial(collate_fn, pad_id=tokenizer.pad_id)

    def _compute_src_lengths(ds: How2SignDataset) -> list[int]:
        lengths: list[int] = []
        for sample in ds.samples:
            npy_path = sample["npy_path"]
            try:
                arr = np.load(npy_path, mmap_mode="r")
                seq_len = int(arr.shape[0])
            except Exception:
                seq_len = 0
            if max_src_len:
                seq_len = min(seq_len, max_src_len)
            lengths.append(seq_len)
        return lengths

    def _compute_landmark_stats(samples_list, lm_dir):
        """
        Compute per-feature mean/std across the dataset (flattened feature dim).
        Returns numpy arrays of shape (FEAT_DIM,)
        """
        ssum = None
        ssq = None
        count = 0
        for sample in samples_list:
            npy_path = Path(sample["npy_path"]) if isinstance(sample, dict) else Path(sample)
            try:
                arr = np.load(npy_path, mmap_mode="r")
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0], -1)
                data = arr
                if ssum is None:
                    ssum = data.sum(axis=0, dtype=float)
                    ssq = (data ** 2).sum(axis=0, dtype=float)
                else:
                    ssum += data.sum(axis=0, dtype=float)
                    ssq += (data ** 2).sum(axis=0, dtype=float)
                count += data.shape[0]
            except Exception:
                continue
        if count == 0:
            raise RuntimeError("Unable to compute landmark stats: no frames found")
        mean = ssum / count
        var = (ssq / count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)

    def _make_loader(csv_path, shuffle, lm_dir, subset_fraction, max_samples, normalize_stats=None):
        ds = How2SignDataset(
            csv_path=csv_path,
            landmarks_dir=lm_dir,
            tokenizer=tokenizer,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            pose_weight=pose_weight,
            hand_weight=hand_weight,
            face_weight=face_weight,
            normalize_stats=normalize_stats,
            use_hand_relative_norm=use_hand_relative_norm,
            augment=shuffle,
            thumb_dropout_prob=thumb_dropout_prob,
            hand_landmark_dropout_prob=hand_landmark_dropout_prob,
            hand_noise_std=hand_noise_std,
            subset_fraction=subset_fraction,
            max_samples=max_samples,
            sample_seed=subset_seed,
        )
        if use_bucketing and shuffle:
            lengths = _compute_src_lengths(ds)
            batch_sampler = BucketingBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                bucket_size=bucket_size,
                shuffle=True,
                drop_last=drop_last,
                seed=subset_seed,
            )
            return DataLoader(
                ds,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=_collate,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    # Compute normalization stats from the training set and pass to datasets
    normalize_stats = None
    try:
        train_samples = []
        with open(train_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sentence_name = row["SENTENCE_NAME"].strip()
                sentence_id = row["SENTENCE_ID"].strip()
                npy_path = Path(train_landmarks_dir) / f"{sentence_id}_{sentence_name}_landmarks.npy"
                if npy_path.exists():
                    train_samples.append({"npy_path": str(npy_path)})
        if len(train_samples) > 0:
            print(f"[build_dataloaders] Computing landmark normalization stats from {len(train_samples)} files...")
            mean, std = _compute_landmark_stats(train_samples, train_landmarks_dir)
            normalize_stats = {"mean": mean, "std": std}
            print("[build_dataloaders] Landmark stats computed")
    except Exception as e:
        print(f"[build_dataloaders] Warning: could not compute normalize stats: {e}")

    loaders = {
        "train": _make_loader(
            train_csv,
            shuffle=True,
            lm_dir=train_landmarks_dir,
            subset_fraction=train_subset_fraction,
            max_samples=train_max_samples,
            normalize_stats=normalize_stats,
        ),
        "val":   _make_loader(
            val_csv,
            shuffle=False,
            lm_dir=val_landmarks_dir,
            subset_fraction=val_subset_fraction,
            max_samples=val_max_samples,
            normalize_stats=normalize_stats,
        ),
    }
    if test_csv:
        loaders["test"] = _make_loader(
            test_csv,
            shuffle=False,
            lm_dir=test_landmarks_dir or val_landmarks_dir,
            subset_fraction=val_subset_fraction,
            max_samples=val_max_samples,
        )

    return loaders