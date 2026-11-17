"""
phoenix_loader.py

Dataset loader per PHOENIX-2014-T con landmarks MediaPipe pre-estratti.

Struttura su disco:
  landmarks_dir/
      <name>_landmarks.npy        # shape: (T, 94, 4)

  CSV (pipe-separated, con header):
      name | video | start | end | speaker | orth | translation

Landmark layout (94 punti × 4 dimensioni [x, y, z, conf]):
  [ 0-16]  pose upper body (17 pt)
  [17-37]  left hand       (21 pt)
  [38-58]  right hand      (21 pt)
  [59-70]  mouth           (12 pt)
  [71-78]  left eye        ( 8 pt)
  [79-86]  right eye       ( 8 pt)
  [87]     nose tip        ( 1 pt)
  [88-90]  left eyebrow    ( 3 pt)
  [91-93]  right eyebrow   ( 3 pt)

Normalizzazione:
  - x, y, z sono già in [0, 1] come output di MediaPipe (normalizzati per
    dimensione immagine). Si applica una clip a [0, 1] per sicurezza e poi
    una standardizzazione per-feature (media/std calcolata sul training set).
  - Il canale confidence (indice 3) non viene standardizzato: è già in [0, 1].
  - Normalizzazione relativa alle mani (wrist-relative) applicabile opzionalmente.
"""

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────
N_LANDMARKS = 94
N_DIMS      = 4                          # x, y, z, confidence
FEAT_DIM    = N_LANDMARKS * N_DIMS       # 376

# Slices per gruppo di landmark
POSE_LANDMARKS       = 17
LEFT_HAND_LANDMARKS  = 21
RIGHT_HAND_LANDMARKS = 21
FACE_LANDMARKS       = 35               # mouth+eyes+nose+eyebrows = 12+8+8+1+3+3

POSE_SLICE       = slice(0,  17)
LEFT_HAND_SLICE  = slice(17, 38)
RIGHT_HAND_SLICE = slice(38, 59)
FACE_SLICE       = slice(59, 94)


# ─────────────────────────────────────────────
# Tokenizer (word-level, identico al loader originale)
# ─────────────────────────────────────────────
class SentenceTokenizer:
    """
    Tokenizer word-level semplice.
    Costruisce il vocabolario dai dati di training; i token speciali
    sono inseriti agli indici 0-3 per compatibilità con il decoder.

    Args:
        min_freq: scarta parole che appaiono meno di min_freq volte.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}
        self._build_specials()

    _TOKEN_RE = re.compile(r"[a-z0-9äöüß]+(?:['-][a-z0-9äöüß]+)*|[^\w\s]")

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
        """Costruisce il vocabolario da una lista di frasi (tedeschi inclusi)."""
        from collections import Counter
        word_counts: Counter = Counter()
        for sent in sentences:
            for word in self._split(sent):
                word_counts[word] += 1
        for word, count in word_counts.most_common():
            if count >= self.min_freq:
                idx = len(self.token2idx)
                self.token2idx[word] = idx
                self.idx2token[idx] = word

    @staticmethod
    def _split(sentence: str) -> list[str]:
        return SentenceTokenizer._TOKEN_RE.findall(sentence.lower())

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
        return self._detokenize(tokens)

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        closing_punct = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "'", '"'}
        hyphen_like   = {"-", "–", "—"}
        out: list[str] = []
        for tok in tokens:
            if not out:
                out.append(tok)
                continue
            if tok in closing_punct or tok in hyphen_like:
                out[-1] = out[-1] + tok
            else:
                out.append(tok)
        return " ".join(out)

    # ── salvataggio / caricamento ─────────────
    def save(self, path: str | Path):
        import json
        meta = {"min_freq": self.min_freq, "vocab": self.token2idx}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SentenceTokenizer":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "vocab" in data:
            min_freq = data.get("min_freq", 1)
            vocab = data["vocab"]
        else:
            min_freq = 1
            vocab = data
        tok = cls(min_freq=min_freq)
        tok.token2idx = vocab
        tok.idx2token = {v: k for k, v in tok.token2idx.items()}
        return tok


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class PhoenixDataset(Dataset):
    """
    Dataset PHOENIX-2014-T con landmarks MediaPipe pre-estratti.

    Args:
        csv_path:          percorso al CSV (pipe-separated).
        landmarks_dir:     cartella con file <name>_landmarks.npy.
        tokenizer:         istanza SentenceTokenizer già costruita.
        target_field:      'translation' (default, testo tedesco normalizzato)
                           oppure 'orth' (gloss in maiuscolo).
        max_src_len:       numero massimo di frame (None = nessun limite).
        max_tgt_len:       numero massimo di token target (None = nessun limite).
        flatten_landmarks: True → output (T, 376); False → output (T, 94, 4).
        pose_weight:       peso applicato ai landmark di posa dopo normalizzazione.
        hand_weight:       peso applicato ai landmark delle mani dopo normalizzazione.
        face_weight:       peso applicato ai landmark del viso dopo normalizzazione.
        normalize_stats:   dict con 'mean' e 'std' numpy arrays di shape (376,)
                           calcolati sul training set (coordinate x/y/z).
                           Se None non viene applicata la standardizzazione.
        use_hand_relative_norm: se True, i landmark delle mani vengono
                           espressi relativamente al polso corrispondente.
        subset_fraction:   frazione del dataset da usare (es. 0.1 = 10%).
        max_samples:       numero massimo di campioni.
        sample_seed:       seed per il campionamento riproducibile.
    """

    def __init__(
        self,
        csv_path: str | Path,
        landmarks_dir: str | Path,
        tokenizer: SentenceTokenizer,
        target_field: str = "translation",
        max_src_len: Optional[int] = 512,
        max_tgt_len: Optional[int] = 128,
        flatten_landmarks: bool = True,
        pose_weight: float = 1.0,
        hand_weight: float = 1.0,
        face_weight: float = 1.0,
        normalize_stats: Optional[dict] = None,
        use_hand_relative_norm: bool = True,
        subset_fraction: Optional[float] = None,
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
    ):
        self.landmarks_dir   = Path(landmarks_dir)
        self.tokenizer       = tokenizer
        self.target_field    = target_field
        self.max_src_len     = max_src_len
        self.max_tgt_len     = max_tgt_len
        self.flatten         = flatten_landmarks
        self.pose_weight     = pose_weight
        self.hand_weight     = hand_weight
        self.face_weight     = face_weight
        self.subset_fraction = subset_fraction
        self.max_samples     = max_samples
        self.sample_seed     = sample_seed
        self.use_hand_relative_norm = use_hand_relative_norm

        # Normalizzazione per-feature (solo sui canali x/y/z, non su confidence)
        self.normalize_stats = normalize_stats
        if normalize_stats is not None:
            mean = normalize_stats.get("mean")
            std  = normalize_stats.get("std")
            if mean is None or std is None:
                raise ValueError("normalize_stats deve contenere 'mean' e 'std'")
            self._norm_mean = np.asarray(mean, dtype=np.float32)  # (376,)
            self._norm_std  = np.asarray(std,  dtype=np.float32)
        else:
            self._norm_mean = None
            self._norm_std  = None

        self.samples: list[dict] = []
        self._load_csv(csv_path)
        self._apply_subset()

        # Validazione dimensione normalize_stats vs dataset
        if self.samples and self._norm_mean is not None:
            try:
                first = np.load(self.samples[0]["npy_path"])
                feat_dim_ds = int(first.shape[1] if first.ndim == 2
                                  else first.shape[1] * first.shape[2])
                if self._norm_mean.shape[0] != feat_dim_ds:
                    print(
                        f"[PhoenixDataset] Warning: normalize_stats dim "
                        f"({self._norm_mean.shape[0]}) ≠ feat_dim ({feat_dim_ds}). "
                        f"Standardizzazione disabilitata."
                    )
                    self._norm_mean = None
                    self._norm_std  = None
            except Exception as e:
                print(f"[PhoenixDataset] Impossibile validare feat_dim: {e}")

    # ── caricamento CSV ───────────────────────────────────────────────
    def _load_csv(self, csv_path: str | Path):
        """
        Legge il CSV pipe-separated di PHOENIX-2014-T.
        Header: name | video | start | end | speaker | orth | translation
        """
        csv_path = Path(csv_path)
        skipped  = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                name = row["name"].strip()
                # Target: 'translation' (tedesco normalizzato) o 'orth' (gloss)
                target = row.get(self.target_field, "").strip()
                if not target:
                    skipped += 1
                    continue

                npy_path = self.landmarks_dir / f"{name}_landmarks.npy"
                if not npy_path.exists():
                    skipped += 1
                    continue

                self.samples.append({
                    "name":     name,
                    "npy_path": npy_path,
                    "target":   target,
                    "speaker":  row.get("speaker", "").strip(),
                    "orth":     row.get("orth", "").strip(),
                })

        if skipped:
            print(f"[PhoenixDataset] Saltati {skipped} campioni "
                  f"(.npy mancante o campo target vuoto)")
        print(f"[PhoenixDataset] Caricati {len(self.samples)} campioni "
              f"da {csv_path.name} (target='{self.target_field}')")

    # ── subset ───────────────────────────────────────────────────────
    def _apply_subset(self):
        if not self.samples:
            return
        total  = len(self.samples)
        target = total
        if self.subset_fraction is not None and self.subset_fraction < 1.0:
            target = max(1, int(total * self.subset_fraction))
        if self.max_samples is not None:
            target = min(target, self.max_samples)
        if target < total:
            rng = random.Random(self.sample_seed)
            self.samples = rng.sample(self.samples, k=target)
            print(f"[PhoenixDataset] Subset: {target}/{total} campioni")

    # ── normalizzazione wrist-relative ───────────────────────────────
    @staticmethod
    def _apply_hand_relative_normalization(lm: Tensor) -> Tensor:
        """
        Esprime le coordinate x/y/z delle mani relativamente al polso (indice 0
        di ciascuna mano). Il canale confidence rimane invariato.

        Args:
            lm: Tensor di shape (T, 94, 4)
        Returns:
            Tensor di shape (T, 94, 4) con mani wrist-relative.
        """
        lm = lm.clone()
        # Polso sinistro: primo landmark del gruppo left_hand (indice 17 globale)
        left_wrist  = lm[:, LEFT_HAND_SLICE.start : LEFT_HAND_SLICE.start + 1, :3]
        # Polso destro: primo landmark del gruppo right_hand (indice 38 globale)
        right_wrist = lm[:, RIGHT_HAND_SLICE.start : RIGHT_HAND_SLICE.start + 1, :3]

        lm[:, LEFT_HAND_SLICE,  :3] -= left_wrist
        lm[:, RIGHT_HAND_SLICE, :3] -= right_wrist
        return lm

    # ── __len__ / __getitem__ ─────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── Carica landmarks ──────────────────────────────────────────
        lm = np.load(sample["npy_path"])          # (T, 94, 4) o (T, 376)
        lm = torch.from_numpy(lm).float()

        # Normalizza a (T, 94, 4)
        if lm.ndim == 2:
            T = lm.shape[0]
            if lm.shape[1] % N_DIMS != 0:
                raise ValueError(
                    f"Feature dim {lm.shape[1]} non divisibile per N_DIMS={N_DIMS} "
                    f"in {sample['npy_path']}"
                )
            lm = lm.view(T, lm.shape[1] // N_DIMS, N_DIMS)
        elif lm.ndim == 3:
            pass  # già (T, N, 4)
        else:
            raise ValueError(
                f"Shape non supportata {tuple(lm.shape)} in {sample['npy_path']}"
            )

        n_landmarks = lm.shape[1]
        feat_dim    = n_landmarks * N_DIMS

        # ── Clip coordinate al range [0, 1] ───────────────────────────
        # MediaPipe emette x, y, z già normalizzati per dimensione immagine,
        # ma possono occasionalmente sforare. La clip garantisce il range.
        # Il canale confidence (dim 3) è già in [0, 1], non viene toccato.
        lm[:, :, :3] = lm[:, :, :3].clamp(0.0, 1.0)

        # ── Troncamento temporale ─────────────────────────────────────
        if self.max_src_len and lm.shape[0] > self.max_src_len:
            lm = lm[: self.max_src_len]

        # ── Standardizzazione per-feature (solo x/y/z) ───────────────
        # Applica mean/std calcolati sul training set. La standardizzazione
        # usa la rappresentazione flat ma maschera il canale confidence.
        if self._norm_mean is not None and self._norm_std is not None:
            T    = lm.shape[0]
            flat = lm.view(T, feat_dim).numpy()                    # (T, 376)
            flat = (flat - self._norm_mean) / (self._norm_std + 1e-6)
            lm   = torch.from_numpy(flat).float().view(T, n_landmarks, N_DIMS)
            # Ripristina il canale confidence al valore originale non standardizzato
            # (già caricato prima della standardizzazione):
            # Il canale conf è indice 3 → rimane quello prodotto dalla clip,
            # ma viene sovrascritto dalla standardizzazione. Lo rimettiamo a posto.
            lm_raw = torch.from_numpy(
                np.load(sample["npy_path"]).astype(np.float32)
            )
            if lm_raw.ndim == 2:
                lm_raw = lm_raw.view(T, n_landmarks, N_DIMS)
            lm[:, :, 3] = lm_raw[: T, :, 3].clamp(0.0, 1.0)

        # ── Normalizzazione relativa alle mani ────────────────────────
        if self.use_hand_relative_norm:
            lm = self._apply_hand_relative_normalization(lm)

        # ── Pesi per gruppo di landmark ───────────────────────────────
        if self.pose_weight != 1.0:
            lm[:, POSE_SLICE,  :] *= self.pose_weight
        if self.hand_weight != 1.0:
            lm[:, LEFT_HAND_SLICE,  :] *= self.hand_weight
            lm[:, RIGHT_HAND_SLICE, :] *= self.hand_weight
        if self.face_weight != 1.0:
            lm[:, FACE_SLICE, :] *= self.face_weight

        # ── Output shape sorgente ─────────────────────────────────────
        if self.flatten:
            T  = lm.shape[0]
            lm = lm.view(T, feat_dim)              # (T, 376)

        # ── Tokenizzazione target ─────────────────────────────────────
        tgt_ids = self.tokenizer.encode(sample["target"], add_special_tokens=True)
        if self.max_tgt_len and len(tgt_ids) > self.max_tgt_len:
            tgt_ids = tgt_ids[: self.max_tgt_len - 1] + [self.tokenizer.eos_id]
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

        return {
            "src":      lm,        # (T, 376) o (T, 94, 4)
            "tgt":      tgt_ids,   # (L,)
            "name":     sample["name"],
            "target":   sample["target"],
            "orth":     sample["orth"],
            "speaker":  sample["speaker"],
            "npy_path": str(sample["npy_path"]),
        }


# ─────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────
def collate_fn(
    batch: list[dict],
    pad_id: int = 0,
    bos_id: int = 1,
    unk_id: int = 3,
    eos_id: int = 2,
) -> dict:
    """
    Padda sorgente e target alla lunghezza massima del batch.

    Restituisce:
        src:                    (B, T_max, feat_dim)
        src_key_padding_mask:   (B, T_max)   True = padding
        tgt:                    (B, L_max)
        tgt_key_padding_mask:   (B, L_max)   True = padding
        tgt_input:              (B, L_max-1) — input decoder (senza <eos>)
        tgt_output:             (B, L_max-1) — target loss  (senza <bos>)
        tgt_in_key_padding_mask:  (B, L_max-1)
        tgt_out_key_padding_mask: (B, L_max-1)
        src_lens:               (B,)
        tgt_lens:               (B,)
        names:                  list[str]
        targets:                list[str]
        orths:                  list[str]
        npy_paths:              list[str]
    """
    srcs      = [item["src"]  for item in batch]
    tgts      = [item["tgt"]  for item in batch]
    names     = [item["name"] for item in batch]
    targets   = [item["target"]   for item in batch]
    orths     = [item["orth"]     for item in batch]
    npy_paths = [item["npy_path"] for item in batch]

    src_lens = [s.shape[0] for s in srcs]
    tgt_lens = [t.shape[0] for t in tgts]

    T_max = max(src_lens)
    L_max = max(tgt_lens)
    B     = len(batch)
    D     = srcs[0].shape[-1]

    # ── Sorgente ─────────────────────────────
    src_padded = torch.zeros(B, T_max, D)
    src_mask   = torch.ones(B, T_max, dtype=torch.bool)   # True = padding
    for i, (s, l) in enumerate(zip(srcs, src_lens)):
        src_padded[i, :l] = s
        src_mask[i,   :l] = False

    # ── Target ───────────────────────────────
    tgt_padded = torch.full((B, L_max), fill_value=pad_id, dtype=torch.long)
    tgt_mask   = torch.ones(B, L_max, dtype=torch.bool)
    for i, (t, l) in enumerate(zip(tgts, tgt_lens)):
        tgt_padded[i, :l] = t
        tgt_mask[i,   :l] = False

    # ── Shift per teacher forcing ─────────────
    tgt_input  = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]
    tgt_in_mask  = tgt_mask[:, :-1]
    tgt_out_mask = tgt_mask[:, 1:]

    return {
        "src":                      src_padded,     # (B, T, D)
        "src_key_padding_mask":     src_mask,        # (B, T)
        "tgt":                      tgt_padded,      # (B, L)
        "tgt_key_padding_mask":     tgt_mask,        # (B, L)
        "tgt_input":                tgt_input,       # (B, L-1)
        "tgt_output":               tgt_output,      # (B, L-1)
        "tgt_in_key_padding_mask":  tgt_in_mask,
        "tgt_out_key_padding_mask": tgt_out_mask,
        "src_lens":                 torch.tensor(src_lens),
        "tgt_lens":                 torch.tensor(tgt_lens),
        "names":                    names,
        "targets":                  targets,
        "orths":                    orths,
        "npy_paths":                npy_paths,
    }


# ─────────────────────────────────────────────
# Bucketing sampler (riduce padding nel batch)
# ─────────────────────────────────────────────
class BucketingBatchSampler(Sampler[list[int]]):
    """Raggruppa sequenze di lunghezza simile per minimizzare il padding."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_size: int = 200,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.lengths    = lengths
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, bucket_size)
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self.seed       = seed
        self._epoch     = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng     = random.Random(self.seed + self._epoch)
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        buckets = [
            indices[i : i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]
        if self.shuffle:
            for b in buckets:
                rng.shuffle(b)
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
        n = len(self.lengths)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


# ─────────────────────────────────────────────
# Funzioni factory
# ─────────────────────────────────────────────
def build_tokenizer(
    train_csv: str | Path,
    save_path: Optional[str | Path] = None,
    min_freq: int = 1,
    target_field: str = "translation",
) -> SentenceTokenizer:
    """
    Costruisce il tokenizer dal CSV di training di PHOENIX-2014-T.

    Args:
        train_csv:    path al CSV pipe-separated.
        save_path:    path dove salvare il tokenizer (.json).
        min_freq:     frequenza minima per includere una parola nel vocabolario.
        target_field: 'translation' (default) o 'orth'.
    """
    tokenizer = SentenceTokenizer(min_freq=min_freq)
    sentences: list[str] = []
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            s = row.get(target_field, "").strip()
            if s:
                sentences.append(s)
    tokenizer.build_from_sentences(sentences)
    print(
        f"[build_tokenizer] Vocabolario: {tokenizer.vocab_size} token "
        f"da {len(sentences)} frasi (target='{target_field}', min_freq={min_freq})"
    )
    if save_path:
        tokenizer.save(save_path)
        print(f"[build_tokenizer] Salvato in {save_path}")
    return tokenizer


def _compute_landmark_stats(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola mean/std per-feature (shape: FEAT_DIM=376) sul dataset.
    Solo i frame validi (non-zero) contribuiscono alle statistiche.
    """
    ssum  = np.zeros(FEAT_DIM, dtype=np.float64)
    ssq   = np.zeros(FEAT_DIM, dtype=np.float64)
    count = 0
    for sample in samples:
        try:
            arr = np.load(sample["npy_path"], mmap_mode="r").astype(np.float32)
            if arr.ndim == 3:
                arr = arr.reshape(arr.shape[0], -1)   # (T, 376)
            ssum  += arr.sum(axis=0, dtype=np.float64)
            ssq   += (arr ** 2).sum(axis=0, dtype=np.float64)
            count += arr.shape[0]
        except Exception:
            continue
    if count == 0:
        raise RuntimeError("Nessun frame valido trovato per il calcolo delle stats")
    mean = (ssum / count).astype(np.float32)
    var  = (ssq / count - (ssum / count) ** 2).astype(np.float32)
    std  = np.sqrt(np.maximum(var, 1e-12))
    return mean, std


def build_dataloaders(
    train_csv:           str | Path,
    val_csv:             str | Path,
    train_landmarks_dir: str | Path,
    val_landmarks_dir:   str | Path,
    tokenizer:           SentenceTokenizer,
    batch_size:          int = 32,
    num_workers:         int = 4,
    max_src_len:         int = 512,
    max_tgt_len:         int = 128,
    target_field:        str = "translation",
    pose_weight:         float = 1.0,
    hand_weight:         float = 1.0,
    face_weight:         float = 1.0,
    pin_memory:          bool = True,
    test_csv:            Optional[str | Path] = None,
    test_landmarks_dir:  Optional[str | Path] = None,
    train_subset_fraction: Optional[float] = None,
    val_subset_fraction:   Optional[float] = None,
    train_max_samples:   Optional[int] = None,
    val_max_samples:     Optional[int] = None,
    subset_seed:         int = 42,
    use_bucketing:       bool = False,
    bucket_size:         int = 200,
    drop_last:           bool = False,
    use_hand_relative_norm: bool = True,
    flatten_landmarks:   bool = True,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> dict[str, DataLoader]:
    """
    Restituisce {"train": DataLoader, "val": DataLoader, ["test": DataLoader]}.

    Calcola automaticamente le statistiche di normalizzazione dal training set
    e le passa a tutti i dataset.
    """
    from functools import partial

    _collate = partial(
        collate_fn,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        unk_id=tokenizer.unk_id,
        eos_id=tokenizer.eos_id,
    )

    # ── Calcolo stats di normalizzazione dal training set ─────────────
    normalize_stats = None
    try:
        # Carica i sample del training senza costruire il Dataset completo
        train_samples: list[dict] = []
        with open(train_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                name = row["name"].strip()
                npy  = Path(train_landmarks_dir) / f"{name}_landmarks.npy"
                if npy.exists():
                    train_samples.append({"npy_path": str(npy)})

        if train_samples:
            print(
                f"[build_dataloaders] Calcolo stats normalizzazione "
                f"su {len(train_samples)} file..."
            )
            mean, std = _compute_landmark_stats(train_samples)
            normalize_stats = {"mean": mean, "std": std}
            print("[build_dataloaders] Stats calcolate.")
    except Exception as e:
        print(f"[build_dataloaders] Warning: impossibile calcolare le stats: {e}")

    # ── Helper interno ─────────────────────────────────────────────────
    def _src_lengths(ds: PhoenixDataset) -> list[int]:
        lengths = []
        for s in ds.samples:
            try:
                arr = np.load(s["npy_path"], mmap_mode="r")
                l   = int(arr.shape[0])
            except Exception:
                l   = 0
            lengths.append(min(l, max_src_len) if max_src_len else l)
        return lengths

    def _make_loader(
        csv_path: str | Path,
        lm_dir:   str | Path,
        shuffle:  bool,
        subset_fraction: Optional[float],
        max_samples: Optional[int],
    ) -> DataLoader:
        ds = PhoenixDataset(
            csv_path=csv_path,
            landmarks_dir=lm_dir,
            tokenizer=tokenizer,
            target_field=target_field,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            flatten_landmarks=flatten_landmarks,
            pose_weight=pose_weight,
            hand_weight=hand_weight,
            face_weight=face_weight,
            normalize_stats=normalize_stats,
            use_hand_relative_norm=use_hand_relative_norm,
            subset_fraction=subset_fraction,
            max_samples=max_samples,
            sample_seed=subset_seed,
        )

        if use_bucketing:
            lengths       = _src_lengths(ds)
            batch_sampler = BucketingBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                bucket_size=bucket_size,
                shuffle=shuffle,
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
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
                generator=generator,
                worker_init_fn=worker_init_fn,
        )

    loaders: dict[str, DataLoader] = {
        "train": _make_loader(
            train_csv, train_landmarks_dir,
            shuffle=True,
            subset_fraction=train_subset_fraction,
            max_samples=train_max_samples,
        ),
        "val": _make_loader(
            val_csv, val_landmarks_dir,
            shuffle=False,
            subset_fraction=val_subset_fraction,
            max_samples=val_max_samples,
        ),
    }

    if test_csv:
        loaders["test"] = _make_loader(
            test_csv,
            test_landmarks_dir or val_landmarks_dir,
            shuffle=False,
            subset_fraction=val_subset_fraction,
            max_samples=val_max_samples,
        )

    return loaders