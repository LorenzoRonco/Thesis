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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────
N_LANDMARKS  = 527
N_DIMS       = 4          # x, y, z, visibility/presence
FEAT_DIM     = N_LANDMARKS * N_DIMS   # 2108


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
    ):
        self.landmarks_dir  = Path(landmarks_dir)
        self.tokenizer      = tokenizer
        self.max_src_len    = max_src_len
        self.max_tgt_len    = max_tgt_len
        self.flatten        = flatten_landmarks

        self.samples: list[dict] = []
        self._load_csv(csv_path)

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
        "src_lens":   torch.tensor(src_lens),
        "tgt_lens":   torch.tensor(tgt_lens),
    }


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
    landmarks_dir: str | Path,
    tokenizer:     SentenceTokenizer,
    batch_size:    int = 32,
    num_workers:   int = 4,
    max_src_len:   int = 512,
    max_tgt_len:   int = 128,
    pin_memory:    bool = True,
    test_csv:      Optional[str | Path] = None,
) -> dict[str, DataLoader]:
    """
    Restituisce un dizionario {"train": ..., "val": ..., "test": ...}.
    """
    from functools import partial
    _collate = partial(collate_fn, pad_id=tokenizer.pad_id)

    def _make_loader(csv_path, shuffle):
        ds = How2SignDataset(
            csv_path=csv_path,
            landmarks_dir=landmarks_dir,
            tokenizer=tokenizer,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
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

    loaders = {
        "train": _make_loader(train_csv,  shuffle=True),
        "val":   _make_loader(val_csv,         shuffle=False),
    }
    if test_csv:
        loaders["test"] = _make_loader(test_csv, shuffle=False)

    return loaders