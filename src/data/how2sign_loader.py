"""
DATASET LOADER - Real Data Integration
========================================

Modulo per caricare dati reali dal dataset How2Sign per il training Phase 1.

Include funzioni per:
  1. Caricamento landmarks normalizzati
  2. Caricamento cropping video RGB
  3. Caricamento e processamento label testuali
  4. Creazione DataLoader compatibile con train_phase1.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional, List
import cv2


class How2SignDataset(Dataset):
    """
    Dataset per How2Sign.

    Ogni sample restituisce:
        landmarks   : Tensor [T, 2108]         — landmark appiattiti per frame
        video_frames: Tensor [T, 3, 224, 224]  — frame RGB normalizzati
        sentence    : str                       — testo originale (tokenizzato fuori)
        length      : int                       — numero frame reali (senza padding)

    Se max_frames e' impostato e un video ha T > max_frames, viene effettuato
    un campionamento uniforme sull'intera sequenza (non un taglio dei primi frame).
    """
    
    def __init__(
        self,
        csv_path: Path,
        landmarks_dir: Path,
        cropped_dir: Path,
        max_frames: int = 150,
        img_size: Tuple[int, int] = (224, 224),
        num_samples: Optional[int] = None,
        require_video: bool = True,
    ):
        """
        Inizializza dataset.
        """
        self.csv_path = Path(csv_path)
        self.landmarks_dir = Path(landmarks_dir)
        self.cropped_dir = Path(cropped_dir)
        self.max_frames = max_frames
        self.img_size = img_size
        self.require_video = require_video
        
        df = pd.read_csv(csv_path, sep='\t')

        # I file landmark seguono il pattern:
        #   {SENTENCE_ID}_{SENTENCE_NAME}_landmarks.npy
        # es: --7E2sU6zP4_10_--7E2sU6zP4_10-5-rgb_front_landmarks.npy
        df['npy_path'] = df.apply(
            lambda row: self.landmarks_dir / f"{row['SENTENCE_ID']}_{row['SENTENCE_NAME']}_landmarks.npy",
            axis=1,
        )
        df['video_path'] = df.apply(
            lambda row: self.cropped_dir / f"{row['SENTENCE_ID']}_{row['SENTENCE_NAME']}_cropped.mp4",
            axis=1,
        )

        landmark_exists = df['npy_path'].apply(lambda p: p.exists())
        n_missing_landmarks = int((~landmark_exists).sum())
        if n_missing_landmarks > 0:
            print(f"[Dataset] {n_missing_landmarks} file .npy mancanti, ignorati.")

        if self.require_video:
            video_exists = df['video_path'].apply(lambda p: p.exists())
            n_missing_videos = int((~video_exists).sum())
            if n_missing_videos > 0:
                print(f"[Dataset] {n_missing_videos} video .mp4 mancanti, ignorati.")
            valid_mask = landmark_exists & video_exists
        else:
            valid_mask = landmark_exists

        self.df = df[valid_mask].reset_index(drop=True)

        if num_samples is not None:
            # iloc estrae intervalli di un dataframe
            self.df = self.df.iloc[:num_samples].reset_index(drop=True) 

        if len(self.df) == 0:
            raise ValueError(
                "Nessun sample valido trovato. Verifica csv_path/landmarks_dir "
                "e il naming dei file landmark."
            )

        print(f"[Dataset] {len(self.df)} sample pronti.")
        self._print_frame_length_summary()

    def __len__(self) -> int:
        return len(self.df)

    def _print_frame_length_summary(self) -> None:
        """
        Stampa statistiche sui frame originali dei landmark e sui frame usati
        dopo l'eventuale limitazione con max_frames.
        """
        frame_counts: List[int] = []
        for npy_path in self.df['npy_path']:
            try:
                arr = np.load(npy_path, mmap_mode='r')
                frame_counts.append(int(arr.shape[0]))
            except Exception:
                continue

        if not frame_counts:
            print("[Dataset] Statistiche frame non disponibili (nessun .npy leggibile).")
            return

        counts = np.asarray(frame_counts, dtype=np.int64)
        p50 = int(np.percentile(counts, 50))
        p90 = int(np.percentile(counts, 90))
        p95 = int(np.percentile(counts, 95))
        print(
            "[Dataset] Frame originali (landmark): "
            f"min={int(counts.min())}, p50={p50}, p90={p90}, p95={p95}, max={int(counts.max())}"
        )

        if self.max_frames is None:
            print("[Dataset] max_frames=None: nessun downsampling temporale applicato.")
            return

        used_counts = np.minimum(counts, self.max_frames)
        dropped_counts = counts - used_counts
        truncated_mask = counts > self.max_frames
        n_truncated = int(truncated_mask.sum())
        pct_truncated = 100.0 * n_truncated / max(1, len(counts))
        mean_dropped = float(dropped_counts.mean())
        mean_dropped_truncated = (
            float(dropped_counts[truncated_mask].mean()) if n_truncated > 0 else 0.0
        )

        print(
            "[Dataset] Uso frame con max_frames="
            f"{self.max_frames}: campioni oltre soglia={n_truncated}/{len(counts)} "
            f"({pct_truncated:.1f}%), frame medi non usati={mean_dropped:.1f} "
            f"(solo oltre soglia: {mean_dropped_truncated:.1f})."
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        landmarks, frame_indices = self._load_landmarks(row['npy_path'])
        video_frames = self._load_video_frames(video_path=row['video_path'], frame_indices=frame_indices)
        length = landmarks.shape[0]
        sentence = row['SENTENCE']

        return landmarks, video_frames, sentence, length

    # ------------------------------------------------------------------

    def _load_landmarks(self, npy_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Carica e appiattisce i landmarks, con eventuale campionamento temporale.

        Returns:
            landmarks     : [T_sel, 2108]
            frame_indices : indici frame selezionati rispetto alla sequenza originale
        """
        data = np.load(npy_path).astype(np.float32)  # [T, 527, 4]
        data = data.reshape(data.shape[0], -1)        # [T, 2108]
        total_length = data.shape[0]

        if self.max_frames is None or total_length <= self.max_frames:
            frame_indices = np.arange(total_length, dtype=np.int64)
        else:
            # Copre l'intera sequenza in modo uniforme, evitando bias sui soli frame iniziali.
            frame_indices = np.linspace(
                0,
                total_length - 1,
                num=self.max_frames,
                dtype=np.int64,
            )

        data = data[frame_indices]

        return torch.from_numpy(data), frame_indices


    def _load_video_frames(self, video_path: Path, frame_indices: np.ndarray) -> torch.Tensor:
        """
        Carica frame da video .mp4 croppato.

        Args:
            video_path: path al file .mp4 croppato
            frame_indices: indici frame da estrarre (allineati ai landmark)

        Returns:
            Tensor [T_sel, 3, H, W]
        """
        target_length = int(frame_indices.shape[0])
        
        if not video_path.exists():
            # Fallback utile se require_video=False.
            return torch.zeros(target_length, 3, *self.img_size)

        cap = cv2.VideoCapture(str(video_path))
        video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if target_length > 0 and video_total > 0 and int(frame_indices[-1]) >= video_total:
            print(
                "[Dataset][WARN] Possibile mismatch landmarks/video: "
                f"{video_path.name} ha {video_total} frame, "
                f"ma i landmarks richiedono indice massimo {int(frame_indices[-1])}."
            )
        frames = []
        wanted_ptr = 0
        current_idx = 0

        while wanted_ptr < target_length:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx != int(frame_indices[wanted_ptr]):
                current_idx += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.img_size)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(torch.from_numpy(frame))
            wanted_ptr += 1
            current_idx += 1

        cap.release()

        # Se il video ha meno frame del richiesto, padda con zeri
        if len(frames) < target_length:
            padding = torch.zeros(target_length - len(frames), 3, *self.img_size)
            if frames:
                return torch.cat([torch.stack(frames), padding], dim=0)
            else:
                return padding

        # Caso standard: numero frame sufficiente.
        return torch.stack(frames)
       
# ----------------------------------------------------------------------
# collate_fn — padding dinamico per batch
# ----------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
    """
    Padda landmarks e frame alla lunghezza massima del batch.

    Returns:
        landmarks_padded  : [B, T_max, 2108]
        frames_padded     : [B, T_max, 3, H, W]
        padding_mask      : [B, T_max]  — True sui frame di padding
        sentences         : List[str]
        lengths           : List[int]
    """
    landmarks_list, frames_list, sentences, lengths = zip(*batch)

    T_max = max(lengths)
    B = len(lengths)
    _, H, W = frames_list[0].shape[1], frames_list[0].shape[2], frames_list[0].shape[3]
    landmark_dim = landmarks_list[0].shape[1]

    landmarks_padded = torch.zeros(B, T_max, landmark_dim)
    frames_padded    = torch.zeros(B, T_max, 3, H, W)
    padding_mask     = torch.ones(B, T_max, dtype=torch.bool)  # True = padding

    for i, (lm, fr, length) in enumerate(zip(landmarks_list, frames_list, lengths)):
        landmarks_padded[i, :length] = lm
        frames_padded[i, :length]    = fr
        padding_mask[i, :length]     = False  # False = frame valido

    return landmarks_padded, frames_padded, padding_mask, list(sentences), list(lengths)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_dataloader(
    csv_path: Path,
    landmarks_dir: Path,
    cropped_dir: Path,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    num_samples: Optional[int] = None,
    max_frames: Optional[int] = None,
    require_video: bool = True,
) -> DataLoader:
    dataset = How2SignDataset(
        csv_path=csv_path,
        landmarks_dir=landmarks_dir,
        cropped_dir=cropped_dir,
        max_frames=max_frames,
        num_samples=num_samples,
        require_video=require_video,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )