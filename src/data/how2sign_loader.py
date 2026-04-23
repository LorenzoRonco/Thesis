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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        landmarks, length = self._load_landmarks(row['npy_path'])
        video_frames = self._load_video_frames(video_path=row['video_path'], length=length)
        sentence = row['SENTENCE']

        return landmarks, video_frames, sentence, length

    # ------------------------------------------------------------------

    def _load_landmarks(self, npy_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Carica e appiattisce i landmarks.

        Returns:
            landmarks : [T, 2108]
            length    : numero frame reali (prima del taglio)
        """
        data = np.load(npy_path).astype(np.float32)  # [T, 527, 4]
        data = data.reshape(data.shape[0], -1)        # [T, 2108]

        length = data.shape[0]

        if self.max_frames is not None and length > self.max_frames:
            data = data[:self.max_frames]
            length = self.max_frames

        return torch.from_numpy(data), length


    def _load_video_frames(self, video_path: Path, length: int) -> torch.Tensor:
        """
        Carica frame da video .mp4 croppato.

        Args:
            video_path: path al file .mp4 croppato
            length    : numero frame da caricare (allineato ai landmark)

        Returns:
            Tensor [length, 3, H, W]
        """
        
        if not video_path.exists():
            # Fallback utile se require_video=False.
            return torch.zeros(length, 3, *self.img_size)

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while len(frames) < length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.img_size)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(torch.from_numpy(frame))

        cap.release()

        # Se il video ha meno frame dei landmark, padda con zeri
        if len(frames) < length:
            padding = torch.zeros(length - len(frames), 3, *self.img_size)
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