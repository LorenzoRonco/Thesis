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
    Dataset per How2Sign video con landmarks.
    
    Structure assunto:
      dataset/
        how2sign_realigned_train.csv  (metadata)
        landmarks_normalized/
          <video_id>_landmarks.npy
        segmented/
          <video_id>_frame_<n>.jpg
    """
    
    def __init__(
        self,
        csv_path: Path,
        landmarks_dir: Path,
        segmented_dir: Path,
        max_frames: int = 150,
        img_size: Tuple[int, int] = (224, 224),
        num_samples: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Inizializza dataset.
        
        Args:
            csv_path: Path al CSV with metadata
            landmarks_dir: Path a directory landmarks_normalized
            segmented_dir: Path a directory con video segmented
            max_frames: Numero massimo frame per padding
            img_size: Dimensione risize frame
            num_samples: Limit numero campioni (per testing)
            device: Device per tensori
        """
        self.csv_path = Path(csv_path)
        self.landmarks_dir = Path(landmarks_dir)
        self.segmented_dir = Path(segmented_dir)
        self.max_frames = max_frames
        self.img_size = img_size
        self.device = device
        
        # Carica metadata
        self.metadata = pd.read_csv(self.csv_path)
        if num_samples:
            self.metadata = self.metadata.iloc[:num_samples]
        
        print(f"[Dataset] Loaded {len(self.metadata)} samples from CSV")
    
    def load_landmarks(self, video_id: str, max_frames: int) -> torch.Tensor:
        """
        Carica landmarks normalizzati.
        
        Args:
            video_id: ID video
            max_frames: Numero frame max
        
        Returns:
            Tensor (max_frames, 2108) con padding se necessario
        """
        landmarks_file = self.landmarks_dir / f"{video_id}_landmarks.npy"
        
        try:
            landmarks = np.load(landmarks_file)  # (T, 2108)
            
            # Padding se necessario
            if landmarks.shape[0] < max_frames:
                padding = np.zeros(
                    (max_frames - landmarks.shape[0], landmarks.shape[1]),
                    dtype=landmarks.dtype
                )
                landmarks = np.vstack([landmarks, padding])
            else:
                landmarks = landmarks[:max_frames]
            
            return torch.from_numpy(landmarks).float().to(self.device)
        
        except FileNotFoundError:
            print(f"⚠️  Landmarks not found: {video_id}")
            return torch.zeros(max_frames, 2108).to(self.device)
    
    def load_video_frames(self, video_id: str, max_frames: int) -> torch.Tensor:
        """
        Carica frame video segmented.
        
        Args:
            video_id: ID video
            max_frames: Numero frame max
        
        Returns:
            Tensor (max_frames, 3, 224, 224)
        """
        frames = []
        
        # Cerca file frame
        frame_pattern = f"{video_id}_frame_*.jpg"
        frame_files = sorted(self.segmented_dir.glob(frame_pattern))
        
        if not frame_files:
            print(f"⚠️  Frames not found: {video_id}")
            return torch.zeros(max_frames, 3, *self.img_size).to(self.device)
        
        for frame_path in frame_files[:max_frames]:
            try:
                # Carica immagine
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                
                # Converti BGR → RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, self.img_size)
                
                # Normalizzazione [0, 1]
                img = img.astype(np.float32) / 255.0
                
                # Transponi a (3, H, W)
                img = np.transpose(img, (2, 0, 1))
                
                frames.append(torch.from_numpy(img).float())
            
            except Exception as e:
                print(f"⚠️  Error loading frame {frame_path}: {e}")
                continue
        
        # Stack frames
        if frames:
            video = torch.stack(frames)  # (num_loaded, 3, H, W)
        else:
            video = torch.zeros(1, 3, *self.img_size).float()
        
        # Padding frame
        if video.shape[0] < max_frames:
            padding_frames = torch.zeros(
                max_frames - video.shape[0],
                3,
                *self.img_size
            )
            video = torch.cat([video, padding_frames], dim=0)
        else:
            video = video[:max_frames]
        
        return video.to(self.device)
    
    def load_translation(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carica traduzione (sarà implementato basato su dataset reale).
        
        Per ora, ritorna tensor casuali.
        In produzione, integra tokenizer real text.
        
        Returns:
            (target_tokens, target_labels)
        """
        # TODO: Implementa caricamento real testo
        # Placeholder per testing
        text_max_len = 512
        vocab_size = 10000
        
        target_tokens = torch.zeros(text_max_len, dtype=torch.long)
        target_tokens[0] = 1  # BOS
        target_tokens[1:10] = torch.randint(2, vocab_size, (9,))
        target_tokens[10] = 2  # EOS
        
        target_labels = torch.zeros_like(target_tokens)
        target_labels[:-1] = target_tokens[1:]
        target_labels[-1] = 2  # EOS
        
        return target_tokens, target_labels
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ritorna: (landmarks, video_frames, target_tokens, target_labels)
        """
        row = self.metadata.iloc[idx]
        video_id = row['video_id']  # Adjust column name based on CSV
        
        landmarks = self.load_landmarks(video_id, self.max_frames)
        video_frames = self.load_video_frames(video_id, self.max_frames)
        target_tokens, target_labels = self.load_translation(row)
        
        return landmarks, video_frames, target_tokens, target_labels


def create_real_dataloader(
    csv_path: Path,
    landmarks_dir: Path,
    segmented_dir: Path,
    batch_size: int = 32,
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    device: torch.device = torch.device('cpu'),
) -> DataLoader:
    """
    Crea DataLoader con dati reali How2Sign.
    
    Args:
        csv_path: Path to metadata CSV
        landmarks_dir: Path to landmarks directory
        segmented_dir: Path to segmented frames directory
        batch_size: Batch size
        num_samples: Limit numero samples (for testing)
        shuffle: Shuffle dataset
        device: Device for tensors
    
    Returns:
        DataLoader
    """
    dataset = How2SignDataset(
        csv_path=csv_path,
        landmarks_dir=landmarks_dir,
        segmented_dir=segmented_dir,
        num_samples=num_samples,
        device=device,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    """
    Esempio di utilizzo del dataset reale.
    """
    
    from pathlib import Path
    
    # Percorsi (adjust based on your structure)
    dataset_dir = Path("dataset")
    csv_path = dataset_dir / "how2sign_realigned_train.csv"
    landmarks_dir = dataset_dir / "landmarks_normalized"
    segmented_dir = dataset_dir / "segmented"
    
    # Crea dataset
    print("Creating real How2Sign dataset...")
    dataset = How2SignDataset(
        csv_path=csv_path,
        landmarks_dir=landmarks_dir,
        segmented_dir=segmented_dir,
        num_samples=1000,  # Test con 1000 campioni
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    
    # Crea DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test caricamento batch
    print(f"Dataset size: {len(dataset)}")
    print("Loading sample batch...")
    
    for landmarks, video_frames, target_tokens, target_labels in loader:
        print(f"✓ Landmarks shape: {landmarks.shape}")
        print(f"✓ Video frames shape: {video_frames.shape}")
        print(f"✓ Target tokens shape: {target_tokens.shape}")
        print(f"✓ Target labels shape: {target_labels.shape}")
        break
