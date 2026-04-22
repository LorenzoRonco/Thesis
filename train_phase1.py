"""
FASE 1 DI ADDESTRAMENTO - Sign Language Translation
=====================================================

Script per l'addestramento della prima fase del modello di traduzione.

OBIETTIVO DELLA FASE 1:
  - Addestrare il modello su 1000 campioni per stabilizzare gli embeddings
  - Congelare i pesi di MobileNet (2D-CNN) per mantenere feature visive pre-addestrate
  - Addestrare i componenti principali (Transformer encoder, 1D-CNN, GRU, Dense, Decoder)
  - Bilanciare i pesi tra i due rami (geometrico e visuale)
  - Preparare il modello per il fine-tuning totale (Fase 2)

ARCHITETTURA ATTIVATA:
  ┌──────────────────────────────────────────────────────────────┐
  │ INPUT: Landmarks (B, T, 2108) + Video RGB (B, T, 3, H, W)   │
  └────┬────────────────────────────────────────────────┬────────┘
       │                                                │
   [CONGELATO]                              [ADDESTRAMENTO]
  ┌────▼──────────┐                      ┌───────────────▼──────┐
  │ MobileNet     │ ← FROZEN              │ 1D-CNN               │
  │ (2D-CNN)      │                       │ Video processing     │
  └────┬──────────┘                       │                      │
       │                                  │ GRU Sequential       │
       │                          ┌───────▼──────────────────┐  │
       │                 ┌────────┤ Transformer Encoder      │◄─┘
       │                 │        │ (Landmarks)             │
       │                 │        └───────┬──────────────────┘
       │                 │                │
       │         ┌───────▼────────────────▼──────┐
       │         │ Fusion Module                  │
       │         │ (Weighted combination)         │
       │         └───────┬──────────────────────┬─┘
       └─────────────────┤ Memory (B, T, 512)  │
                         └───────┬──────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │ Transformer Decoder  │
                      │ (Text Generation)    │
                      └──────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Output: Logits          │
                    │ (B, T_text, vocab_size)│
                    └────────────┬────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │ Cross-Entropy Loss   │
                      │ Backprop             │
                      └──────────────────────┘

CONFIGURAZIONE TRAINING:
  - Dataset: 1000 campioni
  - Batch Size: 32
  - Epochs: 50-100
  - Learning Rate: 1e-3 (AdamW)
  - Loss Function: CrossEntropyLoss
  - Optimizer: AdamW (ideale per Transformer)
  - Learning Rate Scheduler: ReduceLROnPlateau
  - Gradient Clipping: norm=1.0
  - Validation Split: 80/20
  - Checkpoint: Salva il modello migliore

PARAMETRI CONGELATI:
  - MobileNet backbone: trainable=False
  
PARAMETRI ATTIVI:
  - Transformer Encoder (landmarks): trainable=True
  - Conv1D (video): trainable=True
  - GRU (video): trainable=True
  - Fusion Module: trainable=True
  - Dense Layers: trainable=True
  - Transformer Decoder: trainable=True

Autore: Thesis Project
Data: 2026
"""

import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torch.amp import autocast, GradScaler  # Mixed Precision Training (AMP)

try:
    import cv2
except ImportError:
    cv2 = None

# Import modelli
from src.models.end_to_end_seq2seq import SignLanguageTranslationModel


# =============================================================================
# CONFIGURAZIONE TRAINING
# =============================================================================

class TrainingConfig:
    """
    Configurazione per la Phase 1 Training.
    """
    
    def __init__(self):
        # Dataset e Dataloader
        self.num_samples = 2000  # Numero campioni per Phase 1 (aumentato da 1000 a 2000)
        self.batch_size = 16  # OTTIMIZZATO: ridotto da 32 per evitare OOM (estrai 2x meno campioni per volta)
        self.num_workers = 4  # Parallel loading
        self.val_split = 0.2  # 80/20 train/validation SPLIT
        
        # Model Architecture
        self.video_hidden_dim = 512
        self.landmark_dim = 2108  # MediaPipe holistic landmarks
        self.num_encoder_layers = 4  # Transformer encoder blocks
        self.num_decoder_layers = 4  # Transformer decoder blocks
        self.num_heads = 8  # Multi-head attention heads
        
        # Vocabulary and Sequence lengths
        self.text_vocab_size = 10000  # Dimensione vocabolario (adjust based on dataset)
        self.text_max_len = 256  # OTTIMIZZATO: ridotto da 512 per Phase 1 (sequenze di testo tipicamente più corte)
        self.video_max_frames = 100  # OTTIMIZZATO: ridotto da 150 per ridurre memoria video (mantiene qualità)
        
        # Training Hyperparameters
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5  # L2 regularization
        self.gradient_clip_norm = 1.0
        
        # Learning Rate Scheduler
        self.use_scheduler = True
        self.scheduler_factor = 0.5  # Riduce LR di 0.5x quando patience esaurito
        self.scheduler_patience = 3  # Riduci LR dopo 3 epoch senza miglioramento (era 5)
        self.scheduler_min_lr = 5e-5

        # LR Warmup
        self.use_lr_warmup = True
        self.warmup_epochs = 5
        self.warmup_start_factor = 0.2

        # Debug/verification
        self.verify_backprop = False
        self.verify_backprop_every_n_batches = 50
        self.verify_backprop_max_checks_per_epoch = 2
        
        # Freezing Strategy
        self.freeze_mobilenet = True  # ← PRINCIPALE: Congela MobileNet
        self.freeze_encoder_initially = False  # Non congela Transformer encoder
        
        # Mixed Precision Training (AMP)
        self.use_mixed_precision = True  # Riduce memoria GPU di ~30-40% con zero overhead
        self.use_gradient_accumulation = False  # Disabilitato (non necessario con batch ridotto)
        
        # Checkpointing
        self.checkpoint_dir = Path("checkpoints/phase1")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = True
        self.save_frequency = 1  # Salva ogni N epoch
        
        # Logging
        self.log_dir = Path("logs/phase1")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_frequency = 50  # Log ogni N batch

        # Validation pacing (periodic checks on subset)
        self.val_check_interval = 10  # Valida ogni 10 epoche
        self.val_max_batches = 10  # Limite batch per validation (10 batch = ~160 campioni, ridotto per memoria)

        # Early stopping (in numero di validation check, non epoche)
        self.early_stopping_patience_checks = 3
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Seed
        self.seed = 42
    
    def to_dict(self) -> Dict:
        """Converti config a dizionario per logging."""
        return {
            k: v for k, v in self.__dict__.items()
            if not isinstance(v, (Path,))
        }


# =============================================================================
# DATASET REAL + MOCK
# =============================================================================

class RealHow2SignDataset(Dataset):
    """
    Dataset reale How2Sign basato su CSV realigned + landmarks_normalized + cropped.
    Ritorna sample compatibili con il training loop esistente.
    """

    def __init__(
        self,
        csv_path: Path,
        landmarks_dir: Path,
        cropped_dir: Path,
        max_samples: Optional[int] = None,
        video_max_frames: int = 150,
        landmark_dim: int = 2108,
        text_max_len: int = 512,
        vocab_size: int = 10000,
        shared_word2idx: Optional[Dict[str, int]] = None,
        shared_idx2word: Optional[Dict[int, str]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.landmarks_dir = Path(landmarks_dir)
        self.cropped_dir = Path(cropped_dir)
        self.video_max_frames = video_max_frames
        self.landmark_dim = landmark_dim
        self.text_max_len = text_max_len
        self.vocab_size = vocab_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV non trovato: {self.csv_path}")
        if not self.landmarks_dir.exists():
            raise FileNotFoundError(f"Cartella landmarks non trovata: {self.landmarks_dir}")
        if not self.cropped_dir.exists():
            raise FileNotFoundError(f"Cartella video cropped non trovata: {self.cropped_dir}")

        df = pd.read_csv(self.csv_path, sep="\t")
        required_cols = ["SENTENCE_ID", "SENTENCE_NAME", "SENTENCE"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Colonna obbligatoria mancante nel CSV: {col}")

        self.samples = []
        for row in df.itertuples(index=False):
            sentence_id = str(getattr(row, "SENTENCE_ID"))
            sentence_name = str(getattr(row, "SENTENCE_NAME"))
            sentence = str(getattr(row, "SENTENCE"))

            landmark_file = self.landmarks_dir / f"{sentence_id}_{sentence_name}_landmarks.npy"
            video_file = self.cropped_dir / f"{sentence_id}_{sentence_name}_cropped.mp4"

            if not landmark_file.exists() or not video_file.exists():
                continue
            if not sentence or sentence == "nan":
                continue

            self.samples.append(
                {
                    'landmark_file': landmark_file,
                    'video_file': video_file,
                    'sentence': sentence,
                }
            )

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise RuntimeError(
                "Nessun campione valido trovato. Verifica naming files tra CSV, landmarks e cropped."
            )

        # IMPORTANT: train e validation devono condividere lo stesso vocabolario,
        # altrimenti gli indici target non corrispondono ai logits del decoder.
        if shared_word2idx is not None and shared_idx2word is not None:
            self.word2idx = dict(shared_word2idx)
            self.idx2word = dict(shared_idx2word)
            self.vocab_size = max(self.vocab_size, len(self.word2idx))
            self.using_shared_vocab = True
        else:
            self.word2idx = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
            self.idx2word = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
            self._build_vocab()
            self.using_shared_vocab = False

        print(f"[Dataset] ✓ RealHow2SignDataset con {len(self.samples)} campioni validi")
        print(f"  - CSV: {self.csv_path}")
        print(f"  - Landmarks dir: {self.landmarks_dir}")
        print(f"  - Cropped dir: {self.cropped_dir}")
        print(f"  - Vocab size effettivo: {len(self.word2idx)}")
        if self.using_shared_vocab:
            print("  - Vocab source: shared from train dataset")
        if cv2 is None:
            print("  - OpenCV non installato: video frames verranno riempiti con zeri")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def _build_vocab(self):
        freq = {}
        for sample in self.samples:
            for tok in self._tokenize(sample['sentence']):
                freq[tok] = freq.get(tok, 0) + 1

        max_new_tokens = max(0, self.vocab_size - len(self.word2idx))
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for tok, _ in sorted_tokens[:max_new_tokens]:
            idx = len(self.word2idx)
            self.word2idx[tok] = idx
            self.idx2word[idx] = tok

    def _encode_sentence(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenize(sentence)
        token_ids = [self.word2idx['<bos>']]
        token_ids.extend(self.word2idx.get(tok, self.word2idx['<unk>']) for tok in tokens)
        token_ids.append(self.word2idx['<eos>'])

        if len(token_ids) > self.text_max_len:
            token_ids = token_ids[: self.text_max_len]
            token_ids[-1] = self.word2idx['<eos>']

        target_tokens = torch.zeros(self.text_max_len, dtype=torch.long)
        target_tokens[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)

        target_labels = torch.zeros(self.text_max_len, dtype=torch.long)
        if len(token_ids) > 1:
            target_labels[: len(token_ids) - 1] = torch.tensor(token_ids[1:], dtype=torch.long)
        target_labels[len(token_ids) - 1] = self.word2idx['<eos>']

        return target_tokens, target_labels

    def _load_landmarks(self, file_path: Path) -> torch.Tensor:
        lm = np.load(file_path)
        if lm.ndim == 1:
            lm = lm.reshape(1, -1)
        elif lm.ndim > 2:
            lm = lm.reshape(lm.shape[0], -1)

        if lm.shape[1] < self.landmark_dim:
            pad = np.zeros((lm.shape[0], self.landmark_dim - lm.shape[1]), dtype=np.float32)
            lm = np.concatenate([lm.astype(np.float32), pad], axis=1)
        else:
            lm = lm[:, : self.landmark_dim].astype(np.float32)

        if lm.shape[0] > self.video_max_frames:
            idxs = np.linspace(0, lm.shape[0] - 1, self.video_max_frames).astype(int)
            lm = lm[idxs]
        elif lm.shape[0] < self.video_max_frames:
            pad = np.zeros((self.video_max_frames - lm.shape[0], self.landmark_dim), dtype=np.float32)
            lm = np.concatenate([lm, pad], axis=0)

        return torch.from_numpy(lm)

    def _load_video_frames(self, file_path: Path) -> torch.Tensor:
        out = torch.zeros((self.video_max_frames, 3, 224, 224), dtype=torch.float32)
        if cv2 is None:
            return out

        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                print(f"[⚠] Video non aperto: {file_path.name}")
                return out

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                print(f"[⚠] Video con 0 frame: {file_path.name}")
                return out

            if total_frames > self.video_max_frames:
                frame_idxs = np.linspace(0, total_frames - 1, self.video_max_frames).astype(int)
            else:
                frame_idxs = np.arange(total_frames)

            frames_read = 0
            for i, frame_idx in enumerate(frame_idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok:
                    continue

                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
                    frame = frame.astype(np.float32) / 255.0
                    out[i] = torch.from_numpy(frame).permute(2, 0, 1)
                    frames_read += 1
                except Exception as e:
                    print(f"[⚠] Errore processando frame {i} da {file_path.name}: {e}")
                    continue

            cap.release()
            
            if frames_read == 0:
                print(f"[✗] CORRUPTED: {file_path.name} - nessun frame letto")
                return torch.zeros((self.video_max_frames, 3, 224, 224), dtype=torch.float32)
            
            if frames_read < len(frame_idxs) * 0.5:
                print(f"[⚠] PARTIAL: {file_path.name} - solo {frames_read}/{len(frame_idxs)} frame")
            
            return out
            
        except Exception as e:
            print(f"[✗] ERRORE VIDEO: {file_path.name} - {str(e)}")
            return torch.zeros((self.video_max_frames, 3, 224, 224), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                sample = self.samples[idx]
                landmarks = self._load_landmarks(sample['landmark_file'])
                video_frames = self._load_video_frames(sample['video_file'])
                
                # Check se il video è vuoto (segno di corruzione)
                if video_frames.sum() < 1.0:  # Se quasi tutti zeri, è corrupto
                    print(f"[🔄] Retry {attempt+1}/{max_retries}: video corrotto, carico un altro...")
                    idx = np.random.randint(0, len(self.samples))
                    continue
                
                target_tokens, target_labels = self._encode_sentence(sample['sentence'])

                return {
                    'landmarks': landmarks,
                    'video_frames': video_frames,
                    'target_tokens': target_tokens,
                    'target_labels': target_labels,
                }
            except Exception as e:
                print(f"[🔄] Errore al caricamento sample {idx} (tentativo {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    idx = np.random.randint(0, len(self.samples))
                else:
                    # Fallback: ritorna dati vuoti/dummy per evitare blocco
                    print(f"[⚠] Max retries raggiunto, ritorno dati dummy")
                    return {
                        'landmarks': torch.zeros(self.video_max_frames, self.landmark_dim, dtype=torch.float32),
                        'video_frames': torch.ones(self.video_max_frames, 3, 224, 224, dtype=torch.float32) * 0.5,
                        'target_tokens': torch.zeros(self.text_max_len, dtype=torch.long),
                        'target_labels': torch.zeros(self.text_max_len, dtype=torch.long),
                    }
        
        # Fallback finale (non dovrebbe mai arrivare qui)
        return {
            'landmarks': torch.zeros(self.video_max_frames, self.landmark_dim, dtype=torch.float32),
            'video_frames': torch.ones(self.video_max_frames, 3, 224, 224, dtype=torch.float32) * 0.5,
            'target_tokens': torch.zeros(self.text_max_len, dtype=torch.long),
            'target_labels': torch.zeros(self.text_max_len, dtype=torch.long),
        }


def collate_fn_real(batch):
    """Collate function per RealHow2SignDataset."""
    landmarks = torch.stack([item['landmarks'] for item in batch])
    video_frames = torch.stack([item['video_frames'] for item in batch])
    target_tokens = torch.stack([item['target_tokens'] for item in batch])
    target_labels = torch.stack([item['target_labels'] for item in batch])

    return {
        'landmarks': landmarks,
        'video_frames': video_frames,
        'target_tokens': target_tokens,
        'target_labels': target_labels,
    }


class MockSignLanguageDataset(Dataset):
    """
    Dataset mock per testing/demo con lazy loading.
    Genera dati on-the-fly per ogni batch, non pre-alloca tutto in memoria.
    In produzione, carica dati reali da disco (landmarks + video + labels).
    """
    
    def __init__(
        self,
        num_samples: int,
        video_max_frames: int = 50,  # Ridotto da 150 per efficienza
        landmark_dim: int = 2108,
        text_max_len: int = 128,  # Ridotto da 512 per efficienza mock
        vocab_size: int = 10000,
        device: torch.device = torch.device('cpu')
    ):
        """
        Crea dataset mock con lazy loading.
        
        Args:
            num_samples: Numero campioni totali
            video_max_frames: Numero frame per video
            landmark_dim: Dimensione vettore landmarks
            text_max_len: Max lunghezza sequenza testo target
            vocab_size: Dimensione vocabolario
            device: Device per tensori (ignorato - dati generati durante __getitem__)
        """
        
        self.num_samples = num_samples
        self.video_max_frames = video_max_frames
        self.landmark_dim = landmark_dim
        self.text_max_len = text_max_len
        self.vocab_size = vocab_size
        self.device = device
        
        # Seed per reproducibilità
        self.seed = 42
        
        print(f"[Dataset] ✓ MockDataset con {num_samples} campioni (lazy loading)")
        print(f"  - Video max frames: {video_max_frames}")
        print(f"  - Landmark dim: {landmark_dim}")
        print(f"  - Text max len: {text_max_len}")
        print(f"  - Vocab size: {vocab_size}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Genera un campione random al volo.
        Questo approccio evita di pre-allocare tutto in memoria.
        """
        # Set seed per reproducibilità
        torch.manual_seed(self.seed + idx)
        np.random.seed(self.seed + idx)
        
        # 1. Landmarks: (video_max_frames, landmark_dim)
        landmarks = torch.randn(
            self.video_max_frames,
            self.landmark_dim
        ) * 0.1  # Scale piccolo
        
        # 2. Video frames: (video_max_frames, 3, 224, 224)
        video_frames = torch.randn(
            self.video_max_frames,
            3,
            224,
            224
        ) * 0.5 + 0.5  # Normalizzazione [0, 1]
        
        # 3. Target tokens: (text_max_len,)
        target_tokens = torch.zeros(self.text_max_len, dtype=torch.long)
        seq_len = np.random.randint(10, self.text_max_len - 10)
        
        # BOS token (index 1)
        target_tokens[0] = 1
        # Contenuto: token casuali
        target_tokens[1:seq_len] = torch.randint(
            2,  # Start da 2 (1 è BOS, 0 è padding)
            self.vocab_size,
            (seq_len - 1,)
        )
        # EOS token
        target_tokens[seq_len] = 2
        
        # 4. Target labels: (text_max_len,)
        #    Shift di 1 per teacher forcing
        target_labels = torch.zeros_like(target_tokens)
        target_labels[:-1] = target_tokens[1:]
        target_labels[-1] = 2
        
        return {
            'landmarks': landmarks,
            'video_frames': video_frames,
            'target_tokens': target_tokens,
            'target_labels': target_labels
        }


# Custom collate function per gestire i dict
def collate_fn_mock(batch):
    """Collate function per MockSignLanguageDataset."""
    landmarks = torch.stack([item['landmarks'] for item in batch])
    video_frames = torch.stack([item['video_frames'] for item in batch])
    target_tokens = torch.stack([item['target_tokens'] for item in batch])
    target_labels = torch.stack([item['target_labels'] for item in batch])
    
    return {
        'landmarks': landmarks,
        'video_frames': video_frames,
        'target_tokens': target_tokens,
        'target_labels': target_labels
    }



# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class TrainingLogger:
    """
    Logger per metriche di training.
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Dizionari per accumulare metriche
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Header
        self.write_line("=== TRAINING LOG - PHASE 1 ===")
        self.write_line(f"Started: {datetime.now().isoformat()}")
        self.write_line("")
    
    def write_line(self, message: str):
        """Scrivi linea nel log."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        learning_rate: float,
        elapsed_time: float,
        val_accuracy: Optional[float] = None
    ):
        """Log statistiche epoch."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)
        if val_accuracy is not None:
            self.val_accuracies.append(val_accuracy)
        
        if val_loss is not None:
            val_text = f"Val Loss: {val_loss:.4f}"
        else:
            val_text = "Val Loss: skipped"

        message = (
            f"[Epoch {epoch:3d}] "
            f"Train Loss: {train_loss:.4f} | {val_text} | "
            f"LR: {learning_rate:.2e} | "
            f"Time: {elapsed_time:.2f}s"
        )
        if val_accuracy is not None:
            message += f" | Val Acc: {val_accuracy:.4f}"
        
        self.write_line(message)
    
    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        learning_rate: float
    ):
        """Log batch durante training (periodico)."""
        message = (
            f"[Epoch {epoch}] [{batch:4d}/{total_batches}] "
            f"Loss: {loss:.4f} | LR: {learning_rate:.2e}"
        )
        self.write_line(message)
    
    def save_summary(self):
        """Salva summary finale."""
        self.write_line("")
        self.write_line("=== TRAINING SUMMARY ===")
        self.write_line(f"Total epochs: {len(self.train_losses)}")
        if self.val_losses:
            self.write_line(f"Best val loss: {min(self.val_losses):.4f}")
        else:
            self.write_line("Best val loss: N/A (validation non eseguita)")
        self.write_line(f"Final train loss: {self.train_losses[-1]:.4f}")
        
        if self.val_accuracies:
            self.write_line(f"Best val accuracy: {max(self.val_accuracies):.4f}")


class MetricsCalculator:
    """
    Calcola metriche durante validation.
    """
    
    @staticmethod
    def token_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcola accuratezza token-level escludendo padding.
        
        Args:
            predictions: (batch, seq_len, vocab_size) logits
            targets: (batch, seq_len) indici token
        
        Returns:
            Accuratezza (0-1)
        """
        # Prendi token con massima probabilità
        pred_tokens = torch.argmax(predictions, dim=-1)  # (batch, seq_len)
        
        # Maschera padding (token 0)
        mask = targets != 0  # (batch, seq_len)
        
        # Calcola match
        matches = (pred_tokens == targets) & mask
        accuracy = matches.sum().float() / mask.sum().float()
        
        return accuracy.item()


# =============================================================================
# PHASE 1 TRAINING LOOP
# =============================================================================

class Phase1Trainer:
    """
    Trainer per la Phase 1 di addestramento.
    
    Gestisce:
      - Congelamento selettivo di MobileNet
      - Training loop con validation
      - Checkpointing e salvataggio modelli
      - Logging metriche
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device
    ):
        """
        Inizializza il trainer.
        
        Args:
            model: SignLanguageTranslationModel
            config: TrainingConfig
            device: torch.device (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = TrainingLogger(config.log_dir)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignora padding
            reduction='mean'
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.base_learning_rate = config.learning_rate
        
        # Learning Rate Scheduler
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.scheduler_min_lr
            )
        
        # Mixed Precision Training (AMP) Scaler
        self.scaler = None
        self.use_amp = config.use_mixed_precision and str(device) != 'cpu'
        if self.use_amp:
            self.scaler = GradScaler()
            print("[✓] Mixed Precision Training (AMP) ENABLED - ridurr\u00e0 memoria GPU ~30-40%")
        
        # Tracking miglior validazione
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Log config
        self.logger.write_line("Configuration:")
        for key, value in config.to_dict().items():
            self.logger.write_line(f"  {key}: {value}")
        self.logger.write_line("")

    def _collect_backprop_diagnostics(self, pre_step_params: Dict[str, torch.Tensor]) -> Tuple[float, float, int]:
        """
        Calcola segnali di backprop: norma dei gradienti e variazione pesi.

        Args:
            pre_step_params: snapshot parametri prima di optimizer.step()

        Returns:
            (grad_l2_norm, update_l2_norm, tracked_params)
        """
        grad_sq_sum = 0.0
        update_sq_sum = 0.0
        tracked = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is not None:
                grad_sq_sum += float(torch.sum(param.grad.detach().float() ** 2).item())

            if name in pre_step_params:
                delta = (param.detach().float() - pre_step_params[name]).reshape(-1)
                update_sq_sum += float(torch.sum(delta ** 2).item())
                tracked += 1

        grad_norm = float(np.sqrt(max(grad_sq_sum, 0.0)))
        update_norm = float(np.sqrt(max(update_sq_sum, 0.0)))
        return grad_norm, update_norm, tracked

    def _set_learning_rate(self, lr: float):
        """Imposta learning rate su tutti i parameter groups."""
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def _apply_warmup(self, epoch: int) -> bool:
        """
        Applica linear warmup del learning rate nelle prime epoche.

        Returns:
            True se warmup applicato in questa epoca, False altrimenti.
        """
        if not self.config.use_lr_warmup or self.config.warmup_epochs <= 0:
            return False

        if epoch > self.config.warmup_epochs:
            return False

        start_factor = float(self.config.warmup_start_factor)
        start_factor = max(1e-6, min(start_factor, 1.0))
        start_lr = self.base_learning_rate * start_factor

        if self.config.warmup_epochs == 1:
            warmup_lr = self.base_learning_rate
        else:
            progress = (epoch - 1) / (self.config.warmup_epochs - 1)
            warmup_lr = start_lr + progress * (self.base_learning_rate - start_lr)

        self._set_learning_rate(warmup_lr)
        return True
    
    def freeze_mobilenet(self):
        """
        Congela i pesi di MobileNet (2D-CNN visual branch).
        
        IMPORTANTE: MobileNet è dentro il modulo CNN1DGRUModule,
        che è dentro DualStreamSignLanguageModel (encoder).
        """
        print("\n" + "="*70)
        print("CONGELAMENTO SELETTIVO MOBILENET")
        print("="*70)
        
        # Il backbone reale e' in:
        # model.encoder.video_cnn_gru.feature_extractor.backbone
        newly_frozen_params = 0
        already_frozen_params = 0
        backbone_total_params = 0

        backbone = None
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'video_cnn_gru'):
            video_branch = self.model.encoder.video_cnn_gru
            if hasattr(video_branch, 'feature_extractor') and hasattr(video_branch.feature_extractor, 'backbone'):
                backbone = video_branch.feature_extractor.backbone

        if backbone is not None:
            for name, param in backbone.named_parameters():
                backbone_total_params += param.numel()
                if param.requires_grad:
                    param.requires_grad = False
                    newly_frozen_params += param.numel()
                    print(f"❄️  FROZEN NOW: {name}")
                else:
                    already_frozen_params += param.numel()
        else:
            # Fallback robusto in caso cambi la struttura dei moduli.
            for name, param in self.model.named_parameters():
                if 'feature_extractor.backbone' in name:
                    backbone_total_params += param.numel()
                    if param.requires_grad:
                        param.requires_grad = False
                        newly_frozen_params += param.numel()
                        print(f"❄️  FROZEN NOW: {name}")
                    else:
                        already_frozen_params += param.numel()

        total_frozen = newly_frozen_params + already_frozen_params
        print(f"\n✓ MobileNet backbone params: {backbone_total_params:,}")
        print(f"  - Congelati ora: {newly_frozen_params:,}")
        print(f"  - Gia congelati: {already_frozen_params:,}")
        print(f"  - Totale congelati backbone: {total_frozen:,}")
        print("="*70 + "\n")
    
    def print_trainable_parameters(self):
        """
        Stampa riepilogo parametri allenabili vs congelati.
        """
        trainable_params = 0
        trainable_buffers = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        total_params = trainable_params + frozen_params
        
        self.logger.write_line("\n" + "="*70)
        self.logger.write_line("TRAINABLE vs FROZEN PARAMETERS")
        self.logger.write_line("="*70)
        self.logger.write_line(f"Trainable params:  {trainable_params:>15,}")
        self.logger.write_line(f"Frozen params:     {frozen_params:>15,}")
        self.logger.write_line(f"Total params:      {total_params:>15,}")
        self.logger.write_line(f"Percentage train:  {100*trainable_params/total_params:>14.2f}%")
        self.logger.write_line("="*70 + "\n")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Addestra un epoch.
        
        Args:
            train_loader: DataLoader per training
            epoch: Numero epoch
        
        Returns:
            Loss medio epoch
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        backprop_checks_done = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch (può essere dict o tuple)
            if isinstance(batch, dict):
                landmarks = batch['landmarks'].to(self.device)
                video_frames = batch['video_frames'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                target_labels = batch['target_labels'].to(self.device)
            else:
                # Formato legacy (tuple)
                landmarks, video_frames, target_tokens, target_labels = batch
                landmarks = landmarks.to(self.device)
                video_frames = video_frames.to(self.device)
                target_tokens = target_tokens.to(self.device)
                target_labels = target_labels.to(self.device)
            
            # Forward pass
            try:
                # Mixed Precision: wrap forward in autocast
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(
                            landmarks=landmarks,
                            video_frames=video_frames,
                            target_tokens=target_tokens
                        )  # (batch, seq_len, vocab_size)
                        
                        # Reshape per loss
                        batch_size, seq_len, vocab_size = logits.shape
                        logits_flat = logits.view(batch_size * seq_len, vocab_size)
                        targets_flat = target_labels.view(batch_size * seq_len)
                        
                        # Compute loss
                        loss = self.loss_fn(logits_flat, targets_flat)
                else:
                    logits = self.model(
                        landmarks=landmarks,
                        video_frames=video_frames,
                        target_tokens=target_tokens
                    )  # (batch, seq_len, vocab_size)
                    
                    # Reshape per loss
                    batch_size, seq_len, vocab_size = logits.shape
                    logits_flat = logits.view(batch_size * seq_len, vocab_size)
                    targets_flat = target_labels.view(batch_size * seq_len)
                    
                    # Compute loss
                    loss = self.loss_fn(logits_flat, targets_flat)
                
            except Exception as e:
                print(f"❌ Errore in forward pass batch {batch_idx}: {e}")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()

            run_backprop_check = (
                self.config.verify_backprop
                and (batch_idx + 1) % self.config.verify_backprop_every_n_batches == 0
                and backprop_checks_done < self.config.verify_backprop_max_checks_per_epoch
            )

            pre_step_params: Dict[str, torch.Tensor] = {}
            if run_backprop_check:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        pre_step_params[name] = param.detach().float().clone()
            
            if self.use_amp and self.scaler is not None:
                # Mixed precision: scale loss e backward
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (importantissimo per Transformer)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm
                )
                
                # Optimization step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard backward
                loss.backward()
                
                # Gradient clipping (importantissimo per Transformer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm
                )
                
                # Optimization step
                self.optimizer.step()

            if run_backprop_check:
                grad_norm, update_norm, tracked = self._collect_backprop_diagnostics(pre_step_params)
                backprop_checks_done += 1
                self.logger.write_line(
                    f"[BackpropCheck] epoch={epoch} batch={batch_idx + 1} "
                    f"grad_norm={grad_norm:.3e} update_norm={update_norm:.3e} tracked_params={tracked}"
                )
                if update_norm <= 0.0:
                    self.logger.write_line(
                        "[BackpropCheck][WARN] update_norm e' zero: controlla grad scaler, optimizer, freeze e loss."
                    )
            
            # Accumula loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log periodicamente
            if (batch_idx + 1) % self.config.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_loss = total_loss / num_batches
                self.logger.log_batch(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    total_batches=len(train_loader),
                    loss=avg_loss,
                    learning_rate=current_lr
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Valida il modello.
        
        Args:
            val_loader: DataLoader per validation
        
        Returns:
            (avg_loss, avg_accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Unpack batch (può essere dict o tuple)
                if isinstance(batch, dict):
                    landmarks = batch['landmarks'].to(self.device)
                    video_frames = batch['video_frames'].to(self.device)
                    target_tokens = batch['target_tokens'].to(self.device)
                    target_labels = batch['target_labels'].to(self.device)
                else:
                    # Formato legacy (tuple)
                    landmarks, video_frames, target_tokens, target_labels = batch
                    landmarks = landmarks.to(self.device)
                    video_frames = video_frames.to(self.device)
                    target_tokens = target_tokens.to(self.device)
                    target_labels = target_labels.to(self.device)
                
                # Forward pass con autocast per ridurre memoria
                try:
                    if self.use_amp:
                        with autocast(device_type='cuda', dtype=torch.float16):
                            logits = self.model(
                                landmarks=landmarks,
                                video_frames=video_frames,
                                target_tokens=target_tokens
                            )  # (batch, seq_len, vocab_size)
                            
                            # Reshape per loss
                            batch_size, seq_len, vocab_size = logits.shape
                            logits_flat = logits.view(batch_size * seq_len, vocab_size)
                            targets_flat = target_labels.view(batch_size * seq_len)
                            
                            # Compute loss
                            loss = self.loss_fn(logits_flat, targets_flat)
                    else:
                        logits = self.model(
                            landmarks=landmarks,
                            video_frames=video_frames,
                            target_tokens=target_tokens
                        )  # (batch, seq_len, vocab_size)
                        
                        # Reshape per loss
                        batch_size, seq_len, vocab_size = logits.shape
                        logits_flat = logits.view(batch_size * seq_len, vocab_size)
                        targets_flat = target_labels.view(batch_size * seq_len)
                        
                        # Compute loss
                        loss = self.loss_fn(logits_flat, targets_flat)
                    
                    # Compute accuracy
                    accuracy = MetricsCalculator.token_accuracy(logits, target_labels)
                    
                except Exception as e:
                    print(f"❌ Errore in validation batch {batch_idx}: {e}")
                    continue
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

                if max_batches is not None and num_batches >= max_batches:
                    break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """
        Salva checkpoint del modello.
        
        Args:
            epoch: Numero epoch
            val_loss: Validation loss
            is_best: Se True, salva come best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict(),
        }
        
        # Regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Main training loop per Phase 1.
        
        Args:
            train_loader: DataLoader training
            val_loader: DataLoader validation
        """
        import time
        
        # Stampa config e freeze
        self.freeze_mobilenet()
        self.print_trainable_parameters()
        
        self.logger.write_line("\n" + "="*70)
        self.logger.write_line("INIZIO TRAINING PHASE 1")
        self.logger.write_line("="*70 + "\n")
        
        last_val_loss: Optional[float] = None
        last_val_accuracy: Optional[float] = None
        no_improve_val_checks = 0

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()

            warmup_active = self._apply_warmup(epoch)
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation (periodica su subset se configurata)
            did_validation = False
            if val_loader is not None and (epoch % self.config.val_check_interval == 0):
                # Libera memoria GPU prima della validation
                torch.cuda.empty_cache()
                
                val_loss, val_accuracy = self.validate(
                    val_loader,
                    max_batches=self.config.val_max_batches
                )
                last_val_loss = val_loss
                last_val_accuracy = val_accuracy
                did_validation = True
            else:
                val_loss = None
                val_accuracy = None
            
            epoch_time = time.time() - epoch_start
            
            # Learning rate scheduler step SOLO su validation reale.
            # Con validation ogni 10 epoche, usare la train loss qui porta
            # a riduzioni premature del LR e peggiora la generalizzazione.
            if (
                self.scheduler is not None
                and did_validation
                and val_loss is not None
                and not warmup_active
            ):
                self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                elapsed_time=epoch_time,
                val_accuracy=val_accuracy
            )
            
            # Checkpointing (solo se validation eseguita)
            if did_validation:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    no_improve_val_checks = 0
                else:
                    no_improve_val_checks += 1

                if (epoch % self.config.save_frequency == 0) or is_best:
                    self.save_checkpoint(epoch, val_loss, is_best)

                # Early stopping in numero di validation check consecutivi senza miglioramento
                if no_improve_val_checks >= self.config.early_stopping_patience_checks:
                    self.logger.write_line(
                        f"\n⚠️  Early stopping: no improvement for "
                        f"{self.config.early_stopping_patience_checks} validation checks"
                    )
                    break
            else:
                if val_loader is not None:
                    self.logger.write_line(
                        f"[Epoch {epoch:3d}] Validation skipped (interval={self.config.val_check_interval})"
                    )
        
        self.logger.write_line("\n" + "="*70)
        self.logger.write_line(f"TRAINING COMPLETATO")
        self.logger.write_line(f"Best epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
        self.logger.write_line("="*70 + "\n")
        
        self.logger.save_summary()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Entry point per Phase 1 Training.
    """
    parser = argparse.ArgumentParser(description="Phase 1 Training - Sign Language Translation")
    parser.add_argument("--num_samples", type=int, default=2000, help="Numero campioni training (default: 2000)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (OTTIMIZZATO da 32 a 16 per ridurre OOM)")
    parser.add_argument("--num_epochs", type=int, default=100, help="Numero epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Numero epoche warmup LR (default: 5)")
    parser.add_argument("--warmup_start_factor", type=float, default=0.2, help="Fattore LR iniziale warmup (default: 0.2)")
    parser.add_argument("--disable_warmup", action="store_true", help="Disabilita warmup LR")
    parser.add_argument("--verify_backprop", action="store_true", help="Verifica grad e update pesi durante training")
    parser.add_argument("--verify_backprop_every_n_batches", type=int, default=50, help="Esegui backprop check ogni N batch (default: 50)")
    parser.add_argument("--verify_backprop_max_checks_per_epoch", type=int, default=2, help="Massimo numero di backprop check per epoca (default: 2)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset/how2sign_realigned_train.csv",
        help="Path CSV realigned train",
    )
    parser.add_argument(
        "--landmarks_dir",
        type=str,
        default="dataset/landmarks_normalized",
        help="Cartella landmarks normalizzati",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="dataset/cropped",
        help="Cartella video cropped",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Frazione validation [0,1). Default 0.2 (80/20 train/val), usa 0 per train-only",
    )
    parser.add_argument(
        "--use_separate_val",
        action="store_true",
        help="Usa un dataset validation separato (ignora --val_split)",
    )
    parser.add_argument(
        "--val_csv_path",
        type=str,
        default="dataset/how2sign_realigned_val.csv",
        help="Path CSV realigned validation",
    )
    parser.add_argument(
        "--val_landmarks_dir",
        type=str,
        default="dataset/landmarks_validation_normalized",
        help="Cartella landmarks validation normalizzati",
    )
    parser.add_argument(
        "--val_video_dir",
        type=str,
        default="dataset/cropped_validation",
        help="Cartella video cropped validation",
    )
    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=200,
        help="Numero campioni validation (default: 200, 0=all)",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=10,
        help="Esegui validation ogni N epoch (default: 10)",
    )
    parser.add_argument(
        "--val_max_batches",
        type=int,
        default=10,
        help="Numero massimo di batch per validation (default: 10, 0=full dataset)",
    )
    args = parser.parse_args()
    
    # Configurazione
    config = TrainingConfig()
    config.num_samples = args.num_samples
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.warmup_epochs = max(0, args.warmup_epochs)
    config.warmup_start_factor = max(1e-6, min(args.warmup_start_factor, 1.0))
    config.use_lr_warmup = not args.disable_warmup
    config.verify_backprop = args.verify_backprop
    config.verify_backprop_every_n_batches = max(1, args.verify_backprop_every_n_batches)
    config.verify_backprop_max_checks_per_epoch = max(1, args.verify_backprop_max_checks_per_epoch)
    config.val_split = args.val_split
    config.val_check_interval = max(1, args.val_check_interval)
    config.val_max_batches = None if args.val_max_batches == 0 else args.val_max_batches

    if not (0.0 <= config.val_split < 1.0):
        raise ValueError(f"--val_split deve essere in [0,1). Ricevuto: {config.val_split}")
    
    if args.device:
        config.device = torch.device(args.device)
    
    # Set seed per riproducibilità
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    print("\n" + "="*70)
    print("PHASE 1 TRAINING - Sign Language Translation")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Num samples: {config.num_samples}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Validation split: {config.val_split}")
    print(f"Use separate validation: {args.use_separate_val}")
    if args.use_separate_val:
        print(f"Val CSV path: {args.val_csv_path}")
        print(f"Val landmarks dir: {args.val_landmarks_dir}")
        print(f"Val video dir: {args.val_video_dir}")
        print(f"Val num samples: {args.val_num_samples}")
    print(f"Validation interval: {config.val_check_interval}")
    print(f"Validation max batches: {config.val_max_batches}")
    print(f"CSV path: {args.csv_path}")
    print(f"Landmarks dir: {args.landmarks_dir}")
    print(f"Video dir: {args.video_dir}")
    print("="*70)
    
    # Memory optimization
    if str(config.device) != 'cpu':
        print("\n🚀 MEMORY OPTIMIZATION:")
        print("   ✓ Batch size reduced: 32 → 16")
        print("   ✓ Video frames reduced: 150 → 100")
        print("   ✓ Text max length reduced: 512 → 256")
        print("   ✓ Mixed Precision Training (AMP) ENABLED")
        print("\n⚡ Additional optimization (optional):")
        print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("   This enables expandable memory segments to reduce fragmentation.")
        print()
    
    print("="*70 + "\n")
    
    # Auto-enable separate validation if files exist
    val_csv = Path(args.val_csv_path)
    val_landmarks = Path(args.val_landmarks_dir)
    val_video = Path(args.val_video_dir)
    
    if not args.use_separate_val and all([val_csv.exists(), val_landmarks.exists(), val_video.exists()]):
        print("✓ Validation files found. Auto-enabling separate validation dataset.")
        args.use_separate_val = True
        print(f"  - CSV: {val_csv}")
        print(f"  - Landmarks: {val_landmarks}")
        print(f"  - Video: {val_video}\n")
    
    # Dataset reale
    print("Creating REAL dataset...")
    dataset = RealHow2SignDataset(
        csv_path=Path(args.csv_path),
        landmarks_dir=Path(args.landmarks_dir),
        cropped_dir=Path(args.video_dir),
        max_samples=config.num_samples,
        video_max_frames=config.video_max_frames,
        landmark_dim=config.landmark_dim,
        text_max_len=config.text_max_len,
        vocab_size=config.text_vocab_size,
    )

    val_dataset = None
    if args.use_separate_val:
        val_max_samples = None if args.val_num_samples == 0 else args.val_num_samples
        val_dataset = RealHow2SignDataset(
            csv_path=Path(args.val_csv_path),
            landmarks_dir=Path(args.val_landmarks_dir),
            cropped_dir=Path(args.val_video_dir),
            max_samples=val_max_samples,
            video_max_frames=config.video_max_frames,
            landmark_dim=config.landmark_dim,
            text_max_len=config.text_max_len,
            vocab_size=config.text_vocab_size,
            shared_word2idx=dataset.word2idx,
            shared_idx2word=dataset.idx2word,
        )
        train_dataset = dataset
    else:
        # Split train/val (opzionale)
        if config.val_split > 0.0:
            train_size = int((1 - config.val_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = dataset
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False,
        collate_fn=collate_fn_real
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if config.device.type == 'cuda' else False,
            collate_fn=collate_fn_real
        )
        print(f"✓ Dataset created: {len(train_dataset)} train, {len(val_dataset)} validation\n")
    else:
        print(f"✓ Dataset created: {len(train_dataset)} train, 0 validation (train-only mode)\n")
    
    # Model
    print("Creating model...")
    model = SignLanguageTranslationModel(
        # Encoder params
        video_hidden_dim=config.video_hidden_dim,
        landmark_dim=config.landmark_dim,
        num_encoder_layers=config.num_encoder_layers,
        freeze_mobilenet=config.freeze_mobilenet,
        
        # Decoder params
        text_vocab_size=config.text_vocab_size,
        text_max_len=config.text_max_len,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        
        # Device
        device=config.device,
    )
    model = model.to(config.device)
    print(f"✓ Model created\n")
    
    # Trainer
    trainer = Phase1Trainer(
        model=model,
        config=config,
        device=config.device
    )
    
    # Training loop
    trainer.train(train_loader, val_loader)
    
    print("\n✓ Training completato!")
    print(f"  Miglior validazione loss: {trainer.best_val_loss:.4f} (Epoch {trainer.best_epoch})")
    print(f"  Checkpoints salvati in: {config.checkpoint_dir}")
    print(f"  Logs salvati in: {config.log_dir}")


if __name__ == "__main__":
    main()
