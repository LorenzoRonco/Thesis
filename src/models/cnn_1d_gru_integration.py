"""
Dual-Stream Sign Language Translation Architecture
====================================================

Architettura con dual-stream paralleli per Sign Language Translation:
  - STREAM 1 (Landmarks): Landmarks (MediaPipe) → Transformer Encoder
  - STREAM 2 (Video): RGB Frames → CNN-1D-GRU (2D-CNN + 1D-CNN + GRU)
  - FUSION: Combina i due stream e apprende pesi contestuali
  - OUTPUT: Layer densi per embedding finale

Dimensioni:
    Landmarks: (B, T, 2108) → Transformer → (B, T, 512)
    Video: (B, T, 3, H, W) → CNN-1D-GRU → (B, T, 512)
    Fused: (B, T, 1024) → Linear layers → (B, T, 512)

Author: Thesis Project
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict
from src.models.cnn_1d_gru_module import CNN1DGRUModule, CNN1DGRUConfig
from src.models.transformer_encoder import TransformerEncoder


# =============================================================================
# Dual-Stream Fusion Module (Con weighted combination)
# =============================================================================

class DualStreamFusionModule(nn.Module):
    """
    Modulo di fusione per combinare gli output di due stream paralleli
    tramite layer densi che apprendono i pesi per le rappresentazioni.
    
    Input:
        stream1: (B, T, hidden_dim) da Transformer (landmarks)
        stream2: (B, T, hidden_dim) da CNN-1D-GRU (video)
    
    Output:
        fused: (B, T, hidden_dim) con pesi contestuali
    
    Architettura (OTTIMIZZATA):
        [stream1, stream2] → concatenate → (B, T, 2*hidden_dim)
        ↓
        Linear(2*hidden_dim → hidden_dim) → ReLU → Dropout
        ↓
        Linear(hidden_dim → hidden_dim) → output
    """
    
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Layer densi per il weighting (SEMPLIFICATO: rimosso 1 layer intermedio)
        self.fusion_net = nn.Sequential(
            # Input: concatenazione di stream1 e stream2
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Output: embedding fuso con stessa dimensionalità
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # LayerNorm per stabilità
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        stream1: torch.Tensor,
        stream2: torch.Tensor
    ) -> torch.Tensor:
        """
        Fusion dei due stream tramite layer densi.
        
        Args:
            stream1: (B, T, hidden_dim) da Transformer Encoder (landmarks)
            stream2: (B, T, hidden_dim) da CNN-1D-GRU (video)
        
        Returns:
            fused: (B, T, hidden_dim) tensore fuso e normalizzato
        
        Gestione dimensioni:
            stream1:      (B, T, 512)
            stream2:      (B, T, 512)
                ↓ concatenate su dimensione feature
            concat:       (B, T, 1024)
                ↓ reshape per linear
            reshaped:     (B*T, 1024)
                ↓ fusion_net
            fused:        (B*T, 512)
                ↓ reshape
            output:       (B, T, 512)
                ↓ layer_norm
            final:        (B, T, 512)
        """
        batch_size, time_steps, hidden_dim = stream1.shape
        
        # ========== Concatenate streams ==========
        # Concatenate su feature dimension
        # Input: stream1=(B, T, 512), stream2=(B, T, 512)
        # Output: concat=(B, T, 1024)
        concat = torch.cat([stream1, stream2], dim=-1)
        
        # ========== Reshape for linear layers ==========
        # Linear layers aspettano (batch_size, features)
        # reshape da (B, T, 1024) → (B*T, 1024)
        concat_reshaped = concat.view(batch_size * time_steps, -1)
        
        # ========== Apply fusion network ==========
        # Linear layers calcolare pesi e combinazioni
        # (B*T, 1024) → (B*T, 512)
        fused_flat = self.fusion_net(concat_reshaped)
        
        # ========== Reshape back to sequence ==========
        # (B*T, 512) → (B, T, 512)
        fused = fused_flat.view(batch_size, time_steps, hidden_dim)
        
        # ========== Apply layer normalization ==========
        # Normalizza per stabilità
        fused = self.layer_norm(fused)
        
        return fused


# =============================================================================
# Dual-Stream Sign Language Translation Architecture
# =============================================================================

class DualStreamSignLanguageModel(nn.Module):
    """
    Architettura dual-stream per Sign Language Translation.
    
    Pipeline:
    1. STREAM 1 (Landmarks): 
       Input (B, T, 2108) landmark features
       → TransformerEncoder → (B, T, 512)
    
    2. STREAM 2 (Video RGB):
       Input (B, T, 3, H, W) video frames
       → CNN-1D-GRU (2D-CNN MobileNet + 1D-CNN + GRU) → (B, T, 512)
    
    3. FUSION:
       Concatenate e combina tramite layer densi
       (B, T, 1024) → DualStreamFusionModule → (B, T, 512)
    
    4. CLASSIFICATION HEAD (opzionale):
       Per-frame o aggregated classification
    
    Args:
        hidden_dim (int): Dimensione latente attraverso il pipeline
        landmark_dim (int): Dimensione input landmarks (default: 2108)
        vocab_size (int): Output vocabulary size
        num_transformer_layers (int): Numero layer Transformer
        use_classification_head (bool): Aggiungi classification head
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        landmark_dim: int = 2108,
        vocab_size: int = 1000,
        num_transformer_layers: int = 4,
        use_classification_head: bool = True,
        freeze_mobilenet: bool = False,
        bidirectional_gru: bool = False,
        num_gru_layers: int = 1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.landmark_dim = landmark_dim
        self.vocab_size = vocab_size
        
        # ==================== STREAM 1: Landmarks → TransformerEncoder ====================
        # Input: (B, T, 2108) normalized landmarks
        # Output: (B, T, 512)
        self.landmark_transformer = TransformerEncoder(
            landmark_dim=landmark_dim,  # 2108
            hidden_dim=hidden_dim,      # 512
            num_layers=num_transformer_layers,
            num_heads=8,
            dropout=0.1,
        )
        
        # ==================== STREAM 2: Video RGB → CNN-1D-GRU ====================
        # Input: (B, T, 3, H, W) video frames
        # Output: (B, T, 512)
        self.video_cnn_gru = CNN1DGRUModule(
            hidden_dim=hidden_dim,
            num_gru_layers=num_gru_layers,
            gru_dropout=0.1,
            conv1d_dropout=0.1,
            bidirectional_gru=bidirectional_gru,
            freeze_mobilenet=freeze_mobilenet,
            pretrained_mobilenet=True,
        )
        
        # ==================== FUSION: Combina i due stream ====================
        # Input: (B, T, 512) da stream1 + (B, T, 512) da stream2
        # Output: (B, T, 512) fused
        self.fusion_module = DualStreamFusionModule(
            hidden_dim=hidden_dim,
            dropout=0.1,
        )
        
        # ==================== CLASSIFICATION HEAD (opzionale) ====================
        if use_classification_head:
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vocab_size)
            )
        else:
            self.classification_head = None
    
    def forward(
        self,
        landmarks: torch.Tensor,
        video_frames: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass completo attraverso il dual-stream.
        
        Args:
            landmarks: (batch_size, time_steps, 2108) normalized landmarks
            video_frames: (batch_size, time_steps, 3, height, width) video RGB
            return_intermediate: Se True, ritorna tutti gli output intermedi
        
        Returns:
            output: (batch_size, vocab_size) o (batch_size, time_steps, vocab_size)
            [intermediate]: Dict con stream1, stream2, fused se requested
        
        Flusso dimensioni completo:
        
        LANDMARKS PATH:
            Input:        (B, T, 2108)
            TransEnc:     (B, T, 512)
        
        VIDEO PATH:
            Input:        (B, T, 3, 224, 224)
            MobileNet:    (B, T, 1280)
            Conv1D:       (B, T, 512)
            GRU:          (B, T, 512)
        
        FUSION PATH:
            Concat:       (B, T, 1024) = concat[(B,T,512), (B,T,512)]
            Linear/ReLU:  (B, T, 512)
            Output:       (B, T, 512)
        
        CLASSIFICATION (opzionale):
            Mean pool:    (B, 512)
            Linear:       (B, vocab_size)
        """
        
        # ========== STREAM 1: Process Landmarks ==========
        # Input:  (B, T, 2108)
        # Output: (B, T, 512)
        stream1_output = self.landmark_transformer(landmarks)
        
        # ========== STREAM 2: Process Video Frames ==========
        # Input:  (B, T, 3, H, W)
        # Output: (B, T, 512)
        stream2_output, h_n = self.video_cnn_gru(video_frames)
        
        # ========== FUSION: Combine Streams ==========
        # stream1_output: (B, T, 512)
        # stream2_output: (B, T, 512)
        # Output:        (B, T, 512)
        fused_output = self.fusion_module(stream1_output, stream2_output)
        
        # ========== CLASSIFICATION HEAD (opzionale) ==========
        if self.classification_head is not None:
            # Temporal aggregation (media su T)
            # Input:  (B, T, 512)
            # Output: (B, 512)
            pooled = fused_output.mean(dim=1)
            
            # Classification
            # Input:  (B, 512)
            # Output: (B, vocab_size)
            logits = self.classification_head(pooled)
        else:
            # Return fused features without classification
            logits = fused_output
        
        # ========== Return intermediate if requested ==========
        if return_intermediate:
            intermediate = {
                'landmarks': stream1_output,
                'video': stream2_output,
                'fused': fused_output,
            }
            return logits, intermediate
        
        return logits
    
    def freeze_video_backbone(self):
        """Congela MobileNet nel video branch."""
        self.video_cnn_gru.freeze_mobilenet()
    
    def unfreeze_video_backbone(self):
        """Scongela MobileNet nel video branch."""
        self.video_cnn_gru.unfreeze_mobilenet()


# =============================================================================
# Training Utilities
# =============================================================================

class TrainConfig:
    """Configurazione per il training."""
    
    def __init__(
        self,
        num_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        warmup_epochs: int = 2,
        device: str = 'cuda',
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.warmup_epochs = warmup_epochs
        self.device = device
    
    def __str__(self):
        return (
            f"TrainConfig(\n"
            f"  epochs={self.num_epochs},\n"
            f"  batch_size={self.batch_size},\n"
            f"  lr={self.learning_rate},\n"
            f"  weight_decay={self.weight_decay},\n"
            f"  grad_clip={self.grad_clip_norm}\n"
            f")"
        )


class Trainer:
    """Trainer per il modello completo."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Esegui un'epoch di training sul dual-stream."""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (landmarks, frames, targets) in enumerate(train_loader):
            landmarks = landmarks.to(self.device)
            frames = frames.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass attraverso dual-stream
            # landmarks: (B, T, 2108)
            # frames: (B, T, 3, H, W)
            # Ritorna logits: (B, vocab_size)
            logits = self.model(landmarks, frames)
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (importante per GRU)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_norm
            )
            
            self.optimizer.step()
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Esegui la validazione."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        for landmarks, frames, targets in val_loader:
            landmarks = landmarks.to(self.device)
            frames = frames.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(landmarks, frames)
            loss = self.criterion(logits, targets)
            
            val_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_val_loss, accuracy
    
    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Esegui il training completo per num_epochs."""
        print(f"Starting training on {self.device}")
        print(self.config)
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate schedule
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")
                is_best = " ← BEST"
            else:
                is_best = ""
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.1f}%{is_best}")
        
        print("Training complete!")
    
    def save_checkpoint(self, path: str):
        """Salva checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Carica checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


# =============================================================================
# Full Example: End-to-End Training
# =============================================================================

def example_full_pipeline():
    """Esempio completo di training della architettura dual-stream."""
    print("\n" + "=" * 70)
    print(" Dual-Stream Sign Language Translation Pipeline")
    print(" [Landmarks + Video RGB] → Fusion → Classification")
    print("=" * 70)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ===== 1. Create Model =====
    print("\n1. Creating dual-stream model...")
    model = DualStreamSignLanguageModel(
        hidden_dim=512,
        landmark_dim=2108,
        vocab_size=1000,
        num_transformer_layers=4,
        use_classification_head=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params/1e6:.1f}M")
    print(f"   - Landmark Transformer: ~15M")
    print(f"   - Video CNN-1D-GRU: ~15M")
    print(f"   - Fusion module: ~2M")
    print(f"   - Classification head: ~1M")
    
    # ===== 2. Create Synthetic Dataset =====
    print("\n2. Creating synthetic dual-stream dataset...")
    n_train = 400
    n_val = 100
    time_steps = 100
    
    # Landmarks data: (n_samples, time_steps, 2108)
    X_landmarks_train = torch.randn(n_train, time_steps, 2108)
    X_landmarks_val = torch.randn(n_val, time_steps, 2108)
    
    # Video data: (n_samples, time_steps, 3, H, W)
    X_video_train = torch.randn(n_train, time_steps, 3, 224, 224)
    X_video_val = torch.randn(n_val, time_steps, 3, 224, 224)
    
    # Targets
    y_train = torch.randint(0, 1000, (n_train,))
    y_val = torch.randint(0, 1000, (n_val,))
    
    # Dataset with both inputs
    train_dataset = TensorDataset(X_landmarks_train, X_video_train, y_train)
    val_dataset = TensorDataset(X_landmarks_val, X_video_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   - Landmarks shape: {X_landmarks_train.shape}")
    print(f"   - Video shape: {X_video_train.shape}")
    
    # ===== 3. Setup Training =====
    print("\n3. Setting up training...")
    config = TrainConfig(
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        grad_clip_norm=1.0,
        device=str(device),
    )
    
    trainer = Trainer(model, config, device)
    
    # ===== 4. Train =====
    print("\n4. Training dual-stream architecture...")
    trainer.train_full(train_loader, val_loader)
    
    # ===== 5. Final Evaluation =====
    print("\n5. Final Evaluation...")
    model.eval()
    with torch.no_grad():
        final_val_loss, final_val_acc = trainer.validate(val_loader)
    
    print(f"\n   Final validation loss: {final_val_loss:.4f}")
    print(f"   Final validation accuracy: {final_val_acc:.1f}%")
    
    # ===== 6. Inspect Intermediate Outputs =====
    print("\n6. Inspecting intermediate stream outputs...")
    model.eval()
    with torch.no_grad():
        landmarks_batch = X_landmarks_val[:2]
        video_batch = X_video_val[:2]
        
        logits, intermediate = model(
            landmarks_batch.to(device),
            video_batch.to(device),
            return_intermediate=True
        )
        
        print(f"\n   Stream outputs (for 2 samples):")
        print(f"   - Landmarks stream:  {intermediate['landmarks'].shape} (Transformer output)")
        print(f"   - Video stream:      {intermediate['video'].shape} (CNN-1D-GRU output)")
        print(f"   - Fused output:      {intermediate['fused'].shape} (After fusion layer)")
        print(f"   - Final logits:      {logits.shape} (Classification)")
    
    print("\n✓ Dual-stream pipeline example completed!")


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    # Run the full example
    example_full_pipeline()
    
    print("\n" + "=" * 70)
    print(" Dual-Stream Integration complete!")
    print("=" * 70)
    print("\nArchitecture Summary:")
    print("  ✓ STREAM 1: Landmarks → Transformer Encoder")
    print("  ✓ STREAM 2: Video RGB → CNN-1D-GRU (2D-CNN + 1D-CNN + GRU)")
    print("  ✓ FUSION: Concatenate + Linear layers (weighted combination)")
    print("  ✓ OUTPUT: Classification head (or raw fused features)")
    print("\nUse cases:")
    print("  ✓ Sign Language Translation (SLT)")
    print("  ✓ Gesture Recognition")
    print("  ✓ Action Recognition")
    print("  ✓ Video Understanding")
