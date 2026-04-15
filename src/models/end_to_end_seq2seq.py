"""
Integrazione End-to-End: Dual-Stream Encoder + Transformer Decoder
===================================================================

Questo file mostra come integrare il TransformerDecoder con il tuo
modello DualStreamSignLanguageModel per un'architettura completa
di Sign Language Translation (Video + Landmarks → Text).

Architettura Completa:
  INPUT: Video (B, T, 3, H, W) + Landmarks (B, T, 2108)
    ↓
  ENCODER: DualStreamSignLanguageModel
    - Stream 1: Landmarks → TransformerEncoder → (B, T, 512)
    - Stream 2: Video → CNN-1D-GRU → (B, T, 512)
    - Fusion: → (B, T, 512) fused memory
    ↓
  DECODER: TransformerDecoder
    - Input: Memory (B, T_video, 512) + Target tokens (B, T_text)
    - Output: Logits (B, T_text, vocab_size)
    ↓
  LOSS: Cross-Entropy
    ↓
  BACKPROP & UPDATE

Autore: Thesis Project
Data: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


# Import upstream modules
from src.models.cnn_1d_gru_integration import DualStreamSignLanguageModel
from src.models.transformer_decoder import TransformerDecoder, TransformerDecoderConfig


# =============================================================================
# End-to-End Seq2Seq Model für Sign Language Translation
# =============================================================================

class SignLanguageTranslationModel(nn.Module):
    """
    Modello completo Seq2Seq per Sign Language Translation.
    
    Architettura:
    ┌─────────────────────────────────────────────────────────────┐
    │ INPUT: Video + Landmarks                                    │
    │ - Video RGB:  (B, T_video, 3, H, W)                         │
    │ - Landmarks:  (B, T_video, 2108)                            │
    └──────────────────┬──────────────────────────────┬───────────┘
                       │                              │
           ┌───────────▼──────────┐      ┌───────────▼──────────┐
           │ Transformer Encoder   │      │  CNN-1D-GRU Module   │
           │  (Landmarks)          │      │  (Video)             │
           └─────────┬─────────────┘      └───────────┬──────────┘
                     │                                │
                 (B,T,512)                        (B,T,512)
                     │                                │
                     └──────────────┬─────────────────┘
                                    │
                        ┌───────────▼────────┐
                        │ Fusion Module       │
                        │ (B, T_video, 512)  │
                        └───────────┬────────┘
                                    │
                        ┌───────────▼──────────────────┐
                        │ Transformer Decoder           │
                        │ - Memory: (B, T_video, 512)  │
                        │ - Targets: (B, T_text, vocab)│
                        │ - Output: (B, T_text, V)     │
                        └───────────┬──────────────────┘
                                    │
                        ┌───────────▼────────┐
                        │ Cross-Entropy Loss  │
                        │ Backprop            │
                        └────────────────────┘
    
    Args:
        # Encoder params
        video_hidden_dim (int): Dimensione hidden per video stream
        landmark_dim (int): Dimensione input landmarks (2108)
        num_encoder_layers (int): Transformer encoder layers
        freeze_mobilenet (bool): Congela backbone del video
        
        # Decoder params
        text_vocab_size (int): Dimensione vocabolario testo
        text_max_len (int): Lunghezza massima sequenza testo
        num_decoder_layers (int): Transformer decoder layers
        
        # Generale
        device (torch.device): Device per i tensori
    """
    
    def __init__(
        self,
        # Encoder params
        video_hidden_dim: int = 512,
        landmark_dim: int = 2108,
        num_encoder_layers: int = 4,
        freeze_mobilenet: bool = False,
        
        # Decoder params
        text_vocab_size: int = 10000,
        text_max_len: int = 512,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        
        # General
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        
        self.device = device
        self.video_hidden_dim = video_hidden_dim
        self.landmark_dim = landmark_dim
        self.text_vocab_size = text_vocab_size
        
        # ========== ENCODER: Dual-Stream (Video + Landmarks) ==========
        # Produce: (B, T_video, hidden_dim) fused representation
        self.encoder = DualStreamSignLanguageModel(
            hidden_dim=video_hidden_dim,
            landmark_dim=landmark_dim,
            num_transformer_layers=num_encoder_layers,
            use_classification_head=False,  # ← IMPORTANTE: No classification head!
            freeze_mobilenet=freeze_mobilenet,
        )
        
        # ========== DECODER: Transformer (Text Generation) ==========
        # Input: Memory (B, T_video, hidden_dim) + Target tokens
        # Output: (B, T_text, vocab_size) logits
        self.decoder = TransformerDecoder(
            hidden_dim=video_hidden_dim,
            vocab_size=text_vocab_size,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            max_target_len=text_max_len,
            device=device,
        )
    
    def encode(
        self,
        landmarks: torch.Tensor,
        video_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Codifica video + landmarks nel memory multimodale.
        
        Args:
            landmarks: (batch, T_video, 2108)
            video_frames: (batch, T_video, 3, H, W)
        
        Returns:
            memory: (batch, T_video, hidden_dim)
        """
        # Forward dual-stream
        intermediate_output, _ = self.encoder(
            landmarks=landmarks,
            video_frames=video_frames,
            return_intermediate=True
        )
        
        # intermediate_output è (B, T_video, hidden_dim) fused
        return intermediate_output
    
    def decode(
        self,
        memory: torch.Tensor,
        target_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decodifica da memory a logit per il testo.
        
        Args:
            memory: (batch, T_video, hidden_dim)
            target_tokens: (batch, T_text) con BOS offset
            memory_key_padding_mask: (batch, T_video) opzionale
            target_key_padding_mask: (batch, T_text) opzionale
        
        Returns:
            logits: (batch, T_text, vocab_size)
        """
        logits = self.decoder(
            memory=memory,
            target_tokens=target_tokens,
            memory_key_padding_mask=memory_key_padding_mask,
            target_key_padding_mask=target_key_padding_mask,
        )
        return logits
    
    def forward(
        self,
        landmarks: torch.Tensor,
        video_frames: torch.Tensor,
        target_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass completo (training).
        
        Args:
            landmarks: (batch, T_video, 2108)
            video_frames: (batch, T_video, 3, H, W)
            target_tokens: (batch, T_text) con BOS all'inizio
            memory_key_padding_mask: (batch, T_video) opzionale
            target_key_padding_mask: (batch, T_text) opzionale
        
        Returns:
            logits: (batch, T_text, vocab_size)
        """
        # Encode
        memory = self.encode(landmarks, video_frames)
        
        # Decode
        logits = self.decode(
            memory=memory,
            target_tokens=target_tokens,
            memory_key_padding_mask=memory_key_padding_mask,
            target_key_padding_mask=target_key_padding_mask,
        )
        
        return logits
    
    def generate(
        self,
        landmarks: torch.Tensor,
        video_frames: torch.Tensor,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generazione autoregressiva di sequenze testo da video + landmarks.
        
        Args:
            landmarks: (batch, T_video, 2108)
            video_frames: (batch, T_video, 3, H, W)
            bos_token_id: ID del BOS token (default 1)
            eos_token_id: ID del EOS token (default 2)
            max_length: Lunghezza massima (default 128)
            temperature: Controllo randomicità (default 1.0)
            top_k: Top-k sampling opzionale
            top_p: Nucleus sampling opzionale
        
        Returns:
            generated_ids: (batch, gen_length <= max_length)
        """
        # Encode once
        with torch.no_grad():
            memory = self.encode(landmarks, video_frames)
        
        # Decode autoregressivamente
        generated_ids = self.decoder.generate(
            memory=memory,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        return generated_ids
    
    def freeze_encoder(self):
        """Congela l'encoder (non aggiorna i pesi)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Scongela l'encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class Seq2SeqConfig:
    """Configurazione per training."""
    # Encoder
    video_hidden_dim: int = 512
    landmark_dim: int = 2108
    num_encoder_layers: int = 4
    
    # Decoder
    text_vocab_size: int = 10000
    text_max_len: int = 512
    num_decoder_layers: int = 4
    num_heads: int = 8
    
    # Training
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    
    # Scheduling
    warmup_epochs: int = 5
    device: str = 'cuda'


# =============================================================================
# Trainer
# =============================================================================

class Seq2SeqTrainer:
    """Trainer per il modello Seq2Seq completo."""
    
    def __init__(
        self,
        model: SignLanguageTranslationModel,
        config: Seq2SeqConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler con warmup (simple cosine after warmup)
        total_steps = config.num_epochs * 100  # Assume ~100 batches per epoch
        warmup_steps = config.warmup_epochs * 100
        
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """Train per un'epoch."""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (landmarks, video_frames, target_tokens) in enumerate(train_loader):
            landmarks = landmarks.to(self.device)
            video_frames = video_frames.to(self.device)
            target_tokens = target_tokens.to(self.device)
            
            # Forward
            logits = self.model(
                landmarks=landmarks,
                video_frames=video_frames,
                target_tokens=target_tokens,
            )
            
            # Loss
            logits_flat = logits.view(-1, self.model.text_vocab_size)
            targets_flat = target_tokens.view(-1)
            loss = self.criterion(logits_flat, targets_flat)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip_norm
            )
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validation."""
        self.model.eval()
        val_loss = 0
        
        for landmarks, video_frames, target_tokens in val_loader:
            landmarks = landmarks.to(self.device)
            video_frames = video_frames.to(self.device)
            target_tokens = target_tokens.to(self.device)
            
            logits = self.model(landmarks, video_frames, target_tokens)
            
            logits_flat = logits.view(-1, self.model.text_vocab_size)
            targets_flat = target_tokens.view(-1)
            loss = self.criterion(logits_flat, targets_flat)
            
            val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Training loop completo."""
        print(f"Training on {self.device}")
        print(f"Config: {self.config}")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                is_best = " ← BEST"
            else:
                is_best = ""
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}{is_best}")
        
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
# Example Usage
# =============================================================================

def example_end_to_end():
    """Esempio di utilizzo end-to-end."""
    print("\n" + "="*70)
    print(" END-TO-END EXAMPLE: Video + Landmarks → Text")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create model
    print("\n1. Creating Seq2Seq model...")
    config = Seq2SeqConfig(
        video_hidden_dim=512,
        landmark_dim=2108,
        text_vocab_size=10000,
        device=str(device),
    )
    
    model = SignLanguageTranslationModel(
        video_hidden_dim=config.video_hidden_dim,
        landmark_dim=config.landmark_dim,
        num_encoder_layers=config.num_encoder_layers,
        text_vocab_size=config.text_vocab_size,
        text_max_len=config.text_max_len,
        num_decoder_layers=config.num_decoder_layers,
        device=device,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params/1e6:.1f}M")
    
    # 2. Create dummy dataset
    print("\n2. Creating dummy dataset...")
    num_train = 100
    
    X_landmarks = torch.randn(num_train, 100, 2108)
    X_video = torch.randn(num_train, 100, 3, 224, 224)
    y_text = torch.randint(0, 10000, (num_train, 50))
    y_text[:, 0] = 1  # BOS
    
    train_dataset = TensorDataset(X_landmarks, X_video, y_text)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print(f"   Training samples: {len(train_dataset)}")
    
    # 3. Training
    print("\n3. Training...")
    trainer = Seq2SeqTrainer(model, config, device)
    
    # Train for 2 epochs for demo
    num_batches = min(5, len(train_loader))
    model.train()
    for batch_idx, (landmarks, video, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        landmarks = landmarks.to(device)
        video = video.to(device)
        targets = targets.to(device)
        
        logits = model(landmarks, video, targets)
        
        loss = trainer.criterion(
            logits.view(-1, model.text_vocab_size),
            targets.view(-1)
        )
        
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
        print(f"  Batch {batch_idx+1}: loss={loss.item():.4f}")
    
    # 4. Inference
    print("\n4. Generating sequences...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            landmarks=X_landmarks[:1].to(device),
            video_frames=X_video[:1].to(device),
            max_length=50,
            temperature=0.9,
        )
    
    print(f"   Generated IDs shape: {generated.shape}")
    print(f"   Generated IDs: {generated[0, :].cpu().tolist()}")
    
    print("\n✓ End-to-end example completed!")


if __name__ == "__main__":
    example_end_to_end()
