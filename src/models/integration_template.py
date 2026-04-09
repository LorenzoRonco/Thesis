"""
Integrazione del Transformer Encoder nel Pipeline SLR/SLT Completo
==================================================================

Questo file mostra come integrare il TransformerEncoder per landmarks
con:
  1. CNN backbone per estrazione feature visive dai frame video
  2. Fusione multimodale (landmarks + visual features)
  3. Decoder (Transformer o RNN)
  4. Task-specific head (classification)
  5. Training loop completo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
import numpy as np

from transformer_encoder import TransformerEncoder


# ============================================================================
# PARTE 1: CNN Backbone per Feature Visive (Mock + Real)
# ============================================================================

class MockVideoBackbone(nn.Module):
    """
    CNN mock per testing. Simula estrazione di feature da frame video.
    
    Inputs: (batch, 3, height, width)
    Outputs: (batch, visual_dim=2048)
    """
    
    def __init__(self, visual_dim: int = 2048):
        super().__init__()
        self.visual_dim = visual_dim
        
        # Dummy network: conv → flatten → dense
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, visual_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) video frames
        
        Returns:
            features: (batch, visual_dim)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RealVideoBackbone(nn.Module):
    """
    Real CNN backbone basato su ResNet pretrained.
    
    In produzione, usare:
      - ResNet-50: 2048 dim
      - EfficientNet-B4: 1792 dim
      - ViT-Base: 768 dim
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        visual_dim: int = 2048,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.visual_dim = visual_dim
        
        # Questo è uno stub - in produzione carica un modello vero
        if backbone_name == 'resnet50':
            try:
                from torchvision.models import resnet50
                backbone = resnet50(pretrained=pretrained)
                # Remove classification head
                self.backbone = nn.Sequential(*list(backbone.children())[:-1])
                self.fc = nn.Linear(2048, visual_dim) if visual_dim != 2048 else nn.Identity()
            except ImportError:
                print("torchvision not available, using mock backbone")
                self.backbone = MockVideoBackbone(visual_dim)
                self.fc = nn.Identity()
        else:
            # Default to mock
            self.backbone = MockVideoBackbone(visual_dim)
            self.fc = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) o (batch, frames, 3, H, W)
        
        Returns:
            features: (batch, visual_dim) o (batch, frames, visual_dim)
        """
        if x.dim() == 5:  # (batch, frames, 3, H, W)
            b, f, c, h, w = x.shape
            x = x.view(b * f, c, h, w)
            x = self.backbone(x)
            x = x.view(b * f, -1)
            x = self.fc(x)
            x = x.view(b, f, -1)
            return x.mean(dim=1)  # Pool over time
        else:
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


# ============================================================================
# PARTE 2: Multimodal Fusion
# ============================================================================

class MultimodalFusion(nn.Module):
    """
    Fusione di features da landmarks (temporal) e visual (spatial).
    
    Strategie disponibili:
      1. Concatenation: (landmark_dim + visual_dim) → fusion_dim
      2. Addition: Richiede dimensioni uguali
      3. Bilinear: Interazione quadratica
      4. Attention: Fusion ponderato tramite attention
    """
    
    def __init__(
        self,
        landmark_dim: int = 512,
        visual_dim: int = 2048,
        fusion_dim: int = 512,
        strategy: str = 'concat'
    ):
        super().__init__()
        
        self.landmark_dim = landmark_dim
        self.visual_dim = visual_dim
        self.fusion_dim = fusion_dim
        self.strategy = strategy
        
        if strategy == 'concat':
            self.fusion_layer = nn.Linear(landmark_dim + visual_dim, fusion_dim)
        
        elif strategy == 'add':
            # Richiede dim uguali dopo proiezione
            assert landmark_dim == visual_dim, "Add strategy requires equal dimensions"
            self.fusion_layer = nn.Identity()
        
        elif strategy == 'bilinear':
            # Interazione: landmark ⊙ W ⊙ visual
            self.bilinear = nn.Bilinear(landmark_dim, visual_dim, fusion_dim)
            self.fusion_layer = nn.Identity()
        
        elif strategy == 'attention':
            # Usa attention per decidere quale modalità è più importante
            self.landmark_proj = nn.Linear(landmark_dim, fusion_dim)
            self.visual_proj = nn.Linear(visual_dim, fusion_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                batch_first=True
            )
            self.fusion_layer = nn.Identity()
        
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
    
    def forward(
        self,
        landmark_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            landmark_features: (batch, landmark_dim) da pooling
            visual_features: (batch, visual_dim) da CNN
        
        Returns:
            fused: (batch, fusion_dim)
        """
        if self.strategy == 'concat':
            x = torch.cat([landmark_features, visual_features], dim=1)
            return self.fusion_layer(x)
        
        elif self.strategy == 'add':
            return landmark_features + visual_features
        
        elif self.strategy == 'bilinear':
            return self.bilinear(landmark_features, visual_features)
        
        elif self.strategy == 'attention':
            # Tratta landmark e visual come due modalità separate
            lm = self.landmark_proj(landmark_features)  # (batch, fusion_dim)
            vi = self.visual_proj(visual_features)      # (batch, fusion_dim)
            
            # Attention tra le due modalità
            query = lm.unsqueeze(1)  # (batch, 1, fusion_dim)
            key = vi.unsqueeze(1)    # (batch, 1, fusion_dim)
            value = vi.unsqueeze(1)
            
            out, _ = self.attention(query, key, value)  # (batch, 1, fusion_dim)
            return out.squeeze(1)


# ============================================================================
# PARTE 3: Classification Head
# ============================================================================

class ClassificationHead(nn.Module):
    """
    Head per task di classificazione/traduzioni.
    
    Mappa fused features → logits per vocabolario.
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dims: list = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256]
        
        layers = []
        prev_dim = input_dim
        
        # Strati intermedi
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Strato di output
        layers.append(nn.Linear(prev_dim, vocab_size))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) fused features
        
        Returns:
            logits: (batch, vocab_size)
        """
        return self.head(x)


# ============================================================================
# PARTE 4: Modello Completo Dual-Stream
# ============================================================================

class DualStreamSignLanguageModel(nn.Module):
    """
    Modello complete for Sign Language Recognition/Translation.
    
    Architecture:
        Landmarks (frames, 2108) → TransformerEncoder → Pool → (512)
                                                                    ↓
        Video (frames, 3, H, W) → CNNBackbone → (2048) ─────────────→
                                                                    ↓
                                            Fusion Layer → (512)
                                                                    ↓
                                            Classification Head → (vocab_size)
    
    Args:
        vocab_size (int): Dimensione vocabolario
        landmark_dim (int): Dimensione landmarks (2108)
        encoder_config (dict): Config per TransformerEncoder
        video_backbone (str or nn.Module): CNN backbone
        fusion_strategy (str): Strategia fusione
        freeze_video_backbone (bool): Congela CNN
    """
    
    def __init__(
        self,
        vocab_size: int,
        landmark_dim: int = 2108,
        encoder_config: dict = None,
        video_backbone: str = 'mock',
        fusion_strategy: str = 'concat',
        freeze_video_backbone: bool = False,
        visual_dim: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 1. Transformer Encoder per landmarks
        if encoder_config is None:
            encoder_config = {
                'landmark_dim': landmark_dim,
                'hidden_dim': 512,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1
            }
        
        self.landmark_encoder = TransformerEncoder(**encoder_config)
        landmark_hidden_dim = encoder_config.get('hidden_dim', 512)
        
        # 2. CNN Backbone for visual features
        if isinstance(video_backbone, str):
            if video_backbone == 'mock':
                self.video_backbone = MockVideoBackbone(visual_dim)
            else:
                self.video_backbone = RealVideoBackbone(
                    backbone_name=video_backbone,
                    visual_dim=visual_dim,
                    freeze_backbone=freeze_video_backbone
                )
        else:
            self.video_backbone = video_backbone
        
        # 3. Pooling per landmarks
        self.pooling = 'mean'  # o 'max', 'cls', etc.
        
        # 4. Multimodal Fusion
        self.fusion = MultimodalFusion(
            landmark_dim=landmark_hidden_dim,
            visual_dim=visual_dim,
            fusion_dim=landmark_hidden_dim,
            strategy=fusion_strategy
        )
        
        # 5. Classification head
        self.classifier = ClassificationHead(
            input_dim=landmark_hidden_dim,
            vocab_size=vocab_size,
            hidden_dims=[256],
            dropout=0.1
        )
    
    def forward(
        self,
        landmarks: torch.Tensor,
        video_frames: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass completo.
        
        Args:
            landmarks: (batch, num_frames, 2108)
            video_frames: (batch, 3, H, W) o (batch, num_frames, 3, H, W)
            seq_lens: (batch,) lunghezze effettive
        
        Returns:
            logits: (batch, vocab_size)
        """
        # 1. Encode landmarks
        landmark_features = self.landmark_encoder(landmarks, seq_lens=seq_lens)
        # (batch, num_frames, hidden_dim)
        
        # 2. Pool over time
        if self.pooling == 'mean':
            landmark_pooled = landmark_features.mean(dim=1)
        elif self.pooling == 'max':
            landmark_pooled, _ = landmark_features.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        # (batch, hidden_dim)
        
        # 3. Extract visual features
        if video_frames is not None:
            visual_features = self.video_backbone(video_frames)  # (batch, visual_dim)
        else:
            # Se non disponibili, usa zero
            visual_features = torch.zeros(
                landmark_pooled.size(0),
                2048,
                device=landmark_pooled.device
            )
        
        # 4. Fuse features
        fused = self.fusion(landmark_pooled, visual_features)
        # (batch, hidden_dim)
        
        # 5. Classify
        logits = self.classifier(fused)
        # (batch, vocab_size)
        
        return logits


# ============================================================================
# PARTE 5: Dataset e DataLoader per Training
# ============================================================================

class SignLanguageDataset(Dataset):
    """
    Dataset per Sign Language.
    
    In pratica carica:
    - Landmarks normalizzati (.npy o memoria)
    - Video frames (opzionale)
    - Labels/Translations
    """
    
    def __init__(
        self,
        landmarks_list: list,
        labels_list: list,
        video_list: list = None,
        max_len: int = 200
    ):
        """
        Args:
            landmarks_list: Lista di (frames, 2108) arrays
            labels_list: Lista di token ids (o gloss ids se SLR)
            video_list: Lista di (frames, 3, H, W) arrays (optional)
            max_len: Max sequence length per padding
        """
        self.landmarks = landmarks_list
        self.labels = labels_list
        self.videos = video_list or [None] * len(landmarks_list)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, idx):
        landmarks = torch.from_numpy(self.landmarks[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Padding/truncation
        if landmarks.size(0) < self.max_len:
            padding = torch.zeros(
                self.max_len - landmarks.size(0),
                landmarks.size(1)
            )
            landmarks = torch.cat([landmarks, padding], dim=0)
        else:
            landmarks = landmarks[:self.max_len]
        
        seq_len = min(self.landmarks[idx].shape[0], self.max_len)
        
        result = {
            'landmarks': landmarks,
            'label': label,
            'seq_len': seq_len
        }
        
        if self.videos[idx] is not None:
            video = torch.from_numpy(self.videos[idx]).float()
            result['video'] = video
        
        return result


def collate_fn(batch):
    """Custom collate per batch variabili."""
    landmarks = torch.stack([item['landmarks'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    seq_lens = torch.tensor([item['seq_len'] for item in batch])
    
    result = {
        'landmarks': landmarks,
        'labels': labels,
        'seq_lens': seq_lens
    }
    
    if 'video' in batch[0]:
        videos = torch.stack([item['video'] for item in batch])
        result['video'] = videos
    
    return result


# ============================================================================
# PARTE 6: Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda'
) -> float:
    """
    Un epoch di training.
    
    Returns:
        avg_loss: Loss medio dell'epoca
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        landmarks = batch['landmarks'].to(device)
        labels = batch['labels'].to(device)
        seq_lens = batch['seq_lens'].to(device)
        
        video = batch.get('video')
        if video is not None:
            video = video.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(landmarks, video, seq_lens)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}: loss={loss.item():.4f}")
    
    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Valutazione.
    
    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            landmarks = batch['landmarks'].to(device)
            labels = batch['labels'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            
            video = batch.get('video')
            if video is not None:
                video = video.to(device)
            
            logits = model(landmarks, video, seq_lens)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def main_training_example():
    """
    Esempio di training loop completo.
    """
    print("=" * 70)
    print("Sign Language Translation - Complete Training Example")
    print("=" * 70)
    
    # Configurazione
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 5000
    batch_size = 8
    num_epochs = 10
    
    print(f"Device: {device}")
    print()
    
    # Crea modello
    model = DualStreamSignLanguageModel(
        vocab_size=vocab_size,
        video_backbone='mock',
        fusion_strategy='concat'
    ).to(device)
    
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Crea dummy dataset
    num_samples = 100
    landmarks_list = [
        np.random.randn(np.random.randint(80, 150), 2108)
        for _ in range(num_samples)
    ]
    labels_list = np.random.randint(0, vocab_size, num_samples)
    
    dataset = SignLanguageDataset(
        landmarks_list=landmarks_list,
        labels_list=labels_list,
        max_len=150
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Optimizer e loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Average loss: {avg_loss:.4f}")
        print()
    
    print("Training complete!")


if __name__ == '__main__':
    main_training_example()
