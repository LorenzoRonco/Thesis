"""
CNN-1D-GRU Sequential Feature Extraction Module
================================================

Questo modulo implementa un'architettura di feature extraction e sequential modeling
che combina:
  1. MobileNet pre-addestrata (2D-CNN) per l'estrazione di feature spaziali
  2. 1D-CNN per un'elaborazione temporale iniziale
  3. GRU per il processamento della sequenza lungo l'asse temporale

L'output di questo modulo è compatibile con un Transformer Encoder per il
processamento ulteriore.

Architettura:
    Frame Input (B, T, 3, H, W)
        ↓
    [2D-CNN: MobileNet v2]  → spatial feature extraction
    Output: (B, T, 1280)
        ↓
    [1D-CNN Layer]  → temporal processing start
    Output: (B, T, hidden_dim)
        ↓
    [GRU Layer]  → sequential modeling
    Output: (B, T, hidden_dim)
        ↓
    Output finale: (B, T, hidden_dim) → Transformer Encoder

Author: Thesis Project
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Optional, Tuple, List


class MobileNetFeatureExtractor(nn.Module):
    """
    Feature Extractor basato su MobileNet v2 pre-addestrata.
    
    Estrae feature spaziali dai frame video utilizzando MobileNet v2.
    Si può scegliere se usare i pesi pre-addestrati o meno.
    
    Args:
        pretrained (bool): Se True, carica pesi ImageNet pre-addestrati
        feature_dim (int): Dimensione delle feature spaziali estratte (default: 1280 per MobileNet)
        freeze_backbone (bool): Se True, non aggiorna i pesi during training
        
    Input Shape:
        (batch_size, time_steps, 3, height, width)
    
    Output Shape:
        (batch_size, time_steps, feature_dim)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 1280,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Carica MobileNet v2
        if pretrained:
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        else:
            self.backbone = mobilenet_v2(weights=None)
        
        # La feature_dim di MobileNet è 1280 per default
        self.feature_dim = feature_dim
        
        # Rimuovi il classification head (mantieni solo le feature convoluzionali)
        # MobileNet ha un avgpool + classifier, manteniamo diverse fasi
        # Features disponibili:
        # - backbone.features: sequenza di layer convoluzionali
        # - Ultima convoluzione produce 1280 canali
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estrae feature spaziali dai frame video.
        
        Args:
            x: Tensor di forma (batch_size, time_steps, 3, height, width)
        
        Returns:
            Tensor di forma (batch_size, time_steps, feature_dim)
        
        Gestione delle dimensioni:
            Input:  (B, T, 3, H, W) = (8, 150, 3, 224, 224)
            ↓
            Reshape: (B*T, 3, H, W) = (1200, 3, 224, 224)
            ↓
            MobileNet: (B*T, 1280, h, w) = (1200, 1280, 7, 7)
            ↓
            Global Average Pool: (B*T, 1280) = (1200, 1280)
            ↓
            Reshape: (B, T, 1280) = (8, 150, 1280)
        """
        batch_size, time_steps, channels, height, width = x.shape
        
        # Reshape: (B, T, 3, H, W) → (B*T, 3, H, W)
        x_reshaped = x.view(batch_size * time_steps, channels, height, width)
        
        # Estrai feature attraverso tutti i layer convoluzionali
        # MobileNet.features contiene tutti i layer convoluzionali
        features = self.backbone.features(x_reshaped)
        # Output: (B*T, 1280, h, w) dove h, w dipendono dall'input
        # Per input 224x224: (B*T, 1280, 7, 7)
        
        # Global Average Pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        # Output: (B*T, 1280, 1, 1)
        
        # Flatten
        features = features.view(batch_size * time_steps, -1)
        # Output: (B*T, 1280)
        
        # Reshape back to sequence: (B*T, 1280) → (B, T, 1280)
        features = features.view(batch_size, time_steps, self.feature_dim)
        
        return features


class Conv1DTemporalBlock(nn.Module):
    """
    Block di convoluzione 1D per l'elaborazione temporale iniziale.
    
    Applica convoluzione 1D sui frame per catturare pattern temporali locali
    prima del processamento GRU.
    
    Args:
        in_channels (int): Numero di canali input
        out_channels (int): Numero di canali output
        kernel_size (int): Dimensione del kernel temporale (default: 3)
        padding (int): Padding temporale (default: 1 per mantenere lunghezza)
        dropout (float): Dropout rate
    
    Input Shape:
        (batch_size, time_steps, in_channels)
    
    Output Shape:
        (batch_size, time_steps, out_channels)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Conv1d richiede input (B, C, T) ma il nostro è (B, T, C)
        # Converremo dentro forward()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Elaborazione 1D-CNN temporale.
        
        Args:
            x: Tensor di forma (batch_size, time_steps, in_channels)
        
        Returns:
            Tensor di forma (batch_size, time_steps, out_channels)
        
        Gestione delle dimensioni:
            Input:  (B, T, C_in) = (8, 150, 1280)
            ↓
            Transpose: (B, T, C_in) → (B, C_in, T) = (8, 1280, 150)
            ↓
            Conv1d: (B, C_in, T) → (B, C_out, T) = (8, 512, 150)
            ↓
            BatchNorm1d: (B, C_out, T) = (8, 512, 150)
            ↓
            ReLU: (B, C_out, T) = (8, 512, 150)
            ↓
            Dropout: (B, C_out, T) = (8, 512, 150)
            ↓
            Transpose: (B, C_out, T) → (B, T, C_out) = (8, 150, 512)
        """
        # Transpose da (B, T, C) a (B, C, T) per Conv1d
        x = x.transpose(1, 2)  # (B, T, C_in) → (B, C_in, T)
        
        # Applica convoluzione 1D
        x = self.conv(x)  # (B, C_in, T) → (B, C_out, T)
        
        # Batch normalization
        x = self.batch_norm(x)  # (B, C_out, T)
        
        # Attivazione
        x = self.relu(x)  # (B, C_out, T)
        
        # Dropout
        x = self.dropout(x)  # (B, C_out, T)
        
        # Transpose back da (B, C, T) a (B, T, C)
        x = x.transpose(1, 2)  # (B, C_out, T) → (B, T, C_out)
        
        return x


class SequenceGRUEncoder(nn.Module):
    """
    GRU Encoder per il processamento sequenziale della temporalità.
    
    Processa la sequenza di feature temporali usando GRU (Gated Recurrent Unit)
    per catturare dipendenze a lungo termine.
    
    Args:
        input_dim (int): Dimensione feature input
        hidden_dim (int): Dimensione dello stato nascosto GRU
        num_layers (int): Numero di layer GRU impilati (default: 1)
        dropout (float): Dropout rate tra layer
        bidirectional (bool): Se True, usa GRU bidirezionale
        
    Input Shape:
        (batch_size, time_steps, input_dim)
    
    Output Shape:
        - Se bidirectional=False: (batch_size, time_steps, hidden_dim)
        - Se bidirectional=True: (batch_size, time_steps, hidden_dim*2)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension (considerando bidirectional)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processamento sequenziale GRU.
        
        Args:
            x: Tensor di forma (batch_size, time_steps, input_dim)
            h: Hidden state iniziale (opzionale)
        
        Returns:
            output: Tensor di forma (batch_size, time_steps, output_dim)
            h_n: Hidden state finale (batch_size, num_layers*num_directions, hidden_dim)
        
        Gestione delle dimensioni:
            Input:    (B, T, C_in) = (8, 150, 512)
            ↓
            GRU (unidirezionale):
            Output:   (B, T, hidden_dim) = (8, 150, 512)
            h_n:      (num_layers, B, hidden_dim) = (1, 8, 512)
            
            GRU (bidirezionale):
            Output:   (B, T, hidden_dim*2) = (8, 150, 1024)
            h_n:      (num_layers*2, B, hidden_dim) = (2, 8, 512)
        """
        # Forward GRU con batch_first=True
        output, h_n = self.gru(x, h)
        # output: (batch_size, time_steps, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        
        return output, h_n


class CNN1DGRUModule(nn.Module):
    """
    Modulo completo di Feature Extraction e Sequential Modeling.
    
    Combina:
    1. MobileNet (2D-CNN) per l'estrazione di feature spaziali dai frame
    2. Conv1D per l'elaborazione temporale iniziale
    3. GRU per il processamento della sequenza temporale
    
    L'output è compatibile con un Transformer Encoder per il processamento
    ulteriore di video o sequenze di frame per task come Sign Language Translation.
    
    Args:
        hidden_dim (int): Dimensione latente comune (default: 512)
        num_gru_layers (int): Numero di layer GRU (default: 1)
        gru_dropout (float): Dropout GRU (default: 0.1)
        conv1d_dropout (float): Dropout Conv1D (default: 0.1)
        bidirectional_gru (bool): GRU bidirezionale (default: False)
        freeze_mobilenet (bool): Congela pesi MobileNet (default: False)
        pretrained_mobilenet (bool): MobileNet pre-addestrata (default: True)
    
    Input Shape:
        Video frames: (batch_size, time_steps, 3, height, width)
        Es: (8, 150, 3, 224, 224)
    
    Output Shape:
        Feature sequence: (batch_size, time_steps, hidden_dim)
        Es: (8, 150, 512)
    
    Utilizzo:
        >>> model = CNN1DGRUModule(hidden_dim=512)
        >>> frames = torch.randn(8, 150, 3, 224, 224)
        >>> features = model(frames)
        >>> print(features.shape)  # torch.Size([8, 150, 512])
        
        # Passaggio a Transformer Encoder
        >>> transformer = TransformerEncoder(
        ...     landmark_dim=512,
        ...     hidden_dim=512,
        ...     num_layers=4
        ... )
        >>> output = transformer(features)
        >>> print(output.shape)  # torch.Size([8, 150, 512])
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_gru_layers: int = 1,
        gru_dropout: float = 0.1,
        conv1d_dropout: float = 0.1,
        bidirectional_gru: bool = False,
        freeze_mobilenet: bool = False,
        pretrained_mobilenet: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional_gru = bidirectional_gru
        
        # ==================== Stage 1: MobileNet (2D-CNN) ====================
        self.feature_extractor = MobileNetFeatureExtractor(
            pretrained=pretrained_mobilenet,
            feature_dim=1280,  # Output di MobileNet
            freeze_backbone=freeze_mobilenet
        )
        
        # ==================== Stage 2: Conv1D Temporal Processing ====================
        self.conv1d_block = Conv1DTemporalBlock(
            in_channels=1280,       # Output MobileNet
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            dropout=conv1d_dropout
        )
        
        # ==================== Stage 3: GRU Sequence Processing ====================
        self.gru_encoder = SequenceGRUEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers,
            dropout=gru_dropout if num_gru_layers > 1 else 0,
            bidirectional=bidirectional_gru
        )
        
        # Se GRU è bidirezionale, aggiungi layer lineare per proiettare
        # dall'output size (hidden_dim*2) a hidden_dim
        if bidirectional_gru:
            self.output_projection = nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
        else:
            self.output_projection = None
        
        # Layer normalization finale
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        frames: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass completo attraverso il modulo CNN-1D-GRU.
        
        Args:
            frames: Tensor di forma (batch_size, time_steps, 3, height, width)
                   Es: (8, 150, 3, 224, 224)
            h_0: Hidden state iniziale GRU (opzionale)
        
        Returns:
            output: Tensor di forma (batch_size, time_steps, hidden_dim)
                   Es: (8, 150, 512)
            h_n: Hidden state finale GRU
        
        Flusso dei dati con dimensioni:
        
        Input frames:
            (B, T, 3, H, W) = (8, 150, 3, 224, 224)
            ↓
        [1. MobileNet 2D-CNN]
            Output: (B, T, 1280)
            ↓
        [2. Conv1D Temporal]
            Output: (B, T, hidden_dim) = (8, 150, 512)
            ↓
        [3. GRU Encoder]
            Output: (B, T, hidden_dim) o (B, T, hidden_dim*2)
            h_n: (num_layers*num_dirs, B, hidden_dim)
            ↓
        [4. Output Projection] (solo se bidirectional)
            Output: (B, T, hidden_dim) = (8, 150, 512)
            ↓
        [5. Layer Normalization]
            Output: (B, T, hidden_dim) = (8, 150, 512)
            ↓
        Output finale:
            (B, T, hidden_dim) = (8, 150, 512) ✓ Compatibile con Transformer
        """
        batch_size, time_steps, channels, height, width = frames.shape
        
        # ========== Stage 1: Feature Extraction con MobileNet ==========
        # Input:  (B, T, 3, H, W)
        # Output: (B, T, 1280)
        spatial_features = self.feature_extractor(frames)
        
        # ========== Stage 2: Conv1D Temporal Processing ==========
        # Input:  (B, T, 1280)
        # Output: (B, T, hidden_dim)
        temporal_features = self.conv1d_block(spatial_features)
        
        # ========== Stage 3: GRU Sequential Processing ==========
        # Input:  (B, T, hidden_dim)
        # Output: (B, T, hidden_dim) o (B, T, hidden_dim*2)
        #         (num_layers*num_directions, B, hidden_dim)
        gru_output, h_n = self.gru_encoder(temporal_features, h_0)
        
        # ========== Stage 4: Output Projection (se bidirezionale) ==========
        if self.bidirectional_gru:
            # Proietta da (B, T, hidden_dim*2) → (B, T, hidden_dim)
            gru_output = self.output_projection(gru_output)
        
        # ========== Stage 5: Final Layer Normalization ==========
        output = self.final_norm(gru_output)
        
        return output, h_n
    
    def get_output_dim(self) -> int:
        """Ritorna la dimensione dell'output finale."""
        return self.hidden_dim
    
    def freeze_mobilenet(self):
        """Congela i parametri di MobileNet."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_mobilenet(self):
        """Scongela i parametri di MobileNet."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def freeze_gru(self):
        """Congela i parametri di GRU."""
        for param in self.gru_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_gru(self):
        """Scongela i parametri di GRU."""
        for param in self.gru_encoder.parameters():
            param.requires_grad = True


# ============================================================================
# Configuration Presets
# ============================================================================

class CNN1DGRUConfig:
    """Configuration preset per diverse esigenze di modellazione."""
    
    @staticmethod
    def light():
        """Configurazione leggera per prototyping rapido."""
        return {
            'hidden_dim': 256,
            'num_gru_layers': 1,
            'gru_dropout': 0.1,
            'conv1d_dropout': 0.1,
            'bidirectional_gru': False,
            'freeze_mobilenet': False,
            'pretrained_mobilenet': True,
        }
    
    @staticmethod
    def standard():
        """Configurazione standard consigliata."""
        return {
            'hidden_dim': 512,
            'num_gru_layers': 1,
            'gru_dropout': 0.1,
            'conv1d_dropout': 0.1,
            'bidirectional_gru': False,
            'freeze_mobilenet': False,
            'pretrained_mobilenet': True,
        }
    
    @staticmethod
    def heavy():
        """Configurazione più pesante per task complessi."""
        return {
            'hidden_dim': 768,
            'num_gru_layers': 2,
            'gru_dropout': 0.2,
            'conv1d_dropout': 0.1,
            'bidirectional_gru': True,
            'freeze_mobilenet': False,
            'pretrained_mobilenet': True,
        }
    
    @staticmethod
    def frozen_backbone():
        """Configurazione con MobileNet congelato (transfer learning)."""
        return {
            'hidden_dim': 512,
            'num_gru_layers': 1,
            'gru_dropout': 0.1,
            'conv1d_dropout': 0.1,
            'bidirectional_gru': False,
            'freeze_mobilenet': True,
            'pretrained_mobilenet': True,
        }


if __name__ == "__main__":
    # Test rapido del modulo
    print("=" * 60)
    print("CNN-1D-GRU Module Test")
    print("=" * 60)
    
    # Crea il modulo
    model = CNN1DGRUModule(hidden_dim=512)
    
    # Mostra l'architettura
    print("\nArchitettura del modulo:")
    print(model)
    
    # Conta i parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParametri totali: {total_params:,}")
    print(f"Parametri trainabili: {trainable_params:,}")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Test Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    time_steps = 75
    height, width = 224, 224
    
    # Input di test
    frames = torch.randn(batch_size, time_steps, 3, height, width)
    print(f"\nInput shape: {frames.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Time steps: {time_steps}")
    print(f"  - Frame dimension: 3 × {height} × {width}")
    
    # Forward pass
    with torch.no_grad():
        output, h_n = model(frames)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  - Expected: ({batch_size}, {time_steps}, 512)")
    print(f"  - Match: {output.shape == torch.Size([batch_size, time_steps, 512])}")
    
    print(f"\nHidden state shape: {h_n.shape}")
    print(f"\nOutput dimension: {model.get_output_dim()}")
    
    print("\n✓ Test completato con successo!")
