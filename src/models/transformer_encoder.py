"""
Transformer Encoder per Sign Language Translation
========================================================

Modulo implementa un Transformer Encoder per l'elaborazione di sequenze
di landmarks normalizzati estratti da video di linguaggio dei segni.

Input: Landmarks normalizzati (batch, num_frames, landmark_dim=2108)
Output: Rappresentazioni contestuali (batch, num_frames, hidden_dim) o aggregati

Architettura:
  1. Embedding lineare: landmark_dim → hidden_dim
  2. Positional Encoding: per frame sequenziale
  3. Transformer Encoder Blocks: Multi-head attention + FFN
  4. Gestione di padding mask per sequenze variabili

Autore: Thesis Project
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class PositionalEncoding(nn.Module):
    """
    Positional Encoding per sequenze di frame video.
    
    Utilizzato per iniettare informazioni posizionali nella sequenza temporal.
    Implementazione: sinusoidale classica di Vaswani et al., con learnable addizionalmente
    disponibile.
    
    Args:
        d_model (int): Dimensione del modello (hidden_dim)
        max_len (int): Lunghezza massima della sequenza (numero massimo di frame)
        learnable (bool): Se True, usa embedding learnable invece di sinusoidale
        dropout (float): Dropout applicato dopo l'encoding
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        learnable: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.learnable = learnable
        self.dropout = nn.Dropout(p=dropout)
        
        if learnable:
            # Learnable positional embeddings per frame
            self.pe = nn.Parameter(torch.randn(max_len, d_model))
            nn.init.normal_(self.pe, mean=0, std=0.02)
        else:
            # Sinusoidale positional encoding (classico Transformer)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            # Calcolo: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            #         PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) *
                -(math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:  # Caso strano: d_model dispari
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            
            # Registra come buffer (non è parametro allenabile)
            self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggiunge positional encoding alla sequenza.
        
        Args:
            x (torch.Tensor): Sequenza (batch, seq_len, d_model)
        
        Returns:
            torch.Tensor: Sequenza con positional encoding aggiunto (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        if self.learnable:
            pos_enc = self.pe[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)
        else:
            pos_enc = self.pe[:, :seq_len, :]  # (1, seq_len, d_model)
        
        x = x + pos_enc
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention per Transformer.
    
    Implementazione standard di Vaswani et al. (Attention is All You Need).
    
    Matematica:
        Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
        MultiHead = Concat(head_1, ..., head_h) · W^O
    
    Args:
        hidden_dim (int): Dimensione del modello (deve essere divisibile per num_heads)
        num_heads (int): Numero di attention heads
        dropout (float): Dropout sui pesi di attenzione
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) deve essere divisibile per num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads  # Dimensione per head
        
        # Proiezioni lineari per Q, K, V (pesi condivisi per tutti gli head)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Proiezione di output
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.attention_weights = None  # Per debug/visualizzazione
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola multi-head attention.
        
        Args:
            query (torch.Tensor): Query (batch, seq_len, hidden_dim)
            key (torch.Tensor): Key (batch, seq_len, hidden_dim)
            value (torch.Tensor): Value (batch, seq_len, hidden_dim)
            mask (Optional[torch.Tensor]): Padding mask (batch, 1, 1, seq_len)
                Se True, il token è valido; se False, è padding.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: (batch, seq_len, hidden_dim)
                - attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.size()
        
        # 1. Proiezione lineare e reshape per multi-head
        #    Da: (batch, seq_len, hidden_dim)
        #    A: (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Calcolo punteggi di attenzione: Q·K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch, num_heads, seq_len, seq_len)
        
        # 3. Applicazione mask al padding
        if mask is not None:
            # Espandi mask per tutti gli head: (batch, 1, 1, seq_len) → (batch, 1, 1, seq_len)
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        # 4. Softmax sui punteggi
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # Salva per debug/visualizzazione
        self.attention_weights = attn_weights.detach()
        
        # 5. Applicazione pesi alle value
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_k)
        
        # 6. Ricombina gli head
        #    Da: (batch, num_heads, seq_len, d_k)
        #    A: (batch, seq_len, num_heads, d_k) → (batch, seq_len, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 7. Proiezione di output
        output = self.W_o(context)  # (batch, seq_len, hidden_dim)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Feed-Forward Network layer per Transformer.
    
    Struttura: Linear → ReLU → Linear
    Implementazione: hidden_dim → d_ff → hidden_dim
    
    Nota sulla densità: La rete espande a d_ff (tipicamente 4x hidden_dim)
    e poi riconduce. Questo consente al modello di catturare
    non-linearità senza aumentare eccessivamente i parametri.
    
    Args:
        hidden_dim (int): Dimensione di input/output
        d_ff (int): Dimensione dello strato intermedio (default: 4*hidden_dim)
        dropout (float): Dropout dopo ReLU
    """
    
    def __init__(
        self,
        hidden_dim: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * hidden_dim  # Espansione standard
        
        self.fc1 = nn.Linear(hidden_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applica feed-forward network.
        
        Args:
            x (torch.Tensor): Input (batch, seq_len, hidden_dim)
        
        Returns:
            torch.Tensor: Output (batch, seq_len, hidden_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Singolo blocco encoder del Transformer.
    
    Struttura:
        Input → Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm → Output
    
    Ogni sotto-layer ha residual connection + layer normalization (post-norm).
    
    Args:
        hidden_dim (int): Dimensione del modello
        num_heads (int): Numero di attention heads
        d_ff (int): Dimensione dello strato intermedio FFN
        dropout (float): Dropout rate
        activation (str): Funzione di attivazione ('relu' o 'gelu')
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # 1. Multi-Head Attention Block
        self.self_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 2. Feed-Forward Block
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            d_ff=d_ff,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applica un blocco encoder.
        
        Args:
            x (torch.Tensor): Input (batch, seq_len, hidden_dim)
            mask (Optional[torch.Tensor]): Padding mask (batch, 1, 1, seq_len)
        
        Returns:
            torch.Tensor: Output (batch, seq_len, hidden_dim)
        """
        # 1. Self-Attention with Residual Connection
        #    Post-norm: x → attn → norm(x + attn(x))
        attn_out, _ = self.self_attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Feed-Forward with Residual Connection
        ffn_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class LandmarkEmbedding(nn.Module):
    """
    Embedding layer per landmarks normalizzati.
    
    Trasforma vettori di landmarks (2108 dim) in rappresentazioni latenti
    (hidden_dim), aggiungendo informazioni posizionali.
    
    Flusso:
        landmarks (2108) → Linear projection → hidden_dim
                        → Positional Encoding
                        → LayerNorm + Dropout
    
    Args:
        landmark_dim (int): Dimensione dei landmarks (default: 2108)
        hidden_dim (int): Dimensione latente di output
        dropout (float): Dropout rate
        learnable_pos_encoding (bool): Se True, usa pos encoding learnable
    """
    
    def __init__(
        self,
        landmark_dim: int = 2108,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        learnable_pos_encoding: bool = False,
        max_seq_len: int = 5000
    ):
        super().__init__()
        
        # Proiezione lineare: landmarks → hidden_dim
        self.projection = nn.Linear(landmark_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim,
            max_len=max_seq_len,
            learnable=learnable_pos_encoding,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Elabora landmarks in rappresentazioni latenti.
        
        Args:
            landmarks (torch.Tensor): Landmarks (batch, num_frames, landmark_dim=2108)
        
        Returns:
            torch.Tensor: Embedding (batch, num_frames, hidden_dim)
        """
        # Proiezione
        x = self.projection(landmarks)  # (batch, num_frames, hidden_dim)
        
        # Positional encoding
        x = self.positional_encoding(x)  # (batch, num_frames, hidden_dim)
        
        # Normalizzazione e dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder per sequenze di landmarks.
    
    Architettura completa:
        1. LandmarkEmbedding: landmarks → latent space
        2. Stack di N TransformerEncoderBlocks
        3. Opzionale: pooling (mean/max/cls) per aggregazione temporale
    
    Input: Landmarks normalizzati (batch, num_frames, 2108)
    Output: Rappresentazioni contestuali (batch, num_frames, hidden_dim)
           o aggregati tempolarmente (batch, hidden_dim)
    
    Args:
        landmark_dim (int): Dimensione dei landmarks (default: 2108)
        hidden_dim (int): Dimensione latente
        num_layers (int): Numero di blocchi encoder
        num_heads (int): Numero di attention heads
        d_ff (int): Dimensione intermedia FFN (default: 4*hidden_dim)
        dropout (float): Dropout rate
        learnable_pos_encoding (bool): Usa positional encoding learnable
        activation (str): Funzione attivazione ('relu' o 'gelu')
        max_seq_len (int): Lunghezza massima sequenza
    """
    
    def __init__(
        self,
        landmark_dim: int = 2108,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        learnable_pos_encoding: bool = False,
        activation: str = 'relu',
        max_seq_len: int = 5000
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.landmark_dim = landmark_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = LandmarkEmbedding(
            landmark_dim=landmark_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_pos_encoding=learnable_pos_encoding,
            max_seq_len=max_seq_len
        )
        
        # Stack di encoder blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Layer finale di normalizzazione
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def create_mask(
        self,
        seq_lens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        max_len: int = None,
        device: torch.device = None
    ) -> Optional[torch.Tensor]:
        """
        Crea padding mask per sequenze di lunghezze variabili.
        
        Args:
            seq_lens (Optional[torch.Tensor]): Lunghezze effettive (batch,)
            batch_size (int): Dimensione batch
            max_len (int): Lunghezza massima (se None, deducita da seq_lens)
            device (torch.device): Device su cui creare il tensor
        
        Returns:
            Optional[torch.Tensor]: Mask (batch, 1, 1, max_len)
                True per token validi, False per padding
                Se seq_lens è None, ritorna None (no masking)
        """
        if seq_lens is None:
            return None
        
        if max_len is None:
            max_len = seq_lens.max().item()
        
        if device is None:
            device = seq_lens.device
        
        # Crea indici di posizione: (max_len,)
        positions = torch.arange(max_len, device=device)
        
        # Espandi per il batch: (batch, 1, max_len)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Confronta con lunghezze effettive: (batch, max_len)
        mask = positions < seq_lens.unsqueeze(1)
        
        # Reshape per l'attenzione: (batch, 1, 1, max_len)
        mask = mask.unsqueeze(1).unsqueeze(1)
        
        return mask
    
    def forward(
        self,
        landmarks: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Elabora landmarks tramite Transformer Encoder.
        
        Args:
            landmarks (torch.Tensor): Landmarks (batch, num_frames, 2108)
            seq_lens (Optional[torch.Tensor]): Lunghezze effettive (batch,)
                Se None, assume che non ci sia padding
            return_all_layers (bool): Se True, ritorna output di tutti gli strati
        
        Returns:
            torch.Tensor: Output contestuali (batch, num_frames, hidden_dim)
            List[torch.Tensor]: Se return_all_layers=True, lista degli output intermedi
        """
        batch_size, num_frames, _ = landmarks.size()
        device = landmarks.device
        
        # Crea mask se sequenze hanno lunghezze variabili
        mask = self.create_mask(seq_lens, batch_size, num_frames, device)
        
        # Embedding
        x = self.embedding(landmarks)  # (batch, num_frames, hidden_dim)
        
        # Stack di encoder layers
        layer_outputs = [x] if return_all_layers else None
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=mask)
            if return_all_layers:
                layer_outputs.append(x)
        
        # Normalizzazione finale
        x = self.final_norm(x)
        
        if return_all_layers:
            return layer_outputs
        
        return x
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Estrai i pesi di attenzione (per visualizzazione/debug).
        
        Returns:
            List[torch.Tensor]: Pesi di attenzione da tutti gli strati e head
        """
        attention_weights = []
        for layer in self.encoder_layers:
            weights = layer.self_attention.attention_weights
            if weights is not None:
                attention_weights.append(weights)
        return attention_weights


# Utility functions per facilità di uso
def create_transformer_encoder(
    config: dict = None,
    **kwargs
) -> TransformerEncoder:
    """
    Factory function per creare un TransformerEncoder con configurazione.
    
    Args:
        config (dict): Configurazione (sovrascrive kwargs)
        **kwargs: Parametri individuali
    
    Returns:
        TransformerEncoder: Istanza del modello
    """
    default_config = {
        'landmark_dim': 2108,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'd_ff': None,
        'dropout': 0.1,
        'learnable_pos_encoding': False,
    }
    
    if config is not None:
        default_config.update(config)
    
    default_config.update(kwargs)
    
    return TransformerEncoder(**default_config)


if __name__ == '__main__':
    # Esempio di utilizzo
    print("=== Transformer Encoder per Sign Language Translation ===\n")
    
    # Configurazione
    batch_size = 4
    num_frames = 150
    landmark_dim = 2108
    hidden_dim = 512
    
    # Crea modello
    encoder = TransformerEncoder(
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    # Dummy input
    landmarks = torch.randn(batch_size, num_frames, landmark_dim)
    
    # Forward pass
    output = encoder(landmarks)
    
    print(f"Input shape: {landmarks.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test con sequenze variabili
    print("\n=== Test con padding mask ===")
    seq_lens = torch.tensor([150, 120, 100, 80])
    output_masked = encoder(landmarks, seq_lens=seq_lens)
    print(f"Output con mask: {output_masked.shape}")
    
    # Test return_all_layers
    print("\n=== Test return_all_layers ===")
    all_outputs = encoder(landmarks, return_all_layers=True)
    print(f"Numero di strati: {len(all_outputs)}")
    for i, layer_out in enumerate(all_outputs):
        print(f"  Layer {i}: {layer_out.shape}")
