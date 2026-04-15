"""
Transformer Decoder per Sign Language Translation (Video-to-Text)
==================================================================

Modulo implementa un Transformer Decoder autoregressivo per la generazione di testo
a partire da embedding multimodale continuo (fusion di video + landmarks).

Architettura:
  1. Token Embedding: ID token → hidden_dim
  2. Positional Encoding: Posizioni sequenziali
  3. Decoder Blocks con:
     - Masked Self-Attention: Sulla sequenza target (autoregressiva)
     - Cross-Attention: Tra target e memoria multimodale
     - Feed-Forward Network
  4. Linear Projection: hidden_dim → vocab_size (logit)

Input:
  - Memory (Context): (batch, video_frames, hidden_dim) embedding multimodale
  - Target tokens: (batch, seq_len) indici token
  
Output:
  - Logits: (batch, seq_len, vocab_size) logit per item vocabolario

Autore: Thesis Project
Data: 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import json


# =============================================================================
# Positional Encoding per Sequenze di Testo
# =============================================================================

class TextPositionalEncoding(nn.Module):
    """
    Positional Encoding per sequenze di token (testo generato).
    
    Basato sulla formula sinusoidale classica di Vaswani et al., adattato per
    sequenze di testo che possono essere molto più lunghe rispetto ai frame video.
    
    Args:
        d_model (int): Dimensione del modello (hidden_dim)
        max_len (int): Lunghezza massima della sequenza di testo
        dropout (float): Dropout dopo l'encoding
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Pre-calcola positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registra come buffer (non è parametro)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass per adding positional encoding.
        
        Args:
            x: (batch, seq_len, d_model) embedded tokens
        
        Returns:
            x + pos_encoding: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        pos_enc = self.pe[:, :seq_len, :]
        x = x + pos_enc
        return self.dropout(x)


# =============================================================================
# Causal Mask Generation
# =============================================================================

def _generate_causal_mask(
    seq_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Genera una maschera triangolare inferiore per garantire autoregressività.
    
    La maschera previene che la posizione t guardi alle posizioni future t+1, t+2, ...
    Questo è essenziale per la generazione autoregressiva dove ogni token dipende
    solo dai token precedenti.
    
    Args:
        seq_len (int): Lunghezza della sequenza target
        device (torch.device): Device (CPU o CUDA)
    
    Returns:
        mask: (seq_len, seq_len) triangular lower mask
        Con valore indicato come:
          - 0 (False): Attendere questo token (è consentito guardarvi)
          - float('-inf'): Non attendere questo token (nasconderlo)
    
    Esempio con seq_len=3:
        [[0,      -inf, -inf],
         [0,      0,    -inf],
         [0,      0,    0   ]]
        
        Token 0 può guardarsi solo
        Token 1 può guardare 0 e 1
        Token 2 può guardare 0, 1 e 2
    """
    # Crea triangolo superiore
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1  # Ignora la diagonale
    )
    
    return mask


# =============================================================================
# Transformer Decoder
# =============================================================================

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder autoregressivo per Sign Language Translation (Video→Text).
    
    Riceve come input:
      1. Memory (Contesto Multimodale): (B, T_video, hidden_dim)
         Embedding fuso da video (R(2+1)D CNN) + landmarks (Transformer Encoder)
      
      2. Target Tokens: (B, T_text) indici dei token target
         Sequenza di testo da generare (shifted right con BOS token)
    
    Produce:
      - Logits: (B, T_text, vocab_size) per ogni token
    
    Architettura Interna:
      ┌─ Token Embedding (vocab_size → hidden_dim)
      │
      ├─ Positional Encoding (per posizioni nel testo)
      │
      ├─ Decoder Blocks (num_decoder_layers volte):
      │  ├─ Masked Self-Attention (sul testo)
      │  ├─ Cross-Attention (testo × video)
      │  └─ Feed-Forward (2 layer densi)
      │
      └─ Output Linear (hidden_dim → vocab_size)
    
    Args:
        hidden_dim (int): Dimensione degli embedding e degli stati hidden
        vocab_size (int): Dimensione totale vocabolario
        num_decoder_layers (int): Numero di decoder blocks
        num_heads (int): Numero di teste nella multi-head attention
        dim_feedforward (int): Dimensione layer denso intermedio in FFN
        dropout (float): Dropout rate applicato ovunque
        max_target_len (int): Lunghezza massima sequenza target
        device (torch.device): Device per i tensori
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_target_len: int = 512,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.device = device
        
        # ========== 1. Token Embedding ==========
        # Mappa indici token (0..vocab_size-1) → embedding (hidden_dim)
        # Input:  (batch, seq_len) int64 indici
        # Output: (batch, seq_len, hidden_dim) float32 embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # ========== 2. Positional Encoding ==========
        # Aggiunge informazioni di posizione nella sequenza
        # Input:  (batch, seq_len, hidden_dim) embedded token
        # Output: (batch, seq_len, hidden_dim) con PE aggiunto
        self.positional_encoding = TextPositionalEncoding(
            d_model=hidden_dim,
            max_len=max_target_len,
            dropout=dropout
        )
        
        # ========== 3. Decoder Blocks ==========
        # Crea i layer standard PyTorch
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Attende (batch, seq, features) anzichè (seq, batch, features)
            activation='relu'
        )
        
        # Combina i layer in un modulo
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # ========== 4. Output Linear Layer ==========
        # Mappa gli embedding nascosti al vocabolario
        # Input:  (batch, seq_len, hidden_dim)
        # Output: (batch, seq_len, vocab_size) logit
        self.output_linear = nn.Linear(hidden_dim, vocab_size)
        
        # Inizializzazione pesi
        self._init_weights()
    
    def _init_weights(self):
        """Inizializza i pesi nel decoder."""
        # Embeddings con distribuzione normale
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Output linear con xavier uniform
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
    
    def forward(
        self,
        memory: torch.Tensor,
        target_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass attraverso il decoder.
        
        Args:
            memory: (batch, video_frames, hidden_dim)
                    Output fuso da dual-stream upstream (video + landmarks)
                    Rappresenta il CONTESTO multimodale su cui il decoder attende
            
            target_tokens: (batch, seq_len_target) dtype=int64
                          Indici di token della sequenza target
                          DEVE includere BOS token all'inizio (offset right)
            
            memory_key_padding_mask: (batch, video_frames) bool opzionale
                                     True dove sono pad tokens nel video context
                                     Default: None (no padding)
            
            target_key_padding_mask: (batch, seq_len_target) bool opzionale
                                    True dove sono pad tokens nel target
                                    Default: None (no padding)
        
        Returns:
            logits: (batch, seq_len_target, vocab_size)
                   Logit grezzi per ogni token nel target
                   Sui quali applicare cross_entropy loss
        
        Dettagli dei passaggi dimensionali:
        ────────────────────────────────────
        
        MEMORY (contesto multimodale):
            Input:  (B, T_video, 512)  dalla fusione dual-stream
            No change: viene usato così come è
        
        TARGET (sequenza tokens):
            Input:           (B, T_text) indici int64
            Token Embedding: (B, T_text, 512)
            Pos Encoding:    (B, T_text, 512)
        
        CAUSAL MASKING (autoregressività):
            Genera mask (T_text, T_text) triangolare inferiore
            Blocca l'attenzione verso i token futuri
        
        DECODER (4 blocks nel default):
            Ogni block ha:
              - Self-Attention: Su target tokens (con causal mask)
              - Cross-Attention: Target × Memory
              - Feed-Forward: 512 → 2048 → 512
            
            Output: (B, T_text, 512)
        
        OUTPUT LINEAR:
            Input:  (B, T_text, 512)
            Output: (B, T_text, vocab_size) LOGIT GREZZI
        
        Cross-Entropy Loss (calcolato esternamente):
            Reshape logits a (B*T_text, vocab_size)
            Reshape target a (B*T_text,)
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        """
        
        batch_size = target_tokens.size(0)
        seq_len_target = target_tokens.size(1)
        
        # ========== Step 1: Embed Target Tokens ==========
        # (B, T_text) → (B, T_text, hidden_dim)
        # Prende gli indici e ritorna embedding vettori
        target_embedded = self.token_embedding(target_tokens)
        
        # ========== Step 2: Add Positional Encoding ==========
        # (B, T_text, hidden_dim) + sinusoidale PE → (B, T_text, hidden_dim)
        # Inietta posizioni sequenziali
        target_with_pos = self.positional_encoding(target_embedded)
        
        # ========== Step 3: Generate Causal Attention Mask ==========
        # Crea maschera triangolare per garantire autoregressività
        # (T_text, T_text) - Dice quali token possono guardarsi
        tgt_mask = _generate_causal_mask(
            seq_len=seq_len_target,
            device=self.device
        )
        
        # ========== Step 4: Apply Transformer Decoder ==========
        # Riceve:
        #   - tgt: (B, T_text, hidden_dim) sequenza di testo
        #   - memory: (B, T_video, hidden_dim) context multimodale
        #   - tgt_mask: (T_text, T_text) causal mask
        #   - memory_key_padding_mask: (B, T_video) padding mask per memory
        #   - tgt_key_padding_mask: (B, T_text) padding mask per target
        #
        # Ritorna: (B, T_text, hidden_dim) oggetti decoder
        decoder_output = self.transformer_decoder(
            tgt=target_with_pos,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=target_key_padding_mask,
        )
        
        # ========== Step 5: Project to Vocabulary ==========
        # (B, T_text, hidden_dim) → (B, T_text, vocab_size)
        # Mappa ogni position del hidden state al vocabolario
        logits = self.output_linear(decoder_output)
        
        return logits
    
    def generate(
        self,
        memory: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generazione autoregressiva di sequenze.
        
        Inizia con BOS token e genera i token successivi uno per volta,
        condizionati su tutti i token precedenti (causal).
        
        Args:
            memory: (batch, video_frames, hidden_dim) contesto multimodale
            bos_token_id: ID del BOS token (tipicamente 1 o 2)
            eos_token_id: ID del EOS token (fine sequenza)
            max_length: Lunghezza massima della sequenza generata
            temperature: Controllo della "randomicità" della sampling
                        > 1: Più random
                        < 1: Meno random (greedy con temperature → greedy puro)
                        = 1: Default
            top_k: Se fornito, campiona solo da top-k token più probabili
            top_p: Se fornito, campiona da token fino a cumulative prob p
        
        Returns:
            generated_ids: (batch, generated_seq_len) indici token generati
        
        Meccanismo:
            1. Inizia con [BOS] per ogni sample nel batch
            2. Loop su max_length passi:
               - Forward pass decoded input + memory
               - Prendi logit dell'ultimo token predetto
               - Applica sampling (temperature, top-k, top-p)
               - Aggiungi token generato ai precedenti
               - Se tutti i sample hanno EOS, stop
            3. Ritorna le sequenze generate (pos padding con PAD token)
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Inizia con BOS token per tutti
        # (batch,) → (batch, 1)
        generated = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Track which sequences hanno raggiunto EOS
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length - 1):
            # Forward pass: decoder vede tutti i token generati finora
            logits = self.forward(
                memory=memory,
                target_tokens=generated,
            )
            
            # Prendi i logit solo dell'ultimo token (quello che prediamo dopo)
            # (batch, seq_len, vocab_size) → (batch, vocab_size)
            next_token_logits = logits[:, -1, :]
            
            # Applica temperature scaling
            next_token_logits = next_token_logits / (temperature + 1e-8)
            
            # Normalizza a probabilità
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Top-k filtering (opzionale)
            if top_k is not None:
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Top-p (nucleus) filtering (opzionale)
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Campiona dal'istribuzione
            next_tokens = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Rimpiazza i token per sequenze finite con PAD (ID 0)
            next_tokens[finished] = 0
            
            # Aggiungi ai token generati
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Aggiorna il flag per sequenze finite
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            
            # Se tutte le sequenze sono finite, stop
            if finished.all():
                break
        
        return generated


# =============================================================================
# Configuration Class
# =============================================================================

class TransformerDecoderConfig:
    """
    Configurazione per TransformerDecoder.
    
    Utile per salvare/caricare hyperparameter in JSON.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_target_len: int = 512,
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_target_len = max_target_len
    
    def to_dict(self) -> dict:
        """Converte config a dict."""
        return {
            'hidden_dim': self.hidden_dim,
            'vocab_size': self.vocab_size,
            'num_decoder_layers': self.num_decoder_layers,
            'num_heads': self.num_heads,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'max_target_len': self.max_target_len,
        }
    
    def save(self, path: str):
        """Salva config in JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Carica config da JSON."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls(**d)
    
    def __str__(self):
        return (
            f"TransformerDecoderConfig(\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  num_decoder_layers={self.num_decoder_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  dim_feedforward={self.dim_feedforward},\n"
            f"  dropout={self.dropout},\n"
            f"  max_target_len={self.max_target_len}\n"
            f")"
        )
