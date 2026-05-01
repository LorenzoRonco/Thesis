"""
model.py

Architettura Transformer encoder-decoder per Sign Language Translation.

Sorgente: sequenza di landmarks MediaPipe  (B, T, 2108)
Target:   sequenza di token testuali       (B, L)

Componenti:
  - LandmarkEmbedding  : proietta (T, 2108) → (T, d_model) con pos. encoding
  - TransformerEncoder : N layer encoder con self-attention
  - TransformerDecoder : M layer decoder con masked self-attn + cross-attn
  - OutputProjection   : (B, L, d_model) → (B, L, vocab_size)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────
# Positional Encoding sinusoidale
# ──────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """
    PE classico di Vaswani et al. (2017).
    Supporta sequenze fino a max_len frame/token.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # (L, D)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L,1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))             # (1, L, D)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────
# Proiezione landmarks → d_model
# ──────────────────────────────────────────────
class LandmarkEmbedding(nn.Module):
    """
    Proietta i landmarks piatti (T, feat_dim=2108) nello spazio d_model
    tramite un piccolo MLP + Layer Norm + Positional Encoding.

    Opzionalmente impara un embedding per il tipo di landmark
    (face / pose / left_hand / right_hand) sommato all'input.
    """

    def __init__(
        self,
        feat_dim: int,
        d_model: int,
        max_len: int = 4096,
        dropout: float = 0.1,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

    def forward(self, src: Tensor) -> Tensor:
        """src: (B, T, feat_dim) → (B, T, d_model)"""
        x = self.proj(src)
        x = self.norm(x)
        return self.pos_enc(x)


# ──────────────────────────────────────────────
# Token Embedding per il decoder
# ──────────────────────────────────────────────
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale     = math.sqrt(d_model)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len, dropout)

    def forward(self, tgt: Tensor) -> Tensor:
        """tgt: (B, L) → (B, L, d_model)"""
        x = self.embedding(tgt) * self.scale
        return self.pos_enc(x)


# ──────────────────────────────────────────────
# Modello principale
# ──────────────────────────────────────────────
class SignLanguageTransformer(nn.Module):
    """
    Transformer encoder-decoder per Sign Language Translation.

    Args:
        feat_dim:       dimensione dell'input (default 2108 = 527×4)
        vocab_size:     dimensione vocabolario
        d_model:        dimensione nascosta del transformer
        nhead:          numero di teste di attenzione
        num_enc_layers: layer encoder
        num_dec_layers: layer decoder
        dim_feedforward: dimensione FFN interna
        dropout:        dropout
        max_src_len:    lunghezza massima sorgente (frame)
        max_tgt_len:    lunghezza massima target (token)
        pad_id:         indice del token di padding
        label_smoothing: smoothing per la cross-entropy loss
    """

    def __init__(
        self,
        feat_dim:       int   = 2108,
        vocab_size:     int   = 5000,
        d_model:        int   = 512,
        nhead:          int   = 8,
        num_enc_layers: int   = 6,
        num_dec_layers: int   = 6,
        dim_feedforward: int  = 2048,
        dropout:        float = 0.1,
        max_src_len:    int   = 1024,
        max_tgt_len:    int   = 256,
        pad_id:         int   = 0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.d_model   = d_model
        self.pad_id    = pad_id

        # ── Embedding ─────────────────────────
        self.src_embed = LandmarkEmbedding(
            feat_dim=feat_dim,
            d_model=d_model,
            max_len=max_src_len,
            dropout=dropout,
        )
        self.tgt_embed = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_tgt_len,
            dropout=dropout,
        )

        # ── Encoder ───────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN: più stabile in training
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_enc_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Decoder ───────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=num_dec_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Proiezione output ─────────────────
        self.output_proj = nn.Linear(d_model, vocab_size)

        # ── Loss ──────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ── Causal mask ───────────────────────────
    @staticmethod
    def _make_causal_mask(size: int, device: torch.device) -> Tensor:
        """Maschera causale (upper-triangular) per il decoder."""
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )

    # ── Forward ───────────────────────────────
    def forward(
        self,
        src:                    Tensor,                  # (B, T, feat_dim)
        tgt_input:              Tensor,                  # (B, L-1)
        tgt_output:             Tensor,                  # (B, L-1)
        src_key_padding_mask:   Tensor | None = None,   # (B, T) True=pad
        tgt_in_key_padding_mask: Tensor | None = None,  # (B, L-1)
    ) -> tuple[Tensor, Tensor]:
        """
        Ritorna:
            logits: (B, L-1, vocab_size)
            loss:   scalar
        """
        B, T, _ = src.shape
        L = tgt_input.shape[1]
        device   = src.device

        # ── Encode ────────────────────────────
        memory = self.encode(src, src_key_padding_mask)   # (B, T, d_model)

        # ── Decode ────────────────────────────
        causal_mask = self._make_causal_mask(L, device)   # (L, L)
        tgt_emb = self.tgt_embed(tgt_input)               # (B, L, d_model)

        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_in_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )                                                  # (B, L, d_model)

        logits = self.output_proj(dec_out)                 # (B, L, V)

        # ── Loss ──────────────────────────────
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),           # (B*L, V)
            tgt_output.reshape(-1),                        # (B*L,)
        )

        return logits, loss

    def encode(
        self,
        src: Tensor,                              # (B, T, feat_dim)
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Restituisce le rappresentazioni encoder: (B, T, d_model)"""
        src_emb = self.src_embed(src)             # (B, T, d_model)
        return self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

    # ── Greedy decoding ───────────────────────
    @torch.no_grad()
    def greedy_decode(
        self,
        src:                  Tensor,             # (B, T, feat_dim)
        bos_id:               int,
        eos_id:               int,
        max_len:              int = 128,
        src_key_padding_mask: Tensor | None = None,
    ) -> list[list[int]]:
        """
        Decodifica greedy (per inferenza / validazione rapida).
        Ritorna lista di liste di token id (senza padding).
        """
        device  = src.device
        B       = src.size(0)
        memory  = self.encode(src, src_key_padding_mask)    # (B, T, d_model)

        # Inizializza con <bos>
        ys          = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished    = torch.zeros(B, dtype=torch.bool, device=device)
        results     = [[] for _ in range(B)]

        for step in range(max_len):
            L = ys.size(1)
            causal = self._make_causal_mask(L, device)
            tgt_emb = self.tgt_embed(ys)
            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits  = self.output_proj(dec_out[:, -1, :])  # (B, V)
            next_id = logits.argmax(dim=-1)                 # (B,)

            for b in range(B):
                if not finished[b]:
                    tok = next_id[b].item()
                    if tok == eos_id:
                        finished[b] = True
                    else:
                        results[b].append(tok)

            if finished.all():
                break

            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

        return results

    # ── Beam search ───────────────────────────
    @torch.no_grad()
    def beam_search(
        self,
        src:                  Tensor,             # (1, T, feat_dim)  — singolo esempio
        bos_id:               int,
        eos_id:               int,
        beam_size:            int  = 4,
        max_len:              int  = 128,
        length_penalty:       float = 0.6,
        src_key_padding_mask: Tensor | None = None,
    ) -> list[int]:
        """
        Beam search per singolo esempio.
        Ritorna la sequenza di token id migliore.
        """
        assert src.size(0) == 1, "beam_search accetta un solo esempio alla volta"
        device = src.device
        memory = self.encode(src, src_key_padding_mask)   # (1, T, d_model)

        # Beam: lista di (score, [token_ids])
        beams: list[tuple[float, list[int]]] = [(0.0, [bos_id])]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            candidates: list[tuple[float, list[int]]] = []
            for score, seq in beams:
                if seq[-1] == eos_id:
                    completed.append((score, seq))
                    continue
                ys = torch.tensor([seq], dtype=torch.long, device=device)
                L  = ys.size(1)
                causal  = self._make_causal_mask(L, device)
                tgt_emb = self.tgt_embed(ys)
                dec_out = self.decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=causal,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                log_probs = F.log_softmax(self.output_proj(dec_out[:, -1, :]), dim=-1)
                top_logp, top_ids = log_probs.topk(beam_size, dim=-1)   # (1, k)
                for lp, tid in zip(top_logp[0].tolist(), top_ids[0].tolist()):
                    candidates.append((score + lp, seq + [tid]))

            if not candidates:
                break

            # Mantieni i beam migliori (con length penalty)
            candidates.sort(
                key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                reverse=True,
            )
            beams = candidates[:beam_size]

        completed += [(s, seq) for s, seq in beams if seq[-1] != eos_id]
        if not completed:
            return []
        completed.sort(
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True,
        )
        best_seq = completed[0][1]
        # Rimuovi <bos> e <eos>
        return [t for t in best_seq if t not in (bos_id, eos_id)]

    # ── Numero parametri ──────────────────────
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())