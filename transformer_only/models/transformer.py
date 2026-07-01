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
# Temporal CNN feature extractor (Conv1d)
# ──────────────────────────────────────────────
class TemporalConvBlock(nn.Module):
    """
    Residual Conv1d block for local temporal motion modeling.
    Input/Output: (B, C, T)
    """

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.GroupNorm(1, channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.act(x)
        return x


class TemporalConvFeatureExtractor(nn.Module):
    """
    Converts (B, T, F) continuous landmarks into (B, T, d_model) temporal motion features.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        conv_dim: int,
        num_blocks: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.in_proj = nn.Conv1d(input_dim, conv_dim, kernel_size=kernel_size, padding=padding)
        self.in_bn = nn.GroupNorm(1, conv_dim)
        self.in_act = nn.GELU()
        self.blocks = nn.Sequential(
            *[TemporalConvBlock(conv_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv1d(conv_dim, d_model, kernel_size=1)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, src: Tensor) -> Tensor:
        # src: (B, T, F) -> Conv1d expects (B, F, T)
        x = src.transpose(1, 2)
        x = self.in_proj(x)
        x = self.in_bn(x)
        x = self.in_act(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        # back to (B, T, D)
        x = x.transpose(1, 2)
        x = self.out_norm(x)
        return x


# ──────────────────────────────────────────────
# Proiezione landmarks → d_model
# ──────────────────────────────────────────────
class LandmarkEmbedding(nn.Module):
    """
    Proietta i landmarks piatti (T, feat_dim=2108) nello spazio d_model,
    applica un vettore di pesatura apprendibile SiLT e poi inietta
    informazione temporale con una Bi-GRU.

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
        embedding_type: str = "mlp",
        temporal_kernel_size: int = 5,
        temporal_blocks: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding_type = embedding_type
        self.proj = None
        self.temporal_extractor = None
        if embedding_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, d_model),
            )
        elif embedding_type == "temporal_cnn":
            self.temporal_extractor = TemporalConvFeatureExtractor(
                input_dim=feat_dim,
                d_model=d_model,
                conv_dim=hidden_dim,
                num_blocks=temporal_blocks,
                kernel_size=temporal_kernel_size,
                dropout=dropout,
            )
    
        # PE sinusoidale esplicito, uguale a quello del decoder
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        if self.embedding_type == "mlp":
            x = self.proj(src)
        else:
            x = self.temporal_extractor(src)
    
        x = x + self.pe[:, :src.size(1)]
        return self.norm(self.dropout(x))


# ──────────────────────────────────────────────
# Token Embedding per il decoder
# ──────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for autoregressive decoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Target sequence length {x.size(1)} exceeds max_len={self.pe.size(1)} for positional encoding"
            )
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale     = math.sqrt(d_model)
        self.positional = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

    def forward(self, tgt: Tensor, tgt_key_padding_mask: Tensor | None = None) -> Tensor:
        """tgt: (B, L) → (B, L, d_model)"""
        x = self.embedding(tgt) * self.scale
        x = self.positional(x)

        if tgt_key_padding_mask is not None:
            x = x.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)

        return x


# ──────────────────────────────────────────────
# Transformer decoder con ritorno delle attention map
# ──────────────────────────────────────────────
class TransformerDecoderLayerWithAttn(nn.Module):
    """Decoder layer che può restituire le cross-attention medie dell'ultimo layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def _sa_block(
        self,
        x: Tensor,
        tgt_mask: Tensor | None,
        tgt_key_padding_mask: Tensor | None,
    ) -> Tensor:
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(attn_out)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        memory_mask: Tensor | None,
        memory_key_padding_mask: Tensor | None,
        return_attn: bool,
    ) -> tuple[Tensor, Tensor | None]:
        attn_out, attn_weights = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        return self.dropout2(attn_out), attn_weights if return_attn else None

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        x = tgt
        attn_weights = None

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            mha_out, attn_weights = self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                return_attn,
            )
            x = x + mha_out
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            mha_out, attn_weights = self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                return_attn,
            )
            x = self.norm2(x + mha_out)
            x = self.norm3(x + self._ff_block(x))

        return x, attn_weights


class TransformerDecoderWithAttn(nn.Module):
    """Stack di decoder layers che può restituire le attention map dell'ultimo layer."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayerWithAttn,
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer if i == 0 else self._clone_layer(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def _clone_layer(layer: TransformerDecoderLayerWithAttn) -> TransformerDecoderLayerWithAttn:
        import copy

        return copy.deepcopy(layer)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        output = tgt
        last_attn = None

        for layer_idx, layer in enumerate(self.layers):
            output, last_attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_attn=return_attn and (layer_idx == len(self.layers) - 1),
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, last_attn if return_attn else None


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
        num_enc_layers: int   = 3,
        num_dec_layers: int   = 3,
        dim_feedforward: int  =1024,
        dropout:        float = 0.1,
        max_src_len:    int   = 1024,
        max_tgt_len:    int   = 256,
        pad_id:         int   = 0,
        label_smoothing: float = 0.1,   # era 0.3: più alto confonde la predizione di EOS → più inserzioni
        src_embedding_type: str = "mlp",
        temporal_kernel_size: int = 5,
        temporal_blocks: int = 3,
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
            embedding_type=src_embedding_type,
            temporal_kernel_size=temporal_kernel_size,
            temporal_blocks=temporal_blocks,
            hidden_dim=d_model,
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

        # ── CTC head ──────────────────────────
        # CTC requires an extra blank symbol. We make the head emit
        # (vocab_size + 1) classes and reserve the last index as the blank.
        self.ctc_blank = vocab_size
        self.ctc_head = nn.Linear(d_model, vocab_size + 1)

        # ── Decoder ───────────────────────────
        dec_layer = TransformerDecoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = TransformerDecoderWithAttn(
            dec_layer,
            num_layers=num_dec_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Proiezione output ─────────────────
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Keep PyTorch default init for GRU params (better suited for gated RNNs).
            if "bigru" in name:
                continue
            # Token embedding: initialize with normal std = d_model^-0.5 (common practice)
            if "tgt_embed.embedding.weight" in name:
                nn.init.normal_(p, mean=0.0, std=self.d_model ** -0.5)
                continue
            # Linear and multi-dim weights
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                continue
            # Biases
            if "bias" in name:
                nn.init.zeros_(p)
                continue
            # LayerNorm / 1-d params: keep defaults but ensure sensible init
            if "norm" in name or "ln" in name or p.dim() == 1:
                try:
                    nn.init.ones_(p)
                except Exception:
                    pass

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
        return_attn: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
        """
        Ritorna:
            ce_logits:  (B, L-1, vocab_size)
                ctc_logits: (B, T, vocab_size+1)  # includes blank as last index
            attn:       (B, L-1, T) se return_attn=True
        """
        B, T, _ = src.shape
        L = tgt_input.shape[1]
        device   = src.device

        # ── Encode ────────────────────────────
        memory = self.encode(src, src_key_padding_mask)   # (B, T, d_model)
        ctc_logits = self.ctc_head(memory)                # (B, T, V+1) - last index is blank

        # ── Decode ────────────────────────────
        causal_mask = self._make_causal_mask(L, device)   # (L, L)
        tgt_emb = self.tgt_embed(tgt_input, tgt_in_key_padding_mask)               # (B, L, d_model)

        dec_out, cross_attn = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_in_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            return_attn=return_attn,
        )                                                  # (B, L, d_model)

        ce_logits = self.output_proj(dec_out)             # (B, L, V)

        if return_attn:
            return ce_logits, ctc_logits, cross_attn
        return ce_logits, ctc_logits

    def encode(
        self,
        src: Tensor,                              # (B, T, feat_dim)
        src_key_padding_mask: Tensor | None = None,
        return_ctc_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Restituisce le rappresentazioni encoder: (B, T, d_model).

        Se return_ctc_logits=True, ritorna anche i logit della testa CTC
        applicata direttamente sull'output dell'encoder.
        """
        src_emb = self.src_embed(src, src_key_padding_mask)             # (B, T, d_model)

        # Ensure padded positions remain zeroed after the CNN/embedding pipeline so
        # they cannot leak non-zero values (due to conv/bias) into attention.
        # src_key_padding_mask: (B, T) with True==pad
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(-1)  # (B, T, 1)
            src_emb = src_emb.masked_fill(mask, 0.0)

        # Optional diagnostic printing: show stats of raw src, src_emb and memory once
        if getattr(self, "debug_log_stats", False) and not getattr(self, "_logged_stats_once", False):
            try:
                print("[Model Debug] src stats: mean={:.6f}, std={:.6f}".format(src.mean().item(), src.std().item()))
                print("[Model Debug] src_emb stats: mean={:.6f}, std={:.6f}".format(src_emb.mean().item(), src_emb.std().item()))
            except Exception:
                pass

        # Optional positional encoding debug: print info once when requested
        if getattr(self, "debug_positional_encoding", False) and not getattr(self, "_logged_pos_enc_once", False):
            try:
                pe = None
                if hasattr(self, "tgt_embed") and hasattr(self.tgt_embed, "positional") and hasattr(self.tgt_embed.positional, "pe"):
                    pe = self.tgt_embed.positional.pe
                if pe is not None:
                    print(f"[Model Debug] Positional encoding buffer shape: {tuple(pe.shape)}  dtype={pe.dtype}  device={pe.device}")
                    print(f"[Model Debug] PosEnc stats: mean={pe.mean().item():.6f}, std={pe.std().item():.6f}")
                else:
                    print("[Model Debug] Positional encoding buffer not found on model.tgt_embed.positional.pe")
            except Exception:
                pass
            self._logged_pos_enc_once = True

        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Also force encoder outputs (memory) at padded positions to zero so that
        # cross-attention cannot attend to "dirty" positions created by convs.
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(-1)
            memory = memory.masked_fill(mask, 0.0)

        if getattr(self, "debug_log_stats", False) and not getattr(self, "_logged_stats_once", False):
            try:
                print("[Model Debug] memory stats: mean={:.6f}, std={:.6f}".format(memory.mean().item(), memory.std().item()))
            except Exception:
                pass
            self._logged_stats_once = True

        if return_ctc_logits:
            return memory, self.ctc_head(memory)
        return memory

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
            dec_out, _ = self.decoder(
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
                dec_out, _ = self.decoder(
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
            # Use sequence length excluding <bos> for length penalty (off-by-one fix).
            candidates.sort(
                key=lambda x: x[0] / (max(1, (len(x[1]) - 1)) ** length_penalty),
                reverse=True,
            )
            beams = candidates[:beam_size]

        completed += [(s, seq) for s, seq in beams if seq[-1] != eos_id]
        if not completed:
            return []
        # Final sort: use length excluding <bos> to compute length-penalty correctly.
        completed.sort(
            key=lambda x: x[0] / (max(1, (len(x[1]) - 1)) ** length_penalty),
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