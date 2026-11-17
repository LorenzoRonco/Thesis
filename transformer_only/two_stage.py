"""
two_stage.py

Architettura a due stadi per Sign Language Translation su PHOENIX-2014-T.

Stadio 1 — SignLanguageTransformer (transformer.py, invariato):
    landmark → gloss (orth)
    Input:  (B, T, feat_dim)
    Output: sequenza di gloss token

Stadio 2 — GlossToTextTransformer (questo file):
    gloss → translation (tedesco)
    Input:  (B, G) token id di gloss
    Output: (B, L) token id di traduzione

Il training dei due modelli è completamente indipendente:
  - Modello 1 si allena su target_field='orth'
  - Modello 2 si allena su coppie (orth, translation) dal CSV

Durante l'inferenza, TwoStagePipeline:
  1. Chiama greedy_decode di Modello 1 → lista di gloss id
  2. Ri-tokenizza i gloss (opzionalmente con il tokenizer del Modello 2)
  3. Passa la sequenza al Modello 2 → traduzione finale
"""

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Riutilizziamo i building block del transformer esistente
from .models.transformer import (
    TransformerDecoderLayerWithAttn,
    TransformerDecoderWithAttn,
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    SignLanguageTransformer,
)


# ─────────────────────────────────────────────
# Encoder testuale per gloss
# ─────────────────────────────────────────────
class GlossEncoder(nn.Module):
    """
    Encoder che trasforma una sequenza di gloss token in rappresentazioni
    contestuali (B, G, d_model).

    È un semplice Transformer encoder su embedding di token, identico al
    decoder encoder del Modello 1 ma applicato ai gloss.
    """

    def __init__(
        self,
        gloss_vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(gloss_vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.norm_in = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        gloss_ids: Tensor,                     # (B, G)
        gloss_key_padding_mask: Tensor | None = None,  # (B, G) True=pad
    ) -> Tensor:                               # (B, G, d_model)
        x = self.embedding(gloss_ids) * self.scale
        x = self.pos_enc(x)
        x = self.norm_in(x)
        if gloss_key_padding_mask is not None:
            x = x.masked_fill(gloss_key_padding_mask.unsqueeze(-1), 0.0)
        memory = self.encoder(x, src_key_padding_mask=gloss_key_padding_mask)
        return memory


# ─────────────────────────────────────────────
# Modello 2: Gloss → Translation
# ─────────────────────────────────────────────
class GlossToTextTransformer(nn.Module):
    """
    Seq2seq Transformer: gloss (orth) → translation (tedesco).

    Architettura identica al decoder del Modello 1, ma:
      - L'encoder elabora gloss token (non landmark)
      - Il decoder genera token di traduzione
      - Vocabolari sorgente e target possono essere diversi
        (gloss_vocab != translation_vocab)

    Args:
        gloss_vocab_size:    dimensione vocabolario gloss (Modello 1)
        trans_vocab_size:    dimensione vocabolario traduzione
        d_model:             dimensione nascosta
        nhead:               teste di attenzione
        num_enc_layers:      layer encoder gloss
        num_dec_layers:      layer decoder traduzione
        dim_feedforward:     dimensione FFN
        dropout:             dropout
        max_gloss_len:       lunghezza massima sequenza gloss
        max_trans_len:       lunghezza massima traduzione
        gloss_pad_id:        pad id nel vocabolario gloss
        trans_pad_id:        pad id nel vocabolario traduzione
        label_smoothing:     smoothing cross-entropy
        share_vocab:         se True i due vocabolari sono identici e
                             si condivide la embedding matrix
                             (usabile solo se gloss_vocab == trans_vocab)
    """

    def __init__(
        self,
        gloss_vocab_size:  int,
        trans_vocab_size:  int,
        d_model:           int   = 256,
        nhead:             int   = 4,
        num_enc_layers:    int   = 2,
        num_dec_layers:    int   = 2,
        dim_feedforward:   int   = 512,
        dropout:           float = 0.1,
        max_gloss_len:     int   = 128,
        max_trans_len:     int   = 128,
        gloss_pad_id:      int   = 0,
        trans_pad_id:      int   = 0,
        label_smoothing:   float = 0.1,
        share_vocab:       bool  = False,
    ):
        super().__init__()

        self.d_model       = d_model
        self.gloss_pad_id  = gloss_pad_id
        self.trans_pad_id  = trans_pad_id
        self.label_smoothing = label_smoothing
        self.share_vocab   = share_vocab

        # ── Encoder gloss ──────────────────────
        self.gloss_encoder = GlossEncoder(
            gloss_vocab_size=gloss_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_enc_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_gloss_len,
            pad_id=gloss_pad_id,
        )

        # ── Decoder traduzione ─────────────────
        self.tgt_embed = TokenEmbedding(
            vocab_size=trans_vocab_size,
            d_model=d_model,
            max_len=max_trans_len,
            dropout=dropout,
        )

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

        # ── Proiezione output ──────────────────
        self.output_proj = nn.Linear(d_model, trans_vocab_size)

        # Weight tying opzionale (solo se vocabolari identici)
        if share_vocab:
            assert gloss_vocab_size == trans_vocab_size, (
                "share_vocab=True richiede gloss_vocab_size == trans_vocab_size"
            )
            self.gloss_encoder.embedding.weight = self.tgt_embed.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    @staticmethod
    def _make_causal_mask(size: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )

    # ── Forward (training) ────────────────────
    def forward(
        self,
        gloss_ids:               Tensor,               # (B, G)
        tgt_input:               Tensor,               # (B, L-1)
        tgt_output:              Tensor,               # (B, L-1)  ← usato per la loss
        gloss_key_padding_mask:  Tensor | None = None, # (B, G)
        tgt_in_key_padding_mask: Tensor | None = None, # (B, L-1)
        return_attn:             bool = False,
    ) -> tuple:
        """
        Ritorna:
            logits: (B, L-1, trans_vocab_size)
            loss:   scalar CE loss
            attn:   (B, L-1, G) se return_attn=True, altrimenti None
        """
        B, G = gloss_ids.shape
        L    = tgt_input.shape[1]
        device = gloss_ids.device

        # Encode gloss
        memory = self.gloss_encoder(gloss_ids, gloss_key_padding_mask)  # (B, G, d_model)

        # Decode
        causal_mask = self._make_causal_mask(L, device)
        tgt_emb = self.tgt_embed(tgt_input, tgt_in_key_padding_mask)    # (B, L-1, d_model)

        dec_out, cross_attn = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_in_key_padding_mask,
            memory_key_padding_mask=gloss_key_padding_mask,
            return_attn=return_attn,
        )                                                                # (B, L-1, d_model)

        logits = self.output_proj(dec_out)                               # (B, L-1, V)

        # Loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.trans_pad_id,
            label_smoothing=self.label_smoothing,
        )

        if return_attn:
            return logits, loss, cross_attn
        return logits, loss, None

    # ── Encode gloss ──────────────────────────
    def encode(
        self,
        gloss_ids: Tensor,
        gloss_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Restituisce le rappresentazioni encoder: (B, G, d_model)."""
        return self.gloss_encoder(gloss_ids, gloss_key_padding_mask)

    # ── Greedy decode ─────────────────────────
    @torch.no_grad()
    def greedy_decode(
        self,
        gloss_ids:              Tensor,              # (B, G)
        bos_id:                 int,
        eos_id:                 int,
        max_len:                int = 128,
        gloss_key_padding_mask: Tensor | None = None,
    ) -> list[list[int]]:
        device  = gloss_ids.device
        B       = gloss_ids.size(0)
        memory  = self.encode(gloss_ids, gloss_key_padding_mask)  # (B, G, d_model)

        ys       = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results  = [[] for _ in range(B)]

        for _ in range(max_len):
            L = ys.size(1)
            causal  = self._make_causal_mask(L, device)
            tgt_emb = self.tgt_embed(ys)
            dec_out, _ = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal,
                memory_key_padding_mask=gloss_key_padding_mask,
            )
            logits  = self.output_proj(dec_out[:, -1, :])   # (B, V)
            next_id = logits.argmax(dim=-1)                  # (B,)

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
        gloss_ids:              Tensor,              # (1, G)
        bos_id:                 int,
        eos_id:                 int,
        beam_size:              int   = 4,
        max_len:                int   = 128,
        length_penalty:         float = 0.6,
        gloss_key_padding_mask: Tensor | None = None,
    ) -> list[int]:
        assert gloss_ids.size(0) == 1
        device = gloss_ids.device
        memory = self.encode(gloss_ids, gloss_key_padding_mask)

        beams: list[tuple[float, list[int]]] = [(0.0, [bos_id])]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            candidates: list[tuple[float, list[int]]] = []
            for score, seq in beams:
                if seq[-1] == eos_id:
                    completed.append((score, seq))
                    continue
                ys = torch.tensor([seq], dtype=torch.long, device=device)
                causal  = self._make_causal_mask(ys.size(1), device)
                tgt_emb = self.tgt_embed(ys)
                dec_out, _ = self.decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=causal,
                    memory_key_padding_mask=gloss_key_padding_mask,
                )
                log_probs = F.log_softmax(
                    self.output_proj(dec_out[:, -1, :]), dim=-1
                )
                top_logp, top_ids = log_probs.topk(beam_size, dim=-1)
                for lp, tid in zip(top_logp[0].tolist(), top_ids[0].tolist()):
                    candidates.append((score + lp, seq + [tid]))

            if not candidates:
                break
            candidates.sort(
                key=lambda x: x[0] / (max(1, len(x[1]) - 1) ** length_penalty),
                reverse=True,
            )
            beams = candidates[:beam_size]

        completed += [(s, seq) for s, seq in beams if seq[-1] != eos_id]
        if not completed:
            return []
        completed.sort(
            key=lambda x: x[0] / (max(1, len(x[1]) - 1) ** length_penalty),
            reverse=True,
        )
        return [t for t in completed[0][1] if t not in (bos_id, eos_id)]

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# Pipeline di inferenza a due stadi
# ─────────────────────────────────────────────
class TwoStagePipeline(nn.Module):
    """
    Wrapper di inferenza che concatena Modello 1 e Modello 2.

    NON ha parametri propri: orchestra i due modelli già addestrati.
    NON viene usato durante il training (i due modelli si allenano separatamente).

    Args:
        model1:           SignLanguageTransformer (landmark → gloss)
        model2:           GlossToTextTransformer (gloss → translation)
        gloss_tokenizer:  tokenizer usato da Modello 1 (target_field='orth')
        trans_tokenizer:  tokenizer usato da Modello 2
                          Se identico a gloss_tokenizer, passare lo stesso oggetto.
        gloss_bos_id:     bos del vocabolario gloss
        gloss_eos_id:     eos del vocabolario gloss
        trans_bos_id:     bos del vocabolario traduzione
        trans_eos_id:     eos del vocabolario traduzione
    """

    def __init__(
        self,
        model1:          SignLanguageTransformer,
        model2:          GlossToTextTransformer,
        gloss_tokenizer,
        trans_tokenizer,
        gloss_bos_id:    int,
        gloss_eos_id:    int,
        trans_bos_id:    int,
        trans_eos_id:    int,
        gloss_pad_id:    int = 0,
    ):
        super().__init__()
        self.model1         = model1
        self.model2         = model2
        self.gloss_tok      = gloss_tokenizer
        self.trans_tok      = trans_tokenizer
        self.gloss_bos_id   = gloss_bos_id
        self.gloss_eos_id   = gloss_eos_id
        self.trans_bos_id   = trans_bos_id
        self.trans_eos_id   = trans_eos_id
        self.gloss_pad_id   = gloss_pad_id

    @torch.no_grad()
    def forward(
        self,
        src:                  Tensor,              # (B, T, feat_dim)
        src_key_padding_mask: Tensor | None = None,
        max_gloss_len:        int = 64,
        max_trans_len:        int = 128,
        decode_mode:          str = "greedy",      # "greedy" | "beam"
        beam_size:            int = 4,
        length_penalty:       float = 0.6,
    ) -> dict:
        """
        Inferenza completa landmark → gloss → translation.

        Ritorna:
            {
              "gloss_ids":    list[list[int]],   # token id gloss per campione
              "gloss_texts":  list[str],          # gloss decodificati
              "trans_ids":    list[list[int]],   # token id traduzione
              "trans_texts":  list[str],          # traduzioni finali
            }
        """
        device = src.device
        B = src.size(0)

        # ── Stadio 1: landmark → gloss ────────
        self.model1.eval()
        gloss_ids_list = self.model1.greedy_decode(
            src=src,
            bos_id=self.gloss_bos_id,
            eos_id=self.gloss_eos_id,
            max_len=max_gloss_len,
            src_key_padding_mask=src_key_padding_mask,
        )  # list[list[int]], lunghezza variabile

        gloss_texts = [self.gloss_tok.decode(ids) for ids in gloss_ids_list]

        # ── Padding gloss per Modello 2 ───────
        # Aggiungi bos/eos e padda al batch
        gloss_with_special = [
            [self.gloss_bos_id] + ids + [self.gloss_eos_id]
            for ids in gloss_ids_list
        ]
        max_g = max(len(g) for g in gloss_with_special)
        gloss_padded = torch.full(
            (B, max_g), self.gloss_pad_id, dtype=torch.long, device=device
        )
        gloss_mask = torch.ones(B, max_g, dtype=torch.bool, device=device)
        for i, g in enumerate(gloss_with_special):
            gloss_padded[i, : len(g)] = torch.tensor(g, dtype=torch.long, device=device)
            gloss_mask[i, : len(g)] = False   # False = non padding

        # ── Stadio 2: gloss → translation ─────
        self.model2.eval()
        if decode_mode == "beam":
            # Beam search: un esempio alla volta
            trans_ids_list = []
            for i in range(B):
                ids = self.model2.beam_search(
                    gloss_ids=gloss_padded[i : i + 1],
                    bos_id=self.trans_bos_id,
                    eos_id=self.trans_eos_id,
                    beam_size=beam_size,
                    max_len=max_trans_len,
                    length_penalty=length_penalty,
                    gloss_key_padding_mask=gloss_mask[i : i + 1],
                )
                trans_ids_list.append(ids)
        else:
            trans_ids_list = self.model2.greedy_decode(
                gloss_ids=gloss_padded,
                bos_id=self.trans_bos_id,
                eos_id=self.trans_eos_id,
                max_len=max_trans_len,
                gloss_key_padding_mask=gloss_mask,
            )

        trans_texts = [self.trans_tok.decode(ids) for ids in trans_ids_list]

        return {
            "gloss_ids":   gloss_ids_list,
            "gloss_texts": gloss_texts,
            "trans_ids":   trans_ids_list,
            "trans_texts": trans_texts,
        }
