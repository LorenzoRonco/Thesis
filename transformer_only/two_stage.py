"""
two_stage_hybrid.py

Architettura ibrida per lo Stadio 2: gloss → translation (tedesco)

    GlossEncoder custom  →  Bridge (d_model → 1024)  →  mBART Decoder + LoRA

Perché funziona meglio di mBART puro:
  - L'encoder è addestrato da zero appositamente per le gloss PHOENIX
    (parole tedesche in maiuscolo, telegrafiche) → nessun domain gap sul lato encoder
  - Il decoder è mBART pre-addestrato sul tedesco → genera tedesco fluente
    senza doverlo imparare da zero
  - Il bridge lineare (256→1024 + LayerNorm) adatta le dimensioni
  - LoRA sul solo decoder: i pesi dell'encoder mBART vengono caricati ma
    congelati e mai eseguiti (bypassati tramite encoder_outputs)

Classi esportate:
  - GlossEncoder           : encoder custom per token di gloss
  - HybridGlossToText      : GlossEncoder + bridge + mBART decoder
  - TwoStagePipelineHybrid : inferenza landmark → gloss → tedesco
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .models.transformer import (
    SinusoidalPositionalEncoding,
    SignLanguageTransformer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder gloss custom (identico a quello di two_stage.py originale)
# ─────────────────────────────────────────────────────────────────────────────

class GlossEncoder(nn.Module):
    """
    Encoder Transformer per sequenze di gloss token.

    Trasforma (B, G) token id → (B, G, d_model) rappresentazioni contestuali.
    Addestrato da zero su PHOENIX: impara la distribuzione specifica delle
    gloss (parole tedesche in maiuscolo, abbreviate, telegrafiche).
    """

    def __init__(
        self,
        gloss_vocab_size: int,
        d_model:          int,
        nhead:            int,
        num_layers:       int,
        dim_feedforward:  int,
        dropout:          float,
        max_len:          int,
        pad_id:           int,
    ):
        super().__init__()
        self.pad_id    = pad_id
        self.embedding = nn.Embedding(gloss_vocab_size, d_model, padding_idx=pad_id)
        self.scale     = math.sqrt(d_model)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.norm_in   = nn.LayerNorm(d_model)

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
        gloss_ids:               Tensor,            # (B, G)
        gloss_key_padding_mask:  Tensor | None = None,  # (B, G) True=pad
    ) -> Tensor:                                    # (B, G, d_model)
        x = self.embedding(gloss_ids) * self.scale
        x = self.pos_enc(x)
        x = self.norm_in(x)
        if gloss_key_padding_mask is not None:
            x = x.masked_fill(gloss_key_padding_mask.unsqueeze(-1), 0.0)
        return self.encoder(x, src_key_padding_mask=gloss_key_padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Architettura ibrida: GlossEncoder + Bridge + mBART Decoder
# ─────────────────────────────────────────────────────────────────────────────

class HybridGlossToText(nn.Module):
    """
    GlossEncoder custom + Bridge lineare + mBART Decoder con LoRA.

    Flusso forward (training):
        gloss_ids  ──► GlossEncoder ──► (B, G, d_model)
                                          │
                                        Bridge  ──► (B, G, 1024)
                                          │
                              encoder_outputs ──► mBART Decoder ──► loss / logits

    Il decoder mBART riceve le rappresentazioni gloss proiettate come se
    fossero l'output del suo encoder originale (via BaseModelOutput).
    L'encoder mBART è caricato ma congelato e mai eseguito.

    Args:
        gloss_vocab_size:        dimensione vocabolario gloss
        d_model:                 dim nascosta del GlossEncoder
        nhead:                   teste di attenzione del GlossEncoder
        num_enc_layers:          layer del GlossEncoder
        dim_feedforward:         dim FFN del GlossEncoder
        dropout:                 dropout
        max_gloss_len:           lunghezza max sequenza gloss
        gloss_pad_id:            pad id nel vocabolario gloss
        lora_r:                  rank LoRA per il decoder mBART
        lora_alpha:              scaling LoRA (tipicamente 2×lora_r)
        lora_dropout:            dropout negli adapter LoRA
        lora_target_modules:     moduli del decoder su cui applicare LoRA
        forced_bos_token_id:     id de_DE per forzare generazione in tedesco
        gradient_checkpointing:  abilita gradient checkpointing sul decoder
    """

    MBART_NAME    = "facebook/mbart-large-cc25"
    MBART_D_MODEL = 1024   # dimensione nascosta di mBART-large

    def __init__(
        self,
        gloss_vocab_size:     int,
        d_model:              int   = 256,
        nhead:                int   = 4,
        num_enc_layers:       int   = 2,
        dim_feedforward:      int   = 512,
        dropout:              float = 0.1,
        max_gloss_len:        int   = 64,
        gloss_pad_id:         int   = 0,
        lora_r:               int   = 16,
        lora_alpha:           int   = 32,
        lora_dropout:         float = 0.1,
        lora_target_modules:  list  = None,
        forced_bos_token_id:  int   = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.gloss_pad_id = gloss_pad_id
        self.d_model      = d_model

        # ── 1. Encoder gloss (addestrato da zero su PHOENIX) ──────────────────
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

        # ── 2. Bridge: d_model → mBART hidden (1024) ─────────────────────────
        # LayerNorm stabilizza la scala prima di entrare nel decoder mBART
        # (che si aspetta input della stessa distribuzione dei suoi encoder output)
        self.bridge = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, self.MBART_D_MODEL),
            nn.LayerNorm(self.MBART_D_MODEL),
        )

        # ── 3. mBART con LoRA sul solo decoder ────────────────────────────────
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        base = MBartForConditionalGeneration.from_pretrained(self.MBART_NAME)

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.mbart = get_peft_model(base, lora_cfg)

        # Congela encoder mBART (encoder + eventuali LoRA su encoder):
        # viene caricato ma mai eseguito (usiamo encoder_outputs per bypassarlo)
        for name, param in self.mbart.named_parameters():
            if "model.encoder" in name:
                param.requires_grad = False

        # Gradient checkpointing sul decoder mBART
        if gradient_checkpointing:
            self.mbart.enable_input_require_grads()
            self.mbart.gradient_checkpointing_enable()

        self.forced_bos_token_id = forced_bos_token_id
        self.trans_pad_id        = self.mbart.config.pad_token_id

    # ── Encode gloss con encoder custom + bridge ──────────────────────────────
    def _encode(
        self,
        gloss_ids:              Tensor,            # (B, G)
        gloss_key_padding_mask: Tensor | None,     # (B, G) True=pad
    ) -> tuple[Tensor, Tensor]:
        """
        Ritorna (encoder_hidden_states, encoder_attention_mask):
          - encoder_hidden_states: (B, G, 1024)  — input per il decoder mBART
          - encoder_attention_mask: (B, G) long  — 1=token reale, 0=padding
        """
        B, G    = gloss_ids.shape
        device  = gloss_ids.device

        memory      = self.gloss_encoder(gloss_ids, gloss_key_padding_mask)  # (B, G, d_model)
        memory_proj = self.bridge(memory)                                      # (B, G, 1024)

        if gloss_key_padding_mask is not None:
            # HF usa 1=reale / 0=pad, inverse rispetto a key_padding_mask (True=pad)
            enc_attn_mask = (~gloss_key_padding_mask).long()
        else:
            enc_attn_mask = torch.ones(B, G, dtype=torch.long, device=device)

        return memory_proj, enc_attn_mask

    # ── Forward (training con teacher forcing) ────────────────────────────────
    def forward(
        self,
        gloss_ids:              Tensor,            # (B, G)
        gloss_key_padding_mask: Tensor | None,     # (B, G) True=pad
        labels:                 Tensor,            # (B, L) -100 su padding
    ) -> tuple:
        """
        Ritorna (logits, loss, None).
        loss è calcolata da mBART internamente sulle posizioni != -100.
        """
        memory_proj, enc_attn_mask = self._encode(gloss_ids, gloss_key_padding_mask)

        out = self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory_proj),
            attention_mask=enc_attn_mask,
            labels=labels,
            label_smoothing=0.1,
        )
        return out.logits, out.loss, None

    # ── Generazione autoregressiva ────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        gloss_ids:              Tensor,
        gloss_key_padding_mask: Tensor | None,
        max_new_tokens:         int   = 128,
        num_beams:              int   = 4,
        length_penalty:         float = 0.6,
    ) -> Tensor:
        """
        Genera la traduzione. Greedy con num_beams=1, beam search con >1.
        """
        memory_proj, enc_attn_mask = self._encode(gloss_ids, gloss_key_padding_mask)

        return self.mbart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory_proj),
            attention_mask=enc_attn_mask,
            forced_bos_token_id=self.forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            max_length=None,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=(num_beams > 1),
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    # ── Utilità ───────────────────────────────────────────────────────────────
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def print_trainable_parameters(self):
        enc_params = sum(p.numel() for p in self.gloss_encoder.parameters())
        bri_params = sum(p.numel() for p in self.bridge.parameters())
        lora_params = sum(
            p.numel() for n, p in self.mbart.named_parameters()
            if p.requires_grad and "model.encoder" not in n
        )
        total = self.num_parameters(trainable_only=False)
        trainable = enc_params + bri_params + lora_params
        print(
            f"[HybridGlossToText] Parametri trainable:\n"
            f"  GlossEncoder : {enc_params:>12,}\n"
            f"  Bridge       : {bri_params:>12,}\n"
            f"  mBART LoRA   : {lora_params:>12,}\n"
            f"  ─────────────────────────────\n"
            f"  Trainable    : {trainable:>12,}  ({100*trainable/total:.2f}% del totale)\n"
            f"  Totale       : {total:>12,}"
        )

    # ── Salvataggio ───────────────────────────────────────────────────────────
    def save(self, checkpoint_dir: str | Path):
        """
        Salva tutti i pesi trainable:
          - {checkpoint_dir}/encoder_bridge.pt  : GlossEncoder + bridge
          - {checkpoint_dir}/lora/               : adapter LoRA del decoder mBART
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "gloss_encoder": self.gloss_encoder.state_dict(),
                "bridge":        self.bridge.state_dict(),
            },
            checkpoint_dir / "encoder_bridge.pt",
        )
        self.mbart.save_pretrained(str(checkpoint_dir / "lora"))

    @classmethod
    def load(
        cls,
        checkpoint_dir:      str | Path,
        gloss_vocab_size:    int,
        forced_bos_token_id: int   = None,
        d_model:             int   = 256,
        nhead:               int   = 4,
        num_enc_layers:      int   = 2,
        dim_feedforward:     int   = 512,
        dropout:             float = 0.1,
        max_gloss_len:       int   = 64,
        gloss_pad_id:        int   = 0,
        lora_r:              int   = 16,
        lora_alpha:          int   = 32,
        lora_dropout:        float = 0.1,
        lora_target_modules: list  = None,
    ) -> "HybridGlossToText":
        """Carica il modello completo da un checkpoint salvato con save()."""
        checkpoint_dir = Path(checkpoint_dir)

        model = cls(
            gloss_vocab_size=gloss_vocab_size,
            d_model=d_model, nhead=nhead, num_enc_layers=num_enc_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            max_gloss_len=max_gloss_len, gloss_pad_id=gloss_pad_id,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            forced_bos_token_id=forced_bos_token_id,
            gradient_checkpointing=False,
        )

        enc_bri = torch.load(checkpoint_dir / "encoder_bridge.pt", map_location="cpu")
        model.gloss_encoder.load_state_dict(enc_bri["gloss_encoder"])
        model.bridge.load_state_dict(enc_bri["bridge"])

        # Ricarica mBART base + adapter LoRA
        base = MBartForConditionalGeneration.from_pretrained(cls.MBART_NAME)
        model.mbart = PeftModel.from_pretrained(base, str(checkpoint_dir / "lora"))

        return model
    


    def init_gloss_embeddings_from_mbart(self, gloss_tokenizer) -> int:
        """
        Copia i pesi embedding di mBART nel GlossEncoder per i token in comune.
        Le gloss PHOENIX sono parole tedesche uppercase: mBART conosce le loro
        versioni lowercase, il che fornisce un'inizializzazione semanticamente ricca.
        Ritorna il numero di token inizializzati con successo.
        """
        mbart_tok  = MBartTokenizer.from_pretrained(self.MBART_NAME)
        # accesso ai pesi embedding del modello base sotto PEFT
        mbart_emb  = self.mbart.get_input_embeddings().weight.data  # (vocab, 1024)

        # proiezione 1024 → d_model tramite inversa lineare approssimata del bridge
        with torch.no_grad():
            bridge_weight = self.bridge[3].weight.data @ self.bridge[0].weight.data
            proj = torch.linalg.pinv(bridge_weight)  # (d_model, 1024)

        hits = 0
        with torch.no_grad():
            for token, idx in gloss_tokenizer.token2idx.items():
                # prova sia uppercase che lowercase
                for variant in [token.lower(), token.capitalize(), token]:
                    mbart_id = mbart_tok.convert_tokens_to_ids(f"▁{variant}")
                    if mbart_id == mbart_tok.unk_token_id:
                        mbart_id = mbart_tok.convert_tokens_to_ids(variant)
                    if mbart_id != mbart_tok.unk_token_id:
                        src = mbart_emb[mbart_id]          # (1024,)
                        self.gloss_encoder.embedding.weight.data[idx] = proj @ src
                        hits += 1
                        break
        print(f"[init_embeddings] {hits}/{len(gloss_tokenizer.token2idx)} token inizializzati da mBART")
        return hits


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline di inferenza a due stadi (versione ibrida)
# ─────────────────────────────────────────────────────────────────────────────

class TwoStagePipelineHybrid(nn.Module):
    """
    Inferenza completa: landmark → gloss (Modello 1) → tedesco (ibrido).

    Modello 1 (SignLanguageTransformer) rimane invariato.
    Modello 2 è HybridGlossToText.

    Args:
        model1:          SignLanguageTransformer (landmark → gloss)
        model2:          HybridGlossToText (gloss → traduzione)
        gloss_tokenizer: SentenceTokenizer del Modello 1
        mbart_tokenizer: MBartTokenizer (per decodificare l'output)
        gloss_bos_id:    bos id vocabolario gloss
        gloss_eos_id:    eos id vocabolario gloss
        gloss_pad_id:    pad id vocabolario gloss
    """

    def __init__(
        self,
        model1:          SignLanguageTransformer,
        model2:          HybridGlossToText,
        gloss_tokenizer,
        mbart_tokenizer: MBartTokenizer,
        gloss_bos_id:    int,
        gloss_eos_id:    int,
        gloss_pad_id:    int = 0,
    ):
        super().__init__()
        self.model1       = model1
        self.model2       = model2
        self.gloss_tok    = gloss_tokenizer
        self.mbart_tok    = mbart_tokenizer
        self.gloss_bos_id = gloss_bos_id
        self.gloss_eos_id = gloss_eos_id
        self.gloss_pad_id = gloss_pad_id

    @torch.no_grad()
    def forward(
        self,
        src:                  Tensor,              # (B, T, feat_dim)
        src_key_padding_mask: Tensor | None = None,
        max_gloss_len:        int   = 64,
        max_trans_len:        int   = 128,
        num_beams:            int   = 4,
        length_penalty:       float = 0.6,
    ) -> dict:
        device = src.device
        B      = src.size(0)

        # ── Stadio 1: landmark → gloss (Modello 1 invariato) ──────────────────
        self.model1.eval()
        gloss_ids_list = self.model1.greedy_decode(
            src=src,
            bos_id=self.gloss_bos_id,
            eos_id=self.gloss_eos_id,
            max_len=max_gloss_len,
            src_key_padding_mask=src_key_padding_mask,
        )
        gloss_texts = [self.gloss_tok.decode(ids) for ids in gloss_ids_list]

        # ── Padding gloss con token speciali per il Modello 2 ─────────────────
        gloss_with_special = [
            [self.gloss_bos_id] + ids + [self.gloss_eos_id]
            for ids in gloss_ids_list
        ]
        max_g        = max(len(g) for g in gloss_with_special)
        gloss_padded = torch.full((B, max_g), self.gloss_pad_id, dtype=torch.long, device=device)
        gloss_mask   = torch.ones(B, max_g, dtype=torch.bool, device=device)
        for i, g in enumerate(gloss_with_special):
            gloss_padded[i, :len(g)] = torch.tensor(g, dtype=torch.long, device=device)
            gloss_mask[i, :len(g)]   = False

        # ── Stadio 2: gloss → traduzione (ibrido) ─────────────────────────────
        self.model2.eval()
        generated = self.model2.generate(
            gloss_ids=gloss_padded,
            gloss_key_padding_mask=gloss_mask,
            max_new_tokens=max_trans_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        trans_texts = self.mbart_tok.batch_decode(generated, skip_special_tokens=True)

        return {
            "gloss_texts": gloss_texts,
            "trans_texts": trans_texts,
        }