import torch
import torch.nn as nn
from transformers import BartTokenizer
from .transformer_encoder import PositionalEncoding


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder autoregressivo per generazione testo.

    Riceve il tensore fuso [B, T, 512] dal FusionModule e genera
    la traduzione token per token.

    Training : teacher forcing — riceve i token corretti come input
    Inferenza : greedy decoding — prende il token più probabile ad ogni step
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # Tokenizer e vocabolario
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.vocab_size = self.tokenizer.vocab_size  # 50265
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Embedding dei token testuali
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=d_model,
            padding_idx=self.pad_token_id,
        )

        # Encoding posizionale per la sequenza testuale
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        # Decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Proiezione finale → logits sul vocabolario
        self.output_projection = nn.Linear(d_model, self.vocab_size)

    def forward(
        self,
        memory: torch.Tensor,
        target_tokens: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Modalità training con teacher forcing.

        Args:
            memory                  : [B, T, 512] — output FusionModule
            target_tokens           : [B, L]       — token target (dal tokenizer)
            memory_key_padding_mask : [B, T] bool  — padding mask dei frame video
                                      (stessa del Transformer Encoder e CNN Branch)
        Returns:
            logits: [B, L, vocab_size]
        """
        L = target_tokens.size(1)

        # Maschera causale: ogni token vede solo i token precedenti
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=target_tokens.device
        )

        # Maschera padding sui token target
        target_padding_mask = (target_tokens == self.pad_token_id)  # [B, L]

        # Embedding + encoding posizionale
        x = self.token_embedding(target_tokens)  # [B, L, 512]
        x = self.pos_encoding(x)                 # [B, L, 512]

        # Decoder: self-attention sui token + cross-attention su memory
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,                        # causale sui token
            tgt_key_padding_mask=target_padding_mask,    # padding sui token
            memory_key_padding_mask=memory_key_padding_mask,  # padding sui frame
        )  # [B, L, 512]

        return self.output_projection(x)  # [B, L, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        max_new_tokens: int = 100,
    ) -> list[str]:
        """
        Greedy decoding — genera la traduzione token per token.

        Args:
            memory                  : [B, T, 512]
            memory_key_padding_mask : [B, T] bool
            max_new_tokens          : numero massimo token generati
        Returns:
            Lista di stringhe tradotte, una per elemento del batch
        """
        B = memory.size(0)
        device = memory.device

        # Inizializza con il token BOS per ogni elemento del batch
        generated = torch.full(
            (B, 1), self.bos_token_id, dtype=torch.long, device=device
        )
        # Tiene traccia di quali sequenze hanno già generato EOS
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(
                memory=memory,
                target_tokens=generated,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # [B, current_len, vocab_size]

            # Prende il token più probabile all'ultimo step
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)  # [B, current_len+1]

            # Marca come finished le sequenze che hanno generato EOS
            finished |= (next_token.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break

        # Decodifica i token in stringhe
        translations = []
        for i in range(B):
            tokens = generated[i].tolist()
            # Rimuove BOS e tronca all'EOS
            if self.eos_token_id in tokens:
                tokens = tokens[1:tokens.index(self.eos_token_id)]
            else:
                tokens = tokens[1:]
            translations.append(self.tokenizer.decode(tokens, skip_special_tokens=True))

        return translations