import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Encoding posizionale sinusoidale sulla dimensione temporale (frame index).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        # dropout a ogni passaggio del training spegne casualmente alcuni neuroni in un livello della rete
        # serve per prevenire overfitting
        self.dropout = nn.Dropout(p=dropout)
        
        #crea una matrice vuota di [5000, 512] -> 1 riga x ogni frame
        # 1 colonna per ogni dimensione del modello
        pe = torch.zeros(max_len, d_model)

        # crea vettore di posizioni [0, 1, 2, ..., 4999] e lo trasforma in colonna [5000, 1]
        # rappresenta l'indice di ogni frame
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # calcola vettore frequenze per ogni dimensione del modello
        # prime dimensioni hanno frequenze alte -> x distinguere posizioni vicine
        # ultime dim hanno frequenze basse -> x distinguere posizioni lontane
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # colonne pari riempite con senoidi, colonne dispari con cosenoidi
        # per identificare posizione frame in sequenza
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LandmarkTransformerEncoder(nn.Module):
    """
    Transformer Encoder per sequenze di landmark MediaPipe.

    Input : [B, T, 2108]   — landmark appiattiti per frame (dal DataLoader)
    Output: [B, T, d_model] — rappresentazione contestuale per frame
    """

    def __init__(
        self,
        input_dim: int = 2108,        # 527 landmarks × 4 dimensioni
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()

        # Proiezione dallo spazio landmark al d_model
        # landmark in input sono vettori a 2108 valori
        # Transformer lavora a 512 dimensioni (d_model) -> linear fa proiezione imparando quali combinazioni
        # di landmark sono informative
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model), # normalizzazione per stabilizzare il training
        )

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)


        # definisco singolo layer encoder.
        # prima self attention, poi rete feed forward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [B, T, d_model] -> prima dimensione del tensore è il batch
            norm_first=True,   # Pre-LN: più stabile nel training
        )

        # impilo 6 layer dell'encoder appena definito
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model), #stabilizza output di tutto l'encoder
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x                   : [B, T, 2108]
            src_key_padding_mask: [B, T] bool — True sui frame di padding
                                  (viene direttamente dalla padding_mask del collate_fn)
        Returns:
            [B, T, d_model]
        """
        x = self.input_projection(x)    # [B, T, d_model]
        x = self.pos_encoding(x)        # [B, T, d_model]

        # passa attraverso i 6 layer dell'encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x # [B, T, d_model] per ogni frame video