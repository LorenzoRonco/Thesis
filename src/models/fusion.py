import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """
    Fonde le rappresentazioni del Transformer Encoder e della CNN Branch.

    Pipeline:
        Concatenazione        [B, T, 1024]
        Linear + LayerNorm    [B, T, 512]   — proiezione nello spazio comune
        Deep layers           [B, T, 512]   — calcolo pesi delle rappresentazioni
        Output                [B, T, 512]   — pronto per il Transformer Decoder

    Input:
        landmark_features : [B, T, 512]  — output Transformer Encoder
        video_features    : [B, T, 512]  — output CNN Branch
    Output:
        [B, T, 512]
    """

    def __init__(
        self,
        d_model: int = 512,
        num_deep_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Proiezione iniziale: concatenazione → spazio comune
        # ------------------------------------------------------------------
        self.input_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 1024 → 512
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Deep layers: imparano quanto pesare le due rappresentazioni
        # Ogni layer è un blocco residuale — mantiene l'informazione
        # originale e aggiunge solo il delta appreso
        # ------------------------------------------------------------------
        self.deep_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_deep_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        landmark_features: torch.Tensor,
        video_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            landmark_features : [B, T, 512]
            video_features    : [B, T, 512]
            padding_mask      : [B, T] bool — True sui frame di padding
        Returns:
            [B, T, 512]
        """

        # --- Concatenazione -----------------------------------------------
        x = torch.cat([landmark_features, video_features], dim=-1)  # [B, T, 1024]

        # --- Proiezione nello spazio comune --------------------------------
        x = self.input_projection(x)   # [B, T, 512]

        # --- Deep layers con connessioni residuali -------------------------
        for layer in self.deep_layers:
            residual = x
            x = layer(x)
            x = self.dropout(x + residual)  # connessione residuale

        # --- Azzera i frame di padding ------------------------------------
        # Evita che il decoder veda rappresentazioni spurie sui frame vuoti
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x  # [B, T, 512]