import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class CNNBranch(nn.Module):
    """
    Branch CNN per estrazione feature da frame video croppati.

    Pipeline:
        MobileNetV3  — feature spaziali per frame      [B, T, 960]
        1D-CNN       — pattern temporali locali         [B, T, 512]
        GRU          — dipendenze temporali long-term   [B, T, 512]

    Input : [B, T, 3, 224, 224]
    Output: [B, T, 512]
    """

    def __init__(
        self,
        d_model: int = 512,
        gru_layers: int = 2,
        dropout: float = 0.1,
        frame_chunk_size: int | None = 16,
    ):
        super().__init__()
        self.frame_chunk_size = frame_chunk_size

        # ------------------------------------------------------------------
        # 2D-CNN: MobileNetV3 Large pretrained
        # Rimuoviamo il classifier finale — ci serve solo il feature extractor
        # Output per frame: [B, 960, 7, 7] → average pooling → [B, 960]
        # ------------------------------------------------------------------
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        # features contiene tutti i layer convoluzionali + avgpool
        self.spatial_encoder = nn.Sequential(
            mobilenet.features,   # [B, 960, 7, 7]
            mobilenet.avgpool,    # [B, 960, 1, 1]
            nn.Flatten(),         # [B, 960]
        )
        mobilenet_out_dim = 960

        # ------------------------------------------------------------------
        # 1D-CNN: pattern temporali locali
        # Lavora sulla dimensione T — ogni kernel "vede" kernel_size frame
        # consecutivi e ne estrae un pattern.
        # in_channels=960  (feature MobileNet)
        # out_channels=512 (d_model)
        # kernel_size=3    → ogni frame guarda sé stesso + 1 frame a sx e dx
        # padding=1        → mantiene la lunghezza T invariata
        # ------------------------------------------------------------------
        self.temporal_local = nn.Sequential(
            nn.Conv1d(
                in_channels=mobilenet_out_dim,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # ------------------------------------------------------------------
        # GRU: dipendenze temporali a lungo termine
        # batch_first=True → input [B, T, d_model]
        # unidirezionale   → compatibile con uso real-time
        # ------------------------------------------------------------------
        self.temporal_global = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x                   : [B, T, 3, 224, 224]
            src_key_padding_mask: [B, T] bool — True sui frame di padding
                                  (stessa mask del Transformer Encoder)
        Returns:
            [B, T, 512]
        """
        B, T, C, H, W = x.shape

        # --- MobileNetV3: processa i frame in chunk per ridurre il picco VRAM
        # Mantiene tutta la sequenza temporale, ma evita il batch B*T in un colpo solo.
        x = x.view(B * T, C, H, W)          # [B*T, 3, 224, 224]
        if self.frame_chunk_size is None or self.frame_chunk_size <= 0:
            x = self.spatial_encoder(x)      # [B*T, 960]
        else:
            chunks = []
            for start in range(0, x.size(0), self.frame_chunk_size):
                end = min(start + self.frame_chunk_size, x.size(0))
                chunks.append(self.spatial_encoder(x[start:end]))
            x = torch.cat(chunks, dim=0)     # [B*T, 960]

        x = x.view(B, T, -1)                 # [B, T, 960]

        # --- 1D-CNN: pattern temporali locali ------------------------------
        # Conv1d si aspetta [B, C, T], quindi trasponiamo
        x = x.permute(0, 2, 1)              # [B, 960, T]
        x = self.temporal_local(x)          # [B, 512, T]
        x = x.permute(0, 2, 1)              # [B, T, 512]

        # --- GRU: dipendenze temporali a lungo termine ---------------------
        x, _ = self.temporal_global(x)      # [B, T, 512]

        return x