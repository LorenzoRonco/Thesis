import torch
import torch.nn as nn
from .transformer_encoder import LandmarkTransformerEncoder
from .cnn_branch import CNNBranch
from .fusion import FusionModule
from .transformer_decoder import TransformerDecoder


class SignLanguageTranslator(nn.Module):
    """
    Modello completo per Sign Language Translation.

    Pipeline:
        Landmark branch : LandmarkTransformerEncoder  [B, T, 512]
        Video branch    : CNNBranch                   [B, T, 512]
        Fusione         : FusionModule                [B, T, 512]
        Decoder         : TransformerDecoder          [B, L, vocab_size]

    Input:
        landmarks    : [B, T, 2108]       — landmark appiattiti per frame
        video_frames : [B, T, 3, 224, 224] — frame RGB normalizzati
        padding_mask : [B, T] bool         — True sui frame di padding
    """

    def __init__(
        self,
        # Encoder
        landmark_input_dim: int = 2108,
        d_model: int = 512,
        encoder_heads: int = 8,
        encoder_layers: int = 6,
        encoder_ffn_dim: int = 2048,
        # CNN Branch
        gru_layers: int = 2,
        # Fusion
        fusion_deep_layers: int = 2,
        # Decoder
        decoder_heads: int = 8,
        decoder_layers: int = 6,
        decoder_ffn_dim: int = 2048,
        max_text_len: int = 512,
        # Comune
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = LandmarkTransformerEncoder(
            input_dim=landmark_input_dim,
            d_model=d_model,
            nhead=encoder_heads,
            num_layers=encoder_layers,
            dim_feedforward=encoder_ffn_dim,
            dropout=dropout,
        )

        self.cnn_branch = CNNBranch(
            d_model=d_model,
            gru_layers=gru_layers,
            dropout=dropout,
        )

        self.fusion = FusionModule(
            d_model=d_model,
            num_deep_layers=fusion_deep_layers,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            max_seq_len=max_text_len,
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        video_frames: torch.Tensor,
        padding_mask: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modalità training con teacher forcing.

        Args:
            landmarks     : [B, T, 2108]
            video_frames  : [B, T, 3, 224, 224]
            padding_mask  : [B, T] bool — True sui frame di padding
            target_tokens : [B, L]      — token target per teacher forcing
        Returns:
            logits: [B, L, vocab_size]
        """
        # Branch landmark
        landmark_features = self.encoder(
            landmarks,
            src_key_padding_mask=padding_mask,
        )  # [B, T, 512]

        # Branch video
        video_features = self.cnn_branch(
            video_frames,
            src_key_padding_mask=padding_mask,
        )  # [B, T, 512]

        # Fusione
        memory = self.fusion(
            landmark_features=landmark_features,
            video_features=video_features,
            padding_mask=padding_mask,
        )  # [B, T, 512]

        # Decoder
        logits = self.decoder(
            memory=memory,
            target_tokens=target_tokens,
            memory_key_padding_mask=padding_mask,
        )  # [B, L, vocab_size]

        return logits

    @torch.no_grad()
    def generate(
        self,
        landmarks: torch.Tensor,
        video_frames: torch.Tensor,
        padding_mask: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> list[str]:
        """
        Inferenza con greedy decoding.

        Args:
            landmarks     : [B, T, 2108]
            video_frames  : [B, T, 3, 224, 224]
            padding_mask  : [B, T] bool
            max_new_tokens: numero massimo token generati
        Returns:
            Lista di stringhe tradotte, una per elemento del batch
        """
        landmark_features = self.encoder(
            landmarks,
            src_key_padding_mask=padding_mask,
        )
        video_features = self.cnn_branch(
            video_frames,
            src_key_padding_mask=padding_mask,
        )
        memory = self.fusion(
            landmark_features=landmark_features,
            video_features=video_features,
            padding_mask=padding_mask,
        )
        return self.decoder.generate(
            memory=memory,
            memory_key_padding_mask=padding_mask,
            max_new_tokens=max_new_tokens,
        )