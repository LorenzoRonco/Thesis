"""
Esempi di utilizzo del Transformer Encoder per Sign Language Translation
=========================================================================

Questo file contiene 6 esempi progressivi di come usare il TransformerEncoder
per processare landmarks di linguaggio dei segni.

1. Basic usage
2. Variable sequence lengths with padding masking
3. Accessing intermediate layer outputs
4. Extracting attention weights for analysis
5. Training setup with loss computation
6. Fine-tuning sulla task specifica
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformer_encoder import (
    TransformerEncoder,
    create_transformer_encoder,
    LandmarkEmbedding,
    PositionalEncoding
)


# ============================================================================
# EXAMPLE 1: Basic Usage - Forward Pass on Normalized Landmarks
# ============================================================================

def example_1_basic_usage():
    """
    Uso base: elaborare landmarks normalizzati attraverso l'encoder.
    
    Input: Landmarks estratti da MediaPipe e già normalizzati
    Output: Rappresentazioni contestuali per ogni frame
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    # Parametri
    batch_size = 8
    num_frames = 150
    landmark_dim = 2108  # MediaPipe landmarks flattened
    hidden_dim = 512
    
    # Istanzia il modello
    encoder = TransformerEncoder(
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    # Dummy input: landmarks normalizzati [0, 1]
    # (batch, num_frames, landmark_dim)
    landmarks = torch.randn(batch_size, num_frames, landmark_dim)
    
    # Forward pass
    output = encoder(landmarks)
    
    # Risultati
    print(f"Input shape  (batch, frames, landmarks): {landmarks.shape}")
    print(f"Output shape (batch, frames, hidden):    {output.shape}")
    print(f"Model size:  {sum(p.numel() for p in encoder.parameters()):,} parameters")
    print()


# ============================================================================
# EXAMPLE 2: Variable Sequence Lengths with Padding Mask
# ============================================================================

def example_2_variable_lengths():
    """
    Gestire sequenze di lunghezze variabili con padding mask.
    
    Scenario: Video di diversa durata nel batch
    Soluzione: Padding mask che impedisce all'attenzione di guardare
               oltre la lunghezza effettiva di ogni sequenza
    """
    print("=" * 70)
    print("EXAMPLE 2: Variable Sequence Lengths with Padding Mask")
    print("=" * 70)
    
    # Configurazione
    batch_size = 4
    max_frames = 200
    landmark_dim = 2108
    hidden_dim = 512
    
    # Lunghezze effettive variabili
    seq_lens = torch.tensor([200, 150, 100, 80])
    
    encoder = TransformerEncoder(
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=8
    )
    
    # Input padded
    landmarks = torch.randn(batch_size, max_frames, landmark_dim)
    
    # Forward pass con mask
    output = encoder(landmarks, seq_lens=seq_lens)
    
    print(f"Sequence lengths: {seq_lens.tolist()}")
    print(f"Max length:       {max_frames}")
    print(f"Output shape:     {output.shape}")
    print("\nNote: Attention in the encoder only attends to valid positions")
    print("      (masked positions are set to -inf before softmax)")
    print()


# ============================================================================
# EXAMPLE 3: Intermediate Layer Outputs
# ============================================================================

def example_3_intermediate_layers():
    """
    Accedere agli output intermedi di ogni strato dell'encoder.
    
    Utile per:
    - Visualizzare l'evoluzione delle rappresentazioni
    - Estrarre feature da layer intermedi
    - Implementare skip connections
    """
    print("=" * 70)
    print("EXAMPLE 3: Intermediate Layer Outputs")
    print("=" * 70)
    
    batch_size = 2
    num_frames = 100
    landmark_dim = 2108
    hidden_dim = 256
    
    encoder = TransformerEncoder(
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=8
    )
    
    landmarks = torch.randn(batch_size, num_frames, landmark_dim)
    
    # Ritorna output di tutti gli strati
    all_outputs = encoder(landmarks, return_all_layers=True)
    
    print(f"Numero totale di strati: {len(all_outputs)} (embedding + 3 encoders)")
    print("\nLayer-wise output shapes:")
    for i, layer_output in enumerate(all_outputs):
        print(f"  Layer {i:d}: {layer_output.shape}")
    
    print("\nUSE CASES:")
    print("  - Visualizzazione: Traccia come i landmarks evolvono")
    print("  - Feature extraction: Usa layer intermedi come feature")
    print("  - Probing: Analizza cosa impara ogni strato")
    print()


# ============================================================================
# EXAMPLE 4: Attention Weights Analysis
# ============================================================================

def example_4_attention_weights():
    """
    Estrai e analizza i pesi di attenzione.
    
    Utile per:
    - Interpretabilità: vedere quale frames l'encoder sta focalizzando
    - Debug: verificare che l'attenzione sia ragionevole
    - Visualizzazione: heat map dei pesi di attenzione
    """
    print("=" * 70)
    print("EXAMPLE 4: Analyzing Attention Weights")
    print("=" * 70)
    
    batch_size = 1
    num_frames = 50
    landmark_dim = 2108
    hidden_dim = 256
    
    encoder = TransformerEncoder(
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4
    )
    
    landmarks = torch.randn(batch_size, num_frames, landmark_dim)
    output = encoder(landmarks)
    
    # Estrai pesi di attenzione
    attention_weights = encoder.get_attention_weights()
    
    print(f"Numero di strati encoder: {len(attention_weights)}")
    for layer_idx, weights in enumerate(attention_weights):
        print(f"\nLayer {layer_idx}:")
        print(f"  Shape: {weights.shape} (batch, heads, seq_len, seq_len)")
        print(f"  Value range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  Mean attention per position:")
        
        # Media su batch e heads
        mean_attn = weights.mean(dim=(0, 1))  # (seq_len, seq_len)
        
        # Mostra quali posizioni l'encoder focalizza in media
        for pos in range(0, num_frames, 10):
            mean_weight = mean_attn[pos].mean().item()
            print(f"    Position {pos:3d}: {mean_weight:.4f}")
    
    print("\nINTERPRETABILITY TIP:")
    print("  Argmax per posizione ti dice quale frame è più importante")
    print("  per ogni step temporale")
    print()


# ============================================================================
# EXAMPLE 5: Training Loop
# ============================================================================

def example_5_training_setup():
    """
    Setup completo di training con decoder dummy.
    
    Mostra come:
    - Integrare l'encoder nel pipeline di training
    - Computare loss e backprop
    - Gestire sequence lengths durante training
    """
    print("=" * 70)
    print("EXAMPLE 5: Training Setup")
    print("=" * 70)
    
    # Modello
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    # Dummy decoder: media pooling + linear layer
    # (In pratica: RNN, Transformer Decoder, o regressore per testo)
    class DummyDecoder(nn.Module):
        def __init__(self, hidden_dim, vocab_size):
            super().__init__()
            self.fc = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, encoder_out):
            # Media pooling su frame
            pooled = encoder_out.mean(dim=1)  # (batch, hidden_dim)
            return self.fc(pooled)  # (batch, vocab_size)
    
    vocab_size = 5000
    decoder = DummyDecoder(512, vocab_size)
    
    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy training data
    batch_size = 8
    num_frames = 150
    
    landmarks = torch.randn(batch_size, num_frames, 2108)
    targets = torch.randint(0, vocab_size, (batch_size,))
    
    # Training step
    optimizer.zero_grad()
    
    encoder_out = encoder(landmarks)
    logits = decoder(encoder_out)
    loss = criterion(logits, targets)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    
    print(f"Encoder output shape: {encoder_out.shape}")
    print(f"Logits shape:         {logits.shape}")
    print(f"Loss:                 {loss.item():.4f}")
    print(f"\nTraining Step Complete!")
    print(f"  Forward pass: ✓")
    print(f"  Loss computation: ✓")
    print(f"  Backward propagation: ✓")
    print(f"  Gradient clipping: ✓ (max_norm=1.0)")
    print()


# ============================================================================
# EXAMPLE 6: Transfer Learning / Fine-tuning
# ============================================================================

def example_6_finetuning():
    """
    Fine-tuning: congela l'encoder pre-trainato e allena solo il decoder.
    
    Scenario:
    - Encoder pre-trainato su grande dataset
    - Decoder da addestrare su task specifica
    """
    print("=" * 70)
    print("EXAMPLE 6: Fine-tuning Workflow")
    print("=" * 70)
    
    # Encoder pre-trainato (simulato)
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=512,
        num_layers=4,
        num_heads=8
    )
    
    # Congela i parametri dell'encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Decoder per il task specifico
    decoder = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 5000)  # vocab_size
    )
    
    # Solo il decoder è trainabile
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    
    print("Encoder configuration:")
    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Trainable parameters: 0 (frozen)")
    print()
    print("Decoder configuration:")
    pred_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"  Total parameters: {pred_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    print("FINE-TUNING BENEFITS:")
    print("  ✓ Faster training (fewer parameters)")
    print("  ✓ Lower memory requirements")
    print("  ✓ Leverages pre-trained features")
    print("  ✓ Better generalization on small datasets")
    print()


# ============================================================================
# EXAMPLE 7: Different Configurations
# ============================================================================

def example_7_model_configs():
    """
    Configurazioni pre-built per diversi use case.
    """
    print("=" * 70)
    print("EXAMPLE 7: Pre-built Model Configurations")
    print("=" * 70)
    
    configs = {
        'light': {
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
        },
        'standard': {
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
        },
        'heavy': {
            'hidden_dim': 768,
            'num_layers': 6,
            'num_heads': 12,
        }
    }
    
    batch_size = 4
    num_frames = 150
    landmark_dim = 2108
    
    for config_name, config_params in configs.items():
        encoder = TransformerEncoder(
            landmark_dim=landmark_dim,
            **config_params
        )
        
        landmarks = torch.randn(batch_size, num_frames, landmark_dim)
        output = encoder(landmarks)
        
        num_params = sum(p.numel() for p in encoder.parameters())
        
        print(f"\n{config_name.upper()} Configuration:")
        print(f"  Hidden dim: {config_params['hidden_dim']}")
        print(f"  Num layers: {config_params['num_layers']}")
        print(f"  Num heads:  {config_params['num_heads']}")
        print(f"  Total params: {num_params:,}")
        print(f"  Output shape: {output.shape}")
    
    print("\nRECOMMENDATIONS:")
    print("  LIGHT:    Prototyping, limited GPU memory, fast training")
    print("  STANDARD: Production, balanced performance/efficiency")
    print("  HEAVY:    Large datasets, high-end GPUs, maximum capacity")
    print()


def main():
    """Esegui tutti gli esempi."""
    example_1_basic_usage()
    example_2_variable_lengths()
    example_3_intermediate_layers()
    example_4_attention_weights()
    example_5_training_setup()
    example_6_finetuning()
    example_7_model_configs()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
