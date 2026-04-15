"""
Transformer Decoder - Esempi Pratici di Utilizzo
=================================================

Questo file contiene 5 esempi progressivi che vanno da zero al training completo
di un modello end-to-end per Sign Language Translation (Video→Text).

Esempi:
  1. Basic Forward Pass - Generazione semplice di logit
  2. Training Loop - Allenamento con una mini batch
  3. Generazione Autoregressiva - Inferenza passo-passo
  4. Integrazione Dual-Stream - Con encoder upstream
  5. Full Training Pipeline - Training/Validation/Checkpointing

Autore: Thesis Project
Data: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer_decoder import TransformerDecoder, TransformerDecoderConfig


# =============================================================================
# EXAMPLE 1: Basic Forward Pass
# =============================================================================

def example_1_basic_forward():
    """
    Esempio minimale: forward pass del decoder.
    
    Non occorre nulla tranne il decoder e i tensori di input.
    """
    print("\n" + "="*70)
    print(" EXAMPLE 1: Basic Forward Pass")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 1. Crea il decoder
    decoder = TransformerDecoder(
        hidden_dim=512,
        vocab_size=10000,
        num_decoder_layers=4,
        num_heads=8,
        device=device
    ).to(device)
    
    # 2. Crea dati dummy
    batch_size = 2
    video_frames = 100
    seq_len = 50
    
    # Memory: output della fusione dual-stream upstream
    memory = torch.randn(batch_size, video_frames, 512, device=device)
    
    # Target tokens: sequenza di indici con BOS all'inizio
    target_tokens = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    target_tokens[:, 0] = 1  # BOS token in posizione 0
    
    # 3. Forward pass
    decoder.eval()
    with torch.no_grad():
        logits = decoder(memory=memory, target_tokens=target_tokens)
    
    # 4. Ispeziona output
    print(f"Memory shape: {memory.shape}")
    print(f"Target tokens shape: {target_tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"\nLogits sono i logit grezzi per il vocabolario.")
    print(f"Ogni posizione nel testo ha {logits.size(-1)} probabilità (una per token vocabolario).")
    
    # 5. Vedi i top-5 token predetti per la prima sequenza, prima posizione
    probs = F.softmax(logits[0, 0, :], dim=-1)
    top_5_probs, top_5_indices = torch.topk(probs, k=5)
    print(f"\nTop-5 token predetti per (batch=0, pos=0):")
    for idx, prob in zip(top_5_indices.cpu(), top_5_probs.cpu()):
        print(f"  Token {idx.item():5d}: {prob.item():.4f}")
    
    print("\n✓ Basic forward pass completato!")


# =============================================================================
# EXAMPLE 2: Training Loop con Mini Batch
# =============================================================================

def example_2_training_loop():
    """
    Esempio di training loop per un'epoch.
    
    Mostra come:
    - Calcolare la loss
    - Backprop
    - Gradient clipping
    - Aggiornamento pesi
    """
    print("\n" + "="*70)
    print(" EXAMPLE 2: Training Loop")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup modello
    decoder = TransformerDecoder(
        hidden_dim=512,
        vocab_size=5000,  # Vocab size ridotto per demo
        num_decoder_layers=2,  # Fewer layers per velocità
        device=device
    ).to(device)
    
    # Setup optimizer e loss
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD token
    
    # Crea mini dataset
    num_batches = 5
    batch_size = 4
    
    print(f"\nTraining per {num_batches} batch...")
    
    total_loss = 0
    for batch_idx in range(num_batches):
        # Crea batch casuale
        memory = torch.randn(batch_size, 50, 512, device=device)
        target_tokens = torch.randint(0, 5000, (batch_size, 30), device=device)
        target_tokens[:, 0] = 1  # BOS
        
        # ========== Forward Pass ==========
        decoder.train()
        logits = decoder(memory=memory, target_tokens=target_tokens)
        
        # ========== Calcola Loss ==========
        # Reshape per cross_entropy: (B, T, V) → (B*T, V)
        logits_flat = logits.view(-1, decoder.vocab_size)
        targets_flat = target_tokens.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        # ========== Backward Pass ==========
        optimizer.zero_grad()
        loss.backward()
        
        # ========== Gradient Clipping ==========
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        # ========== Update Weights ==========
        optimizer.step()
        
        total_loss += loss.item()
        
        print(f"  Batch {batch_idx+1}/{num_batches}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"\nAverage loss over {num_batches} batches: {avg_loss:.4f}")
    print("✓ Training loop completato!")


# =============================================================================
# EXAMPLE 3: Generazione Autoregressiva
# =============================================================================

def example_3_autoregressive_generation():
    """
    Esempio di generazione autoregressiva (inferenza).
    
    Il decoder genera token uno per volta, condizionato su tutti
    i token precedenti (causal).
    """
    print("\n" + "="*70)
    print(" EXAMPLE 3: Autoregressive Generation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup decoder
    decoder = TransformerDecoder(
        hidden_dim=512,
        vocab_size=100,  # Small vocab per demo
        device=device
    ).to(device)
    
    decoder.eval()
    
    # Memory (contesto multimodale)
    memory = torch.randn(1, 50, 512, device=device)  # Batch size = 1
    
    print("\nGenerazione autoregressiva con diversi parametri:\n")
    
    # ========== Setup 1: Temperature = 0.8 (deterministic) ==========
    print("Setup 1: Temperature = 0.8 (Less random)")
    with torch.no_grad():
        generated_ids = decoder.generate(
            memory=memory,
            bos_token_id=1,
            eos_token_id=2,
            max_length=20,
            temperature=0.8,
        )
    print(f"  Generated IDs: {generated_ids[0, :].cpu().tolist()}")
    print(f"  Length: {generated_ids.size(1)}")
    
    # ========== Setup 2: Temperature = 1.5 (random) ==========
    print("\nSetup 2: Temperature = 1.5 (More random)")
    with torch.no_grad():
        generated_ids = decoder.generate(
            memory=memory,
            bos_token_id=1,
            eos_token_id=2,
            max_length=20,
            temperature=1.5,
        )
    print(f"  Generated IDs: {generated_ids[0, :].cpu().tolist()}")
    print(f"  Length: {generated_ids.size(1)}")
    
    # ========== Setup 3: Top-k Sampling ==========
    print("\nSetup 3: Top-k Sampling (k=10)")
    with torch.no_grad():
        generated_ids = decoder.generate(
            memory=memory,
            bos_token_id=1,
            eos_token_id=2,
            max_length=20,
            temperature=1.0,
            top_k=10,
        )
    print(f"  Generated IDs: {generated_ids[0, :].cpu().tolist()}")
    print(f"  Length: {generated_ids.size(1)}")
    
    # ========== Setup 4: Nucleus Sampling (top_p) ==========
    print("\nSetup 4: Nucleus Sampling (top_p=0.9)")
    with torch.no_grad():
        generated_ids = decoder.generate(
            memory=memory,
            bos_token_id=1,
            eos_token_id=2,
            max_length=20,
            temperature=1.0,
            top_p=0.9,
        )
    print(f"  Generated IDs: {generated_ids[0, :].cpu().tolist()}")
    print(f"  Length: {generated_ids.size(1)}")
    
    print("\n✓ Generazione autoregressiva completata!")


# =============================================================================
# EXAMPLE 4: Integration con Dual-Stream
# =============================================================================

def example_4_dual_stream_integration():
    """
    Esempio di come integrare il decoder con il dual-stream upstream.
    
    Simula il flusso completo:
    Video + Landmarks → Encoder → Memory → Decoder → Logits
    """
    print("\n" + "="*70)
    print(" EXAMPLE 4: Dual-Stream Integration")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class SimplifiedDualStreamEncoder(nn.Module):
        """
        Versione semplificata del dual-stream da cnn_1d_gru_integration.
        Per demo, usiamo solo layer lineari.
        """
        def __init__(self, hidden_dim=512):
            super().__init__()
            self.landmark_proj = nn.Linear(2108, hidden_dim)
            self.video_proj = nn.Linear(256, hidden_dim)  # Assume video features 256D
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        def forward(self, landmarks, video_features):
            # Projecta inputs
            lm = self.landmark_proj(landmarks)  # (B, T, 512)
            vf = self.video_proj(video_features)  # (B, T, 512)
            
            # Fuse
            combined = torch.cat([lm, vf], dim=-1)  # (B, T, 1024)
            fused = self.fusion(combined)  # (B, T, 512)
            
            return fused
    
    # Setup
    encoder = SimplifiedDualStreamEncoder(hidden_dim=512).to(device)
    decoder = TransformerDecoder(
        hidden_dim=512,
        vocab_size=5000,
        num_decoder_layers=2,
        device=device
    ).to(device)
    
    # Dati demo
    batch_size = 2
    landmarks = torch.randn(batch_size, 100, 2108, device=device)
    video_features = torch.randn(batch_size, 100, 256, device=device)
    target_tokens = torch.randint(0, 5000, (batch_size, 30), device=device)
    target_tokens[:, 0] = 1  # BOS
    
    print("\nPipeline:")
    print(f"  Landmarks:      {landmarks.shape} (raw landmarks)")
    print(f"  Video features: {video_features.shape} (raw video features)")
    
    # ========== Encoder produce Memory ==========
    decoder.eval()
    encoder.eval()
    with torch.no_grad():
        memory = encoder(landmarks, video_features)
    print(f"  Memory (fused): {memory.shape} (per il decoder)")
    
    # ========== Decoder produce Logits ==========
    with torch.no_grad():
        logits = decoder(memory=memory, target_tokens=target_tokens)
    print(f"  Logits:         {logits.shape} (per il testo output)")
    
    # ========== Loss Calculation ==========
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    logits_flat = logits.view(-1, decoder.vocab_size)
    targets_flat = target_tokens.view(-1)
    loss = criterion(logits_flat, targets_flat)
    
    print(f"\nCross-Entropy Loss: {loss.item():.4f}")
    print("✓ Dual-stream integration completato!")


# =============================================================================
# EXAMPLE 5: Full Training Pipeline
# =============================================================================

def example_5_full_training_pipeline():
    """
    Esempio completo di training/validation/checkpointing.
    
    Mostra:
    - Learning rate scheduling
    - Checkpointing (salva best model)
    - Validation loop
    - Loss tracking
    """
    print("\n" + "="*70)
    print(" EXAMPLE 5: Full Training Pipeline")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = TransformerDecoderConfig(
        hidden_dim=256,
        vocab_size=5000,
        num_decoder_layers=2,
        num_heads=4,
    )
    
    print(f"\nModel Configuration:")
    print(config)
    
    # Create model
    decoder = TransformerDecoder(**config.to_dict(), device=device).to(device)
    
    # Setup training components
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.98)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,  # 10 epochs
        eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Create synthetic dataset
    num_train = 50
    num_val = 10
    batch_size = 4
    
    # Training data
    X_train_memory = torch.randn(num_train, 50, 256)
    X_train_target = torch.randint(0, 5000, (num_train, 30))
    X_train_target[:, 0] = 1  # BOS
    
    train_dataset = TensorDataset(X_train_memory, X_train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation data
    X_val_memory = torch.randn(num_val, 50, 256)
    X_val_target = torch.randint(0, 5000, (num_val, 30))
    X_val_target[:, 0] = 1
    
    val_dataset = TensorDataset(X_val_memory, X_val_target)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Batch size: {batch_size}\n")
    
    # Training loop
    num_epochs = 5
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # ========== Training ==========
        decoder.train()
        epoch_loss = 0
        
        for memory, target_tokens in train_loader:
            memory = memory.to(device)
            target_tokens = target_tokens.to(device)
            
            # Forward
            logits = decoder(memory=memory, target_tokens=target_tokens)
            
            # Loss
            logits_flat = logits.view(-1, decoder.vocab_size)
            targets_flat = target_tokens.view(-1)
            loss = criterion(logits_flat, targets_flat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # ========== Validation ==========
        decoder.eval()
        val_loss = 0
        
        with torch.no_grad():
            for memory, target_tokens in val_loader:
                memory = memory.to(device)
                target_tokens = target_tokens.to(device)
                
                logits = decoder(memory=memory, target_tokens=target_tokens)
                logits_flat = logits.view(-1, decoder.vocab_size)
                targets_flat = target_tokens.view(-1)
                loss = criterion(logits_flat, targets_flat)
                
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # ========== Scheduler ==========
        scheduler.step()
        
        # ========== Logging ==========
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {lr:.2e}")
        
        # ========== Checkpointing ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # In real scenario: torch.save(decoder.state_dict(), 'best_model.pt')
            print(f"  ✓ Best model (val_loss={val_loss:.4f})")
    
    print(f"\n✓ Training pipeline completato!")
    print(f"Final metrics:")
    print(f"  Best val loss: {min(val_losses):.4f}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")


# =============================================================================
# Run All Examples
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" TRANSFORMER DECODER - PRACTICAL EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_basic_forward()
    example_2_training_loop()
    example_3_autoregressive_generation()
    example_4_dual_stream_integration()
    example_5_full_training_pipeline()
    
    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nProssimi step:")
    print("  1. Adatta gli esempi ai tuoi dati reali")
    print("  2. Togli il use_classification_head dal dual-stream encoder")
    print("  3. Allenati su How2Sign dataset")
    print("  4. Valuta con metriche (BLEU, METEOR, CER)")
    print("="*70 + "\n")
