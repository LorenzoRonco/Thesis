"""
CNN-1D-GRU Module: Progressive Examples
========================================

6 Examples progressivi che mostrano come usare il modulo CNN-1D-GRU
da casi basilari a scenari di training completi.

Esegui con: python src/models/cnn_1d_gru_examples.py

Author: Thesis Project
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.cnn_1d_gru_module import CNN1DGRUModule, CNN1DGRUConfig


def print_header(title):
    """Stampa un header formattato."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_tensor_info(name, tensor):
    """Stampa info dettagliate su un tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Value range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"  Memory: {tensor.numel() * tensor.element_size() / 1024**2:.2f} MB")


# =============================================================================
# EXAMPLE 1: Basic Forward Pass
# =============================================================================

def example_1_basic_forward():
    """
    Esempio 1: Forward pass basilare.
    
    Mostra come:
    - Creare il modulo
    - Passare input attraverso il modello
    - Interpretare l'output
    """
    print_header("Example 1: Basic Forward Pass")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 1. Crea il modulo
    model = CNN1DGRUModule(hidden_dim=512)
    model = model.to(device)
    print("Model created:")
    print(f"  - Architecture: 2D-CNN (MobileNet) → 1D-CNN → GRU")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Crea input di test
    batch_size = 8
    time_steps = 150
    height, width = 224, 224
    frames = torch.randn(batch_size, time_steps, 3, height, width).to(device)
    print_tensor_info("Input frames", frames)
    
    # 3. Forward pass
    print("\nForward pass...")
    with torch.no_grad():
        output, h_n = model(frames)
    
    # 4. Mostra output
    print_tensor_info("Output features", output)
    print_tensor_info("Hidden state", h_n)
    
    # 5. Verifica dimensioni
    expected_shape = (batch_size, time_steps, 512)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"\n✓ Output shape correct: {output.shape}")
    
    # 6. Costo computazionale
    print("\nComputational cost:")
    params = sum(p.numel() for p in model.parameters())
    flops = batch_size * time_steps * 3.5e9  # Approximate
    print(f"  - Total parameters: {params:,}")
    print(f"  - FLOPs per batch: {flops/1e9:.2f}B")
    print(f"  - Memory per batch: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB")


# =============================================================================
# EXAMPLE 2: Configuration Presets
# =============================================================================

def example_2_configurations():
    """
    Esempio 2: Diverse configurazioni.
    
    Mostra come:
    - Usare i preset di configurazione
    - Confrontare dimensioni e parametri
    - Scegliere configurazione appropriata
    """
    print_header("Example 2: Configuration Presets")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(4, 75, 3, 224, 224).to(device)
    
    # Lista di configurazioni
    configs = {
        'Light': CNN1DGRUConfig.light(),
        'Standard': CNN1DGRUConfig.standard(),
        'Heavy': CNN1DGRUConfig.heavy(),
        'Frozen Backbone': CNN1DGRUConfig.frozen_backbone(),
    }
    
    results = []
    
    for config_name, config in configs.items():
        print(f"\n{config_name}:")
        print(f"  Config: {config}")
        
        # Create model
        model = CNN1DGRUModule(**config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Forward pass (timing)
        with torch.no_grad():
            output, _ = model(dummy_input)
        
        results.append({
            'config': config_name,
            'output_dim': output.shape[-1],
            'total_params': total_params,
            'trainable_params': trainable_params,
            'output_shape': output.shape
        })
        
        print(f"  Output: {output.shape}")
        print(f"  Total params: {total_params/1e6:.1f}M")
        print(f"  Trainable params: {trainable_params/1e6:.1f}M")
    
    # Summary table
    print("\n" + "=" * 70)
    print("Configuration Summary:")
    print("=" * 70)
    print(f"{'Config':<20} {'Params':<15} {'Trainable':<15} {'Output Size':<20}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<20} {r['total_params']/1e6:<14.1f}M {r['trainable_params']/1e6:<14.1f}M {str(r['output_shape']):<20}")


# =============================================================================
# EXAMPLE 3: Integration with Transformer Encoder
# =============================================================================

def example_3_transformer_integration():
    """
    Esempio 3: Integrazione con Transformer Encoder.
    
    Mostra come:
    - Usare CNN-1D-GRU come backbone
    - Passare output a Transformer
    - Full end-to-end processing
    """
    print_header("Example 3: Integration with Transformer Encoder")
    
    try:
        from src.models.transformer_encoder import TransformerEncoder
    except ImportError:
        print("⚠ TransformerEncoder not found. Install it first.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create CNN-1D-GRU
    print("\n1. CNN-1D-GRU Module:")
    cnn_gru = CNN1DGRUModule(hidden_dim=512).to(device)
    cnn_gru_params = sum(p.numel() for p in cnn_gru.parameters())
    print(f"   Parameters: {cnn_gru_params:,}")
    
    # 2. Create Transformer Encoder
    print("\n2. Transformer Encoder:")
    transformer = TransformerEncoder(
        landmark_dim=512,  # Match CNN-1D-GRU output!
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    ).to(device)
    trans_params = sum(p.numel() for p in transformer.parameters())
    print(f"   Parameters: {trans_params:,}")
    
    # 3. Full pipeline
    print("\n3. Full Pipeline Forward Pass:")
    batch_size = 4
    time_steps = 100
    
    frames = torch.randn(batch_size, time_steps, 3, 224, 224).to(device)
    print(f"   Input frames: {frames.shape}")
    
    with torch.no_grad():
        # CNN-1D-GRU extraction
        features, _ = cnn_gru(frames)
        print(f"   After CNN-1D-GRU: {features.shape}")
        
        # Transformer encoding
        encoded = transformer(features)
        print(f"   After Transformer: {encoded.shape}")
        
        # Optional: temporal pooling
        pooled = encoded.mean(dim=1)
        print(f"   After temporal pooling: {pooled.shape}")
    
    print("\n   ✓ Full pipeline works correctly!")
    print(f"\n   Total parameters: {(cnn_gru_params + trans_params)/1e6:.1f}M")


# =============================================================================
# EXAMPLE 4: Gradient Flow and Backpropagation
# =============================================================================

def example_4_gradient_flow():
    """
    Esempio 4: Flusso dei gradienti durante backprop.
    
    Mostra come:
    - Verificare che i gradienti fluiscono correttamente
    - Monitorare dimensioni dei gradienti
    - Applicare gradient clipping
    """
    print_header("Example 4: Gradient Flow and Backpropagation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create model
    model = CNN1DGRUModule(hidden_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 2. Dummy data
    batch_size = 4
    time_steps = 50
    frames = torch.randn(batch_size, time_steps, 3, 224, 224).to(device)
    targets = torch.randn(batch_size, time_steps, 512).to(device)
    
    # 3. Forward + Loss
    print("\n1. Forward pass...")
    output, _ = model(frames)
    loss = criterion(output, targets)
    print(f"   Output shape: {output.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # 4. Backward
    print("\n2. Backward pass...")
    optimizer.zero_grad()
    loss.backward()
    
    # 5. Check gradients
    print("\n3. Gradient statistics:")
    print(f"{'Component':<30} {'Grad norm':<15} {'Requires grad':<15}")
    print("-" * 60)
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm().item()
            grad_norms.append(grad_norm)
            print(f"{name:<30} {grad_norm:<15.6f} {param.requires_grad:<15}")
    
    # 6. Gradient clipping
    print("\n4. Gradient clipping:")
    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    print(f"   Total norm before: {total_norm:.4f}")
    print(f"   Clipping threshold: {max_norm}")
    print(f"   ✓ Gradients clipped successfully")
    
    # 7. Optimizer step
    print("\n5. Optimizer step...")
    optimizer.step()
    print(f"   ✓ Parameters updated")
    
    # 8. Verify learning
    print("\n6. Loss change verification:")
    output_after, _ = model(frames)
    loss_after = criterion(output_after, targets)
    loss_change = loss.item() - loss_after.item()
    print(f"   Loss before: {loss.item():.4f}")
    print(f"   Loss after:  {loss_after.item():.4f}")
    print(f"   Change:      {loss_change:.4f} {'↓ (good)' if loss_change > 0 else '↑ (bad)'}")


# =============================================================================
# EXAMPLE 5: Training Loop with Validation
# =============================================================================

def example_5_training_loop():
    """
    Esempio 5: Loop di training completo con validazione.
    
    Mostra come:
    - Creare dataloader
    - Training loop con loss tracking
    - Validazione
    - Checkpointing
    """
    print_header("Example 5: Complete Training Loop")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    n_samples = 320
    X = torch.randn(n_samples, 100, 3, 224, 224)
    y = torch.randint(0, 10, (n_samples,))  # 10 classes
    
    train_dataset = TensorDataset(X[:250], y[:250])
    val_dataset = TensorDataset(X[250:], y[250:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # 2. Setup model, optimizer, etc
    print("\n2. Setup training...")
    model = CNN1DGRUModule(hidden_dim=512).to(device)
    classifier = nn.Linear(512, 10).to(device)  # Classification head
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # 3. Training loop
    print("\n3. Training for 3 epochs...")
    num_epochs = 3
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        classifier.train()
        train_loss = 0
        
        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward
            features, _ = model(frames)  # (B, T, 512)
            pooled = features.mean(dim=1)  # (B, 512)
            logits = classifier(pooled)  # (B, 10)
            
            loss = criterion(logits, labels)
            train_loss += loss.item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                
                features, _ = model(frames)
                pooled = features.mean(dim=1)
                logits = classifier(pooled)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"   Epoch {epoch+1}: Train loss={avg_train_loss:.4f}, "
              f"Val loss={avg_val_loss:.4f}, Val acc={val_accuracy:.1f}%")
        
        scheduler.step()
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'classifier_state': classifier.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }
            print(f"      → Best checkpoint saved (loss={best_val_loss:.4f})")
    
    print("\n   ✓ Training complete!")


# =============================================================================
# EXAMPLE 6: Transfer Learning (Frozen Backbone)
# =============================================================================

def example_6_transfer_learning():
    """
    Esempio 6: Transfer learning con MobileNet congelato.
    
    Mostra come:
    - Congelare backbone MobileNet
    - Training solo della parte adattatrice (Conv1D + GRU)
    - Scongelamento progressivo
    - Monitoring di quali parametri vengono aggiornati
    """
    print_header("Example 6: Transfer Learning (Frozen Backbone)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create model with frozen backbone
    print("\n1. Create model with frozen MobileNet...")
    model = CNN1DGRUModule(
        hidden_dim=512,
        freeze_mobilenet=True,
        pretrained_mobilenet=True
    ).to(device)
    
    # Count trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params/1e6:.1f}M")
    print(f"   Trainable parameters: {trainable_params/1e6:.1f}M")
    print(f"   Frozen parameters: {(total_params - trainable_params)/1e6:.1f}M")
    
    # 2. Training on limited data
    print("\n2. Phase 1: Training with frozen backbone...")
    
    # Small dataset (limited data scenario)
    X_small = torch.randn(64, 75, 3, 224, 224)
    y_small = torch.randint(0, 10, (64,))
    small_loader = DataLoader(TensorDataset(X_small, y_small), batch_size=16)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # higher LR for smaller param set
    criterion = nn.CrossEntropyLoss()
    classifier = nn.Linear(512, 10).to(device)
    
    for epoch in range(2):
        model.train()
        epoch_loss = 0
        
        for frames, labels in small_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            features, _ = model(frames)
            pooled = features.mean(dim=1)
            logits = classifier(pooled)
            
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"   Epoch {epoch+1}: Loss = {epoch_loss/len(small_loader):.4f}")
    
    # 3. Unfreeze backbone for fine-tuning
    print("\n3. Phase 2: Unfreezing MobileNet for fine-tuning...")
    model.unfreeze_mobilenet()
    
    # Count params again
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters after unfreeze: {trainable_params_after/1e6:.1f}M")
    print(f"   Increase: {(trainable_params_after - trainable_params)/1e6:.1f}M")
    
    # Fine-tuning with lower learning rate
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-5)  # much lower LR
    
    for epoch in range(2):
        model.train()
        epoch_loss = 0
        
        for frames, labels in small_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            features, _ = model(frames)
            pooled = features.mean(dim=1)
            logits = classifier(pooled)
            
            loss = criterion(logits, labels)
            
            optimizer_ft.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_ft.step()
            
            epoch_loss += loss.item()
        
        print(f"   Epoch {epoch+1}: Loss = {epoch_loss/len(small_loader):.4f}")
    
    print("\n   ✓ Transfer learning complete!")


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Esegui tutti gli esempi."""
    print("\n" + "=" * 70)
    print(" CNN-1D-GRU Module: 6 Progressive Examples")
    print("=" * 70)
    
    examples = [
        ("Basic Forward Pass", example_1_basic_forward),
        ("Configuration Presets", example_2_configurations),
        ("Transformer Integration", example_3_transformer_integration),
        ("Gradient Flow & Backprop", example_4_gradient_flow),
        ("Training Loop", example_5_training_loop),
        ("Transfer Learning", example_6_transfer_learning),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
            print(f"\n✓ Example {i} completed successfully")
        except Exception as e:
            print(f"\n✗ Example {i} failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
    
    print("\n" + "=" * 70)
    print(" All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
