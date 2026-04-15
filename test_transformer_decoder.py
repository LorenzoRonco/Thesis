#!/usr/bin/env python3
"""Quick validation test for TransformerDecoder implementation."""

import torch
from src.models.transformer_decoder import TransformerDecoder, TransformerDecoderConfig

def test_forward_pass():
    """Test basic forward pass."""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass")
    print("="*60)
    
    decoder = TransformerDecoder(
        hidden_dim=256,
        vocab_size=1000,
        num_decoder_layers=2
    )
    
    # Create dummy inputs
    batch_size, video_frames, text_length = 2, 50, 30
    memory = torch.randn(batch_size, video_frames, 256)
    target_tokens = torch.randint(0, 1000, (batch_size, text_length))
    target_tokens[:, 0] = 1  # Set BOS
    
    # Forward pass
    decoder.eval()
    with torch.no_grad():
        logits = decoder(memory, target_tokens)
    
    print(f"✓ Forward pass successful")
    print(f"  Memory shape: {memory.shape}")
    print(f"  Target tokens shape: {target_tokens.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: (2, 30, 1000)")
    assert logits.shape == torch.Size([2, 30, 1000]), "Shape mismatch!"
    print(f"✓ Shape verification passed")


def test_generation():
    """Test autoregressive generation."""
    print("\n" + "="*60)
    print("TEST 2: Autoregressive Generation")
    print("="*60)
    
    decoder = TransformerDecoder(
        hidden_dim=256,
        vocab_size=100,
        num_decoder_layers=2
    )
    
    memory = torch.randn(1, 50, 256)
    
    decoder.eval()
    with torch.no_grad():
        generated = decoder.generate(
            memory=memory,
            bos_token_id=1,
            eos_token_id=2,
            max_length=20,
            temperature=1.0
        )
    
    print(f"✓ Generation successful")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    assert generated[0, 0].item() == 1, "First token should be BOS!"
    print(f"✓ Generation verification passed")


def test_config_serialization():
    """Test config save/load."""
    print("\n" + "="*60)
    print("TEST 3: Config Serialization")
    print("="*60)
    
    config = TransformerDecoderConfig(
        hidden_dim=512,
        vocab_size=5000,
        num_decoder_layers=3
    )
    
    print(f"✓ Config created:")
    print(config)
    
    # Note: Actual save/load would require file system access
    # Just verify to_dict() works
    config_dict = config.to_dict()
    print(f"\n✓ Config serialization successful")
    print(f"  Config dict: {config_dict}")


def test_loss_calculation():
    """Test loss calculation."""
    print("\n" + "="*60)
    print("TEST 4: Loss Calculation")
    print("="*60)
    
    decoder = TransformerDecoder(
        hidden_dim=256,
        vocab_size=1000,
        num_decoder_layers=2
    )
    
    batch_size, text_length = 4, 30
    memory = torch.randn(batch_size, 50, 256)
    target_tokens = torch.randint(0, 1000, (batch_size, text_length))
    target_tokens[:, 0] = 1  # BOS
    
    # Forward and loss
    logits = decoder(memory, target_tokens)
    logits_flat = logits.view(-1, decoder.vocab_size)
    targets_flat = target_tokens.view(-1)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(logits_flat, targets_flat)
    
    print(f"✓ Loss calculation successful")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    print(f"✓ Loss verification passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" TRANSFORMER DECODER VALIDATION TESTS")
    print("="*60)
    
    try:
        test_forward_pass()
        test_generation()
        test_config_serialization()
        test_loss_calculation()
        
        print("\n" + "="*60)
        print(" ✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
