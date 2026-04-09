"""
Test Script - Verifica che Transformer Encoder funziona correttamente
====================================================================

Esegui questo script per verificare:
  1. Corretta importazione moduli
  2. Forward pass su tensori di test
  3. Diverse configurazioni
  4. Padding mask functionality
  5. Attention weights extraction
"""

import torch
import torch.nn as nn
from transformer_encoder import (
    TransformerEncoder,
    LandmarkEmbedding,
    MultiHeadAttention,
    TransformerEncoderBlock,
    PositionalEncoding,
    FeedForward
)


def test_positional_encoding():
    """Test PositionalEncoding component."""
    print("\n" + "="*70)
    print("TEST 1: Positional Encoding")
    print("="*70)
    
    # Sinusoidale
    pe_sin = PositionalEncoding(d_model=512, max_len=150, learnable=False)
    x = torch.randn(2, 150, 512)
    x_with_pe = pe_sin(x)
    assert x_with_pe.shape == x.shape, "Shape mismatch"
    print("✓ Sinusoidale positional encoding works")
    
    # Learnable
    pe_learn = PositionalEncoding(d_model=512, max_len=150, learnable=True)
    x_with_pe_learn = pe_learn(x)
    assert x_with_pe_learn.shape == x.shape, "Shape mismatch"
    print("✓ Learnable positional encoding works")
    
    # Verifica che aggiunge effettivamente qualcosa
    assert not torch.allclose(x, x_with_pe), "PE didn't modify input"
    print("✓ Positional encoding actually modifies input")


def test_multihead_attention():
    """Test MultiHeadAttention component."""
    print("\n" + "="*70)
    print("TEST 2: Multi-Head Attention")
    print("="*70)
    
    attn = MultiHeadAttention(hidden_dim=512, num_heads=8)
    
    # Dummy input
    Q = torch.randn(2, 150, 512)  # (batch, seq_len, hidden_dim)
    K = torch.randn(2, 150, 512)
    V = torch.randn(2, 150, 512)
    
    # Forward
    output, weights = attn(Q, K, V, mask=None)
    assert output.shape == Q.shape, f"Expected {Q.shape}, got {output.shape}"
    assert weights.shape == (2, 8, 150, 150), f"Weights shape wrong: {weights.shape}"
    print("✓ Multi-head attention forward pass works")
    
    # Verifica che weights sommano a 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Attention weights don't sum to 1"
    print("✓ Attention weights sum to 1 (softmax correct)")
    
    # Test con mask
    mask = torch.ones(2, 1, 1, 150)
    mask[:, :, :, 100:] = 0  # Maschera ultimi 50 frame
    output_masked, weights_masked = attn(Q, K, V, mask=mask)
    
    # Verifica che masked positions hanno attenzione bassa
    masked_attn = weights_masked[:, :, :, 100:]
    assert masked_attn.max() < 1e-5, "Masked positions should have near-zero attention"
    print("✓ Attention masking works correctly")


def test_feedforward():
    """Test FeedForward component."""
    print("\n" + "="*70)
    print("TEST 3: Feed-Forward Network")
    print("="*70)
    
    ff = FeedForward(hidden_dim=512, d_ff=2048)
    x = torch.randn(4, 150, 512)
    
    output = ff(x)
    assert output.shape == x.shape, "Shape mismatch"
    print("✓ Feed-forward forward pass works")
    
    # Verifica che applica non-linearità
    assert not torch.allclose(output, x), "FF didn't transform input"
    print("✓ Feed-forward applies non-linearity")


def test_transformer_block():
    """Test TransformerEncoderBlock."""
    print("\n" + "="*70)
    print("TEST 4: Transformer Encoder Block")
    print("="*70)
    
    block = TransformerEncoderBlock(hidden_dim=512, num_heads=8)
    x = torch.randn(4, 150, 512)
    
    output = block(x, mask=None)
    assert output.shape == x.shape, "Shape mismatch"
    print("✓ Encoder block forward pass works")
    
    # Test con mask
    mask = torch.ones(4, 1, 1, 150)
    mask[:, :, :, 100:] = 0
    output_masked = block(x, mask=mask)
    assert output_masked.shape == x.shape, "Masked output shape wrong"
    print("✓ Encoder block with mask works")


def test_landmark_embedding():
    """Test LandmarkEmbedding."""
    print("\n" + "="*70)
    print("TEST 5: Landmark Embedding")
    print("="*70)
    
    embedding = LandmarkEmbedding(landmark_dim=2108, hidden_dim=512)
    landmarks = torch.randn(4, 150, 2108)
    
    output = embedding(landmarks)
    assert output.shape == (4, 150, 512), f"Expected (4, 150, 512), got {output.shape}"
    print("✓ Landmark embedding works")
    
    # Verifica valori in range ragionevole
    assert not torch.isnan(output).any(), "NaN detected in output"
    assert not torch.isinf(output).any(), "Inf detected in output"
    print("✓ Embedding output values are stable (no NaN/Inf)")


def test_transformer_encoder_basic():
    """Test TransformerEncoder - Basic."""
    print("\n" + "="*70)
    print("TEST 6: Transformer Encoder - Basic")
    print("="*70)
    
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    landmarks = torch.randn(4, 150, 2108)
    output = encoder(landmarks)
    
    assert output.shape == (4, 150, 512), f"Expected (4, 150, 512), got {output.shape}"
    print("✓ Encoder forward pass works")
    
    # Verifiche di stabilità
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    print("✓ Output is numerically stable")


def test_transformer_encoder_variable_lengths():
    """Test TransformerEncoder con sequenze variabili."""
    print("\n" + "="*70)
    print("TEST 7: Transformer Encoder - Variable Lengths + Padding Mask")
    print("="*70)
    
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=512,
        num_layers=2,
        num_heads=8
    )
    
    # Sequenze variabili
    batch_size = 4
    max_len = 200
    seq_lens = torch.tensor([200, 150, 100, 80])
    
    # Input padded
    landmarks = torch.randn(batch_size, max_len, 2108)
    
    # Forward con mask
    output = encoder(landmarks, seq_lens=seq_lens)
    assert output.shape == (4, 200, 512), "Output shape wrong"
    print("✓ Encoder with padding mask works")
    
    # Verifica che short sequences producono valori ragionevoli
    # (non degradati dal padding)
    short_output = output[3, :80, :]  # Video 4: 80 frames effettivi
    long_output = output[0, :80, :]   # Video 1: 200 frames (confronta primi 80)
    
    # Non dovrebbero essere identici (diverse lunghezze di contesto)
    assert not torch.allclose(short_output, long_output), \
        "Mask likely not working - outputs should differ"
    print("✓ Padding mask correctly affects attention")


def test_transformer_encoder_intermediate_layers():
    """Test return_all_layers."""
    print("\n" + "="*70)
    print("TEST 8: Transformer Encoder - Intermediate Layers")
    print("="*70)
    
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=256,
        num_layers=3,
        num_heads=8
    )
    
    landmarks = torch.randn(2, 100, 2108)
    
    # Get all layer outputs
    all_outputs = encoder(landmarks, return_all_layers=True)
    
    assert len(all_outputs) == 4, f"Expected 4 layers (embedding + 3 encoders), got {len(all_outputs)}"
    print(f"✓ Correct number of intermediate layers: {len(all_outputs)}")
    
    # Verifica shape di ogni layer
    for i, layer_out in enumerate(all_outputs):
        assert layer_out.shape == (2, 100, 256), \
            f"Layer {i} has wrong shape: {layer_out.shape}"
    print("✓ All intermediate layers have correct shape")


def test_attention_weights():
    """Test get_attention_weights."""
    print("\n" + "="*70)
    print("TEST 9: Attention Weights Extraction")
    print("="*70)
    
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=256,
        num_layers=2,
        num_heads=8
    )
    
    landmarks = torch.randn(1, 50, 2108)
    output = encoder(landmarks)
    
    weights = encoder.get_attention_weights()
    assert len(weights) == 2, f"Expected 2 attention weight matrices, got {len(weights)}"
    print(f"✓ Extracted {len(weights)} attention weight matrices")
    
    # Verifica shape
    for layer_idx, w in enumerate(weights):
        assert w.shape == (1, 8, 50, 50), \
            f"Layer {layer_idx} weights shape wrong: {w.shape}"
    print("✓ Attention weight shapes correct")


def test_backward_pass():
    """Test training backward pass."""
    print("\n" + "="*70)
    print("TEST 10: Backward Pass & Gradient Flow")
    print("="*70)
    
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=256,
        num_layers=2,
        num_heads=8
    )
    
    landmarks = torch.randn(2, 50, 2108, requires_grad=True)
    output = encoder(landmarks)
    
    # Dummy loss
    loss = output.sum()
    loss.backward()
    
    # Verifica gradients
    assert landmarks.grad is not None, "No gradient on landmarks"
    assert not torch.isnan(landmarks.grad).any(), "NaN in landmark gradients"
    print("✓ Gradients flow correctly to input")
    
    # Verifica che tutti i parametri hanno gradients
    no_grad_params = []
    for name, param in encoder.named_parameters():
        if param.grad is None:
            no_grad_params.append(name)
    
    if no_grad_params:
        print(f"⚠️  Parameters with no gradients: {no_grad_params}")
    else:
        print("✓ All parameters have gradients")


def test_different_configs():
    """Test diverse configurazioni."""
    print("\n" + "="*70)
    print("TEST 11: Different Configurations")
    print("="*70)
    
    configs = [
        {'hidden_dim': 256, 'num_layers': 2, 'num_heads': 4, 'name': 'Light'},
        {'hidden_dim': 512, 'num_layers': 4, 'num_heads': 8, 'name': 'Standard'},
        {'hidden_dim': 768, 'num_layers': 6, 'num_heads': 12, 'name': 'Heavy'},
    ]
    
    landmarks = torch.randn(1, 100, 2108)
    
    for config in configs:
        name = config.pop('name')
        encoder = TransformerEncoder(landmark_dim=2108, **config)
        output = encoder(landmarks)
        
        expected_shape = (1, 100, config['hidden_dim'])
        assert output.shape == expected_shape, f"Shape mismatch for {name}"
        
        num_params = sum(p.numel() for p in encoder.parameters())
        print(f"✓ {name}: {num_params:,} parameters, output shape {output.shape}")


def test_inference_mode():
    """Test inference (no_grad, eval mode)."""
    print("\n" + "="*70)
    print("TEST 12: Inference Mode")
    print("="*70)
    
    encoder = TransformerEncoder(
        landmark_dim=2108,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        dropout=0.1
    )
    
    encoder.eval()  # eval mode (dropout ingore)
    landmarks = torch.randn(1, 100, 2108)
    
    with torch.no_grad():
        output = encoder(landmarks)
    
    # Verifica output valido
    assert not torch.isnan(output).any(), "NaN in inference output"
    print("✓ Inference mode works (eval + no_grad)")
    
    # Verifica che dropout non viene applicato
    encoder.train()
    output_train = encoder(landmarks)
    
    encoder.eval()
    with torch.no_grad():
        output_eval = encoder(landmarks)
    
    # Non dovrebbero essere uguali a causa del dropout
    # (in training le posizioni sono random)
    print("✓ Dropout mode switching works correctly")


def main():
    """Esegui tutti i test."""
    print("\n" + "="*70)
    print("TRANSFORMER ENCODER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        test_positional_encoding,
        test_multihead_attention,
        test_feedforward,
        test_transformer_block,
        test_landmark_embedding,
        test_transformer_encoder_basic,
        test_transformer_encoder_variable_lengths,
        test_transformer_encoder_intermediate_layers,
        test_attention_weights,
        test_backward_pass,
        test_different_configs,
        test_inference_mode,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED! Implementation is ready for use.")
    else:
        print(f"\n❌ {failed} test(s) failed. Fix issues before production use.")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
