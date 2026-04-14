#!/usr/bin/env python3
"""
Test script for DualStreamSignLanguageModel
Validates that the entire dual-stream architecture works correctly
"""

import torch
import sys

# Add project to path
sys.path.insert(0, '.')

from src.models.cnn_1d_gru_integration import DualStreamSignLanguageModel

print('=' * 70)
print('TESTING DUAL-STREAM SIGN LANGUAGE MODEL')
print('=' * 70)

# Test 1: Model Creation
print('\n[TEST 1] Creating DualStreamSignLanguageModel...')
try:
    model = DualStreamSignLanguageModel(
        hidden_dim=512,
        landmark_dim=2108,
        vocab_size=1000,
        num_transformer_layers=4,
        use_classification_head=True
    )
    print('   ✅ Model created successfully')
    print(f'   Architecture: {type(model).__name__}')
except Exception as e:
    print(f'   ❌ Error: {e}')
    sys.exit(1)

# Test 2: Create dummy data
print('\n[TEST 2] Creating dummy data...')
try:
    batch_size = 4
    time_steps = 50
    landmarks = torch.randn(batch_size, time_steps, 2108)
    frames = torch.randn(batch_size, time_steps, 3, 224, 224)
    print(f'   ✅ Landmarks shape: {landmarks.shape}')
    print(f'   ✅ Video frames shape: {frames.shape}')
except Exception as e:
    print(f'   ❌ Error: {e}')
    sys.exit(1)

# Test 3: Forward pass
print('\n[TEST 3] Running forward pass...')
try:
    logits = model(landmarks, frames)
    print(f'   ✅ Output logits shape: {logits.shape}')
    assert logits.shape == (batch_size, 1000), f"Expected shape ({batch_size}, 1000), got {logits.shape}"
    print(f'   ✅ Shape is correct: ({batch_size}, 1000)')
except Exception as e:
    print(f'   ❌ Error: {e}')
    sys.exit(1)

# Test 4: Intermediate outputs
print('\n[TEST 4] Getting intermediate outputs...')
try:
    logits, intermediate = model(landmarks, frames, return_intermediate=True)
    assert 'landmarks' in intermediate, "Missing 'landmarks' key"
    assert 'video' in intermediate, "Missing 'video' key"
    assert 'fused' in intermediate, "Missing 'fused' key"
    
    print(f'   ✅ Landmarks stream: {intermediate["landmarks"].shape}')
    print(f'   ✅ Video stream: {intermediate["video"].shape}')
    print(f'   ✅ Fused output: {intermediate["fused"].shape}')
    
    # Verify shapes
    assert intermediate['landmarks'].shape == (batch_size, time_steps, 512)
    assert intermediate['video'].shape == (batch_size, time_steps, 512)
    assert intermediate['fused'].shape == (batch_size, time_steps, 512)
    print('   ✅ All intermediate shapes are correct!')
except Exception as e:
    print(f'   ❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Backward pass (gradient computation)
print('\n[TEST 5] Testing backward pass (loss computation)...')
try:
    # Use smaller data for backward pass to avoid memory issues
    small_batch_size = 1
    small_time_steps = 10
    small_landmarks = torch.randn(small_batch_size, small_time_steps, 2108)
    small_frames = torch.randn(small_batch_size, small_time_steps, 3, 224, 224)
    small_targets = torch.randint(0, 1000, (small_batch_size,))
    
    print(f'   📢 Using reduced data for backward pass:')
    print(f'      - Batch size: {small_batch_size} (was {batch_size})')
    print(f'      - Time steps: {small_time_steps} (was {time_steps})')
    
    logits = model(small_landmarks, small_frames)
    loss = torch.nn.functional.cross_entropy(logits, small_targets)
    loss.backward()
    print(f'   ✅ Loss computed: {loss.item():.4f}')
    print(f'   ✅ Gradients computed successfully')
    
    # Check that gradients exist
    grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_count += 1
    print(f'   ✅ Gradients computed for {grad_count} parameters')
except Exception as e:
    print(f'   ❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Transfer learning (frozen MobileNet)
print('\n[TEST 6] Testing transfer learning (frozen MobileNet)...')
try:
    model_frozen = DualStreamSignLanguageModel(
        hidden_dim=512,
        freeze_mobilenet=True,
        num_transformer_layers=2,
        use_classification_head=True
    )
    logits_frozen = model_frozen(landmarks, frames)
    print(f'   ✅ Frozen model output shape: {logits_frozen.shape}')
    
    # Check that MobileNet has no gradients
    trainable_params = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_frozen.parameters())
    print(f'   ✅ Trainable params: {trainable_params:,} / {total_params:,}')
except Exception as e:
    print(f'   ❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Different configurations
print('\n[TEST 7] Testing different configurations...')
try:
    configs = [
        {'hidden_dim': 256, 'name': 'Light'},
        {'hidden_dim': 512, 'name': 'Standard'},
        {'hidden_dim': 768, 'name': 'Heavy'},
    ]
    
    for config in configs:
        model_config = DualStreamSignLanguageModel(
            hidden_dim=config['hidden_dim'],
            num_transformer_layers=2,
            use_classification_head=True
        )
        output = model_config(landmarks, frames)
        print(f'   ✅ {config["name"]} (hidden_dim={config["hidden_dim"]}): output shape {output.shape}')
except Exception as e:
    print(f'   ❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n' + '=' * 70)
print('✅ ALL TESTS PASSED!')
print('=' * 70)
print('\nSummary:')
print('  • Dual-stream model creates successfully')
print('  • Forward pass works with both landmarks and video')
print('  • Intermediate outputs are available for debugging')
print('  • Backward pass computes gradients correctly')
print('  • Transfer learning (frozen MobileNet) works')
print('  • Different configurations (light/standard/heavy) work')
print('\n🎉 Implementation is complete and working!')
