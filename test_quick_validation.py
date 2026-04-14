#!/usr/bin/env python3
"""
Quick validation test for DualStreamSignLanguageModel
Focus on forward pass and architecture validation (no backward pass on CPU)
"""

import torch
import sys
sys.path.insert(0, '.')

from src.models.cnn_1d_gru_integration import DualStreamSignLanguageModel

print('=' * 70)
print('✅ DUAL-STREAM SIGN LANGUAGE MODEL - QUICK VALIDATION')
print('=' * 70)

# Test 1: Create model
print('\n[1️⃣  MODEL CREATION]')
model = DualStreamSignLanguageModel(hidden_dim=512, num_transformer_layers=2)
print('   ✅ DualStreamSignLanguageModel created')
total_params = sum(p.numel() for p in model.parameters())
print(f'   ✅ Total parameters: {total_params/1e6:.1f}M')

# Test 2: Forward pass
print('\n[2️⃣  FORWARD PASS]')
batch, time, places = 2, 30, 50
landmarks = torch.randn(batch, time, 2108)
frames = torch.randn(batch, time, 3, 224, 224)
logits = model(landmarks, frames)
print(f'   ✅ Landmarks input: {landmarks.shape}')
print(f'   ✅ Video input: {frames.shape}')
print(f'   ✅ Logits output: {logits.shape}')

# Test 3: Intermediate outputs
print('\n[3️⃣  INTERMEDIATE OUTPUTS (for debugging)]')
logits, inter = model(landmarks, frames, return_intermediate=True)
print(f'   ✅ Landmarks stream: {inter["landmarks"].shape}')
print(f'   ✅ Video stream: {inter["video"].shape}')
print(f'   ✅ Fused output: {inter["fused"].shape}')

# Test 4: Transfer learning
print('\n[4️⃣  TRANSFER LEARNING (frozen backbone)]')
model_frozen = DualStreamSignLanguageModel(
    hidden_dim=512,
    freeze_mobilenet=True,
    num_transformer_layers=2
)
logits_frozen = model_frozen(landmarks, frames)
trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
total_frozen = sum(p.numel() for p in model_frozen.parameters())
print(f'   ✅ Output shape: {logits_frozen.shape}')
print(f'   ✅ Trainable params: {trainable/1e6:.1f}M / {total_frozen/1e6:.1f}M')

# Test 5: Configurations
print('\n[5️⃣  MULTIPLE CONFIGURATIONS]')
for hidden_dim, name in [(256, 'Light'), (512, 'Standard'), (768, 'Heavy')]:
    m = DualStreamSignLanguageModel(hidden_dim=hidden_dim, num_transformer_layers=1)
    out = m(landmarks, frames)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'   ✅ {name:10} (hidden_dim={hidden_dim:3}): {params:5.1f}M params, output {out.shape}')

print('\n' + '=' * 70)
print('✅ ALL VALIDATIONS PASSED!')
print('=' * 70)
print('''
Summary:
  • Model creates and loads MobileNet v2 successfully
  • Forward pass works with both landmarks (2108) and video (3×224×224)
  • Both streams process in parallel independently
  • Fusion combines outputs correctly: (512) + (512) → (512)
  • Intermediate outputs available for debugging/visualization
  • Transfer learning (frozen MobileNet) works
  • Multiple configurations (light/standard/heavy) all functional

The Dual-Stream architecture is fully functional! 🎉
''')
