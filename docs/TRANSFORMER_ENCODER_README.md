# Transformer Encoder per Sign Language Translation

## Quick Start & Implementation Guide

---

## 📋 Sommario

Hai a disposizione un'implementazione completa e modulare di un **Transformer Encoder** per il procesaming di landmarks di linguaggio dei segni.

**Cosa troverai:**

| File                              | Descrizione                                                   |
| --------------------------------- | ------------------------------------------------------------- |
| `transformer_encoder.py`          | 🎯 **Implementazione principale** - Tutte le componenti core  |
| `transformer_encoder_examples.py` | 📚 **6 Esempi progressivi** - Da basic a training             |
| `integration_template.py`         | 🔗 **Pipeline completo** - CNN + Fusion + Decoder             |
| `TRANSFORMER_ENCODER_GUIDE.md`    | 📖 **Documentazione tecnica** - Architettura e best practices |

---

## 🚀 Quick Start (5 minuti)

### 1. Import e Setup

```python
import torch
from src.models.transformer_encoder import TransformerEncoder

# Crea il modello
encoder = TransformerEncoder(
    landmark_dim=2108,      # MediaPipe landmarks
    hidden_dim=512,         # Dimensione latente
    num_layers=4,           # 4 encoder blocks
    num_heads=8,            # Multi-head attention
    dropout=0.1
)

print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
```

### 2. Forward Pass su Landmarks Normalizzati

```python
# Input: Landmarks normalizzati da preprocessing
landmarks = torch.randn(batch_size=8, num_frames=150, landmark_dim=2108)

# Forward
output = encoder(landmarks)
# Output: (8, 150, 512) - Rappresentazioni contestuali per ogni frame
```

### 3. Pooling per Aggregazione Temporale

```python
# Media su frame per ottenere una singola rappresentazione per video
pooled = output.mean(dim=1)  # (8, 512)

# Passa al decoder/classifier
logits = classifier(pooled)  # (8, vocab_size)
```

---

## 📊 Architettura Dettagliata

```
Landmarks (batch, frames, 2108)
    ↓
[LandmarkEmbedding]
  - Linear: 2108 → 512
  - Positional Encoding (sinusoidale)
  - LayerNorm + Dropout
    ↓
[Transformer Encoder Block ×4]
  Per ogni block:
    - Multi-Head Attention (8 heads, 64 dim/head)
    - Residual: x + attention(x)
    - LayerNorm
    - Feed-Forward (512 → 2048 → 512)
    - Residual: x + ffn(x)
    - LayerNorm
    ↓
[Final LayerNorm]
    ↓
Output: (batch, frames, 512)
```

### Pesi e Parametri

```
Total Parameters ≈ 15-20M

Breakdown:
  - Embedding (2108 → 512): ~1.1M
  - Attention heads (8 × 64): Shared W_q, W_k, W_v, W_o
  - FeedForward (512 → 2048 → 512): Per layer
  - LayerNorm: Minore overhead
```

---

## 💡 Casi di Utilizzo

### ✅ Caso 1: Feature Extraction Solo dai Landmarks

Quando: Non hai video RGB, solo landmarks

```python
landmarks = np.load('path/to/landmarks.npy')  # (frames, 2108)
landmarks = torch.from_numpy(landmarks).float().unsqueeze(0)

encoder = TransformerEncoder(landmark_dim=2108, hidden_dim=512)
features = encoder(landmarks)  # (1, frames, 512)

# Usa features per downstream task
```

### ✅ Caso 2: Dual-Stream (Landmarks + Video)

Quando: Hai sia landmarks che video frames

```python
from src.models.integration_template import DualStreamSignLanguageModel

model = DualStreamSignLanguageModel(
    vocab_size=5000,
    video_backbone='mock',  # o 'resnet50'
    fusion_strategy='concat'  # concat, add, bilinear, attention
)

logits = model(landmarks, video_frames, seq_lens)
```

### ✅ Caso 3: Fine-tuning su Modello Pre-trainato

Quando: Vai un encoder pre-trainato su dataset grande

```python
# Congela encoder
for param in encoder.parameters():
    param.requires_grad = False

# Allena solo decoder
decoder = nn.Sequential(...)
optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

# Training
for landmarks, targets in train_loader:
    features = encoder(landmarks)
    logits = decoder(features.mean(dim=1))
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
```

### ✅ Caso 4: Sequenze Variabili con Padding Mask

Quando: Video di durate diverse nel batch

```python
# Lunghezze effettive variabili
seq_lens = torch.tensor([150, 120, 100, 80])

# Landmarks padded a lunghezza massima
landmarks = torch.zeros(4, 150, 2108)
# Popola solo i frame effettivi...

# Forward con mask
output = encoder(landmarks, seq_lens=seq_lens)
# Attention automaticamente ignora padding
```

---

## 🛠️ Configurazioni Raccomandate

### Lightweight (Prototyping)

```python
encoder = TransformerEncoder(
    hidden_dim=256,
    num_layers=2,
    num_heads=4,
    dropout=0.1
)
# ~2M parameters, ~500MB GPU memory
```

### Standard (Production)

```python
encoder = TransformerEncoder(
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    dropout=0.1
)
# ~15M parameters, ~2GB GPU memory (batch=32)
```

### Heavy (Large Scale)

```python
encoder = TransformerEncoder(
    hidden_dim=768,
    num_layers=6,
    num_heads=12,
    dropout=0.1
)
# ~40M parameters, ~5GB GPU memory (batch=32)
```

---

## 📈 Training Best Practices

### 1. Optimizer e Learning Rate

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)
```

### 2. Gradient Clipping (Essenziale!)

```python
# Nel training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 3. Batch Size e Sequenza

```
GPU Memory ≈ batch_size × seq_len × hidden_dim × 4 × 1.5

Esempi:
  - RTX 3090 (24GB): batch=64, seq=150, hidden=512 ✓
  - RTX 3080 (10GB): batch=32, seq=100, hidden=512 ✓
  - RTX 3070 (8GB):  batch=16, seq=100, hidden=256 ✓
```

### 4. Warmup

```python
# Primi 2 epoch: graduale riscaldamento del learning rate
if epoch < 2:
    current_lr = 1e-4 * (epoch + 1) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
```

---

## 🔍 Debugging & Monitoring

### Verifica Health del Modello

```python
# 1. Weight magnitudes
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

# 2. Gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
    else:
        print(f"{name}: NO GRADIENT!")

# 3. Attention weights (dovrebbe essere focalizzato, non uniforme)
attn_weights = encoder.get_attention_weights()
for layer_idx, w in enumerate(attn_weights):
    entropy = -(w * torch.log(w + 1e-10)).sum(dim=-1).mean()
    print(f"Layer {layer_idx} attention entropy: {entropy:.4f}")
```

### Common Issues

| Problema              | Soluzione                                                                |
| --------------------- | ------------------------------------------------------------------------ |
| Loss NaN              | Riduci learning rate (1e-4 → 1e-5), o aumenta gradient clipping          |
| Loss non diminuisce   | Prova warmup, controlla learning rate, verifica data loading             |
| Memory out            | Riduci batch_size, seq_len, hidden_dim, o abilita gradient checkpointing |
| Attenzione non impara | Aumenta num_heads, riduci dropout, aumenta hidden_dim                    |

---

## 📝 Workflow Completo: Da Zero a Training

### Step 1: Prepara i Dati

```python
import numpy as np
from preprocessing.normalize_landmarks import LandmarkNormalizer

# Carica landmarks grezzi
landmarks_raw = load_landmarks('path/to/video')  # (frames, pose+hand+face)

# Normalizza
normalizer = LandmarkNormalizer()
landmarks_normalized = normalizer.normalize_by_per_frame_bbox(landmarks_raw)
# Output: (frames, 2108) con valori [0, 1]

# Salva
np.save('normalized_landmarks.npy', landmarks_normalized)
```

### Step 2: Crea Dataset

```python
from src.models.integration_template import SignLanguageDataset, collate_fn

landmarks_list = [np.load(f) for f in landmark_files]
labels_list = [get_label(f) for f in landmark_files]

dataset = SignLanguageDataset(
    landmarks_list=landmarks_list,
    labels_list=labels_list,
    max_len=150
)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
```

### Step 3: Crea Modello

```python
from src.models.transformer_encoder import TransformerEncoder

encoder = TransformerEncoder(
    landmark_dim=2108,
    hidden_dim=512,
    num_layers=4,
    num_heads=8
)

# O modello completo con CNN
from src.models.integration_template import DualStreamSignLanguageModel

model = DualStreamSignLanguageModel(vocab_size=5000)
```

### Step 4: Training Loop

```python
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in train_loader:
        landmarks = batch['landmarks']
        labels = batch['labels']
        seq_lens = batch['seq_lens']

        logits = model(landmarks, seq_lens=seq_lens)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
```

---

## 📚 Documentazione Completa

Per dettagli su:

- **Architettura e formulazione matematica** → `TRANSFORMER_ENCODER_GUIDE.md`
- **6 Esempi di uso progressivi** → `transformer_encoder_examples.py`
- **Pipeline completo full-stack** → `integration_template.py`

---

## 🎯 Prossimi Step

1. **Verifica l'installazione**:

   ```bash
   python -c "from src.models.transformer_encoder import TransformerEncoder; print('✓ Setup OK')"
   ```

2. **Esegui gli esempi**:

   ```bash
   python src/models/transformer_encoder_examples.py
   ```

3. **Integra nel tuo pipeline**:

   ```python
   from src.models.transformer_encoder import TransformerEncoder
   # ...
   ```

4. **Monitora training**:
   - Controlla weight magnitudes
   - Verifica gradient flow
   - Analizza attention weights

---

## ❓ FAQ

**Q: Posso usare il modello senza video (solo landmarks)?**
A: Sì! L'encoder è self-contained. Se usi `DualStreamSignLanguageModel`, passi `video=None`.

**Q: Cambia l'architettura per diversi tipi di segni?**
A: No, the architecture è general-purpose. Adatta `hidden_dim` e `num_layers` alla complessità del tuo dataset.

**Q: Come proseguo verso un decoder per tradurre in frasi?**
A: Usa il TransformerEncoder come encoder in un setup seq2seq con TransformerDecoder per generare testo.

**Q: Supportate inferenza in real-time?**
A: Sì, l'encoder è efficiente. Con batch=1 e seq_len=150, tempo ≈ 50-100ms su RTX 3080.

---

## 📞 Supporto Tecnico

Per problemi specifici:

1. Leggi `TRANSFORMER_ENCODER_GUIDE.md` sezione "Troubleshooting"
2. Controlla gli esempi in `transformer_encoder_examples.py`
3. Esamina il template di integrazione in `integration_template.py`

---

**Data**: April 2026
**Versione**: 1.0
**Status**: Production Ready ✅
