# Transformer Decoder - Guida Tecnica e Implementazione Dettagliata

## 📋 Indice

1. [Architettura Dettagliata](#architettura-dettagliata)
2. [Formule Matematiche](#formule-matematiche)
3. [Masked Self-Attention](#masked-self-attention)
4. [Cross-Attention](#cross-attention)
5. [Training Loop Completo](#training-loop-completo)
6. [Generazione Autoregressiva](#generazione-autoregressiva)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Architettura Dettagliata

### Pipeline Completo

```
INPUT MULTIMODALE (upstream)
    ↓
    Landmarks (B, T_video, 2108)  +  Video (B, T_video, 3, H, W)
    ↓
    [Transformer Encoder + CNN-1D-GRU + Fusion]
    ↓
    MEMORY MULTIMODALE
    (B, T_video=100, hidden_dim=512)
    ↓
    ┌─────────────────────────────────┐
    │   TRANSFORMER DECODER           │  ← Questa implementazione
    └─────────────────────────────────┘
    ↓
    INPUT TARGET (training)
    (B, T_text=50) indici token
    ↓
    ┌──────────────────────────────────────┐
    │ 1. Token Embedding Layer             │
    │    (vocab_size=10000) → hidden_dim   │
    │    (B, T_text) → (B, T_text, 512)    │
    └──────────────────────────────────────┘
    ↓
    ┌──────────────────────────────────────┐
    │ 2. Positional Encoding               │
    │    (sinusoidale)                     │
    │    (B, T_text, 512) + PE             │
    └──────────────────────────────────────┘
    ↓
    ┌─────────────────────────────────────────────┐
    │ 3. Transformer Decoder (4 layers default)    │
    │                                              │
    │    Per ogni layer:                           │
    │    ┌────────────────────────────────────┐    │
    │    │ a) Masked Self-Attention           │    │
    │    │    (Token i vede solo token 0..i)  │    │
    │    │    Input:  (B, T_text, 512)        │    │
    │    │    Output: (B, T_text, 512)        │    │
    │    └────────────────────────────────────┘    │
    │    ↓                                          │
    │    ┌────────────────────────────────────┐    │
    │    │ b) Cross-Attention                 │    │
    │    │    (Token i attende a tutti video) │    │
    │    │    Query:   (B, T_text, 512)       │    │
    │    │    Key/Val: (B, T_video, 512)      │    │
    │    │    Output:  (B, T_text, 512)       │    │
    │    └────────────────────────────────────┘    │
    │    ↓                                          │
    │    ┌────────────────────────────────────┐    │
    │    │ c) Feed-Forward Network            │    │
    │    │    Linear(512 → 2048 → 512)        │    │
    │    │    + ReLU activation               │    │
    │    │    Output:  (B, T_text, 512)       │    │
    │    └────────────────────────────────────┘    │
    │                                              │
    │    (Repeat 4 volte con layer normalization) │
    └─────────────────────────────────────────────┘
    ↓
    ┌──────────────────────────────────────┐
    │ 4. Final Linear Layer                │
    │    (hidden_dim → vocab_size)         │
    │    (B, T_text, 512) → (B, T_text, vocab_size) │
    └──────────────────────────────────────┘
    ↓
    OUTPUT: LOGITS GREZZI
    (B, T_text=50, vocab_size=10000)

    (Utilizzare per cross_entropy loss)
```

### Dimensioni Passaggio per Passaggio

**Durante Training con Esempio Concreto:**

```
Parametri:
  - Batch size B = 8
  - T_video = 100 (numero frame nel video)
  - T_text = 50 (lunghezza sequenza target)
  - hidden_dim = 512
  - vocab_size = 10000
  - num_heads = 8

MEMORY (dall'upstream):
  Shape: (8, 100, 512)
  Rappresenta: "Che dice il video in questi 100 frame?"

TARGET TOKENS (ground truth testo):
  Input shape: (8, 50) int64 indices
  Esempio: [1, 234, 567, 890, 345, ...]
           (1 = BOS, resto = token text)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Token Embedding:
  Input:  (8, 50) indici
  Embed lookup: 50 token × 512 dim = 50 embedding vector
  Output: (8, 50, 512)

Positional Encoding:
  Input: (8, 50, 512)
  PE: sinusoidale per posizioni (0..49) nella sequenza testo
  Output: (8, 50, 512) + PE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Decoder Block 1:

  Masked Self-Attention (su target):
    Query:   (8, 50, 512) da testo
    Key:     (8, 50, 512) da testo
    Value:   (8, 50, 512) da testo
    Mask:    (50, 50) triangolare (causal)
    Output:  (8, 50, 512)

  Add & Norm:
    Output + Residual: (8, 50, 512)
    LayerNorm:         (8, 50, 512)

  Cross-Attention (target × memory):
    Query:   (8, 50, 512) da testo
    Key:     (8, 100, 512) da video
    Value:   (8, 100, 512) da video
    Output:  (8, 50, 512)

  Add & Norm:
    Output + Residual: (8, 50, 512)
    LayerNorm:         (8, 50, 512)

  Feed-Forward:
    Linear1: (8, 50, 512) → (8, 50, 2048)
    ReLU:    (8, 50, 2048)
    Linear2: (8, 50, 2048) → (8, 50, 512)
    Output:  (8, 50, 512)

  Add & Norm:
    Output + Residual: (8, 50, 512)
    LayerNorm:         (8, 50, 512)

[Repeat per Decoder Block 2, 3, 4]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output Linear:
  Input:  (8, 50, 512)
  Output: (8, 50, 10000)  ← LOGITS GREZZI

Loss Calculation:
  Reshape logits: (8*50, 10000) = (400, 10000)
  Reshape target: (400,)
  loss = CrossEntropy(logits, target)
```

---

## Formule Matematiche

### Positional Encoding (Sinusoidale)

Per ogni posizione $\text{pos}$ e dimensione $i$:

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Dove:

- $\text{pos}$ = posizione nella sequenza (0 a T-1)
- $i$ = dimensione (0 a 255, per 512 dim)
- $d_{\text{model}}$ = 512 (hidden dimension)

This provides unique position information for every position in the sequence.

### Multi-Head Attention

Per head $h$ (totale 8 heads con dim=512, quindi 64 dim per head):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Con masking (per autoregressività):

$$\text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Dove:

- $Q$ = Query matrix
- $K$ = Key matrix
- $V$ = Value matrix
- $d_k$ = 512/8 = 64 (dimensione per head)
- $M$ = Causal mask (0 per posizioni consentite, $-\infty$ per vietate)

Combined:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_8)W^O$$

### Cross-Attention

$$\text{CrossAttn}_{\text{decoder}}(Q_{\text{target}}, K_{\text{memory}}, V_{\text{memory}}) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$$

Dove:

- $Q$ viene dal target (sequenza testo)
- $K, V$ vengono dalla memoria multimodale (video)

Questo permette a ogni token del testo di "guardare" tutti i frame del video.

### Feed-Forward Network

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Dove:

- $W_1 \in \mathbb{R}^{512 \times 2048}$
- $W_2 \in \mathbb{R}^{2048 \times 512}$

### Cross-Entropy Loss (per training)

$$L = -\sum_{b,t} \log P(y_{b,t} | \text{logits}_{b,t})$$

$$= \sum_{b,t} \text{CrossEntropy}(\text{logits}_{b,t}, y_{b,t})$$

---

## Masked Self-Attention

### Perché la Masking è Cruciale

Durante il training, il modello deve imparare a predire ogni token basandosi **solo** sui token precedenti. Questo è garantito dalla **causal mask (autoregressive mask)**.

```
Senza maschera (SBAGLIATO - il modello vede il futuro):
  Position 0: può guardare 0, 1, 2, 3, 4  ← Vede il futuro! ❌
  Position 1: può guardare 0, 1, 2, 3, 4
  Position 2: può guardare 0, 1, 2, 3, 4
  ...

Con causal mask (GIUSTO - autoregressivo):
  Position 0: può guardare solo 0        ✓
  Position 1: può guardare 0, 1          ✓
  Position 2: può guardare 0, 1, 2       ✓
  Position 3: può guardare 0, 1, 2, 3    ✓
  Position 4: può guardare 0, 1, 2, 3, 4 ✓
```

### Implementazione della Mask

```python
def _generate_causal_mask(seq_len, device):
    # Crea triangolo superiore di -inf
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1
    )
    return mask

# Risultato per seq_len=3:
# [[0.0,      -inf, -inf],
#  [0.0,      0.0, -inf],
#  [0.0,      0.0,  0.0]]
```

### Applicazione negli Attention Scores

```python
# Before softmax:
scores = (Q @ K.T) / sqrt(d_k)  # (3, 3) se seq_len=3

# Applicare mask:
scores = scores + mask  # Posizioni vietate diventano -inf

# Softmax:
attn_weights = softmax(scores)
# -inf → softmax(−inf) = 0, non contribuisce

# Result: Ogni posizione attende solo al suo passato
```

---

## Cross-Attention

### Come Funziona

Il decoder attende a **due sorgenti** diverse:

```
Token nel target:  "the man walks"

Self-Attention (su target):
  "the" attende a: ["the"]
  "man" attende a: ["the", "man"]
  "walks" attende a: ["the", "man", "walks"]

  → Informazione sequenziale nel testo (autoregressiva)

Cross-Attention (su memory video):
  "the" attende a: [frame_0, frame_1, ..., frame_99]
  "man" attende a: [frame_0, frame_1, ..., frame_99]
  "walks" attende a: [frame_0, frame_1, ..., frame_99]

  → Informazione dal video (contesto visivo)

Combinato:
  Ogni token nel testo guarda:
    1. Il suo contesto testuale (che l'ha preceduto)
    2. L'intero video per contesto visivo

  Questo consente una generazione coerente e consapevole del contesto.
```

### Implementazione in PyTorch

```python
class MyDecoderLayer(nn.Module):
    def forward(self, tgt, memory, tgt_mask=None):
        # Self-attention su target
        tgt2 = self.self_attn(
            query=tgt,              # (B, T_text, dim)
            key=tgt,                # (B, T_text, dim)
            value=tgt,              # (B, T_text, dim)
            attn_mask=tgt_mask      # Causal mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention: target attende a memory
        tgt2 = self.multihead_attn(
            query=tgt,              # (B, T_text, dim)
            key=memory,             # (B, T_video, dim)
            value=memory,           # (B, T_video, dim)
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
```

---

## Training Loop Completo

### Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.transformer_decoder import TransformerDecoder

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modello
decoder = TransformerDecoder(
    hidden_dim=512,
    vocab_size=10000,
    num_decoder_layers=4,
    num_heads=8,
    device=device
).to(device)

# Optimizer con weight decay
optimizer = optim.AdamW(
    decoder.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
    betas=(0.9, 0.98)
)

# Scheduler con warmup
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # epochs
    eta_min=1e-6
)

# Loss (ignora padding)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD token

# Carica dataset doppio stream
# Assumi: dataloader fornisce (memory, target_tokens, target_shifted)
```

### Training Loop

```python
def train_epoch(
    decoder,
    train_loader,
    optimizer,
    criterion,
    device,
    grad_clip_norm=1.0
):
    """Esegui un'epoch di training."""
    decoder.train()
    total_loss = 0

    for batch_idx, (memory, target_tokens, _) in enumerate(train_loader):
        memory = memory.to(device)          # (B, T_video, 512)
        target_tokens = target_tokens.to(device)  # (B, T_text)

        # Forward pass
        logits = decoder(
            memory=memory,
            target_tokens=target_tokens
        )  # (B, T_text, vocab_size)

        # Calcola loss
        # Reshape per cross_entropy
        logits_flat = logits.view(-1, decoder.vocab_size)  # (B*T_text, vocab_size)
        targets_flat = target_tokens.view(-1)  # (B*T_text)

        loss = criterion(logits_flat, targets_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (importante per RNN-like models)
        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(),
            max_norm=grad_clip_norm
        )

        # Update
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def validate(
    decoder,
    val_loader,
    criterion,
    device
):
    """Valida il modello."""
    decoder.eval()
    total_loss = 0

    for memory, target_tokens, _ in val_loader:
        memory = memory.to(device)
        target_tokens = target_tokens.to(device)

        logits = decoder(memory=memory, target_tokens=target_tokens)

        logits_flat = logits.view(-1, decoder.vocab_size)
        targets_flat = target_tokens.view(-1)
        loss = criterion(logits_flat, targets_flat)

        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


# Training
num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*50}")

    # Training
    train_loss = train_epoch(
        decoder, train_loader, optimizer, criterion, device
    )

    # Validation
    val_loss = validate(decoder, val_loader, criterion, device)

    # Learning rate scheduling
    scheduler.step()

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(decoder.state_dict(), 'decoder_best.pt')
        print(f"✓ Best model saved (val_loss={val_loss:.4f})")

    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
```

---

## Generazione Autoregressiva

### Loop di Generazione Passo per Passo

```
Memoria video: (1, 100, 512)
BOS token ID: 1
EOS token ID: 2
Max length: 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Passo 1:
  Input: [1] (BOS)
  Decoder vede: BOS attende solo se stesso
  Output logits: (1, 10000) per la posizione BOS
  Sample token: T=0.9 → token "234" ("the")
  Sequenza: [1, 234]

Passo 2:
  Input: [1, 234]
  Decoder vede:
    - Pos 0 (BOS) attende solo se stesso
    - Pos 1 ("the") attende BOS e se stesso
  Output logits: (2, 10000) per ogni posizione
  Prendi ultimo logit (pos 1): sample token "567" ("man")
  Sequenza: [1, 234, 567]

Passo 3:
  Input: [1, 234, 567]
  Decoder output: (3, 10000)
  Prendi ultimo logit: sample token "890" ("walks")
  Sequenza: [1, 234, 567, 890]

...

Passo N:
  Sample token: 2 (EOS)
  Stop! Sequenza completa: [1, 234, 567, 890, ..., 2]
```

### Campionamento con Temperatura

```python
# Logits grezzi
logits = decoder_output[:, -1, :]  # (batch, vocab_size)

# Temperatura < 1: Meno casuale (più confident)
logits = logits / 0.7
probs = softmax(logits)
# → Distribuzioni più piccanti, meno diversità

# Temperatura = 1: "Normale"
probs = softmax(logits)

# Temperatura > 1: Più casuale (meno confident)
logits = logits / 1.5
probs = softmax(logits)
# → Distribuzioni più piatte, più diversità

next_token = torch.multinomial(probs, num_samples=1)
```

### Top-k Sampling

```python
# Prendi i k token più probabili
k = 50
top_k_probs, top_k_indices = torch.topk(probs, k=k)

# Set altre probabilità a 0
probs_new = torch.zeros_like(probs)
probs_new.scatter_(1, top_k_indices, top_k_probs)

# Rinormalizza
probs_new = probs_new / probs_new.sum(dim=-1, keepdim=True)

# Sample da top-k
next_token = torch.multinomial(probs_new, num_samples=1)
```

### Nucleus Sampling (Top-p)

```python
# Ordina probabilità in descending order
sorted_probs, sorted_indices = torch.sort(
    probs, descending=True, dim=-1
)

# Cumulative sum
cumsum = torch.cumsum(sorted_probs, dim=-1)

# Rimuovi token che superano la soglia
sorted_indices_to_remove = cumsum > 0.95  # p=0.95
sorted_indices_to_remove[..., 0] = False  # Mantieni almeno 1

# Convert back to original indices
indices_to_remove = sorted_indices_to_remove.scatter(
    -1, sorted_indices, sorted_indices_to_remove
)
probs[indices_to_remove] = 0
probs = probs / probs.sum(dim=-1, keepdim=True)

next_token = torch.multinomial(probs, num_samples=1)
```

---

## Best Practices

### 1. Preparazione Dati

```python
# Target tokens DEVONO avere BOS all'inizio (shift destra)
# Esempio raw data: "the man walks"
# Con IDs: [234, 567, 890]

# Per training:
target_tokens = [1, 234, 567, 890]  # [BOS, word1, word2, EOS]
                                      # Lunghezza: 4

# Lunghezze assicurate:
max_len = 100
target_tokens = torch.nn.functional.pad(
    target_tokens,
    (0, max_len - len(target_tokens)),
    value=0  # 0 = PAD token (ignore in loss)
)
```

### 2. Inizializzazione Pesi

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

decoder.apply(init_weights)
```

### 3. Learning Rate Scheduling

```python
# Warmup + Cosine Annealing
def get_scheduler(
    optimizer,
    num_epochs,
    warmup_epochs=5,
    device='cuda'
):
    """Crea scheduler con warmup."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Cosine annealing dopo warmup
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 4. Gradient Checkpointing (per grandi modelli)

```python
# Salva memoria ricomputando gradualmente
from torch.utils.checkpoint import checkpoint

class CheckpointedDecoder(nn.Module):
    def forward(self, tgt, memory, tgt_mask=None):
        for layer in self.layers:
            tgt = checkpoint(layer, tgt, memory, tgt_mask)
        return tgt
```

### 5. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in train_loader:
        with autocast():
            logits = decoder(memory, target_tokens)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

---

## Troubleshooting

### Problema: Loss Non Scende

**Cause e Soluzioni:**

1. **Learning Rate troppo piccolo**

   ```python
   # Prova lr schedule con warmup
   optimizer = optim.AdamW(params, lr=1e-4)
   scheduler = get_warmup_scheduler(optimizer, warmup_epochs=5)
   ```

2. **Batch Size troppo piccolo**

   ```python
   # Aumenta batch size (16 → 32 → 64)
   # Batch normalization like effects
   dataloader = DataLoader(dataset, batch_size=64)
   ```

3. **Inizializzazione pesi**
   ```python
   # Xavier/Kaiming initialization
   nn.init.xavier_uniform_(decoder.token_embedding.weight)
   ```

### Problema: Generazione Ripete Token

**Cause:**

- Causal mask non applicata correttamente
- Temperature troppo bassa
- Logit non diversi

**Soluzione:**

```python
# Aumenta temperature durante generation
generated = decoder.generate(
    memory=memory,
    temperature=1.2,  # Aumenta randomicità
    top_k=50,         # Limita a top-k
    top_p=0.9         # Nucleus sampling
)
```

### Problema: Out-of-Memory (OOM)

**Soluzioni in ordine:**

```python
# 1. Riduci batch size
batch_size = 4  # era 32

# 2. Riduci lunghezza sequenze
max_target_len = 128  # era 256

# 3. Riduci dimensione modello
decoder = TransformerDecoder(
    hidden_dim=256,           # era 512
    num_decoder_layers=2,     # era 4
)

# 4. Usa Mixed Precision (16-bit)
from torch.cuda.amp import autocast
with autocast():
    logits = decoder(memory, target_tokens)

# 5. Gradient Checkpointing
from torch.utils.checkpoint import checkpoint_sequential
decoder.transformer_decoder = checkpoint_sequential(
    decoder.transformer_decoder,
    2  # segments
)
```

### Problema: Attenzione Non Convergente

**Debug Attention Weights:**

```python
# Estrai pesi di attention da un layer
decoder.eval()
with torch.no_grad():
    logits = decoder(memory, target_tokens)

# Interno a TransformerDecoderLayer (serve custom hook):
def get_attention_weights(decoder_layer):
    hooks = []
    attention_weights = []

    def hook(module, input, output):
        # output è (attention_output, attention_weights)
        attention_weights.append(output[1])

    for layer in decoder.transformer_decoder.layers:
        h = layer.self_attn.register_forward_hook(hook)
        hooks.append(h)

    return attention_weights, hooks
```

---

## Riferimenti

- Vaswani et al., "Attention is All You Need" (2017)
- Huang et al., "Attention is Not Only a Weight" (2020)
- PyTorch Transformer Documentation
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
