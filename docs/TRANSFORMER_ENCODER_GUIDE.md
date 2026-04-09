# Transformer Encoder per Sign Language Translation

## Documentazione Tecnica Completa

---

## Indice

1. [Introduzione e Architettura](#introduzione-e-architettura)
2. [Componenti Modulo](#componenti-modulo)
3. [Matematica e Formula](#matematica-e-formula)
4. [Gestione dei Pesi](#gestione-dei-pesi)
5. [Padding Mask e Sequenze Variabili](#padding-mask-e-sequenze-variabili)
6. [Best Practices](#best-practices)
7. [Performance e Ottimizzazioni](#performance-e-ottimizzazioni)
8. [Troubleshooting](#troubleshooting)

---

## Introduzione e Architettura

### Problema

Tradurre video di linguaggio dei segni a testo richiede di catturare:

- **Informazioni spaziali**: Posizioni delle mani, viso, corpo (landmarks)
- **Informazioni temporali**: Sequenze di gesti nel tempo
- **Dipendenze a lungo raggio**: Gestazioni lunghe che si estendono su molti frame

### Soluzione: Transformer Encoder

Il **Transformer Encoder** elabora landmarks sequenziali con:

- ✅ **Multi-Head Attention** per catturare dipendenze temporali
- ✅ **Positional Encoding** per iniettare informazione posizionale
- ✅ **Feed-Forward Networks** per apprendere trasformazioni non-lineari
- ✅ **Residual Connections + Layer Norm** per stabilità di training

### Flusso Dati Completo

```
Video RGB
    ↓
[MediaPipe Extraction] → Landmarks (pose, hand, face)
    ↓
[Normalization] → Landmarks normalizzati [0, 1]
    ↓
Landmarks (batch, num_frames, 2108)
    ↓
┌─────────────────────────────────────────┐
│   TRANSFORMER ENCODER (questo modulo)   │
├─────────────────────────────────────────┤
│ [1] Landmark Embedding: 2108 → 512     │
│     + Positional Encoding                │
│     + LayerNorm + Dropout                │
│                ↓                         │
│ [2] Transformer Encoder Blocks (×4)    │
│     - Multi-Head Attention 8 heads      │
│     - Feed-Forward 512→2048→512         │
│     - Residual + LayerNorm               │
│     - Dropout                            │
│                ↓                         │
│ [3] Output: (batch, frames, 512)       │
└─────────────────────────────────────────┘
    ↓
[Pooling] → (batch, 512) aggregazione temporale
    ↓
[Classification Head] → Logits (batch, vocab_size)
    ↓
Testo tradotto
```

---

## Componenti Modulo

### 1. PositionalEncoding

**Scopo**: Iniettare informazioni posizionali nella sequenza

**Due modalità**:

#### A) Sinusoidale (Classica - Vaswani 2017)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Proprietà**:

- Non-parametrica (fisso durante training)
- Predittibile su lunghezze più lunghe che visto in training
- Frequenze diverse per ogni dimensione

**Quando usare**: Ambienti di produzione, stabilità preferita

#### B) Learnable (Apprendibile)

```
PE = nn.Parameter(torch.randn(max_len, d_model))
```

**Proprietà**:

- Parametri allenabili
- Specifici del modello
- Potenzialmente migliori su dataset specifici

**Quando usare**: Modelli specializzati, dataset fissi

### 2. MultiHeadAttention

**Matematica**:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

MultiHead = Concat(head_1, ..., head_h) · W^O

dove:
- Q = x · W^Q  (batch, seq, hidden_dim)
- K = x · W^K  (batch, seq, hidden_dim)
- V = x · W^V  (batch, seq, hidden_dim)
- d_k = hidden_dim / num_heads
```

**Implementazione**:

```python
# Proiezione a multi-head (8 heads, 512 dim → 64 dim per head)
Q = W_q(x).view(batch, seq, 8, 64).transpose(1, 2)  # (batch, 8, seq, 64)
K = W_k(x).view(batch, seq, 8, 64).transpose(1, 2)
V = W_v(x).view(batch, seq, 8, 64).transpose(1, 2)

# Attention score
scores = Q @ K.T / sqrt(64)  # (batch, 8, seq, seq)

# With masking
scores[~mask] = -inf

# Softmax + dropout
attn_weights = softmax(scores)  # (batch, 8, seq, seq)

# Apply to values
context = attn_weights @ V  # (batch, 8, seq, 64)

# Ricombina heads
output = concat(context) @ W_o  # (batch, seq, 512)
```

**Vantaggi multi-head**:

- ✅ Diversi "spazi di rappresentazione"
- ✅ Parallelizzabile
- ✅ Cattura relazioni a diverse scale temporali

### 3. FeedForward

**Struttura**:

```
x → Linear(512, 2048) → ReLU → Dropout → Linear(2048, 512) → x
```

**Scopo**: Trasformazioni non-lineari

**Perché l'espansione a 2048**?:

- Aumenta la capacità di apprendimento
- Ogni feed-forward ha ~2M parametri (per 8 heads)
- Trade-off: Più compute ma migliore espressività

### 4. TransformerEncoderBlock

**Architettura**:

```
Input x (batch, seq, 512)
  ↓
[Multi-Head Attention]
  ↓
x + Dropout(attn_output)  ← Residual connection
  ↓
LayerNorm  ← Post-norm
  ↓
[Feed-Forward]
  ↓
x + Dropout(ffn_output)  ← Residual connection
  ↓
LayerNorm
  ↓
Output (batch, seq, 512)
```

**Residual Connections**:

- Permettono gradienti di fluire direttamente
- Essenziali per modelli profondi (>4 layer)

**Layer Normalization vs Batch Norm**:

- LayerNorm: Norma per ogni sample (indipendente dal batch)
- ModernTransformers: LayerNorm preferito per sequenze variabili

### 5. LandmarkEmbedding

**Trasformazione**:

```
landmarks (batch, frames, 2108)
    ↓
[Linear: 2108 → 512]
    ↓
[Positional Encoding]
    ↓
[LayerNorm]
    ↓
[Dropout(0.1)]
    ↓
x_embedded (batch, frames, 512)
```

**Perché normalizzare i landmarks inizialmente**?

- Stabilizza la distribuzione prima del transformer
- Previene exploding/vanishing gradients nei pesi condivisi

---

## Matematica e Formula

### Attention Score Calculation

Dato query `q`, key `k`, value `v`:

```
score(q_i, k_j) = (q_i · k_j) / √d_k

Dove:
- q_i: query vector per posizione i
- k_j: key vector per posizione j
- d_k: dimensione per head (512 / 8 = 64)
```

**Scaling factor √d_k**:

- Previene dominanza di grandi valori
- Mantiene softmax in regime stabile

### Softmax su Attention

```
α(q, K) = softmax(QK^T / √d_k)

softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

Con masking:
softmax_masked = softmax(scores[~mask] = -∞)
               = softmax(scores) dove masked = 0
```

### Positional Encoding

**Sinusoidale**:

```
PE_pos,2i   = sin(pos / 10000^(2i/d_model))
PE_pos,2i+1 = cos(pos / 10000^(2i/d_model))

Frequenza: 10000^(2i/d_model) per dimensioni pari/dispari
```

**Intuizione**:

- Bassa frequenza (i=0): Varia lentamente (posizione globale)
- Alta frequenza (i=d/2): Varia velocemente (dettagli locali)

---

## Gestione dei Pesi

### Inizializzazione Parametri

```python
# Pesi lineari: Xavier uniform
W = nn.Linear(in_features, out_features)
# Default: Uniform(-√k, √k) dove k = 1/(in_features)

# Bias: Zero initialization
self.bias = nn.Parameter(torch.zeros(...))

# LayerNorm: γ=1, β=0 (default in PyTorch)

# Positional Embeddings (se learnable):
self.pe = nn.Parameter(torch.randn(max_len, d_model))
nn.init.normal_(self.pe, mean=0, std=0.02)
```

### Weight Magnitude Tracking

Durante training, monitorare:

```python
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
```

**Segnali d'allarme**:

- ⚠️ std → 0: Weights collassano
- ⚠️ std → ∞: Exploding gradients
- ✅ std stabile: Normal behavior

### Gradient Clipping

**Essenziale per sequenze lunghe**:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Perché**?

- Transformer softmax può produrre gradienti grandi
- Su sequenze lunghe (150+ frames) il problema è amplificato

---

## Padding Mask e Sequenze Variabili

### Il Problema

Nel batch abbiamo video di durate diverse:

```
Video 1: 200 frames ✓
Video 2: 150 frames ✓
Video 3: 100 frames → Paddiamo a 200
Video 4: 80 frames  → Paddiamo a 200
```

L'attenzione NON deve "guardarsi il padding".

### Soluzione: Attention Mask

```python
# seq_lens = [200, 150, 100, 80]
# max_len = 200

mask = torch.zeros(batch=4, 1, 1, 200)
mask[0, :, :, :200] = 1  # Video 1: tutto valido
mask[1, :, :, :150] = 1  # Video 2: solo primi 150
mask[2, :, :, :100] = 1  # Video 3: solo primi 100
mask[3, :, :, :80]  = 1  # Video 4: solo primi 80

# Nel calcolo di attention:
scores[~mask] = -inf
attn = softmax(scores)  # padding → attn_peso ≈ 0
```

### Implementazione nel Codice

```python
def create_mask(seq_lens, batch_size, max_len, device):
    """
    Args:
        seq_lens: (batch,) lunghezze effettive
        batch_size: dimensione batch
        max_len: lunghezza max

    Returns:
        mask: (batch, 1, 1, max_len) bool tensor
              True = valido, False = padding
    """
    positions = torch.arange(max_len, device=device)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(1)
```

### Risultato

```
Senza mask:
  Attenzione "guarda" il padding → Degrada le rappresentazioni

Con mask:
  Attenzione ignora padding → Rappresentazioni pulite per ogni video
```

---

## Best Practices

### 1. Configurazione Dimensioni

| Aspetto      | Raccomandazione           |
| ------------ | ------------------------- |
| `hidden_dim` | 256, 512, o 768           |
| `num_heads`  | 8 in base hidden_dim      |
| `num_layers` | 4-6 per performance buone |
| `d_ff`       | 4 × hidden_dim (default)  |
| `dropout`    | 0.1 (standard)            |

### 2. Learning Rate e Optimizer

```python
# Adam è quasi sempre la scelta migliore
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Con learning rate schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)
```

### 3. Batch Size

```
GPU Memory ≈ batch_size × seq_len × hidden_dim × 4 bytes × factor

Esempio (batch=32, seq=150, hidden=512):
  ≈ 32 × 150 × 512 × 4 × 1.5 ≈ 3.7 GB

# Scegliere batch_size in base a GPU disponibile
```

### 4. Warmup e Gradient Accumulation

```python
# Warmup per primi epoch
for epoch in range(num_epochs):
    if epoch < 2:
        current_lr = 1e-4 * (epoch + 1) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
```

### 5. Dropout e Regularizzazione

```python
# Training time: Dropout attivo
model.train()
output = model(landmarks)  # Dropout applicato

# Eval time: Dropout disattivo
model.eval()
with torch.no_grad():
    output = model(landmarks)  # Dropout non applicato
```

---

## Performance e Ottimizzazioni

### Complessità Computazionale

**Attention** (dominante):

```
O(batch × seq_len² × hidden_dim)

Esempio: 32 × 150² × 512 ≈ 368M operazioni (per layer)
Con 4 layer: ≈ 1.5B operazioni per batch
```

**Strategie di ottimizzazione**:

1. **Sparse Attention** (per sequenze lunghe >500 frame):

   ```python
   # Attenzione solo a vicinato locale
   # Riduce: O(seq_len²) → O(seq_len × window_size)
   ```

2. **Gradient Checkpointing** (memoria):

   ```python
   # Ricomputa attivazioni in backward pass
   # Riduce memoria ma aumenta compute
   ```

3. **Mixed Precision Training**:

   ```python
   from torch.cuda.amp import autocast

   with autocast():
       output = model(landmarks)
       loss = criterion(output, targets)
   ```

### Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
) as prof:
    output = model(landmarks)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Troubleshooting

### Problema 1: Loss non diminuisce

```
Sintomi: Loss rimane costante o aumenta

Soluzioni:
  1. Verifica learning rate (prova 1e-3, 1e-4, 1e-5)
  2. Batch normalization vs solo LayerNorm
  3. Gradient clipping: max_norm=1.0 essenziale
  4. Warm-up nei primi epoch
  5. Shuffle dataset nel DataLoader
```

### Problema 2: NaN loss

```
Sintomi: Loss diventa NaN durante training

Cause:
  1. Learning rate troppo alto
  2. Gradienti esplodono (⚠️ verify con torch.nn.utils.clip_grad_norm_)
  3. Overflow numerico

Soluzioni:
  1. Riduci learning rate (di 10x)
  2. Aumenta gradient clipping max_norm (da 1.0 a 5.0)
  3. Usa mixed precision training con loss scaling

Debug:
  for name, param in model.named_parameters():
      if param.grad is not None:
          print(f"{name}: grad_norm={param.grad.norm()}")
```

### Problema 3: Memory out of memory

```
Sintomi: CUDA out of memory durante forward/backward

Soluzioni:
  1. Riduci batch_size (32 → 16 → 8)
  2. Riduci seq_len (150 → 100)
  3. Abilita gradient_checkpointing
  4. Riduci hidden_dim (512 → 256)
```

### Problema 4: Attenzione non impara

```
Sintomi: Attention weights rimangono uniformi

Debug:
  encoder = TransformerEncoder(...)

  # Dopo forward pass
  attn_weights = encoder.get_attention_weights()
  for layer_idx, weights in enumerate(attn_weights):
      entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
      print(f"Layer {layer_idx} entropy: {entropy:.4f}")

  # Entropia alta = attenzione non focalizzata
  # Entropia bassa = attenzione focalizzata

Soluzioni:
  1. Aumenta num_heads (4 → 8)
  2. Riduci dropout
  3. Aumenta hidden_dim (capacità maggiore)
```

---

## Integrazione nel Pipeline SLR/SLT

### Step-by-step

```python
# 1. Carica landmarks normalizzati
landmarks = np.load('path/to/normalized_landmarks.npy')  # (frames, 2108)
landmarks = torch.from_numpy(landmarks).float().unsqueeze(0)  # (1, frames, 2108)

# 2. Crea encoder
encoder = TransformerEncoder(
    landmark_dim=2108,
    hidden_dim=512,
    num_layers=4,
    num_heads=8
)

# 3. Forward pass
encoder_output = encoder(landmarks)  # (1, frames, 512)

# 4. Aggregazione temporale (media)
pooled = encoder_output.mean(dim=1)  # (1, 512)

# 5. Classificazione (es. Transformer decoder)
decoder = TransformerDecoder(...)
logits = decoder(pooled)  # (1, vocab_size)

# 6. Estrai testo
pred_tokens = logits.argmax(dim=-1)
text = tokenizer.decode(pred_tokens)
```

---

## Risorse Esterne

1. **Vaswani et al., 2017** - Attention is All You Need
   - Articolo originale Transformer
   - https://arxiv.org/abs/1706.03762

2. **The Illustrated Transformer by Jay Alammar**
   - Intuizioni visive sull'architettura
   - https://jalammar.github.io/illustrated-transformer/

3. **PyTorch Documentation**
   - nn.MultiheadAttention
   - nn.LayerNorm
   - torch.nn.utils.clip*grad_norm*

---

## Contatto & Support

Per domande sull'implementazione o su problemi di training,
consulta i file di esempio in `transformer_encoder_examples.py`.
