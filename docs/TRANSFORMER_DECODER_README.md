# Transformer Decoder per Sign Language Translation (Video→Text)

## Quick Start & Implementation Guide

---

## 📋 Sommario

Hai a disposizione un'implementazione completa e pronta per l'uso di un **Transformer Decoder autoregressivo** per la generazione di testo a partire da embedding multimodale continuo (fusione di video + landmarks).

**Cosa troverai:**

| File                           | Descrizione                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| `transformer_decoder.py`       | 🎯 **Implementazione principale** - Decoder + Config + Utilities |
| `TRANSFORMER_DECODER_GUIDE.md` | 📖 **Documentazione tecnica** - Architettura e best practices    |
| Questo file                    | 📚 **Quick Start** - Esempi rapidi e integrazione                |

---

## 🚀 Quick Start (10 minuti)

### 1. Import e Setup Minimale

```python
import torch
from src.models.transformer_decoder import TransformerDecoder

# Crea il decoder con configurazione default
decoder = TransformerDecoder(
    hidden_dim=512,           # Deve corrispondere all'output della fusione upstream
    vocab_size=10000,         # Dimensione vocabolario target (testo)
    num_decoder_layers=4,     # 4 decoder blocks standard
    num_heads=8,              # Multi-head attention
    dim_feedforward=2048,     # Dimensione FFN intermedia
    dropout=0.1,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

print(f"Model parameters: {sum(p.numel() for p in decoder.parameters()):,}")
# ~120M parameters
```

### 2. Forward Pass (Training)

```python
# Memory: embedding multimodale dalla fusione dual-stream
# Shape: (batch_size=8, video_frames=100, hidden_dim=512)
memory = torch.randn(8, 100, 512, device=decoder.device)

# Target tokens: sequenza di testo con BOS offset a sinistra
# Shape: (batch_size=8, seq_len=50)
# IMPORTANTE: Deve includere BOS token all'inizio (tipicamente ID 1)
target_tokens = torch.randint(0, 10000, (8, 50), device=decoder.device)

# Forward pass
logits = decoder(
    memory=memory,
    target_tokens=target_tokens
)
# Output shape: (8, 50, 10000) - logit per ogni token nel testo

# Calcola la loss (cross-entropy)
import torch.nn.functional as F
target_flat = target_tokens.view(-1)
logits_flat = logits.view(-1, 10000)
loss = F.cross_entropy(logits_flat, target_flat)
print(f"Training loss: {loss.item():.4f}")
```

### 3. Generazione Autoregressiva (Inference)

```python
# Una volta allenato il modello, generare sequenze
decoder.eval()

with torch.no_grad():
    # Memory dal preprocessing (identico a training)
    memory = torch.randn(1, 100, 512, device=decoder.device)  # Batch size = 1

    # Genera tokens uno per uno (autoregressiva)
    generated_ids = decoder.generate(
        memory=memory,
        bos_token_id=1,      # Begin-of-sequence token ID
        eos_token_id=2,      # End-of-sequence token ID
        max_length=100,      # Massimo 100 token
        temperature=0.9,     # Poco di casualità
        top_k=None,          # Opzionale: top-k sampling
        top_p=None           # Opzionale: nucleus sampling
    )

    # generated_ids: (1, gen_length) dove gen_length ≤ 100
    print(f"Generated sequence length: {generated_ids.size(1)}")
```

---

## 🔗 Integrazione con Dual-Stream Upstream

### Architettura End-to-End

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT MULTIMEDIALE                       │
│         Landmarks (B, T, 2108)  |  Video (B, T, 3, H, W)   │
└──────────────────┬──────────────────────────────┬────────────┘
                   │                              │
        ┌──────────▼────────────┐      ┌──────────▼──────────┐
        │ Transformer Encoder    │      │  CNN-1D-GRU Module │
        │  (landmarks)           │      │  (video)           │
        │ (B, T, 2108) →        │      │ (B, T, 3, H, W) →  │
        │ (B, T, 512)           │      │ (B, T, 512)        │
        └──────────┬────────────┘      └──────────┬──────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  │
                        ┌─────────▼────────┐
                        │ Fusion Module     │
                        │ (B, T, 512)       │
                        └─────────┬────────┘
                                  │
                [MEMORY per Decoder]
                (B, T_video, 512)
                                  │
                        ┌─────────▼──────────────┐
                        │ Transformer Decoder    │ ← TU SEI QUI
                        │ Input:                 │
                        │ - Memory (da sopra)    │
                        │ - Target tokens        │
                        │                        │
                        │ Output:                │
                        │ - Logits per tokens    │
                        └─────────┬──────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │ Cross-Entropy Loss │
                        │                    │
                        │ Backprop & Update  │
                        └────────────────────┘
```

### Codice di Integrazione

```python
import torch
import torch.nn.functional as F
from src.models.cnn_1d_gru_integration import DualStreamSignLanguageModel
from src.models.transformer_decoder import TransformerDecoder

class SignLanguageTranslationModel(torch.nn.Module):
    """Modello completo: Dual-Stream → Decoder → Testo"""

    def __init__(
        self,
        # Dual-stream params
        video_hidden_dim=512,
        landmark_dim=2108,
        num_transformer_layers=4,

        # Decoder params
        vocab_size=10000,
        num_decoder_layers=4,

        device=torch.device('cuda')
    ):
        super().__init__()
        self.device = device

        # Encoder upstream (video + landmarks)
        self.encoder = DualStreamSignLanguageModel(
            hidden_dim=video_hidden_dim,
            landmark_dim=landmark_dim,
            num_transformer_layers=num_transformer_layers,
            use_classification_head=False,  # ← IMPORTANTE: Togli head
        ).to(device)

        # Decoder per generazione testo
        self.decoder = TransformerDecoder(
            hidden_dim=video_hidden_dim,
            vocab_size=vocab_size,
            num_decoder_layers=num_decoder_layers,
            device=device
        ).to(device)

    def forward(
        self,
        landmarks,
        video_frames,
        target_tokens
    ):
        """
        Forward pass completo (training).

        Args:
            landmarks: (B, T_video, 2108)
            video_frames: (B, T_video, 3, H, W)
            target_tokens: (B, T_text) con BOS offset

        Returns:
            logits: (B, T_text, vocab_size)
        """
        # Encoder: produce memoria multimodale
        memory = self.encoder.get_fused_representation(
            landmarks=landmarks,
            video_frames=video_frames
        )  # (B, T_video, 512)

        # Decoder: genera logit per testo
        logits = self.decoder(
            memory=memory,
            target_tokens=target_tokens
        )  # (B, T_text, vocab_size)

        return logits

    def generate(
        self,
        landmarks,
        video_frames,
        bos_token_id=1,
        eos_token_id=2,
        max_length=100
    ):
        """Generate text from video + landmarks."""
        with torch.no_grad():
            # Encode
            memory = self.encoder.get_fused_representation(
                landmarks=landmarks,
                video_frames=video_frames
            )

            # Decode (autoregressiva)
            generated = self.decoder.generate(
                memory=memory,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_length=max_length
            )

        return generated


# Utilizzo
model = SignLanguageTranslationModel(
    vocab_size=10000,
    device=torch.device('cuda')
)

# Training
landmarks = torch.randn(8, 100, 2108).cuda()
video = torch.randn(8, 100, 3, 224, 224).cuda()
target_text = torch.randint(0, 10000, (8, 50)).cuda()

logits = model(landmarks, video, target_text)
loss = F.cross_entropy(logits.view(-1, 10000), target_text.view(-1))
loss.backward()

# Inference
with torch.no_grad():
    generated = model.generate(landmarks, video)
    print(f"Generated IDs shape: {generated.shape}")
```

---

## 📊 Dettagli Architetturali

### Self-Attention Mascherata (Autoregressività)

La causal mask garantisce che durante il training il decoder non guardi i token futuri.

```
Sequenza target: [BOS, "the", "man", "walks", EOS]
                   0     1      2      3       4

Causal mask (quello che ogni token può guardare):
┌     ┐
│ 1 0 0 0 0 │  Token 0 (BOS) vede solo sé stesso
│ 1 1 0 0 0 │  Token 1 vede token 0 e 1
│ 1 1 1 0 0 │  Token 2 vede token 0, 1, 2
│ 1 1 1 1 0 │  Token 3 vede token 0, 1, 2, 3
│ 1 1 1 1 1 │  Token 4 vede tutti
└     ┘

Durante inference, si aggiunge un token per volta:
Passo 1: Decoder vede [BOS] → predice token 1
Passo 2: Decoder vede [BOS, token_1] → predice token 2
Passo 3: Decoder vede [BOS, token_1, token_2] → predice token 3
...

Questo garantisce che la generazione sia rigorosamente autoregressiva.
```

### Cross-Attention (Attending al Contesto Multimodale)

Ogni token nel target attende al memory (video + landmarks fusionati).

```
Memory (video + landmarks):
┌                        ┐
│ Frame 0 embedding      │  (1, 512)
│ Frame 1 embedding      │  (1, 512)
│ ...                    │
│ Frame 99 embedding     │  (1, 512)
└ Total: (B, 100, 512)  ┘

Target token embedding:
┌     ┐
│ BOS │  (1, 512)
│ "t" │  (1, 512)
│ "h" │  (1, 512)
└     ┘

Cross-Attention: Target attende a Memory
  - Query (Q): Da token corrente nel target
  - Key (K): Da tutti i frame nel memory
  - Value (V): Da tutti i frame nel memory

  → Output: Rappresentazione arricchita che combina:
      - Significato del token corrente (self-attn)
      - Contesto di tutti i frame video (cross-attn)
```

---

## 🎛️ Hyperparameter Tuning

### Configurazione Consigliata per Diversi Scenari

**Generazione rapida (inference veloce, minore qualità):**

```python
decoder = TransformerDecoder(
    hidden_dim=256,
    vocab_size=5000,
    num_decoder_layers=2,
    num_heads=4,
    dim_feedforward=1024,
    dropout=0.05
)
# ~10M parameters
```

**Standard (bilanciato):**

```python
decoder = TransformerDecoder(
    hidden_dim=512,
    vocab_size=10000,
    num_decoder_layers=4,
    num_heads=8,
    dim_feedforward=2048,
    dropout=0.1
)
# ~120M parameters
```

**Potente (migliore qualità, più lento):**

```python
decoder = TransformerDecoder(
    hidden_dim=1024,
    vocab_size=20000,
    num_decoder_layers=8,
    num_heads=16,
    dim_feedforward=4096,
    dropout=0.15
)
# ~800M parameters
```

---

## 💾 Salvataggio e Caricamento

```python
from src.models.transformer_decoder import TransformerDecoderConfig

# Salva modello e config
config = TransformerDecoderConfig(
    hidden_dim=512,
    vocab_size=10000,
    num_decoder_layers=4
)
config.save('decoder_config.json')
torch.save(decoder.state_dict(), 'decoder_weights.pt')

# Carica modello e config
config_loaded = TransformerDecoderConfig.load('decoder_config.json')
decoder_loaded = TransformerDecoder(**config_loaded.to_dict())
decoder_loaded.load_state_dict(torch.load('decoder_weights.pt'))
```

---

## 🔍 Debugging e Troubleshooting

### Memoria insufficiente (OOM)

```python
# Riduci dimensioni:
# 1. Diminuisci batch_size
# 2. Riduci seq_len (max_length during generation)
# 3. Usa hidden_dim=256 invece di 512
# 4. Usa num_decoder_layers=2 invece di 4

decoder = TransformerDecoder(
    hidden_dim=256,
    vocab_size=10000,
    num_decoder_layers=2,
    device=device
)
```

### Loss non scende / genera testo banale

```python
# Check:
# 1. Learning rate (prova 1e-4 o 1e-5)
# 2. Gradient clipping (torch.nn.utils.clip_grad_norm_)
# 3. Memory effettivamente processa il contesto?
#    → Inspect intermediate attention weights

# Debug:
decoder.eval()
with torch.no_grad():
    logits, _ = decoder(memory, target_tokens)
    # Stampa i token più probabili per ogni posizione
    top_k_tokens = torch.topk(logits, k=5, dim=-1)
```

---

## 📚 Prossimi Passi

1. **Allenamento**: Vedi `TRANSFORMER_DECODER_GUIDE.md` → Sezione "Training Loop Completo"
2. **Evaluation**: Metriche BLEU, METEOR, CER per testo generato
3. **Fine-tuning**: Su dati reali (How2Sign dataset)
4. **Ottimizzazione**: Quantization, distillation, model pruning

---

## 📖 Documentazione Completa

Vedi [TRANSFORMER_DECODER_GUIDE.md](TRANSFORMER_DECODER_GUIDE.md) per:

- Architettura dettagliata e formule
- Training loop completo con validazione
- Generazione con campionamento (temperature, top-k, nucleus)
- Best practices e ottimizzazioni
- Troubleshooting avanzato
