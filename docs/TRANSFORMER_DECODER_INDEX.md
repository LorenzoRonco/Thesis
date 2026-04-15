# Transformer Decoder - Indice Completo della Documentazione

## 📚 Cosa è Stato Implementato

Una **architettura Transformer Decoder autoregressiva** completa per Sign Language Translation (Video → Text), integrata con il tuo modello dual-stream upstream.

---

## 📁 File Creati

### Implementazione

| File                                                                                          | Descrizione                                                                                                                                                                                                                                                                                         |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`src/models/transformer_decoder.py`](../src/models/transformer_decoder.py)                   | **Core decoder** - Classe principale TransformerDecoder con: <br> - Token embedding e positional encoding <br> - Causal mask per autoregressività <br> - Multi-head self & cross-attention <br> - Generazione con temperature/top-k/nucleus sampling <br> - Configuration class per serializzazione |
| [`src/models/transformer_decoder_examples.py`](../src/models/transformer_decoder_examples.py) | **5 esempi progressivi**: <br> 1. Basic forward pass <br> 2. Training loop <br> 3. Autoregressive generation <br> 4. Dual-stream integration <br> 5. Full pipeline con validation                                                                                                                   |
| [`src/models/end_to_end_seq2seq.py`](../src/models/end_to_end_seq2seq.py)                     | **Modello end-to-end completo** con: <br> - SignLanguageTranslationModel (Encoder + Decoder) <br> - Seq2SeqTrainer (training loop) <br> - Esempio di utilizzo completo                                                                                                                              |

### Documentazione

| File                                       | Descrizione                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [TRANSFORMER_DECODER_README.md](#readmemd) | **Quick start** (10 minuti) - Come usare il decoder subito <br> - Setup minimale <br> - Forward pass training <br> - Generazione inferenza <br> - Integrazione dual-stream <br> - Hyperparameter tuning                                                                                                                                                                                           |
| [TRANSFORMER_DECODER_GUIDE.md](#guideemd)  | **Guida tecnica dettagliata** <br> - Architettura con diagrammi <br> - Dimensioni passaggio per passaggio <br> - Formule matematiche (PE, attention, FFN) <br> - Masked self-attention (autoregressività) <br> - Cross-attention (video context) <br> - Training loop completo (codice) <br> - Generazione autoregressiva (passo per passo) <br> - Best practices <br> - Troubleshooting avanzato |

---

## 🎯 Caratteristiche Implementate

### 1. **Positional Encoding Sinusoidale**

```
Codifica la posizione di ogni token nella sequenza di testo
Permet al modello di capire l'ordine sequenziale
```

### 2. **Causal Mask (Autoregressività Garantita)**

```
Impedisce che il token alla posizione t veda i token futuri t+1, t+2, ...
Garantisce generazione rigorosamente autoregressiva durante training
```

### 3. **Multi-Head Self-Attention (con Masking)**

```
Self-attention sulla sequenza target
Con causal mask per autoregressività
Cada token attende solo ai token precedenti (incluso sé stesso)
```

### 4. **Cross-Attention (Contesto Multimodale)**

```
Il decoder attende ai frame video (memory dall'upstream)
Dire query: sequenza target (testo)
Key/Value: memoria multimodale fusa (video + landmarks)

Permette al testo generato di essere consapevole del contesto visivo
```

### 5. **Feed-Forward Networks**

```
2-layer dense network (512 → 2048 → 512)
ReLU activation
Residual connections e layer normalization
```

### 6. **Generazione Autoregressiva con Campionamento**

```
Temperature scaling: controlla randomicità
Top-k sampling: sampleziona solo dai top-k token
Nucleus (Top-p) sampling: sampleziona da tokens fino a cumulative prob p
```

### 7. **Serializzazione Configurazione**

```
TransformerDecoderConfig per salvare/caricare hyperparameter in JSON
Facilita reproducibility e condivisione modelli
```

---

## 🚀 Quick Start (5 minuti)

### Installation-Free (Usa direttamente)

```python
import torch
from src.models.transformer_decoder import TransformerDecoder

# Crea decoder
decoder = TransformerDecoder(
    hidden_dim=512,
    vocab_size=10000,
    num_decoder_layers=4
)

# Memory da upstream (e.g., fusion di video + landmarks)
memory = torch.randn(8, 100, 512)  # (batch, video_frames, hidden_dim)

# Target tokens (dall'inizio con BOS token)
target_tokens = torch.randint(0, 10000, (8, 50))
target_tokens[:, 0] = 1  # BOS

# Forward pass
logits = decoder(memory=memory, target_tokens=target_tokens)
# Output: (8, 50, 10000) logit per ogni token nel testo

# Loss
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, 10000),
    target_tokens.view(-1)
)
```

### Generazione

```python
decoder.eval()
with torch.no_grad():
    generated = decoder.generate(
        memory=memory,
        bos_token_id=1,
        eos_token_id=2,
        max_length=100,
        temperature=0.9
    )
```

---

## 📊 Dimensioni dei Tensori

### Training Flow

```
INPUT:
  Landmarks:     (B=8, T_video=100, 2108)
  Video:         (B=8, T_video=100, 3, 224, 224)

ENCODER (Dual-Stream):
  Memory:        (B=8, T_video=100, 512)

DECODER:
  Token Embed:   (B=8, T_text=50, 512)
  Self-Attn:     (B=8, T_text=50, 512)
  Cross-Attn:    Query: (B=8, T_text=50, 512) × Key/Val: (B=8, T_video=100, 512)
  Output:        (B=8, T_text=50, 512)

OUTPUT:
  Logits:        (B=8, T_text=50, vocab_size=10000)

LOSS:
  Cross-Entropy  scalar
```

---

## 🔄 Pipeline Completo

### 1. Training

```python
from src.models.end_to_end_seq2seq import SignLanguageTranslationModel, Seq2SeqTrainer, Seq2SeqConfig

# Create model
config = Seq2SeqConfig()
model = SignLanguageTranslationModel(**config.__dict__)

# Define trainer
trainer = Seq2SeqTrainer(model, config, device)

# Train
for epoch in range(config.num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)
    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

# Save best model
trainer.save_checkpoint('best_model.pt')
```

### 2. Inference (Generazione)

```python
model.eval()
with torch.no_grad():
    generated_ids = model.generate(
        landmarks=test_landmarks,  # (1, 100, 2108)
        video_frames=test_video,   # (1, 100, 3, 224, 224)
        max_length=128,
        temperature=0.9,
        top_k=50
    )

# generated_ids: (1, sequence_length)
# Converti a testo usando il vocabolario inverso
predicted_text = id_to_text(generated_ids[0])
```

---

## ✅ Checklist di Utilizzo

### Per Il Training

- [ ] Installa PyTorch con CUDA (se GPU disponibile)
- [ ] Prepara dataset con:
  - [ ] Landmarks (B, T, 2108) normalizzati
  - [ ] Video RGB (B, T, 3, 224, 224)
  - [ ] Target text tokens (B, T_text) con BOS offset
- [ ] Crea DataLoader
- [ ] Istanzia SignLanguageTranslationModel
- [ ] Crea Seq2SeqTrainer
- [ ] Chiama trainer.train_full()

### Per L'Inference

- [ ] Carica modello allenato
- [ ] Prepara input (landmarks + video)
- [ ] Chiama model.generate()
- [ ] Converti token IDs a testo
- [ ] Visualizza/salva risultati

---

## 🎓 Concetti Chiave Spiegati

### Causal Masking

Il modello deve imparare a generare testo **autoregressivamente** (token per token, condizionato solo ai precedenti).

Durante il training con ground truth:

- Posizione 0 (BOS) → predice token 1, non vede token 1
- Posizione 1 → predice token 2, vede token 0 e 1
- Posizione 2 → predice token 3, vede token 0, 1, 2

Questo è forzato dalla **causal mask** che blocca l'attenzione sopra la diagonale.

### Cross-Attention

L'attenzione tra il testo generato e il video context:

```
Per ogni token nel testo generato:
  ↓
Cerca informazioni rilevanti in TUTTI i frame video
  ↓
Accantona l'attenzione sui frame che contengono gesti rilevanti
```

Questo permettono al decoder di "vedere" cosa succede nel video e generare testo coerente.

### Generazione Autoregressiva

Ad inference time:

```
Step 1: Input [BOS] → Decoder → Predi token 1
Step 2: Input [BOS, token_1] → Decoder → Predi token 2
Step 3: Input [BOS, token_1, token_2] → Decoder → Predi token 3
...
Step N: Predice EOS (End-Of-Sequence) → STOP

Output: [BOS, token_1, token_2, ..., EOS]
```

Convertendo IDs a parole si ottiene il testo finale.

---

## 📖 Documentazione Dettagliata

Vedi:

- **[TRANSFORMER_DECODER_README.md](TRANSFORMER_DECODER_README.md)** - Quick start
- **[TRANSFORMER_DECODER_GUIDE.md](TRANSFORMER_DECODER_GUIDE.md)** - Tecnica completa

Per esempi pratici:

- **[transformer_decoder_examples.py](../src/models/transformer_decoder_examples.py)** - 5 esempi
- **[end_to_end_seq2seq.py](../src/models/end_to_end_seq2seq.py)** - Pipeline completo

---

## 🔧 Configurazioni Consigliate

### Per Dispositivi con GPU Limitata (< 8GB VRAM)

```python
decoder = TransformerDecoder(
    hidden_dim=256,
    vocab_size=5000,
    num_decoder_layers=2,
    num_heads=4,
    dim_feedforward=1024,
    dropout=0.05,
)
```

### Standard (Bilanciato)

```python
decoder = TransformerDecoder(
    hidden_dim=512,
    vocab_size=10000,
    num_decoder_layers=4,
    num_heads=8,
    dim_feedforward=2048,
    dropout=0.1,
)
```

### Per Migliore Qualità (GPU potenti)

```python
decoder = TransformerDecoder(
    hidden_dim=1024,
    vocab_size=20000,
    num_decoder_layers=8,
    num_heads=16,
    dim_feedforward=4096,
    dropout=0.15,
)
```

---

## 🐛 Troubleshooting Veloce

| Problema                   | Soluzione                             |
| -------------------------- | ------------------------------------- |
| Loss non scende            | ↑ learning rate, verifica dati        |
| OOM                        | ↓ batch_size, ↓ hidden_dim, ↓ seq_len |
| Generazione ripete         | ↑ temperature, ↑ top_k                |
| Attenzione non convergente | Verifica causal mask applicata        |

Vedi [TRANSFORMER_DECODER_GUIDE.md](TRANSFORMER_DECODER_GUIDE.md#troubleshooting) per troubleshooting dettagliato.

---

## 📝 Prossimi Passi

1. **Allenamento su dati reali** - How2Sign dataset
2. **Evaluation** - Metriche (BLEU, METEOR, CER)
3. **Fine-tuning** - Su annotazioni reali del testo
4. **Ottimizzazione** - Quantization, distillation, pruning
5. **Deploy** - Inference ottimizzato su CPU/mobile

---

## 📚 Riferimenti

- Vaswani et al., "Attention is All You Need" (2017)
- PyTorch Transformer Documentation
- Your Dual-Stream Implementation (cnn_1d_gru_integration.py)

---

## 👨‍💼 Supporto

Per domande o chiarimenti, fare riferimento a:

- **TRANSFORMER_DECODER_GUIDE.md** - Documentazione tecnica
- **transformer_decoder_examples.py** - Esempi di codice
- **end_to_end_seq2seq.py** - Pipeline completo

---

**Implementazione completata il: 2026-04-14**

✨ Buona fortuna con il tuo modello di Sign Language Translation!
