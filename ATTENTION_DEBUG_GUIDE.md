# Cross-Attention Weight Visualization - Guida al Debug

## Cosa Verifica Questo Strumento?

Questo strumento cattura e visualizza i pesi di attenzione della **Cross-Attention** nel decoder. È un indicatore critico se il modello sta realmente "ascoltando" l'encoder o se lo sta ignorando.

## Come Funziona

Durante la validazione (ogni N epoch):
1. Il modello processa un batch completo
2. Per ogni layer del decoder, catturiamo la matrice di attenzione (L, T):
   - Assi X: posizioni dell'encoder (frame di landmarks)
   - Assi Y: posizioni del decoder (token di output)
3. Visualizziamo la matrice come heatmap
4. Stampiamo statistiche diagnostiche

## Cosa Cercare - I 3 Scenari

### ✅ SCENARIO CORRETTO: Attenzione Focalizzata

```
Heatmap: Diagonale chiara / Bande strutturate
Max weight: ~0.8-1.0
Entropy ratio: 10-30%
Diagnosi: ✓ Modello sta ascoltando l'encoder correttamente
```

**Cosa significa:** Il decoder sta apprendendo a mappare specifiche regioni dell'encoder (landmarks di mani/viso) a specifiche token di output (parole). Hai pattern visibili.

---

### ❌ SCENARIO CRITICO #1: Attenzione Uniforme

```
Heatmap: Colore grigio uniforme su tutta la mappa
Max weight: ~0.01 (molto basso)
Entropy ratio: > 90% (vicino 100%)
Diagnosi: ❌ DECODER STA IGNORANDO L'ENCODER
```

**Cosa significa:** Tutti i token encoder hanno pesi quasi uguali (1/T ≈ 0.01 per T=100 frame). È come se il decoder non stesse leggendo affatto l'encoder. Sta generando testo quasi casualmente basato solo sul suo linguaggio model interno.

**Come identificare nel log:**
```
⚠️  ATTENZIONE QUASI UNIFORME! Il decoder potrebbe ignorare l'encoder!
```

---

### ❌ SCENARIO CRITICO #2: Attenzione Collassata

```
Heatmap: Una singola colonna quasi bianca (tutti gli 0-token focalizzati su 1 encoder pos)
Max weight: ~0.99
Min weight: ~0.0
Entropy ratio: < 5%
Diagnosi: ❌ DECODER STA COLLASSANDO SU UN UNICO TOKEN
```

**Cosa significa:** Tutte le query del decoder stanno attendendo lo **stesso** token dell'encoder, tipicamente:
- Token di padding (token.pad_id)
- Token BOS/EOS
- Una posizione fissa non informativa

Il modello ha trovato una "scorciatoia" per ignorare il vero contenuto dell'encoder.

**Come identificare:** Una colonna grigio scuro, il resto nero.

---

## Metriche nel Riassunto

```python
Max weight: [0, 1]
  - 1.0 = attenzione perfettamente focalizzata su 1 token
  - 0.01 = attenzione distribuita uniformemente

Mean entropy: [0, log(T)]
  - 0 = attenzione su un unico token (collapsata)
  - log(T) = attenzione uniforme su tutti i token (ignora encoder)

Entropy ratio: (mean entropy) / log(T) → [0%, 100%]
  - < 10% = focalizzata (BUONO)
  - 40-60% = moderamente dispersa (NORMALE)
  - > 80% = quasi uniforme (MALE)
```

## Quando Abilitare il Debug

Modificare in `train_v2_optimized.py`:

```python
"debug_attention": True,      # Abilita visualizzazione
"debug_attention_interval": 5,  # Ogni 5 epoch
```

Disabilitare per training più veloce:
```python
"debug_attention": False,
```

## Output Generato

Salva heatmap in:
```
outputs/run2_optimized/attention_heatmaps/
├── batch_debug_layer_00_attn.png  # Layer 0 del decoder
├── batch_debug_layer_01_attn.png  # Layer 1 del decoder
└── ...
```

Ogni PNG mostra:
- **Assi X:** Encoder positions (frame di landmarks, 0-256)
- **Assi Y:** Decoder positions (token di output, 0-128)
- **Colore:** Peso di attenzione [0, 1]

## Interpretazione Avanzata

### Evoluzione Durante Training

**Epoch 1:** Attenzione quasi uniforme (modello inizializza casualmente)
**Epoch 5-10:** Iniziano a formarsi pattern (diagonale, bande)
**Epoch 50+:** Pattern stabile se il modello sta imparando

Se l'attenzione rimane uniforme dopo 50 epoch → problema serio.

### Pattern di Successo Tipici

1. **Diagonale principale:** Decoder position i attende principalmente encoder position i
   - Indica: sequenze short-range, preprocessing allineato

2. **Bande orizzontali:** Decoder position attende una regione dell'encoder
   - Indica: query dipende da multiple sorgenti

3. **Blocchi concentrati:** Decoder focalizzato su specifiche feature (mani, viso, corpo)
   - Indica: specializzazione per landmark type

## Debugging Rapido

Esegui una rapida ispezione senza training completo:

```python
from transformer_only.attention_visualizer import debug_attention_on_batch
from transformer_only.data.how2sign_loader import build_dataloaders
import torch

# Carica batch
loaders = build_dataloaders(...)
batch = next(iter(loaders["val"]))

# Visualizza
device = torch.device("cuda")
model.to(device)
debug_attention_on_batch(
    model, batch, tokenizer, device,
    output_dir="./attention_debug"
)
```

## Cosa Fare Se Trovi Problemi

**Se: Entropy ratio > 80%**
→ Prova: aumentare learning rate, provare warmup più lungo, verificare dati di input

**Se: Tutti i pesi su padding token**
→ Prova: verificare src_key_padding_mask, controllare che i landmarks siano non-zero

**Se: Pattern strano ma BLEU migliora**
→ OK: potrebbe essere un pattern valido non-ovvio

**Se: Attenzione perfetta ma BLEU basso**
→ Potrebbero essere altri problemi: decoding, vocabulary, label smoothing eccessivo
