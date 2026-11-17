# Cross-Attention Debugging Toolkit

Strumento per diagnosticare problemi di attenzione nel decoder di un Transformer.

## Files

- **`attention_visualizer.py`** - Modulo di visualizzazione (importato automaticamente)
- **`quick_attention_debug.py`** - Script di debug rapido standalone
- **`ATTENTION_DEBUG_GUIDE.md`** - Guida completa di interpretazione

## Uso Rapido

### 1. Debug Durante Training (Automatico)

Il training script `train_v2_optimized.py` visualizza automaticamente i pesi ogni N epoch:

```bash
python -m transformer_only.train_v2_optimized
```

Disabilita per velocità:
```python
# In train_v2_optimized.py
"debug_attention": False,
```

### 2. Debug Rapido su Checkpoint Esistente

```bash
# Con checkpoint addestrato
python quick_attention_debug.py --checkpoint outputs/run2_optimized/best.pt

# Con modello random (baseline)
python quick_attention_debug.py --random

# Custom output
python quick_attention_debug.py \
    --checkpoint outputs/run2_optimized/best.pt \
    --output_dir my_debug_folder
```

### 3. Debug Programmatico

```python
from transformer_only.attention_visualizer import debug_attention_on_batch
from transformer_only.data.how2sign_loader import build_dataloaders
import torch

# Carica dati
loaders = build_dataloaders(...)
batch = next(iter(loaders["val"]))

# Visualizza
device = torch.device("cuda")
model.to(device)

debug_attention_on_batch(
    model, batch, tokenizer, device,
    output_dir="attention_results"
)
```

## Output

```
outputs/run2_optimized/attention_heatmaps/
├── batch_debug_layer_00_attn.png
├── batch_debug_layer_01_attn.png
├── batch_debug_layer_02_attn.png
├── batch_debug_layer_03_attn.png
├── batch_debug_layer_04_attn.png
└── batch_debug_layer_05_attn.png
```

Console output:
```
================================================================================
CROSS-ATTENTION WEIGHTS SUMMARY
================================================================================

Layer 0:
  Shape: (L=27, T=118)
  Max weight: 0.8432
  Min weight: 0.0012
  Mean weight: 0.0350
  Mean entropy: 2.3421 (max: 4.7693)
  Entropy ratio: 49.10% (50% = uniforme)
  ✓ Attenzione focalizzata su specifiche posizioni encoder

  Top 5 encoder positions per selected decoder tokens:
    Decoder pos   0: [enc[  0]=0.8432] [enc[ 15]=0.1032] [enc[  5]=0.0421] ...
    Decoder pos   6: [enc[  6]=0.7621] [enc[  7]=0.1345] [enc[  8]=0.0812] ...
    ...
```

## Checklist di Interpretazione

Quando leggi i risultati, controlla nell'ordine:

1. **Entropy ratio**
   - < 10% → Attenzione focalizzata (buono)
   - 40-60% → Moderato (normale)
   - > 80% → Quasi uniforme (problema!)

2. **Heatmap visiva**
   - Diagonale/bande → Allineamento sequenziale (buono)
   - Grigio uniforme → Ignora encoder (male!)
   - Una colonna bianca → Collassata su 1 token (male!)

3. **Top encoder positions**
   - Variano per cada decoder position → Buono
   - Sempre lo stesso (es. 0) → Collasso

## Diagnosi Comuni

| Sintomo | Causa | Soluzione |
|---------|-------|-----------|
| Entropy ratio > 90% | Decoder ignora encoder | ↑ LR, ↑ warmup, check padding mask |
| Una colonna bianca | Collasso su 1 token | Check padding, ↑ LR, ↓ label smoothing |
| Attenzione perfetta ma BLEU basso | Altro problema | Check decoding, vocabulary |
| Pattern strano ma BLEU migliora | Valido non-ovvio | Continua training |

## Troubleshooting

### Errore: "matplotlib not installed"
```bash
pip install matplotlib
```

### Errore: "KeyError: 'src_key_padding_mask'"
Il batch non contiene le maschere. Verificare `build_dataloaders()`.

### Le immagini non vengono salvate
Verificare che `output_dir` sia scrivibile:
```bash
mkdir -p outputs/attention_debug
```

## Performance Impact

- **Durante training:** ~10-20ms per batch (trascurabile)
- **Debug rapido:** ~1-5 secondi per batch
- **Visualizzazione heatmap:** ~100ms per layer

## Environment Requirements

```
torch >= 1.9
numpy
matplotlib (opzionale, per heatmap)
```

Se matplotlib non è installato, il tool stampa comunque le statistiche testuali.
