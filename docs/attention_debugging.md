# Cross-Attention Debugging

This guide explains how to inspect the decoder's **cross-attention** weights. They show whether the decoder is actually "listening" to the encoder or ignoring it.

Implementation: [`transformer_only/attention_visualizer.py`](../transformer_only/attention_visualizer.py)

---

## How it works

During validation (every N epochs), for up to `attention_viz_max_batches` batches:

1. The model processes a batch in eval mode.
2. For every decoder layer, the cross-attention matrix of shape `(L, T)` is captured:
   - **X axis:** encoder positions (landmark frames)
   - **Y axis:** decoder positions (output tokens)
3. Each matrix is optionally saved as a heatmap.
4. Diagnostic statistics are printed to the console, including a padding-mask summary.

---

## Enabling it during training

Visualization is controlled by the `cfg` dictionary in [`transformer_only/train.py`](../transformer_only/train.py):

```python
"attention_viz_every":       None,   # set to an int (e.g. 5) to run every N epochs; None disables it
"attention_viz_max_batches": 2,      # validation batches inspected per run
"attention_viz_save":        False,  # True to also save PNG heatmaps (requires matplotlib)
```

Leave `attention_viz_every` set to `None` for the fastest training.

## Programmatic use

The visualizer can also be run on any checkpoint and dataloader built with [`phoenix_loader.py`](../transformer_only/data/phoenix_loader.py):

```python
from transformer_only.attention_visualizer import (
    debug_attention_on_batch,
    run_attention_visualization_on_loader,
)

# One batch
debug_attention_on_batch(model, batch, tokenizer, device, output_dir="outputs/attention_debug")

# Several batches from a loader
run_attention_visualization_on_loader(
    model, val_loader, tokenizer, device,
    output_dir="outputs/attention_debug",
    max_batches=2,
    save_heatmaps=True,
)
```

## Output

Heatmaps are written to `<output_dir>/attention/` (e.g. `outputs/phoenix_run1/attention/`):

```
epoch_005_batch_00_layer_00_attn.png
epoch_005_batch_00_layer_01_attn.png
...
```

Each PNG shows encoder positions on the X axis, decoder positions on the Y axis and the attention weight in `[0, 1]` as colour.

Example console output:

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
  Entropy ratio: 49.10%
  ✓ Attention focused on specific encoder positions

  Top 5 encoder positions per selected decoder tokens:
    Decoder pos   0: [enc[  0]=0.8432] [enc[ 15]=0.1032] [enc[  5]=0.0421] ...
```

If matplotlib is not installed, the statistics are still printed; only the heatmaps are skipped.

---

## Metrics

| Metric | Range | Meaning |
|---|---|---|
| Max weight | `[0, 1]` | `1.0` = fully focused on one encoder position; `≈1/T` = uniform |
| Mean entropy | `[0, log T]` | `0` = collapsed on one position; `log T` = uniform (encoder ignored) |
| Entropy ratio | `[0%, 100%]` | mean entropy / `log T` |

Rule of thumb for the entropy ratio:

- **< 10%** – focused (good)
- **40–60%** – moderately spread (normal)
- **> 80%** – almost uniform (problem)

---

## The three typical scenarios

### ✅ Focused attention (healthy)

```
Heatmap:       clear diagonal or structured bands
Max weight:    ~0.8–1.0
Entropy ratio: 10–30%
```

The decoder is learning to map specific encoder regions (hand / face landmarks) to specific output tokens.

### ❌ Uniform attention (encoder ignored)

```
Heatmap:       uniform grey over the whole map
Max weight:    ~1/T (e.g. ~0.01 for T=100)
Entropy ratio: > 90%
```

All encoder positions receive almost the same weight. The decoder is effectively generating text from its internal language model only. The console prints:

```
⚠️  ATTENZIONE QUASI UNIFORME! Il decoder potrebbe ignorare l'encoder!
```

### ❌ Collapsed attention (shortcut)

```
Heatmap:       a single bright column, everything else dark
Max weight:    ~0.99
Entropy ratio: < 5%
```

Every decoder query attends to the **same** encoder position, typically a padding position, a BOS/EOS-like position, or a fixed uninformative frame. The model has found a shortcut that bypasses the encoder content.

---

## Interpretation over training

- **Epoch 1:** near-uniform attention (random initialisation).
- **Epochs 5–10:** patterns (diagonal, bands) start to form.
- **Epoch 50+:** patterns should be stable if the model is learning.

Attention that is still uniform after ~50 epochs indicates a serious problem.

Typical healthy patterns:

1. **Main diagonal** – decoder position *i* mainly attends to a matching encoder region; indicates monotonic alignment.
2. **Horizontal bands** – a decoder position attends to a whole encoder region.
3. **Concentrated blocks** – specialisation on specific body parts (hands, face, body).

---

## Troubleshooting

| Symptom | Likely cause | What to try |
|---|---|---|
| Entropy ratio > 80–90% | Decoder ignores encoder | Increase LR, longer warmup, check `src_key_padding_mask` and input data |
| Single bright column | Collapse on one position | Check padding mask, verify landmarks are non-zero, increase LR, reduce label smoothing |
| Clean attention but low BLEU | Problem elsewhere | Check decoding, vocabulary, excessive label smoothing |
| Unusual pattern but BLEU improves | Valid non-obvious pattern | Keep training |
| `[WARNING] ALL positions are padding!` | Broken mask / empty sample | Check the dataloader output for that batch |
| Heatmaps not saved | `attention_viz_save` is `False`, matplotlib missing, or output dir not writable | Enable the flag, `pip install matplotlib`, check permissions |

## Performance impact

- Statistics capture: ~10–20 ms per batch (negligible).
- Heatmap rendering: ~100 ms per layer.
