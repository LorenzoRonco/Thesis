# Tecniche di ottimizzazione e training usate

Questo documento riassume le principali tecniche di ottimizzazione, stabilizzazione e riproducibilità utilizzate nel progetto.

**Scopo:** fornire spiegazioni concise, riferimenti al codice e alle impostazioni di configurazione usate per ciascuna tecnica.

**File principali di riferimento:** [transformer_only/train.py](transformer_only/train.py), [transformer_only/train_stage2.py](transformer_only/train_stage2.py)

--

- **Ottimizzatore `AdamW`:** utilizzo dell'ottimizzatore Adam con decoupled weight decay (`torch.optim.AdamW`).
  - Vantaggi: adaptive learning rates + corretto weight decay separato dal primo momento, utile per modelli Transformer.
  - Dove: in [transformer_only/train.py](transformer_only/train.py) e [transformer_only/train_stage2.py](transformer_only/train_stage2.py).
  - Config principali: `lr`, `weight_decay` (es. cfg["lr"], cfg["weight_decay"]).

- **Weight decay (decoupled):** termine di regolarizzazione passato a `AdamW` per limitare l'overfitting.

- **Warmup + Cosine LR schedule:** scheduler custom `WarmupCosineScheduler`.
  - Meccanismo: fase di warmup lineare per `warmup_steps`, seguita da decrescita coseno fino a `min_lr_ratio * base_lr`.
  - Dove: classe `WarmupCosineScheduler` in [transformer_only/train.py](transformer_only/train.py).
  - Config principali: `warmup_steps`, `epochs` (per calcolo `total_steps`), `min_lr_ratio`.

- **Mixed-precision (AMP):** uso di `torch.cuda.amp.autocast` e `GradScaler` per ridurre memoria e accelerare i passaggi.
  - Vantaggi: velocizza training e riduce consumo memoria su GPU mantenendo stabilità numerica tramite `GradScaler`.
  - Dove: import e uso in [transformer_only/train.py](transformer_only/train.py) e [transformer_only/train_stage2.py](transformer_only/train_stage2.py).
  - Config principale: `use_amp` (abilita/disabilita AMP).

- **Clipping dei gradienti:** `torch.nn.utils.clip_grad_norm_` per limitare la norma dei gradienti e prevenire esplosione.
  - Dove: applicato prima dello step dell'ottimizzatore nelle routine di training (es. `clip_norm` in cfg).
  - Config principale: `clip_norm`.

- **Label smoothing:** smoothing della cross-entropy (valore di default `0.1` nel cfg).
  - Vantaggi: migliora generalizzazione e allevia over-confidence del modello.
  - Dove: configurazione `label_smoothing` e passaggio al modello.

- **Loss composita: CE + CTC (pesatura):** combinazione di Cross-Entropy e CTC con coefficiente `lambda_ctc`.
  - Scopo: aggiungere un termine CTC ausiliario per migliorare allineamento e ridurre inserzioni; pesatura controllata da `lambda_ctc`.
  - Dove: calcolo in `_compute_loss` in [transformer_only/train.py](transformer_only/train.py).
  - Config principale: `lambda_ctc`.

- **Data augmentation per landmark:** tecniche usate per robustezza (rumore gaussiano, frame dropout, speed perturbation, flip orizzontale).
  - Dove: funzione `augment_landmarks` in [transformer_only/train.py](transformer_only/train.py) (parametri nel `cfg` come `aug_noise_std`, `aug_frame_drop_prob`, `aug_speed_min`/`max`, `aug_hflip_prob`).

- **Controllo riproducibilità:** insieme di fix per rendere gli esperimenti riproducibili.
  - Azioni: seed globale (`random`, `numpy`, `torch`, `torch.cuda`), `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
  - DataLoader: uso di `torch.Generator` + `worker_init_fn` per seedare i worker.
  - Dove: funzioni `set_global_seed` e `_make_worker_init_fn` in [transformer_only/train.py](transformer_only/train.py) e in [transformer_only/train_stage2.py](transformer_only/train_stage2.py).

- **Batching / Bucketing:** uso di bucketing per ridurre padding e stabilizzare gli aggiornamenti su sequenze di diversa lunghezza.
  - Parametro: `use_bucketing`, `bucket_size`.

- **Checkpointing selettivo:** salvataggio del miglior modello basato su BLEU-4 (metrica scelta) e salvataggio finale di ultima epoca; contiene `optimizer.state_dict()`, `scheduler.state_dict()`, `scaler.state_dict()` per riprendere esattamente il training.

--

Note aggiuntive e riferimenti rapidi:

- Configurazioni rilevanti in `cfg` (esempi): `lr`, `weight_decay`, `clip_norm`, `warmup_steps`, `use_amp`, `label_smoothing`, `lambda_ctc`.
- Per vedere le implementazioni e i commenti: leggere [transformer_only/train.py](transformer_only/train.py) e [transformer_only/train_stage2.py](transformer_only/train_stage2.py).

Se vuoi, posso:
- aggiungere esempi di valori consigliati per ciascuna tecnica;
- generare una tabella comparativa (pro/contro) o una versione in inglese;
- aggiungere riferimenti bibliografici/URL per `AdamW`, warmup+cosine, AMP, ecc.

Fine del documento.
