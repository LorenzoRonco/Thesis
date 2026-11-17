# Video Segmentation - Preprocessing

## Overview

Questo modulo prepara automaticamente i video della How2Sign Dataset segmentandoli basandosi sui tempi forniti nel file CSV. Ogni segmentazione corrisponde a una frase nel linguaggio dei segni.

## Struttura Cartelle

```
Thesis/
├── dataset/
│   ├── how2sign_realigned_train.csv       # Metadata con tempi di segmentazione
│   ├── How2Sign/                          # Video originali (da popolare)
│   └── segmented/                         # Output - video segmentati (creato automaticamente)
├── preprocess_videos.py                   # Script principale di segmentazione
├── setup.py                               # Setup e verifica dipendenze
└── README.md
```

## Setup Iniziale

### 1. Installa le dipendenze

```bash
python setup.py
```

Questo installerà automaticamente:
- `pandas` - Per lettura CSV
- `ffmpeg` - Per segmentazione video (necessario!)
- `opencv-python` - Opzionale, per operazioni avanzate su frame

### 2. Scarica i video

I video di How2Sign devono essere scaricati da [How2Sign Dataset](https://how2sign.github.io/) e inseriti nella cartella `dataset/How2Sign/`.

Il nome dei file dovrà corrispondere alla colonna `VIDEO_NAME` nel CSV.

### 3. Verifica la struttura

```bash
python setup.py
```

Se tutti i file e le cartelle sono al posto giusto, sei pronto!

## Utilizzo

### Modalità TEST (Prime 2 video - Consigliato prima di fare il run completo)

```bash
python preprocess_videos.py test
```

Questa modalità:
- Processa solo i primi 2 video distinti
- Utile per verificare che tutto funzioni
- Consente di identificare problemi prima del processamento completo

### Modalità FULL (Tutti i video)

```bash
python preprocess_videos.py full
```

Questa modalità:
- Processa TUTTI i video dal CSV
- Crea tutti i segmenti
- Può richiedere molto tempo a seconda del numero di video

## Output

Dopo l'esecuzione, nella cartella `dataset/segmented/` troverai:

```
segmented/
├── VIDEO_ID_1_SENTENCE_ID_1-5-rgb_front.mp4
├── VIDEO_ID_1_SENTENCE_ID_2-5-rgb_front.mp4
├── VIDEO_ID_2_SENTENCE_ID_1-5-rgb_front.mp4
└── ...
```

### Formato dei nomi file

- **SENTENCE_ID**: ID univoco della frase  
- **SENTENCE_NAME**: Identificatore della vista del video (es. "-5-rgb_front")
- **.mp4**: Formato video (H.264 video codec, non riencodato per velocità)

**Ogni segmento video contiene solo la porzione temporale specificata nel CSV.**

## Statistiche Dataset

Il script mostra automaticamente:

```
STATISTICHE DATASET
============================================================
Segmentazioni totali: 51,312
Video unici: 2,267
Frasi totali: 10,264

Durata medio segmentazione: 4.23s
Durata min segmentazione: 0.23s
Durata max segmentazione: 42.15s
============================================================
```

## Troubleshooting

### "ffmpeg non trovato"

**Soluzione**: Installa ffmpeg:
- **Windows**: `choco install ffmpeg` o scarica da ffmpeg.org
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### "Video non trovato"

Verifica che:
1. I file video siano nella cartella `dataset/How2Sign/`
2. I nomi corrispondano a quelli nel CSV (colonna `VIDEO_NAME`)
3. L'estensione sia tra quelle supportate

### "Timeout durante segmentazione"

Significa che il video è molto grande. Aumenta il timeout in `preprocess_videos.py` (riga ~130):
```python
timeout=600  # Da 300 a 600 secondi
```

### Il processo è lento

Questo è normale! ffmpeg non riencode i video ma comunque:
- Decodifica e riencodifica richiede I/O
- Segmenti lunghi richiedono più tempo
- Esegui in modalità test prima

**Tip**: Se hai molti video, puoi eseguire script multipli in parallelo modificando `process_sample()`.

## API Script

### Classe `VideoSegmenter`

```python
from preprocess_videos import VideoSegmenter

# Inizializza
segmenter = VideoSegmenter(
    csv_path="dataset/how2sign_realigned_train.csv",
    video_dir="dataset/How2Sign",
    output_dir="dataset/segmented"
)

# Processa
success, errors = segmenter.process_all()

# O un sample
success, errors = segmenter.process_sample(num_videos=5)
```

### Metodi

- `process_all()` - Processa tutti i video
- `process_sample(num_videos: int)` - Processa numero di video specificato
- `segment_video(...)` - Segmenta un singolo video
- `get_statistics()` - Stampa statistiche
- `_find_video_file(video_name)` - Trova file video per nome

## Output per il Neural Network

Dopo questa fase, avrai:

✓ 50,000+ video segmentati (uno per ogni frase)
✓ Ognuno pronto per essere processato dal modello
✓ Struttura ordinata in disco

Prossimamente:
- [ ] Feature extraction da video (optical flow, pose estimation)
- [ ] Loading batch per training
- [ ] Modello LSTM/Transformer per traduzione

## Note

- Il CSV contiene solo il dataset di training
- Considera di segmentare anche test/validation set se disponibili
- Backup i video originali prima di elaborarli
- La segmentazione non modifica i video originali

---

Per dubbi o problemi: vedi i log di esecuzione per dettagli

## Frameworks e dipendenze

Questo progetto utilizza diversi framework e librerie per il preprocessing, l'estrazione dei landmark e l'addestramento dei modelli. Per un elenco completo e per capire dove ciascuna libreria viene usata, vedi il file di riepilogo:

- `docs/FRAMEWORKS_USED.md`

