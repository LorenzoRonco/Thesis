# Thesis - Sign Language Recognition from Video

## Progetto

Modello di neural network per l'analisi di video di linguaggio dei segni (Sign Language) e la loro traduzione in testo, basato sul dataset **How2Sign**.

## Fasi del Progetto

### ✅ Fase 1: Video Segmentation (Preprocessing)

**Stato**: In progress

Segmentazione automatica dei video basandosi su file CSV con timestamp.

**Script disponibili**:

- `preprocess_videos.py` - Segmentazione video principale
- `analyze_dataset.py` - Analisi statistica del dataset
- `setup.py` - Verifica dipendenze

**Documentazione**: Vedi [VIDEO_SEGMENTATION.md](VIDEO_SEGMENTATION.md)

### ⏳ Fase 2: Feature Extraction

Estrazione di feature dai video segmentati (optical flow, pose estimation, hand tracking)

### ⏳ Fase 3: Model Training

Training del modello LSTM/Transformer per la traduzione

### ⏳ Fase 4: Evaluation & Deployment

Valutazione e deployment del modello

## Quick Start

```bash
# Setup iniziale (installa dipendenze)
python setup.py

# Analizza il dataset
python analyze_dataset.py

# Test su pochi video
python preprocess_videos.py test

# Segmenta tutti i video (una volta verificato il test)
python preprocess_videos.py full
```

## Struttura Cartelle

```
Thesis/
├── dataset/
│   ├── how2sign_realigned_train.csv       # Metadata con tempi segmentazioni
│   ├── How2Sign/                          # Video originali (da popolare)
│   └── segmented/                         # Video segmentati (output)
├── preprocess_videos.py                   # Script segmentazione
├── analyze_dataset.py                     # Script analisi
├── setup.py                               # Setup e verifiche
├── VIDEO_SEGMENTATION.md                  # Guida dettagliata preprocessamento
└── README.md
```

## Requisiti

- Python 3.8+
- ffmpeg (per segmentazione video)
- pandas (lettura CSV)
- OpenCV (opzionale)

Installa le dipendenze con:

```bash
pip install -r requirements.txt
```

## File del Progetto

### 📚 Documentazione

- [QUICKSTART.py](QUICKSTART.py) - Guida rapida (esegui: `python QUICKSTART.py`)
- [VIDEO_SEGMENTATION.md](VIDEO_SEGMENTATION.md) - Documentazione completa
- [config.py](config.py) - Configurazione personalizzabile

### 🐍 Script Principali

- [preprocess_videos.py](preprocess_videos.py) - Segmentazione video
- [analyze_dataset.py](analyze_dataset.py) - Analisi e visualizzazioni
- [diagnose.py](diagnose.py) - Diagnostica problemi
- [setup.py](setup.py) - Setup e verifica dipendenze

### 🚀 Launcher

- [run.bat](run.bat) - Menu interattivo (Windows)
- [run.sh](run.sh) - Menu interattivo (Linux/macOS)

### 📦 Configurazione

- [requirements.txt](requirements.txt) - Dipendenze Python
- [.env.example](.env.example) - Template variabili ambiente

## Quick Start

**Modalità rapida (Windows)**:

```bash
run.bat
```

**Modalità rapida (Linux/macOS)**:

```bash
bash run.sh
```

**Manuale**:

```bash
python setup.py          # Setup iniziale
python diagnose.py       # Verifica setup
python analyze_dataset.py # Analizza dataset
python preprocess_videos.py test  # Test
python preprocess_videos.py full  # Segmentazione completa
```

## Prossimi Step

1. [x] Creare script di segmentazione video
2. [x] Creare script di analisi dataset
3. [x] Creare diagnostica e documentazione
4. [ ] Scaricare video How2Sign
5. [ ] Testare pipeline su subset
6. [ ] Segmentare dataset completo
7. [ ] Implementare feature extraction
8. [ ] Trainare modello

---

👉 **Inizia con**: [run.bat](run.bat) o [QUICKSTART.py](QUICKSTART.py)
