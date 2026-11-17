# Frameworks e Dipendenze Usate nel Progetto

Questo documento riepiloga i principali framework, librerie e tool usati nel progetto e indica dove vengono utilizzati (script/file di riferimento).

## Principali librerie/framework

- **PyTorch (`torch`, `torchvision`, `torchaudio`)**: training dei modelli, definizione delle architetture Transformer, dataloaders e utilità di training.
  - File di riferimento: `transformer_only/train.py`, `transformer_only/train_stage2.py`, `transformer_only/two_stage.py`, `transformer_only/models/transformer.py`, `transformer_only/data/phoenix_loader.py`

- **MediaPipe (`mediapipe`, `mediapipe.tasks`)**: estrazione e processing dei landmark facciali e pose per feature extraction.
  - File di riferimento: `scripts/extract_landmarks.py` (uso diretto di MediaPipe Tasks e moduli di vision)

- **OpenCV (`cv2`, `opencv-python`)**: I/O video, lettura/scrittura frame, pre-elaborazione immagini/video.
  - File di riferimento: `scripts/extract_landmarks.py`, `scripts/inspect_landmarks.py`, preprocessing e script di utilità

- **NumPy (`numpy`)**: operazioni numeriche e strutture dati per array; usata trasversalmente in preprocessing, utils e modelli.
  - File di riferimento: molteplici, p.es. `transformer_only/*`, `scripts/*`

- **Pandas (`pandas`)**: lettura e manipolazione CSV (metadati, tabelle di segmentazione).
  - File di riferimento: `scripts/diagnose.py`, file di preprocessing che leggono i corpus

- **Matplotlib / Seaborn**: visualizzazione e diagnostica (plot dei dati, attention visualizer opzionale).
  - File di riferimento: `transformer_only/attention_visualizer.py`

- **scikit-learn (`scikit-learn`) e SciPy**: utilità per metriche, preprocessing statistico e funzioni di utilità.

- **ffmpeg / ffmpeg-python**: segmentazione ed elaborazione video (wrapper Python per ffmpeg). ffmpeg è richiesto come binario di sistema.

- **TensorBoard**: logging e visualizzazione training.

- **tqdm**: barre di progresso negli script di preprocessing e training.

- **pyyaml**: parsing/configuration dei file YAML usati per parametri di esperimenti.

- **transformers, sacrebleu**: utilità per metriche, evaluation e possibili componenti NLP (dichiarate in `requirements_ml.txt`).

## Note su installazione

- Vedi i file `requirements.txt` e `requirements_ml.txt` nella root del progetto per le versioni consigliate e l'elenco completo delle dipendenze.
- Alcune dipendenze (es. `ffmpeg`) richiedono installazione del binario di sistema oltre al pacchetto Python.

## Come verificare dove ogni libreria è usata

Per una lista esaustiva, puoi cercare le istruzioni `import` nel repository (es. `grep -R "import mediapipe" -n .`).

---

Se vuoi, posso:
- Aggiungere i comandi `pip install -r requirements.txt` e `pip install -r requirements_ml.txt` in un file README di setup.
- Generare un ambiente `requirements.lock` o `venv` con le versioni pinnate.
