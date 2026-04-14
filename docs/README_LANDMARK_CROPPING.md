# Video Landmark-Based Cropping per MobileNet

## 📋 Descrizione

Questo modulo implementa un pipeline completo per ritagliare video RGB usando landmarks normalizzati estratti tramite MediaPipe, con l'obiettivo di preparare i dati per l'integrazione con CNN (MobileNet) nel contesto della traduzione del linguaggio dei segni.

### Caratteristiche Principali

✅ **Denormalizzazione**: Converti landmarks da range [0,1] a coordinate pixel  
✅ **Bounding Box Dinamica**: Calcola automaticamente l'area di interesse per ogni frame  
✅ **Padding Intelligente**: Aggiungi margini (10-25%) attorno al bounding box  
✅ **Aspect Ratio Quadrato**: Forza il crop a essere quadrato (1:1)  
✅ **Temporal Smoothing**: Riduci lo "sfarfallamento" del crop tra frame  
✅ **Preprocessing MobileNet**: Resize a 224×224 + normalizzazione ImageNet  
✅ **Stato PyTorch-Ready**: Output come tensore torch.Tensor pronto per inferenza

---

## 📦 Installazione

### Prerequisiti

```bash
pip install opencv-python numpy torch torchvision scipy matplotlib
```

### Struttura del Progetto

```
scripts/
├── crop_video_with_landmarks.py      # Modulo principale
├── landmark_crop_example.ipynb        # Notebook di esempio
└── README_LANDMARK_CROPPING.md        # Questo file
```

---

## 🚀 Utilizzo Rapido

### Metodo 1: Interfaccia Semplificata (Uno-liner)

```python
from preprocessing.crop_video_with_landmarks import crop_video_with_landmarks
import numpy as np

# Carica landmarks normalizzati
landmarks = np.load("path/to/landmarks.npy")  # shape: (frames, num_landmarks, 2)

# Esegui cropping
cropped_tensor = crop_video_with_landmarks(
    video_path="path/to/video.mp4",
    landmarks=landmarks,
    output_path="output_cropped.mp4",  # Opzionale
    padding=0.15,                       # 15% padding
    target_size=(224, 224),             # MobileNet standard
    smoothing="global",                 # "global" o "moving"
    return_format="tensor"              # "tensor" o "array"
)

print(cropped_tensor.shape)  # (frames, 3, 224, 224)
```

### Metodo 2: Interfaccia Avanzata (Più Controllo)

```python
from crop_video_with_landmarks import VideoLandmarkCropper
import numpy as np

# Carica dati
landmarks = np.load("landmarks.npy")

# Crea istanza dello cropper
cropper = VideoLandmarkCropper(
    video_path="video.mp4",
    landmarks=landmarks,
    padding_percent=0.15,
    target_size=(224, 224),
    smoothing_method="global",
    window_size=5  # Per media mobile
)

# Processa video
cropped_tensor = cropper.process_video(
    output_video_path="cropped.mp4",  # Opzionale
    return_tensors=True               # Ritorna torch.Tensor
)

print(f"Cropped shape: {cropped_tensor.shape}")
print(f"Dtype: {cropped_tensor.dtype}")
print(f"Normalized for MobileNet: {cropped_tensor.min() < 0}")
```

---

## 📊 Dettagli Tecnici

### Flusso di Elaborazione

```
1. Denormalizzazione
   landmarks_normalized [0, 1] × (W, H) → landmarks_pixel

2. Bounding Box Dinamica
   Per ogni frame: (x_min, x_max, y_min, y_max) = argmin/argmax(landmarks)

3. Padding e Quadrato
   box_size = max(width, height) × (1 + padding_percent)
   Centra e forza aspect ratio 1:1

4. Temporal Smoothing
   - "global": Media su tutti i frame
   - "moving": Filter media mobile con finestra

5. Crop e Resize
   frame[y_min:y_max, x_min:x_max] → resize(224, 224)

6. Normalizzazione ImageNet
   (x - mean_imagenet) / std_imagenet
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]
```

### Shape dei Dati

**Input Landmarks:**

```
(frames, num_landmarks, 2)

Esempio:
- frames: 150
- num_landmarks: 2108 (MediaPipe Holistic: 33 pose + 468 face + 21*2 hands)
- dimensione: 2 (x, y)
- range: [0, 1] (NORMALIZZATO)
```

**Output Tensor:**

```
(frames, channels, height, width)
= (150, 3, 224, 224)

dtype: torch.float32
range: [-2.5, 2.6] (normalizzazione ImageNet applicata)
```

### Parametri di Cropping

| Parametro          | Default    | Descrizione                                   |
| ------------------ | ---------- | --------------------------------------------- |
| `padding_percent`  | 0.15       | Margine attorno alla bbox (0.1-0.25)          |
| `target_size`      | (224, 224) | Dimensioni output (standard MobileNet)        |
| `smoothing_method` | "global"   | "global" (stabile) o "moving" (dinamico)      |
| `window_size`      | 5          | Finestra media mobile (se smoothing="moving") |

---

## 💻 Riga di Comando

```bash
python crop_video_with_landmarks.py \
    path/to/video.mp4 \
    path/to/landmarks.npy \
    --output-video cropped.mp4 \
    --output-tensor cropped.pt \
    --padding 0.15 \
    --smoothing global \
    --target-size 224 224
```

---

## 📔 Notebook di Esempio

Per una guida interattiva completa, vedi `landmark_crop_example.ipynb`:

```bash
jupyter notebook landmark_crop_example.ipynb
```

Il notebook include:

1. Caricamento e visualizzazione dati
2. Denormalizzazione landmarks
3. Calcolo bounding box
4. Applicazione padding e quadrato
5. Temporal smoothing
6. Visualizzazione prima/dopo
7. Processing completo
8. Verifica normalizzazione ImageNet
9. Test inferenza MobileNet

---

## 🔧 Troubleshooting

### Problema: Crop vuoto o molto piccolo

```
Causa: Landmarks hanno valori NaN o fuori range
Soluzione:
  - Verifica che landmarks siano normalizzati [0, 1]
  - Controlla per NaN: np.isnan(landmarks).any()
  - Aumenta padding_percent (0.15 → 0.25)
```

### Problema: Crop "sfarfalla" troppo tra frame

```
Causa: Landmarks instabili frame-to-frame
Soluzione:
  - Usa smoothing_method="global" (media su intero video)
  - Se preferisci dinamica, usa "moving" con window_size=10
  - Aumenta padding_percent
```

### Problema: Errore di shape mismatch

```
Causa: Landmarks e numero di frame non corrispondono
Soluzione:
  - Verifica: len(landmarks) == video.total_frames
  - Se diverso, ricalcola landmarks con lo stesso video
```

### Problema: Output non normalizzato per MobileNet

```
Causa: return_format="array" mantiene [0, 255]
Soluzione:
  - Usa return_format="tensor" per auto-normalizzazione
  - O normalizza manualmente:
    x = (x - mean) / std
    con mean=[0.485, 0.456, 0.406]
         std=[0.229, 0.224, 0.225]
```

---

## 📈 Performance

**Complessità Computazionale:**

- Per ogni frame: O(num_landmarks) + O(resize) = O(n)
- Complessiva: O(frames × resize_cost)
- Numero frame: ~150, tempo stimato: 5-30 secondi GPU

**Memoria:**

- Input landmarks: ~1 MB (150 frame × 2108 × 2)
- Output tensor: ~200 MB (150 frame × 3 × 224 × 224 × 4 bytes)

**Best Practices:**

- Processa video in batch se possibile
- Usa GPU per resize se disponibile
- Per video molto lunghi, consideri slide-window approach

---

## 🔗 Integrazione con MobileNet

### Estrai features dalla CNN:

```python
from torchvision import models
import torch

# Carica MobileNet
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet.eval()

# Estrai features (prima della classificazione)
cropped_tensor = torch.load("cropped_tensor.pt")

with torch.no_grad():
    # Forward fino a prima della classificazione
    features = mobilenet.features(cropped_tensor)
    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
    features = torch.flatten(features, 1)

print(features.shape)  # (150, 1280) - MobileNet V2
```

---

## 📚 Formule Matematiche Chiave

### Denormalizzazione Landmarks:

$$\text{landmark}_{\text{pixel}} = \text{landmark}_{\text{norm}} \times [W, H]$$

### Bounding Box Dinamica:

$$x_{\min} = \min(\text{landmarks}[:, 0])$$
$$y_{\min} = \min(\text{landmarks}[:, 1])$$

### Padding e Quadrato:

$$\text{box\_size} = \max(\text{width}, \text{height}) \times (1 + \text{padding})$$

### Normalizzazione ImageNet:

$$x_{\text{norm}} = \frac{x_{\text{resized}} / 255 - \mu}{\sigma}$$

dove $\mu = [0.485, 0.456, 0.406]$ e $\sigma = [0.229, 0.224, 0.225]$

---

## 📝 Note Importanti

1. **Landmarks normalizzati**: Assicurati che i landmarks siano nel range [0, 1]
2. **Coordinata Y invertita**: OpenCV usa Y crescente verso il basso, come MediaPipe
3. **Gestione NaN**: La funzione usa `np.nanmin/nanmax` per ignorare landmark non válidi
4. **Thread-safe**: Il modulo non è thread-safe; crea istanze separate per thread
5. **ImageNet normalization**: Applicata automaticamente se `return_tensors=True`

---

## 📄 Licenza e riferimenti

- **MediaPipe**: Documentazione su landmarks: https://mediapipe.dev/
- **MobileNet**: Documentazione torchvision: https://pytorch.org/vision/
- **ImageNet Stats**: https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py

---

## 🤝 Supporto

Per problemi o suggerimenti:

1. Controlla il notebook di esempio: `landmark_crop_example.ipynb`
2. Consulta la sezione Troubleshooting
3. Verifica logs (impostare `logging.level=DEBUG`)

---

**Ultima modifica**: Aprile 2026  
**Versione**: 1.0
