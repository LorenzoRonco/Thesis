"""
Configurazione per Video Segmentation

Modifica questo file per personalizzare il comportamento degli script.
"""

import json
from pathlib import Path


# ==================== PERCORSI ====================
DATASET_DIR = Path(__file__).parent / "dataset"
CSV_PATH = DATASET_DIR / "how2sign_realigned_train.csv"
VIDEO_DIR = DATASET_DIR / "How2Sign"
OUTPUT_DIR = DATASET_DIR / "segmented"
ANALYSIS_DIR = Path(__file__).parent / "analysis"

# ==================== VIDEO ====================

# Estensioni supportate
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

# Timeout per ffmpeg (secondi)
FFMPEG_TIMEOUT = 300  # 5 minuti

# Codec settings per segmentazione
# Opzioni: 'copy' (veloce, no riencode) oppure 'h264', 'libx264', etc.
VIDEO_CODEC = 'copy'  # Consigliato: non riencode
AUDIO_CODEC = 'copy'  # Consigliato: non riencode

# Overwrite esistenti
OVERWRITE_EXISTING = True

# ==================== LOGGING ====================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True
LOG_FILE = Path(__file__).parent / "logs" / "segmentation.log"

# ==================== ANALISI ====================

# Numero di video per sample
SAMPLE_VIDEO_COUNT = 2

# Numero di bin per istogrammi
HISTOGRAM_BINS = 50

# Crea visualizzazioni automaticamente
GENERATE_VISUALIZATIONS = True

# DPI per immagini
FIGURE_DPI = 150

# ==================== PROCESSING ====================

# Numero massimo di segmentazioni per iterazione (0 = infinito)
MAX_SEGMENTATIONS_PER_BATCH = 0

# Mostra informazioni dettagliate durante processing
VERBOSE = True

# ==================== DATABASE (per future expansions) ====================

# Tipo database per tracking: 'sqlite', 'json', None
TRACK_DATABASE = 'json'  # Salva stato elaborazione

# Nome file tracking
TRACKING_FILE = Path(__file__).parent / "processing_state.json"

# ==================== FUNZIONI HELPER ====================


def get_config_dict():
    """Restituisce la configurazione come dizionario."""
    return {
        'paths': {
            'dataset_dir': str(DATASET_DIR),
            'csv_path': str(CSV_PATH),
            'video_dir': str(VIDEO_DIR),
            'output_dir': str(OUTPUT_DIR),
            'analysis_dir': str(ANALYSIS_DIR),
        },
        'video': {
            'supported_extensions': list(SUPPORTED_VIDEO_EXTENSIONS),
            'timeout': FFMPEG_TIMEOUT,
            'video_codec': VIDEO_CODEC,
            'audio_codec': AUDIO_CODEC,
            'overwrite': OVERWRITE_EXISTING,
        },
        'analysis': {
            'sample_count': SAMPLE_VIDEO_COUNT,
            'histogram_bins': HISTOGRAM_BINS,
            'generate_visualizations': GENERATE_VISUALIZATIONS,
            'figure_dpi': FIGURE_DPI,
        }
    }


def save_config_to_file(output_file=None):
    """Salva la configurazione su file JSON."""
    if output_file is None:
        output_file = Path(__file__).parent / "config_current.json"
    
    config = get_config_dict()
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configurazione salvata: {output_file}")


def print_config():
    """Stampa la configurazione corrente."""
    config = get_config_dict()
    print("\n" + "="*70)
    print("CONFIGURAZIONE CORRENTE")
    print("="*70)
    for section, values in config.items():
        print(f"\n{section.upper()}")
        print("-" * 70)
        for key, value in values.items():
            print(f"  {key:30} : {value}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_config()
