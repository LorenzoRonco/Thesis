"""
Setup e Utility per il preprocessing dei video
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Verifica che tutte le dipendenze siano installate."""
    print("Verifica dipendenze...\n")
    
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'cv2': 'opencv-python'  # Opzionale ma utile
    }
    
    missing = []
    
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name if module_name != 'cv2' else 'cv2')
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (MANCANTE)")
            missing.append(package_name)
    
    if missing:
        print(f"\nInstallazione pacchetti mancanti...")
        for package in missing:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Errore nell'installazione di {package}: {e}")
        print("✓ Pacchetti installati (con eventuali errori sopra)")
    else:
        print("\n✓ Tutte le dipendenze Python presenti")
    
    # Verifica ffmpeg separatamente
    print("\nVerifica ffmpeg (sistema)...")
    try:
        subprocess.run(['ffmpeg', '-version'], 
                     capture_output=True, 
                     check=True,
                     timeout=5)
        print("✓ ffmpeg trovato")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ ffmpeg NON trovato")
        print("\nInstallazione ffmpeg:")
        print("  Windows: choco install ffmpeg (o scarica da ffmpeg.org)")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg")


def check_structure():
    """Verifica la struttura cartelle."""
    print("\nVerifica struttura cartelle...\n")
    
    root_dir = Path(__file__).parent.parent  # Sali 2 livelli: scripts -> root
    
    required = {
        'dataset': 'Cartella dataset',
        'dataset/how2sign_realigned_train.csv': 'File CSV',
        'dataset/How2Sign': 'Cartella video originali'
    }
    
    all_ok = True
    
    for path_str, description in required.items():
        path = root_dir / path_str
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / (1024 * 1024)  # MB
                print(f"✓ {description}: {path_str} ({size:.1f} MB)")
            else:
                # Conta file nella cartella
                file_count = len(list(path.glob('*')))
                print(f"✓ {description}: {path_str} ({file_count} file)")
        else:
            print(f"✗ {description} MANCANTE: {path_str}")
            all_ok = False
    
    return all_ok


def create_segmented_folder():
    """Crea la cartella output se non esiste."""
    root_dir = Path(__file__).parent.parent  # Sali 2 livelli: scripts -> root
    output_dir = root_dir / "dataset" / "segmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Cartella output creata: dataset/segmented")


def main():
    """Esegue tutti i controlli."""
    print("="*60)
    print("SETUP VIDEO SEGMENTATION")
    print("="*60 + "\n")
    
    check_dependencies()
    structure_ok = check_structure()
    create_segmented_folder()
    
    print("\n" + "="*60)
    if structure_ok:
        print("✓ Setup completato con successo!")
        print("\nPer avviare il preprocessing:")
        print("  - TEST (prime 2 video): python preprocess_videos.py test")
        print("  - COMPLETO (tutti video): python preprocess_videos.py full")
    else:
        print("✗ Manca qualcosa. Verifica la struttura delle cartelle.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
