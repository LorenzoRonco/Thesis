"""
Diagnosi e utility per il preprocessing
Aiuta a identificare e risolvere problemi di segmentazione
"""

import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd


def check_ffmpeg_info(video_path):
    """Estrae informazioni da un video usando ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration,size:stream=index,codec_type,width,height,r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1:noescapes=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        info = result.stdout.strip().split('\n')
        return {
            'duration': info[0] if len(info) > 0 else 'N/A',
            'size': info[1] if len(info) > 1 else 'N/A',
        }
    except:
        return None


def diagnose_videos():
    """Diagnostica stati dei video nella cartella."""
    
    root_dir = Path(__file__).parent.parent  # Sali 2 livelli: scripts -> root
    video_dir = root_dir / "dataset" / "raw_videos"
    csv_path = root_dir / "dataset" / "how2sign_realigned_train.csv"
    
    print("\n" + "="*70)
    print("DIAGNOSI VIDEO")
    print("="*70 + "\n")
    
    # Carica CSV
    if not csv_path.exists():
        print("❌ CSV non trovato")
        return
    
    df = pd.read_csv(csv_path, sep='\t')
    expected_videos = set(df['VIDEO_NAME'].unique())
    
    print(f"📊 Video attesi nel CSV: {len(expected_videos)}\n")
    
    # Controlla cartella video
    if not video_dir.exists():
        print(f"❌ Cartella non esiste: {video_dir}")
        print("   Crea la cartella e scarica i video da: https://how2sign.github.io/\n")
        return
    
    print(f"📁 Cartella video: {video_dir}\n")
    
    # Trova file video
    supported_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
    found_videos = defaultdict(list)
    
    print("Ricerca video...\n")
    
    for file in video_dir.rglob('*'):
        if file.is_file() and file.suffix.lower() in supported_extensions:
            # Prova a estrarre il nome base
            for expected_name in expected_videos:
                if expected_name in str(file):
                    found_videos[expected_name].append(file)
                    break
    
    # Statistiche
    print(f"✓ Video trovati: {len(found_videos)}/{len(expected_videos)}\n")
    
    # Video mancanti
    missing = expected_videos - set(found_videos.keys())
    if missing:
        print(f"⚠️  Video MANCANTI: {len(missing)}")
        print("   Primi 10 mancanti:")
        for video in sorted(missing)[:10]:
            print(f"     - {video}")
        if len(missing) > 10:
            print(f"     ... e altri {len(missing) - 10}")
    else:
        print("✓ Tutti i video trovati!")
    
    print()
    
    # Video duplicati o con name mismatch
    mismatched = []
    for name, files in found_videos.items():
        if len(files) > 1:
            print(f"⚠️  VIDEO DUPLICATI: {name}")
            for f in files:
                print(f"     {f}")
            mismatched.append(name)
    
    if mismatched:
        print()
    
    # Info dettagli video (primi 5)
    print("📋 Dettagli primi 5 video trovati:")
    print("-" * 70)
    
    for i, (name, files) in enumerate(sorted(found_videos.items())[:5]):
        video_file = files[0]  # Prendi il primo se duplicati
        size_bytes = video_file.stat().st_size
        size_gb = size_bytes / (1024**3)
        
        print(f"\n{i+1}. {name}")
        print(f"   File: {video_file.name}")
        print(f"   Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
        
        # Prova a ottenere info ffmpeg
        info = check_ffmpeg_info(video_file)
        if info:
            try:
                duration_str = info.get('duration', '').strip()
                if duration_str and duration_str != 'N/A':
                    duration = float(duration_str)
                    print(f"   Durata: {duration/60:.1f} minuti ({duration:.1f}s)")
                else:
                    print("   ⚠️  Impossibile leggere durata")
            except (ValueError, TypeError):
                print("   ⚠️  Impossibile leggere durata")
        else:
            print("   ⚠️  Impossibile leggere info (ffprobe non disponibile)")
    
    print("\n" + "="*70 + "\n")
    
    # Raccomandazioni
    print("💡 RACCOMANDAZIONI\n")
    
    if len(missing) > 0 and len(found_videos) == 0:
        print("❌ NESSUN VIDEO TROVATO!")
        print("   1. Scarica il dataset da: https://how2sign.github.io/")
        print("   2. Estrai i file nella cartella: dataset/How2Sign/")
        print("   3. Verifica i nomi corrispondano al CSV")
    elif len(missing) > len(found_videos) / 2:
        print(f"❌ MANCANO TROPPI VIDEO ({len(missing)}/{len(expected_videos)})")
        print("   Scarica i video mancanti da https://how2sign.github.io/")
    elif len(missing) == 0:
        print("✅ TUTTO OK!")
        print("   Puoi procedere con: python preprocess_videos.py test")
    else:
        print(f"⚠️  Alcuni video mancano ({len(missing)}/{len(expected_videos)})")
        print("   Se necessari, scaricali. Altrimenti puoi procedere parzialmente.")
    
    print()


def check_segmented_output():
    """Verifica output della segmentazione."""
    
    root_dir = Path(__file__).parent.parent  # Sali 2 livelli: scripts -> root
    segmented_dir = root_dir / "dataset" / "segmented"
    
    print("\n" + "="*70)
    print("OUTPUT SEGMENTAZIONE")
    print("="*70 + "\n")
    
    if not segmented_dir.exists():
        print("ℹ️  Nessun output segmentazione ancora creato")
        print("   Esegui: python preprocess_videos.py test\n")
        return
    
    # Conta video segmentati
    segmented_videos = list(segmented_dir.glob('*.mp4'))
    total_size = sum(f.stat().st_size for f in segmented_videos)
    total_size_gb = total_size / (1024**3)
    
    print(f"✓ Video segmentati: {len(segmented_videos)}")
    print(f"✓ Spazio totale: {total_size_gb:.2f} GB\n")
    
    if segmented_videos:
        print("Ultimi 5 video segmentati:")
        for f in sorted(segmented_videos)[-5:]:
            size_mb = f.stat().st_size / (1024**2)
            print(f"  {f.name} ({size_mb:.1f} MB)")
    
    print()


def test_ffmpeg():
    """Testa se ffmpeg è disponibile."""
    
    print("Verifica ffmpeg...\n")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        version_line = result.stdout.split('\n')[0]
        print(f"✓ {version_line}\n")
        return True
    except FileNotFoundError:
        print("❌ ffmpeg non trovato\n")
        print("Installazione:")
        print("  Windows: choco install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg")
        print()
        return False


def main():
    """Menu principale."""
    
    print("\n" + "="*70)
    print("DIAGNOSI PREPROCESSING VIDEO")
    print("="*70)
    
    # Test ffmpeg
    if test_ffmpeg():
        print("✓ Dipendenze OK\n")
    
    # Diagnostica video
    diagnose_videos()
    
    # Check output
    check_segmented_output()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
