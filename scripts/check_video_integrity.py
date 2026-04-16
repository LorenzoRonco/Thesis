"""
Verifica integrità video tra CSV e cartella di input
Controlla corrispondenza file e identifica video orfani
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


def check_video_integrity(csv_path, video_dir):
    """
    Verifica integrità video tra CSV e cartella di input.
    
    Args:
        csv_path: Percorso al file CSV
        video_dir: Percorso alla cartella con i video
        
    Returns:
        Dict con statistiche complete
    """
    
    print("\n" + "="*80)
    print("VERIFICA INTEGRITÀ VIDEO")
    print("="*80 + "\n")
    
    # Carica CSV
    if not Path(csv_path).exists():
        print(f"❌ CSV non trovato: {csv_path}\n")
        return None
    
    df = pd.read_csv(csv_path, sep='\t')
    expected_videos = set(df['VIDEO_NAME'].unique())
    
    print(f"📊 CSV ANALYSIS")
    print(f"  Video unici nel CSV: {len(expected_videos):,}")
    print(f"  Total segmentazioni: {len(df):,}\n")
    
    # Cerca video nella cartella
    video_path = Path(video_dir)
    if not video_path.exists():
        print(f"❌ Cartella non trovata: {video_dir}\n")
        return None
    
    supported_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
    found_files = {}
    
    print(f"📁 CARTELLA INPUT ANALYSIS")
    print(f"  Percorso: {video_path}")
    print(f"  Ricerca video in corso...\n")
    
    # Scandisci la cartella
    all_files_in_folder = list(video_path.rglob('*'))
    video_files = [f for f in all_files_in_folder if f.is_file() and f.suffix.lower() in supported_extensions]
    
    print(f"  File video trovati: {len(video_files):,}\n")
    
    # Estrai nomi file senza estensione e crea mapping
    for file in video_files:
        file_name_without_ext = file.stem
        found_files[file_name_without_ext] = file
    
    # ========== ANALISI 1: VIDEO NEL CSV CHE MANCANO NELLA CARTELLA ==========
    print("="*80)
    print("1️⃣  VIDEO NEL CSV MANCANTI NELLA CARTELLA")
    print("="*80 + "\n")
    
    missing_from_folder = sorted(expected_videos - set(found_files.keys()))
    
    if missing_from_folder:
        print(f"❌ Video mancanti: {len(missing_from_folder):,}/{len(expected_videos):,}\n")
        
        # Mostra primi 20
        print("PRIMI 20 VIDEO MANCANTI:")
        print("-" * 80)
        for i, video_name in enumerate(missing_from_folder[:20], 1):
            segmentations = len(df[df['VIDEO_NAME'] == video_name])
            duration = df[df['VIDEO_NAME'] == video_name]['END_REALIGNED'].sum() - \
                      df[df['VIDEO_NAME'] == video_name]['START_REALIGNED'].sum()
            print(f"{i:3}. {video_name:40} | {segmentations:3} segm. | {duration/60:6.1f} min")
        
        if len(missing_from_folder) > 20:
            print(f"... e altri {len(missing_from_folder) - 20} video\n")
        else:
            print()
        
        # Statistiche sui video mancanti
        missing_df = df[df['VIDEO_NAME'].isin(missing_from_folder)]
        missing_duration = missing_df['END_REALIGNED'].sum() - missing_df['START_REALIGNED'].sum()
        
        print(f"STATISTICHE VIDEO MANCANTI:")
        print(f"  Segmentazioni perse: {len(missing_df):,}")
        print(f"  Ore di video perse: {missing_duration / 3600:.1f} ore")
        print(f"  % di dati mancanti: {len(missing_df) / len(df) * 100:.1f}%\n")
    else:
        print(f"✅ PERFETTO! Tutti i {len(expected_videos):,} video nel CSV sono presenti nella cartella!\n")
    
    # ========== ANALISI 2: FILE NELLA CARTELLA CHE NON SONO NEL CSV ==========
    print("="*80)
    print("2️⃣  FILE NELLA CARTELLA NON PRESENTI NEL CSV (VIDEO EXTRA/ORFANI)")
    print("="*80 + "\n")
    
    extra_files = sorted(set(found_files.keys()) - expected_videos)
    
    if extra_files:
        print(f"⚠️  File extra trovati: {len(extra_files):,}\n")
        
        # Calcola spazio occupato
        total_extra_size = sum(found_files[f].stat().st_size for f in extra_files)
        total_extra_size_gb = total_extra_size / (1024**3)
        
        print(f"STATISTICHE FILE EXTRA:")
        print(f"  Numero di file: {len(extra_files):,}")
        print(f"  Spazio occupato: {total_extra_size_gb:.2f} GB\n")
        
        # Mostra primi 20
        print("PRIMI 20 FILE EXTRA:")
        print("-" * 80)
        for i, video_name in enumerate(extra_files[:20], 1):
            file_path = found_files[video_name]
            file_size = file_path.stat().st_size / (1024**2)  # MB
            print(f"{i:3}. {video_name:40} | {file_size:10.1f} MB")
        
        if len(extra_files) > 20:
            remaining_size = sum(found_files[f].stat().st_size 
                               for f in extra_files[20:]) / (1024**2)
            print(f"... e altri {len(extra_files) - 20} file | {remaining_size:10.1f} MB\n")
        else:
            print()
    else:
        print(f"✅ PERFETTO! Non ci sono file extra nella cartella!\n")
    
    # ========== SUMMARY ==========
    print("="*80)
    print("📈 SUMMARY")
    print("="*80 + "\n")
    
    found_count = len(found_files)
    match_count = len(expected_videos & set(found_files.keys()))
    
    print(f"Video nel CSV:              {len(expected_videos):>6,}")
    print(f"Video trovati in cartella:  {found_count:>6,}")
    print(f"Corrispondenze OK:          {match_count:>6,} ({match_count/len(expected_videos)*100:>5.1f}%)")
    print(f"Video mancanti:             {len(missing_from_folder):>6,} ({len(missing_from_folder)/len(expected_videos)*100:>5.1f}%)")
    print(f"File extra (non in CSV):    {len(extra_files):>6,}\n")
    
    # Status finale
    if len(missing_from_folder) == 0 and len(extra_files) == 0:
        print("✅ STATUS: PERFETTO! Dataset completo e coerente.\n")
    elif len(missing_from_folder) == 0:
        print(f"⚠️  STATUS: Tutti i video del CSV sono presenti, ma ci sono {len(extra_files)} file extra nella cartella.\n")
    elif len(extra_files) == 0:
        print(f"⚠️  STATUS: {len(missing_from_folder)} video del CSV mancano nella cartella.\n")
    else:
        print(f"❌ STATUS: {len(missing_from_folder)} video mancano E {len(extra_files)} file extra.\n")
    
    return {
        'csv_videos': len(expected_videos),
        'folder_files': found_count,
        'matches': match_count,
        'missing': len(missing_from_folder),
        'missing_list': missing_from_folder,
        'extra': len(extra_files),
        'extra_list': extra_files,
        'extra_size_gb': total_extra_size_gb if extra_files else 0
    }


def main():
    # Configurazione percorsi
    root_dir = Path(__file__).parent.parent
    csv_path = root_dir / "dataset" / "how2sign_realigned_train.csv"
    video_dir = str(root_dir / "dataset" / "raw_videos")
    
    # Esegui verifica
    result = check_video_integrity(str(csv_path), video_dir)
    
    # Esporta risultati a file (opzionale)
    if result:
        output_file = root_dir / "video_integrity_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VIDEO INTEGRITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"CSV Videos: {result['csv_videos']:,}\n")
            f.write(f"Folder Files: {result['folder_files']:,}\n")
            f.write(f"Matches: {result['matches']:,}\n")
            f.write(f"Missing: {result['missing']:,}\n")
            f.write(f"Extra: {result['extra']:,}\n")
            f.write(f"Extra Size: {result['extra_size_gb']:.2f} GB\n\n")
            
            if result['missing_list']:
                f.write("MISSING VIDEOS:\n")
                for v in result['missing_list']:
                    f.write(f"  {v}\n")
            
            if result['extra_list']:
                f.write("\nEXTRA FILES:\n")
                for v in result['extra_list']:
                    f.write(f"  {v}\n")
        
        print(f"📄 Report salvato: {output_file}\n")


if __name__ == "__main__":
    main()
