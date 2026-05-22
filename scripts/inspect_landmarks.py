#!/usr/bin/env python3
"""
Script di utilità per ispezionare e validare i landmarks estratti.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def inspect_landmarks_file(filepath):
    """
    Ispeziona un singolo file .npy di landmarks.
    
    Args:
        filepath: percorso del file .npy
    """
    try:
        landmarks = np.load(filepath)
        
        print(f"\n📁 File: {os.path.basename(filepath)}")
        print("─" * 60)
        
        # Informazioni base
        info = [
            ["Shape", str(landmarks.shape)],
            ["Tipo di dato", str(landmarks.dtype)],
            ["Dimensione (MB)", f"{landmarks.nbytes / (1024**2):.2f}"],
            ["Num. frame", landmarks.shape[0]],
            ["Landmarks per frame", landmarks.shape[1] if len(landmarks.shape) > 1 else "N/A"],
        ]
        print(tabulate(info, headers=["Proprietà", "Valore"], tablefmt="grid"))
        
        # Statistiche
        print("\n📊 Statistiche:")
        print("─" * 60)
        stats = [
            ["Min", f"{landmarks.min():.6f}"],
            ["Max", f"{landmarks.max():.6f}"],
            ["Media", f"{landmarks.mean():.6f}"],
            ["Std Dev", f"{landmarks.std():.6f}"],
        ]
        print(tabulate(stats, headers=["Metrica", "Valore"], tablefmt="grid"))
        
        # Valori nulli
        zero_count = np.count_nonzero(landmarks == 0)
        total_values = landmarks.size
        zero_percentage = (zero_count / total_values) * 100
        
        print("\n🔍 Validazione:")
        print("─" * 60)
        validation = [
            ["Valori zero", f"{zero_count} ({zero_percentage:.1f}%)"],
            ["Valori NaN", f"{np.isnan(landmarks).sum()}"],
            ["Valori Inf", f"{np.isinf(landmarks).sum()}"],
            ["Valori in [0,1]", f"{np.sum((landmarks >= 0) & (landmarks <= 1)) if zero_percentage < 50 else 'Quasi tutti'}"],
        ]
        print(tabulate(validation, headers=["Controllo", "Risultato"], tablefmt="grid"))
        
    except Exception as e:
        logger.error(f"Errore nell'ispezionare {filepath}: {e}")


def inspect_directory(directory):
    """
    Ispeziona tutti i file .npy in una directory.
    
    Args:
        directory: percorso della directory
    """
    npy_files = sorted(Path(directory).glob('*_landmarks.npy'))
    
    if not npy_files:
        logger.warning(f"Nessun file .npy trovato in {directory}")
        return
    
    logger.info(f"Trovati {len(npy_files)} file landmarks in {directory}")
    
    # Raccogliere informazioni
    total_files = len(npy_files)
    total_frames = 0
    total_size = 0
    files_info = []
    
    for filepath in npy_files:
        try:
            landmarks = np.load(filepath)
            num_frames = landmarks.shape[0]
            size_mb = landmarks.nbytes / (1024**2)
            
            total_frames += num_frames
            total_size += landmarks.nbytes
            
            files_info.append([
                os.path.basename(filepath),
                num_frames,
                f"{size_mb:.2f} MB",
                f"{landmarks.min():.3f} - {landmarks.max():.3f}"
            ])
        except Exception as e:
            logger.warning(f"Errore nel leggere {filepath}: {e}")
    
    # Visualizzare tabella
    print("\n" + "=" * 100)
    print(f"📊 Riepilogo Directory: {directory}")
    print("=" * 100)
    
    print(tabulate(
        files_info,
        headers=["File", "Frame", "Dimensione", "Range valori"],
        tablefmt="grid"
    ))
    
    print("\n📈 Statistiche Totali:")
    print("─" * 60)
    summary = [
        ["File totali", total_files],
        ["Frame totali", total_frames],
        ["Media frame/file", f"{total_frames / total_files:.1f}"],
        ["Dimensione totale", f"{total_size / (1024**2):.2f} MB"],
        ["Media dim/file", f"{(total_size / total_files) / (1024**2):.2f} MB"],
    ]
    print(tabulate(summary, headers=["Metrica", "Valore"], tablefmt="grid"))
    print()


def compare_splits(landmark_dir):
    """
    Confronta i landmark dei split train e dev.
    
    Args:
        landmark_dir: directory principale dei landmarks
    """
    train_dir = os.path.join(landmark_dir, 'train')
    dev_dir = os.path.join(landmark_dir, 'dev')
    
    comparison = []
    
    for split, directory in [("Train", train_dir), ("Dev", dev_dir)]:
        if not os.path.exists(directory):
            logger.warning(f"Directory non trovata: {directory}")
            comparison.append([split, "N/A", "N/A", "N/A"])
            continue
        
        npy_files = list(Path(directory).glob('*_landmarks.npy'))
        
        if npy_files:
            # Caricare il primo file per controllare la dimensione
            first_file = np.load(npy_files[0])
            landmarks_dim = first_file.shape[1] if len(first_file.shape) > 1 else 0
            total_frames = sum(np.load(f).shape[0] for f in npy_files)
            
            comparison.append([
                split,
                len(npy_files),
                total_frames,
                landmarks_dim
            ])
        else:
            comparison.append([split, 0, 0, "N/A"])
    
    print("\n" + "=" * 80)
    print("📊 Confronto Split")
    print("=" * 80)
    print(tabulate(
        comparison,
        headers=["Split", "Num. Frasi", "Num. Frame", "Dim. Landmarks"],
        tablefmt="grid"
    ))
    print()


def main():
    parser = argparse.ArgumentParser(description="Ispeziona i landmarks estratti")
    parser.add_argument(
        '--file',
        help="Ispeziona un singolo file .npy"
    )
    parser.add_argument(
        '--dir',
        help="Ispeziona tutti i file .npy in una directory"
    )
    parser.add_argument(
        '--landmark-dir',
        default='/home/l_ronco/Documents/Thesis/landmarks',
        help="Directory principale dei landmarks (per confrontare split)"
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help="Confronta i split train e dev"
    )
    
    args = parser.parse_args()
    
    if args.file:
        inspect_landmarks_file(args.file)
    elif args.dir:
        inspect_directory(args.dir)
    elif args.compare:
        compare_splits(args.landmark_dir)
    else:
        print("Specificare --file, --dir, o --compare")
        parser.print_help()


if __name__ == "__main__":
    main()
