"""
Video Segmentation Script for Sign Language Dataset
Questo script legge il CSV How2Sign e segmenta i video in base ai tempi forniti.
"""

import os
import pandas as pd
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple
import sys
import argparse

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoSegmenter:
    """Classe per segmentare i video basandosi su un file CSV."""
    
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        output_dir: str,
        target_fps: float | None = None,
        target_height: int | None = None,
    ):
        """
        Inizializza il segmentatore.
        
        Args:
            csv_path: Percorso al file CSV con le segmentazioni
            video_dir: Cartella contenente i video originali
            output_dir: Cartella dove salvare i video segmentati
        """
        self.csv_path = csv_path
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.target_height = target_height
        
        # Crea la cartella output se non esiste
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carica il CSV
        self.df = pd.read_csv(csv_path, sep='\t')
        logger.info(f"Caricato CSV con {len(self.df)} segmentazioni")
        
        # Verifica che ffmpeg sia disponibile
        self._check_ffmpeg()

        if self.target_fps is not None or self.target_height is not None:
            logger.info(
                "Preprocessing output attivo: target_fps=%s, target_height=%s",
                self.target_fps,
                self.target_height,
            )
    
    def _check_ffmpeg(self) -> bool:
        """Verifica che ffmpeg sia installato."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True)
            logger.info("ffmpeg trovato")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ffmpeg non trovato. Installalo con: pip install ffmpeg-python")
            sys.exit(1)
    
    def _find_video_file(self, video_name: str) -> Path:
        """
        Trova il file video nella cartella video_dir.
        Supporta estensioni comuni: mp4, mov, avi, mkv
        
        Args:
            video_name: Nome del video (senza estensione)
            
        Returns:
            Path al file video, None se non trovato
        """
        extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']
        
        for ext in extensions:
            video_path = self.video_dir / f"{video_name}{ext}"
            if video_path.exists():
                return video_path
        
        # Prova cercando nella root della cartella video_dir
        for file in self.video_dir.rglob(f"{video_name}*"):
            if file.suffix.lower() in extensions:
                return file
        
        return None
    
    def segment_video(self, video_path: Path, start_time: float, end_time: float, 
                     output_path: Path) -> bool:
        """
        Segmenta un video usando ffmpeg.
        
        Args:
            video_path: Percorso del video originale
            start_time: Tempo inizio in secondi
            end_time: Tempo fine in secondi
            output_path: Percorso dove salvare il video segmentato
            
        Returns:
            True se riuscito, False altrimenti
        """
        try:
            # Costruisce il comando ffmpeg
            # -ss start_time: da questo tempo
            # -to duration: per questa durata
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', str(video_path),
                '-to', str(end_time - start_time),
            ]

            vf_filters = []
            if self.target_fps is not None:
                vf_filters.append(f"fps={self.target_fps}")
            if self.target_height is not None:
                # Mantiene aspect ratio; width multipla di 2 per codec H.264
                vf_filters.append(f"scale=-2:{self.target_height}")

            if vf_filters:
                # Se applichiamo filtri temporali/spaziali, serve re-encoding.
                cmd.extend([
                    '-vf', ','.join(vf_filters),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                ])
            else:
                # Nessun preprocessing: copia stream senza ricodifica.
                cmd.extend([
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                ])

            cmd.extend([
                '-y',  # Sovrascrivi file esistenti
                str(output_path)
            ])
            
            # Esegui il comando
            subprocess.run(cmd, 
                         capture_output=True,
                         check=True,
                         timeout=300)  # Timeout 5 minuti
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout durante la segmentazione: {output_path}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Errore ffmpeg per {output_path}: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Errore sconosciuto: {e}")
            return False
    
    def process_all(self, verbose: bool = True, num_videos: int | None = None) -> Tuple[int, int]:
        """
        Processa tutte le segmentazioni dal CSV.
        
        Args:
            verbose: Se True, mostra informazioni dettagliate
            
        Returns:
            Tuple (numero di successi, numero di errori)
        """
        success_count = 0
        error_count = 0
        
        # Raggruppa per video per efficienza
        grouped = self.df.groupby('VIDEO_NAME')
        if num_videos is not None:
            selected_videos = self.df['VIDEO_NAME'].drop_duplicates().iloc[:num_videos]
            grouped = grouped.filter(lambda g: g.name in set(selected_videos)).groupby('VIDEO_NAME')
        total_videos = len(grouped)
        
        for video_idx, (video_name, group) in enumerate(grouped, 1):
            logger.info(f"\n[{video_idx}/{total_videos}] Elaborando video: {video_name}")
            
            # Trova il file video
            video_path = self._find_video_file(video_name)
            if video_path is None:
                logger.warning(f"Video non trovato: {video_name}")
                error_count += len(group)
                continue
            
            if not video_path.exists():
                logger.warning(f"File video non accessibile: {video_path}")
                error_count += len(group)
                continue
            
            logger.info(f"Video trovato: {video_path}")
            
            # Processa tutte le segmentazioni di questo video
            for seg_idx, (_, row) in enumerate(group.iterrows(), 1):
                sentence_id = row['SENTENCE_ID']
                start_time = float(row['START_REALIGNED'])
                end_time = float(row['END_REALIGNED'])
                sentence_name = row['SENTENCE_NAME']
                
                # Crea nome output significativo
                output_filename = f"{sentence_id}_{sentence_name}.mp4"
                output_path = self.output_dir / output_filename
                
                # Segmenta il video
                logger.info(f"  [{seg_idx}/{len(group)}] Segmentazione {sentence_id} "
                           f"({start_time:.2f}s - {end_time:.2f}s)")
                
                if self.segment_video(video_path, start_time, end_time, output_path):
                    success_count += 1
                    logger.info(f"    ✓ Salvato: {output_filename}")
                else:
                    error_count += 1
                    logger.error(f"    ✗ Errore: {output_filename}")
        
        return success_count, error_count
    
    def process_sample(self, num_videos: int = 1) -> Tuple[int, int]:
        """
        Processa solo alcuni video per testing.
        
        Args:
            num_videos: Numero di video diversi da processare
            
        Returns:
            Tuple (numero di successi, numero di errori)
        """
        unique_videos = self.df['VIDEO_NAME'].unique()[:num_videos]
        sample_df = self.df[self.df['VIDEO_NAME'].isin(unique_videos)]
        
        logger.info(f"Modalità TEST: Processando {len(sample_df)} segmentazioni "
                   f"da {len(unique_videos)} video")
        
        success_count = 0
        error_count = 0
        
        for video_name in unique_videos:
            group = sample_df[sample_df['VIDEO_NAME'] == video_name]
            video_path = self._find_video_file(video_name)
            
            if video_path is None:
                logger.warning(f"Video non trovato: {video_name}")
                error_count += len(group)
                continue
            
            for _, row in group.iterrows():
                sentence_id = row['SENTENCE_ID']
                start_time = float(row['START_REALIGNED'])
                end_time = float(row['END_REALIGNED'])
                sentence_name = row['SENTENCE_NAME']
                
                output_filename = f"{sentence_id}_{sentence_name}.mp4"
                output_path = self.output_dir / output_filename
                
                if self.segment_video(video_path, start_time, end_time, output_path):
                    success_count += 1
                else:
                    error_count += 1
        
        return success_count, error_count
    
    def get_statistics(self):
        """Stampa statistiche sul dataset."""
        durations = self.df['END_REALIGNED'] - self.df['START_REALIGNED']
        print("\n" + "="*60)
        print("STATISTICHE DATASET")
        print("="*60)
        print(f"Segmentazioni totali: {len(self.df)}")
        print(f"Video unici: {self.df['VIDEO_NAME'].nunique()}")
        print(f"Frasi totali: {self.df['SENTENCE_ID'].nunique()}")
        print(f"\nDurata medio segmentazione: {durations.mean():.2f}s")
        print(f"Durata min segmentazione: {durations.min():.2f}s")
        print(f"Durata max segmentazione: {durations.max():.2f}s")
        if self.target_fps is not None:
            expected_frames = durations * self.target_fps
            print(f"\nCon target_fps={self.target_fps:g}:")
            print(f"Frame medi per segmento (stimati): {expected_frames.mean():.1f}")
            print(f"Frame p50 per segmento (stimati): {expected_frames.quantile(0.5):.1f}")
            print(f"Frame p90 per segmento (stimati): {expected_frames.quantile(0.9):.1f}")
            print(f"Frame max per segmento (stimati): {expected_frames.max():.1f}")
        print("="*60 + "\n")


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Segmenta i video How2Sign in base ai timestamp CSV."
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='sample',
        choices=['sample', 'test', 'full'],
        help="sample/test per debug veloce, full per tutto il dataset",
    )
    parser.add_argument(
        '--target-fps',
        type=float,
        default=None,
        help="FPS output desiderati (es. 30 o 25). Se omesso, mantiene stream originale.",
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=None,
        help="Altezza output video (es. 480). Larghezza adattata mantenendo aspect ratio.",
    )
    parser.add_argument(
        '--num-videos',
        type=int,
        default=None,
        help="Numero massimo di video distinti da preprocessare.",
    )
    args = parser.parse_args()

    # Configurazione percorsi (sali 3 livelli: preprocessing -> src -> root)
    root_dir = Path(__file__).parent.parent.parent
    csv_path = root_dir / "dataset" / "how2sign_realigned_train.csv"
    video_dir = root_dir / "dataset" / "raw_videos"
    output_dir = root_dir / "dataset" / "segmented"
    
    # Verifica che il CSV esista
    if not csv_path.exists():
        logger.error(f"CSV non trovato: {csv_path}")
        sys.exit(1)
    
    # Crea il segmentatore
    segmenter = VideoSegmenter(
        str(csv_path),
        str(video_dir),
        str(output_dir),
        target_fps=args.target_fps,
        target_height=args.target_height,
    )
    
    # Mostra statistiche
    segmenter.get_statistics()
    
    # Modo: scegli tra 'sample' per un test o 'full' per tutto
    mode = args.mode
    
    if mode == 'test' or mode == 'sample':
        sample_videos = args.num_videos if args.num_videos is not None else 2
        logger.info(f"\n🔍 Avviando in MODALITÀ TEST (primi {sample_videos} video)")
        success, errors = segmenter.process_sample(num_videos=sample_videos)
    else:
        logger.info("\n▶️ Avviando processamento COMPLETO")
        success, errors = segmenter.process_all(num_videos=args.num_videos)
    
    # Risultati finali
    print("\n" + "="*60)
    print("RISULTATI")
    print("="*60)
    print(f"✓ Segmentazioni completate: {success}")
    print(f"✗ Errori: {errors}")
    print(f"Cartella output: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
