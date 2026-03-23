"""
Analisi e statistiche del dataset How2Sign
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_csv(csv_path):
    """Analisi completa del CSV."""
    
    df = pd.read_csv(csv_path, sep='\t')
    
    print("\n" + "="*70)
    print("ANALISI DATASET HOW2SIGN")
    print("="*70 + "\n")
    
    # Info generali
    print("📊 INFORMAZIONI GENERALI")
    print("-" * 70)
    print(f"Numero totale righe (segmentazioni): {len(df):,}")
    print(f"Numero colonne: {len(df.columns)}")
    print(f"Numero video unici: {df['VIDEO_NAME'].nunique():,}")
    print(f"Numero frasi uniche: {df['SENTENCE_ID'].nunique():,}")
    
    # Durate
    print("\n⏱️  DURATE SEGMENTAZIONI")
    print("-" * 70)
    durations = df['END_REALIGNED'] - df['START_REALIGNED']
    print(f"Durata media: {durations.mean():.2f}s")
    print(f"Durata mediana: {durations.median():.2f}s")
    print(f"Durata min: {durations.min():.2f}s")
    print(f"Durata max: {durations.max():.2f}s")
    print(f"Deviazione standard: {durations.std():.2f}s")
    print(f"Durata totale dataset: {durations.sum()/3600:.1f} ore")
    
    # Distribuzione
    print("\n📈 DISTRIBUZIONE")
    print("-" * 70)
    segs_per_video = df.groupby('VIDEO_NAME').size()
    print(f"Segmentazioni per video - Media: {segs_per_video.mean():.1f}")
    print(f"Segmentazioni per video - Min: {segs_per_video.min()}")
    print(f"Segmentazioni per video - Max: {segs_per_video.max()}")
    
    # Top video per segmentazioni
    print("\n🎬 TOP 10 VIDEO PIÙ LUNGHI (per numero segmentazioni)")
    print("-" * 70)
    top_videos = segs_per_video.nlargest(10)
    for i, (video, count) in enumerate(top_videos.items(), 1):
        video_duration = df[df['VIDEO_NAME'] == video]['END_REALIGNED'].max()
        print(f"{i:2}. {video[:30]:30} | {count:3} segm. | ~{video_duration:6.1f}s")
    
    # Lunghezza testi
    print("\n📝 LUNGHEZZA TESTI")
    print("-" * 70)
    text_lengths = df['SENTENCE'].str.split().str.len()
    print(f"Parole per frase - Media: {text_lengths.mean():.1f}")
    print(f"Parole per frase - Min: {text_lengths.min()}")
    print(f"Parole per frase - Max: {text_lengths.max()}")
    
    # Correlazione durata - lunghezza testo
    correlation = durations.corr(text_lengths)
    print(f"Correlazione durata video - lunghezza testo: {correlation:.2f}")
    
    # Variabilità
    print("\n🔍 QUALITÀ DATI")
    print("-" * 70)
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("Valori nulli per colonna:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count}")
    else:
        print("✓ Nessun valore nullo rilevato")
    
    # Verifica temporale
    issues = []
    for _, row in df.iterrows():
        if row['START_REALIGNED'] >= row['END_REALIGNED']:
            issues.append(f"Row {_}: START >= END")
    
    if issues:
        print(f"⚠️  {len(issues)} problemi temporali rilevati:")
        for issue in issues[:5]:
            print(f"  - {issue}")
    else:
        print("✓ Nessun problema temporale")
    
    print("\n" + "="*70 + "\n")
    
    return df, durations, text_lengths


def create_visualizations(df, durations, text_lengths, output_dir='analysis'):
    """Crea grafici di analisi."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Creazione visualizzazioni...\n")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Distribuzione durate
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(durations, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax[0].set_xlabel('Durata (secondi)', fontsize=11)
    ax[0].set_ylabel('Frequenza', fontsize=11)
    ax[0].set_title('Distribuzione Durate Segmentazioni', fontsize=12, fontweight='bold')
    ax[0].axvline(durations.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {durations.mean():.2f}s')
    ax[0].legend()
    
    # Box plot
    ax[1].boxplot(durations, vert=True)
    ax[1].set_ylabel('Durata (secondi)', fontsize=11)
    ax[1].set_title('Box Plot Durate', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / '01_durate_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ 01_durate_distribution.png")
    
    # 2. Lunghezza testi
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(text_lengths, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    ax[0].set_xlabel('Numero di parole', fontsize=11)
    ax[0].set_ylabel('Frequenza', fontsize=11)
    ax[0].set_title('Distribuzione Lunghezza Testi', fontsize=12, fontweight='bold')
    
    ax[1].scatter(durations, text_lengths, alpha=0.3, s=10)
    ax[1].set_xlabel('Durata video (s)', fontsize=11)
    ax[1].set_ylabel('Parole nel testo', fontsize=11)
    ax[1].set_title('Correlazione: Durata Video vs Lunghezza Testo', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / '02_text_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ 02_text_analysis.png")
    
    # 3. Top video
    segs_per_video = df.groupby('VIDEO_NAME').size().nlargest(15)
    fig, ax = plt.subplots(figsize=(12, 8))
    segs_per_video.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Numero segmentazioni', fontsize=11)
    ax.set_title('Top 15 Video per Numero di Segmentazioni', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / '03_top_videos.png', dpi=150, bbox_inches='tight')
    print("✓ 03_top_videos.png")
    
    # 4. Statistiche cumulative
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_durations = np.sort(durations)
    cumsum_durations = np.cumsum(sorted_durations)
    ax.plot(cumsum_durations / 3600, linewidth=2, color='darkgreen')
    ax.fill_between(range(len(cumsum_durations)), cumsum_durations / 3600, alpha=0.3, color='lightgreen')
    ax.set_xlabel('Indice segmentazione (ordinata per durata)', fontsize=11)
    ax.set_ylabel('Ore di video cumulative', fontsize=11)
    ax.set_title('Video Totale Cumulativo', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / '04_cumulative_duration.png', dpi=150, bbox_inches='tight')
    print("✓ 04_cumulative_duration.png")
    
    print(f"\nVisualizzazioni salvate in: {output_path}\n")


def export_summary(df, durations, text_lengths, output_file='dataset_summary.txt'):
    """Esporta summary su file."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("DATASET SUMMARY - HOW2SIGN TRAINING SET\n")
        f.write("="*70 + "\n\n")
        
        f.write("GENERALE\n")
        f.write("-"*70 + "\n")
        f.write(f"Segmentazioni totali: {len(df):,}\n")
        f.write(f"Video unici: {df['VIDEO_NAME'].nunique():,}\n")
        f.write(f"Frasi uniche: {df['SENTENCE_ID'].nunique():,}\n")
        f.write(f"Durata totale: {durations.sum()/3600:.1f} ore\n\n")
        
        f.write("DURATE\n")
        f.write("-"*70 + "\n")
        f.write(f"Media: {durations.mean():.2f}s\n")
        f.write(f"Mediana: {durations.median():.2f}s\n")
        f.write(f"Min: {durations.min():.2f}s\n")
        f.write(f"Max: {durations.max():.2f}s\n")
        f.write(f"Std Dev: {durations.std():.2f}s\n\n")
        
        f.write("TESTI\n")
        f.write("-"*70 + "\n")
        f.write(f"Parole per frase (media): {text_lengths.mean():.1f}\n")
        f.write(f"Parole per frase (min): {text_lengths.min()}\n")
        f.write(f"Parole per frase (max): {text_lengths.max()}\n")
        f.write(f"Correlazione durata-lunghezza testo: {durations.corr(text_lengths):.2f}\n")
    
    print(f"✓ Summary esportato: {output_file}")


def main():
    root_dir = Path(__file__).parent.parent.parent  # Sali 3 livelli: preprocessing -> src -> root
    csv_path = root_dir / "dataset" / "how2sign_realigned_train.csv"
    
    if not csv_path.exists():
        print(f"❌ CSV non trovato: {csv_path}")
        return
    
    # Analisi
    df, durations, text_lengths = analyze_csv(str(csv_path))
    
    # Visualizzazioni
    try:
        create_visualizations(df, durations, text_lengths, 'analysis')
    except Exception as e:
        print(f"⚠️  Errore nella creazione dei grafici: {e}")
    
    # Export summary
    export_summary(df, durations, text_lengths, 'dataset_summary.txt')


if __name__ == "__main__":
    main()
