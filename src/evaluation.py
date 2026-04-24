"""
evaluate.py — Valutazione del Sign Language Translator
=======================================================

ESECUZIONE
----------
Dalla root del progetto:

    python -m src.evaluation

PERSONALIZZAZIONI
-----------------
Modifica i parametri nel blocco `evaluate(...)` in fondo al file:

  checkpoint_path : path al checkpoint da valutare.
                    Usa 'checkpoints/best_model.pt' per il modello migliore,
                    o 'checkpoints/checkpoint_epoch_XXX.pt' per un'epoca specifica.

  num_samples     : numero di video da usare per la valutazione.
                    None = tutto il dataset di validazione.

  batch_size      : numero di sample per batch (default: 8).
                    Riduci a 4 se vai in out-of-memory sulla GPU.

  max_frames      : deve corrispondere al valore usato durante il training.

  max_new_tokens  : numero massimo di token generati per traduzione (default: 100).

  d_model         : deve corrispondere al valore usato durante il training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sacrebleu.metrics import BLEU
from transformers import BartTokenizer

from src.models import SignLanguageTranslator
from src.data.how2sign_loader import How2SignDataset, collate_fn


# ----------------------------------------------------------------------
# Utilità
# ----------------------------------------------------------------------

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(predictions, [references])
    return result.score


def load_model(
    checkpoint_path: Path,
    d_model: int,
    dropout: float,
    device: torch.device,
    frame_chunk_size: int | None,
) -> SignLanguageTranslator:
    """
    Carica il modello da checkpoint.
    Gestisce sia best_model.pt (solo state_dict)
    che checkpoint_epoch_XXX.pt (dict completo).
    """
    model = SignLanguageTranslator(
        d_model=d_model,
        dropout=dropout,
        frame_chunk_size=frame_chunk_size,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # best_model.pt contiene solo lo state_dict
    # checkpoint_epoch_XXX.pt contiene un dict con 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"[Evaluate] Caricato checkpoint epoca {epoch}.")
    else:
        model.load_state_dict(checkpoint)
        print(f"[Evaluate] Caricato best_model.pt.")

    model.eval()
    return model


# ----------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    # Paths
    checkpoint_path: Path,
    csv_path: Path = Path('dataset/how2sign_realigned_val.csv'),
    landmarks_dir: Path = Path('dataset/landmarks_validation_normalized'),
    cropped_dir: Path = Path('dataset/cropped_validation'),
    # Dati
    num_samples: int | None = None,
    max_frames: int = 150,
    batch_size: int = 8,
    max_new_tokens: int = 100,
    # Modello
    d_model: int = 512,
    dropout: float = 0.1,
    frame_chunk_size: int | None = 16,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluate] Device: {device}")

    # --- Dataset ----------------------------------------------------------
    dataset = How2SignDataset(
        csv_path=csv_path,
        landmarks_dir=landmarks_dir,
        cropped_dir=cropped_dir,
        max_frames=max_frames,
        num_samples=num_samples,
    )
    print(f"[Evaluate] {len(dataset)} sample di validazione.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Modello ----------------------------------------------------------
    model = load_model(checkpoint_path, d_model, dropout, device, frame_chunk_size)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # --- Loop -------------------------------------------------------------
    total_loss  = 0.0
    total_tokens = 0
    all_predictions = []
    all_references  = []

    for i, (landmarks, video_frames, padding_mask, sentences, lengths) in enumerate(loader):
        landmarks    = landmarks.to(device)
        video_frames = video_frames.to(device)
        padding_mask = padding_mask.to(device)

        # Loss
        encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        target_tokens  = encoded['input_ids'].to(device)
        decoder_input  = target_tokens[:, :-1]
        decoder_target = target_tokens[:, 1:]

        logits = model(
            landmarks=landmarks,
            video_frames=video_frames,
            padding_mask=padding_mask,
            target_tokens=decoder_input,
        )
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1),
        )

        n_tokens = (decoder_target != tokenizer.pad_token_id).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

        # Traduzioni
        predictions = model.generate(
            landmarks=landmarks,
            video_frames=video_frames,
            padding_mask=padding_mask,
            max_new_tokens=max_new_tokens,
        )
        all_predictions.extend(predictions)
        all_references.extend(sentences)

        print(f"  Batch {i+1}/{len(loader)} completato.", end='\r')

    # --- Risultati --------------------------------------------------------
    avg_loss = total_loss / max(1, total_tokens)
    bleu     = compute_bleu(all_predictions, all_references)

    print(f"\n{'='*60}")
    print(f"  RISULTATI VALUTAZIONE")
    print(f"{'='*60}")
    print(f"  Val Loss : {avg_loss:.4f}")
    print(f"  BLEU     : {bleu:.2f}")
    print(f"\n  Esempi:")
    for i in range(min(3, len(all_predictions))):
        print(f"\n  [{i+1}] Predetto : {all_predictions[i]}")
        print(f"       Atteso   : {all_references[i]}")
    print(f"{'='*60}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    evaluate(
        checkpoint_path=Path('checkpoints/best_model.pt'),
        num_samples=None,
        batch_size=8,
        max_new_tokens=100,
        max_frames=64
    )