"""
training.py — Training loop per Sign Language Translator
=========================================================

ESECUZIONE
----------
Dalla root del progetto:

    python -m src.training

PERSONALIZZAZIONI
-----------------
Modifica i parametri nel blocco `train(...)` in fondo al file:

  num_samples   : numero di video da usare per il training.
                  None = tutto il dataset.

  num_epochs    : numero di epoche di training (default: 20).

  batch_size    : numero di sample per batch (default: 8).
                  Riduci a 4 se vai in out-of-memory sulla GPU.

  learning_rate : lr per tutti i moduli tranne MobileNetV3 (default: 1e-4).

  mobilenet_lr  : lr applicato a MobileNetV3 quando viene sbloccato (default: 1e-5).
                  Più basso del lr principale perché i pesi sono già pretrained.

  warmup_epochs : epoche di warmup del learning rate (default: 2).
                  Durante il warmup il lr sale linearmente fino al valore base.

  unfreeze_epoch: epoca a cui MobileNetV3 viene sbloccato per il fine-tuning (default: 4).
                  Nelle prime epoche i pesi di MobileNetV3 sono congelati.

  max_frames    : numero massimo di frame per video (default: 150).
                  Sequenze più lunghe vengono troncate.

  d_model       : dimensione interna del Transformer (default: 512).
                  Riduci a 256 per diminuire i parametri del modello.

  dropout       : dropout globale (default: 0.1).

CHECKPOINTS
-----------
Salvati in checkpoints/ ad ogni epoca.
Le metriche di ogni epoca sono disponibili in checkpoints/metrics_epoch_XXX.json.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from transformers import BartTokenizer

from src.models import SignLanguageTranslator
from src.data.how2sign_loader import How2SignDataset, collate_fn


# ----------------------------------------------------------------------
# Utilità
# ----------------------------------------------------------------------

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def freeze_mobilenet(model: SignLanguageTranslator):
    for param in model.cnn_branch.spatial_encoder.parameters():
        param.requires_grad = False
    print("[Training] MobileNetV3 congelato.")


def unfreeze_mobilenet(model: SignLanguageTranslator, lr: float, optimizer: torch.optim.Optimizer):
    for param in model.cnn_branch.spatial_encoder.parameters():
        param.requires_grad = True
    optimizer.add_param_group({
        'params': model.cnn_branch.spatial_encoder.parameters(),
        'lr': lr,
    })
    print(f"[Training] MobileNetV3 sbloccato con lr={lr}.")


def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + linear decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return max(
            0.0,
            (total_steps - current_step) / max(1, total_steps - warmup_steps)
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: SignLanguageTranslator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    metrics_path = checkpoint_dir / f"metrics_epoch_{epoch:03d}.json"
    with open(metrics_path, 'w') as f:
        json.dump({'epoch': epoch, **metrics}, f, indent=2)
    print(f"[Checkpoint] Salvato: {path.name}")


# ----------------------------------------------------------------------
# Train step
# ----------------------------------------------------------------------

def train_epoch(
    model: SignLanguageTranslator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.CrossEntropyLoss,
    tokenizer: BartTokenizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    for step, (landmarks, video_frames, padding_mask, sentences, lengths) in enumerate(loader):
        landmarks    = landmarks.to(device)
        video_frames = video_frames.to(device)
        padding_mask = padding_mask.to(device)

        # Tokenizza le frasi target
        encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        target_tokens = encoded['input_ids'].to(device)  # [B, L]

        # Input decoder: tutti i token tranne l'ultimo
        # Target loss:   tutti i token tranne il primo (BOS)
        decoder_input  = target_tokens[:, :-1]  # [B, L-1]
        decoder_target = target_tokens[:, 1:]   # [B, L-1]

        logits = model(
            landmarks=landmarks,
            video_frames=video_frames,
            padding_mask=padding_mask,
            target_tokens=decoder_input,
        )  # [B, L-1, vocab_size]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        n_tokens = (decoder_target != tokenizer.pad_token_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if step % 50 == 0:
            elapsed = time.time() - start
            avg_loss = total_loss / max(1, total_tokens)
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Epoca {epoch}/{num_epochs} | "
                f"Step {step}/{len(loader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tempo: {format_time(elapsed)}"
            )

    return {'train_loss': total_loss / max(1, total_tokens)}


# ----------------------------------------------------------------------
# Training loop principale
# ----------------------------------------------------------------------

def train(
    # Paths
    csv_path: Path,
    landmarks_dir: Path,
    cropped_dir: Path,
    checkpoint_dir: Path = Path('checkpoints'),
    # Dati
    num_samples: int | None = None,
    max_frames: int = 150,
    require_video: bool = True,
    # Training
    num_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    mobilenet_lr: float = 1e-5,
    warmup_epochs: int = 2,
    unfreeze_epoch: int = 4,
    # Modello
    d_model: int = 512,
    dropout: float = 0.1,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] Device: {device}")

    # --- Dataset ----------------------------------------------------------
    dataset = How2SignDataset(
        csv_path=csv_path,
        landmarks_dir=landmarks_dir,
        cropped_dir=cropped_dir,
        max_frames=max_frames,
        num_samples=num_samples,
        require_video=require_video,
    )
    print(f"[Training] {len(dataset)} sample di training.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Modello ----------------------------------------------------------
    model = SignLanguageTranslator(d_model=d_model, dropout=dropout).to(device)
    freeze_mobilenet(model)

    trainable_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and 'spatial_encoder' not in name
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

    total_steps  = num_epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # --- Loop -------------------------------------------------------------
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"  EPOCA {epoch}/{num_epochs}")
        print(f"{'='*60}")

        if epoch == unfreeze_epoch:
            unfreeze_mobilenet(model, mobilenet_lr, optimizer)

        epoch_start = time.time()

        metrics = train_epoch(
            model, loader, optimizer, scheduler,
            criterion, tokenizer, device, epoch, num_epochs,
        )

        epoch_time = time.time() - epoch_start
        metrics['epoch_time'] = format_time(epoch_time)

        print(f"\n  Tempo epoca : {format_time(epoch_time)}")
        print(f"  Train Loss  : {metrics['train_loss']:.4f}")

        save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir)

        # Salva il modello con loss migliore
        if metrics['train_loss'] < best_loss:
            best_loss = metrics['train_loss']
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pt')
            print(f"  ★ Nuova best loss: {best_loss:.4f} — salvato best_model.pt")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    train(
        csv_path=Path('dataset/how2sign_realigned_train.csv'),
        landmarks_dir=Path('dataset/landmarks_normalized'),
        cropped_dir=Path('dataset/cropped'),
        checkpoint_dir=Path('checkpoints'),
        num_samples=1000,  # Usa None per tutto il dataset
        num_epochs=20,
        batch_size=8,
        learning_rate=1e-4,
        mobilenet_lr=1e-5,
        warmup_epochs=2,
        unfreeze_epoch=4,
    )