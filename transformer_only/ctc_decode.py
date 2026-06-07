"""
CTC Greedy Decoding Diagnostics

Funzione autonoma per decodificare le predizioni CTC dell'Encoder e diagnosticare
se l'Encoder sta imparando i segni correttamente.
"""

import torch
from typing import List, Dict


def decode_ctc_predictions(
    ctc_logits: torch.Tensor,
    idx2token: Dict[int, str],
    blank_idx: int = 0,
    pad_idx: int = 1,
) -> List[str]:
    """
    CTC Greedy Decoding: converte i logit CTC del frame-by-frame dell'encoder
    in sequenze di token decodificate.

    Args:
        ctc_logits: Tensor di shape [Batch, Time, vocab_size]
                    Logit non-softmax per ogni frame (output diretto dell'encoder)
        idx2token: Dizionario {token_id: "token_string"}
        blank_idx: Indice del token CTC blank (default 0)
        pad_idx: Indice del token padding (default 1)

    Returns:
        List[str]: Lista di stringhe decodificate (una per elemento del batch)

    Logica CTC:
        1. Argmax greedy su vocab_size → sequenza di token per frame
        2. Rimuovi token consecutivi identici (collassali in uno)
        3. Rimuovi blank_idx
        4. Rimuovi pad_idx
        5. Converti indici in token string
        6. Unisci con spazi per formare la frase finale
    """
    # Ensure we're on CPU for numpy operations
    device = ctc_logits.device
    ctc_logits = ctc_logits.detach()

    # Greedy: argmax sulla dimensione vocab
    # Shape: [Batch, Time, vocab_size] → [Batch, Time]
    predicted_indices = torch.argmax(ctc_logits, dim=-1)

    batch_size = predicted_indices.shape[0]
    decoded_sequences = []

    for batch_idx in range(batch_size):
        # Sequenza di indici per questo elemento del batch
        frame_sequence = predicted_indices[batch_idx].cpu().tolist()

        # Step 1: Rimuovi token consecutivi identici (collapse)
        collapsed = []
        for i, token_id in enumerate(frame_sequence):
            # Aggiungi token solo se diverso dal precedente
            if i == 0 or token_id != frame_sequence[i - 1]:
                collapsed.append(token_id)

        # Step 2: Rimuovi blank_idx e pad_idx
        cleaned = [
            token_id
            for token_id in collapsed
            if token_id != blank_idx and token_id != pad_idx
        ]

        # Step 3: Converti indici in stringhe
        tokens_str = []
        for token_id in cleaned:
            token_id = int(token_id)
            # Usa idx2token se disponibile, altrimenti <unk>
            if token_id in idx2token:
                tokens_str.append(idx2token[token_id])
            else:
                tokens_str.append("<unk>")

        # Step 4: Unisci con spazi per formare la frase finale
        decoded_sentence = " ".join(tokens_str)
        decoded_sequences.append(decoded_sentence)

    return decoded_sequences


def decode_ctc_batch_diagnostics(
    ctc_logits: torch.Tensor,
    idx2token: Dict[int, str],
    tokenizer,
    max_examples: int = 5,
    blank_idx: int = 0,
    pad_idx: int = 1,
) -> dict:
    """
    Versione diagnostica che ritorna anche statistiche di decodifica.

    Args:
        ctc_logits: Tensor [Batch, Time, vocab_size]
        idx2token: Dizionario {token_id: "token_string"}
        tokenizer: Tokenizer object (per accedere a special tokens)
        max_examples: Numero massimo di esempi da stampare
        blank_idx, pad_idx: Indici speciali

    Returns:
        dict: {
            'decoded': List[str],
            'stats': {
                'avg_decoded_len': float,
                'num_blanks_removed': int,
                'num_pads_removed': int,
                'avg_collapses_per_seq': float,
            },
            'examples': List[str]  (prime max_examples predizioni)
        }
    """
    decoded = decode_ctc_predictions(ctc_logits, idx2token, blank_idx, pad_idx)

    # Calcola statistiche diagnostiche
    predicted_indices = torch.argmax(ctc_logits, dim=-1).cpu()
    batch_size = predicted_indices.shape[0]

    total_blanks = 0
    total_pads = 0
    total_collapses = 0
    total_decoded_len = 0

    for batch_idx in range(batch_size):
        frame_seq = predicted_indices[batch_idx].tolist()

        # Conta blank e pad
        blanks_in_seq = sum(1 for t in frame_seq if t == blank_idx)
        pads_in_seq = sum(1 for t in frame_seq if t == pad_idx)
        total_blanks += blanks_in_seq
        total_pads += pads_in_seq

        # Conta collapse (token consecutivi identici)
        collapses = sum(1 for i in range(1, len(frame_seq)) if frame_seq[i] == frame_seq[i - 1])
        total_collapses += collapses

        # Lunghezza della sequenza decodificata (numero di token)
        decoded_len = len(decoded[batch_idx].split())
        total_decoded_len += decoded_len

    avg_decoded_len = total_decoded_len / max(1, batch_size)
    avg_collapses = total_collapses / max(1, batch_size)

    stats = {
        "avg_decoded_len": avg_decoded_len,
        "num_blanks_removed": total_blanks,
        "num_pads_removed": total_pads,
        "avg_collapses_per_seq": avg_collapses,
    }

    examples = decoded[: max(1, min(max_examples, len(decoded)))]

    return {
        "decoded": decoded,
        "stats": stats,
        "examples": examples,
    }
