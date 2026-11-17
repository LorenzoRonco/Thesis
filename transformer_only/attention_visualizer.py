"""
attention_visualizer.py

Strumento per ispezionare e visualizzare i pesi di attenzione della Cross-Attention.
Utile per diagnosticare se il decoder sta ignorando l'encoder.

Uso:
    visualizer = AttentionVisualizer(model)
    visualizer.capture_attention(batch, device)
    visualizer.save_heatmaps(output_dir, prefix="epoch_001")
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

try:
    # Force a non-interactive backend to avoid Tkinter/thread issues in training workers.
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[AttentionVisualizer] Warning: matplotlib not installed. Heatmaps will not be saved.")


class AttentionVisualizer:
    """Cattura e visualizza i pesi di Cross-Attention durante la forward pass."""
    
    def __init__(self, model):
        """
        Args:
            model: SignLanguageTransformer
        """
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Registra hooks su tutti i layer decoder per catturare i pesi di attenzione."""
        for layer_idx, dec_layer in enumerate(self.model.decoder.layers):
            # Hook sulla multihead attention della cross-attention
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output è il risultato dell'attenzione
                    # Per multihead attention, vogliamo i pesi prima del concat
                    # Utilizziamo l'input del modulo per ottenere le attention weights
                    # In PyTorch, per MultiheadAttention non è diretto, quindi usiamo
                    # il metodo alternativo: salvare lo stato durante forward di attenzione manuale
                    pass
                return hook
            
            # Registra hook sull'output della multihead attention
            # Nota: PyTorch MultiheadAttention non espone direttamente i pesi,
            # quindi useremo un approccio alternativo: catturiamo attentione manualmente
        
        # Alternativa: salviamo gli stati interni durante forward
    
    def capture_attention_manual(self, src, tgt_input, src_mask, tgt_mask, device):
        """
        Cattura i pesi di attenzione eseguendo forward pass manuale del decoder.
        
        Args:
            src: (B, T, feat_dim)
            tgt_input: (B, L)
            src_mask: (B, T) bool tensor
            tgt_mask: (L, L) bool tensor
            device: torch device
        
        Returns:
            Dict con i pesi di attenzione per layer
        """
        self.model.eval()
        
        attn_weights_per_layer = {}
        
        with torch.no_grad():
            # Encode
            src_emb = self.model.src_embed(src)  # (B, T, d_model)
            memory = self.model.encoder(src_emb, src_key_padding_mask=src_mask)  # (B, T, d_model)
            
            # Embed target
            tgt_emb = self.model.tgt_embed(tgt_input)  # (B, L, d_model)
            
            # Forward attraverso ogni layer del decoder manualmente
            # per poter catturare i pesi di cross-attention
            dec_input = tgt_emb
            
            for layer_idx, dec_layer in enumerate(self.model.decoder.layers):
                # Prendiamo la cross-attention e facciamo il calcolo manualmente
                cross_attn = dec_layer.multihead_attn
                B, L, D = dec_input.shape
                _, T, _ = memory.shape
                num_heads = cross_attn.num_heads
                head_dim = D // num_heads
                
                # Skip self-attention, andiamo diretto alla cross-attention
                # (in un vero forward pass dovremmo fare anche self-attention+FFN, 
                # ma per il debug vogliamo solo i pesi di cross-attention)
                
                # Cross-attention: Query=decoder, Key=Value=encoder
                # Apply layer norm first (pre-LN)
                norm_dec = dec_layer.norm2(dec_input)
                norm_mem = memory  # Encoder è già normalizzato
                
                # Linear projections per Q, K, V
                q_weight, k_weight, v_weight = torch.split(cross_attn.in_proj_weight, D)
                q_bias, k_bias, v_bias = torch.split(cross_attn.in_proj_bias, D) if cross_attn.in_proj_bias is not None else (None, None, None)
                
                Q = F.linear(norm_dec, q_weight, q_bias)  # (B, L, D)
                K = F.linear(norm_mem, k_weight, k_bias)   # (B, T, D)
                V = F.linear(norm_mem, v_weight, v_bias)   # (B, T, D)
                
                # Reshape for multi-head: (B, seq, D) -> (B, seq, H, d_h) -> (B, H, seq, d_h)
                Q = Q.reshape(B, L, num_heads, head_dim).transpose(1, 2)  # (B, H, L, d_h)
                K = K.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # (B, H, T, d_h)
                V = V.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # (B, H, T, d_h)
                
                # Scaled dot-product attention
                scaling = 1.0 / (head_dim ** 0.5)
                scores = torch.matmul(Q, K.transpose(-2, -1)) * scaling  # (B, H, L, T)
                
                # Apply key padding mask (src_mask: True = padding)
                if src_mask is not None:
                    # src_mask shape: (B, T), True = padding to mask out
                    # Expand to (B, 1, 1, T) for broadcasting
                    mask_src = src_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
                    scores = scores.masked_fill(mask_src, float('-inf'))
                
                # Softmax
                attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, T)
                
                # Replace NaN with 0 (emerge dalle masked positions)
                attn_weights = torch.where(
                    torch.isnan(attn_weights),
                    torch.zeros_like(attn_weights),
                    attn_weights
                )
                
                # Media su batch e heads per visualizzazione
                attn_weights_avg = attn_weights.mean(dim=(0, 1))  # (L, T)
                
                attn_weights_per_layer[layer_idx] = {
                    "full": attn_weights.cpu().numpy(),   # (B, H, L, T)
                    "avg": attn_weights_avg.cpu().numpy(),  # (L, T)
                }
                
                # Applica attenzione a values
                context = torch.matmul(attn_weights, V)  # (B, H, L, d_h)
                
                # Concat e reshape: (B, H, L, d_h) -> (B, L, H, d_h) -> (B, L, D)
                context = context.transpose(1, 2).contiguous().reshape(B, L, D)  # (B, L, D)
                
                # Output projection
                attn_out = F.linear(context, cross_attn.out_proj.weight, cross_attn.out_proj.bias)
                
                # Per il forward pass semplificato, aggiorna dec_input
                # (In realtà dovremmo anche fare self-attention, ma per il debug
                # ci basta solo la cross-attention)
                dec_input = attn_out
        
        self.attention_weights = attn_weights_per_layer
        return attn_weights_per_layer
    
    def save_heatmaps(self, output_dir: Path, prefix: str = ""):
        """
        Salva le heatmap di attenzione come immagini PNG.
        
        Args:
            output_dir: directory di output
            prefix: prefisso per il nome file (es 'epoch_001')
        """
        if not HAS_MATPLOTLIB:
            print("[AttentionVisualizer] matplotlib not available, skipping heatmap save")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_idx, weights_dict in self.attention_weights.items():
            attn_avg = weights_dict["avg"]  # (L, T)
            
            # Crea figura
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Visualizza heatmap
            im = ax.imshow(attn_avg, cmap='viridis', aspect='auto', origin='lower')
            
            ax.set_xlabel('Encoder Positions (Source Landmarks)')
            ax.set_ylabel('Decoder Positions (Target Tokens)')
            ax.set_title(f'Cross-Attention Weights - Decoder Layer {layer_idx}')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attention Weight')
            
            # Salva
            fname = output_dir / f"{prefix}_layer_{layer_idx:02d}_attn.png"
            plt.savefig(fname, dpi=100, bbox_inches='tight')
            print(f"[AttentionVisualizer] Saved: {fname}")
            
            plt.close(fig)
    
    def print_padding_mask_info(self, src_mask):
        """
        Stampa informazioni diagnostiche sul padding mask.
        
        Args:
            src_mask: (B, T) bool tensor, True = padding
        """
        if src_mask is None:
            print("[AttentionVisualizer] src_mask is None (no padding)")
            return
        
        print("\n" + "-"*80)
        print("PADDING MASK DIAGNOSTICS")
        print("-"*80)
        
        B, T = src_mask.shape
        print(f"Mask shape: (B={B}, T={T})")
        
        # Numero di padding per batch
        padding_per_batch = src_mask.sum(dim=1)
        print(f"Padding per batch: {padding_per_batch.tolist()}")
        print(f"Total padding: {src_mask.sum().item()}/{B*T}")
        
        # Ratio
        ratio = src_mask.sum().item() / (B*T)
        print(f"Padding ratio: {ratio:.1%}")
        
        # Check if all True (tutto padding - problema!)
        if src_mask.all():
            print("[WARNING] ⚠️  ALL positions are padding! Attention will be all NaN!")
        
        # Check if all False (nessun padding)
        if not src_mask.any():
            print("[INFO] No padding detected")
        
        print()
    
    def print_attention_summary(self, top_k: int = 5):
        """
        Stampa un riassunto dei pesi di attenzione.
        Utile per diagnosticare problemi.
        
        Args:
            top_k: quanti token visualizzare per ogni posizione del decoder
        """
        print("\n" + "="*80)
        print("CROSS-ATTENTION WEIGHTS SUMMARY")
        print("="*80)
        
        for layer_idx, weights_dict in self.attention_weights.items():
            attn_avg = weights_dict["avg"]  # (L, T)
            L, T = attn_avg.shape
            
            print(f"\nLayer {layer_idx}:")
            print(f"  Shape: (L={L}, T={T})")
            
            # Statistiche globali
            max_weight = attn_avg.max()
            min_weight = attn_avg.min()
            mean_weight = attn_avg.mean()
            
            print(f"  Max weight: {max_weight:.4f}")
            print(f"  Min weight: {min_weight:.4f}")
            print(f"  Mean weight: {mean_weight:.4f}")
            
            # Controlla se i pesi sono uniformi (indizio di problema)
            entropy = -np.sum(attn_avg * np.log(attn_avg + 1e-10), axis=-1)  # Per ogni posizione decoder
            mean_entropy = entropy.mean()
            max_entropy = np.log(T)  # Entropia massima per distribuzione uniforme
            entropy_ratio = mean_entropy / max_entropy
            
            print(f"  Mean entropy: {mean_entropy:.4f} (max: {max_entropy:.4f})")
            print(f"  Entropy ratio: {entropy_ratio:.2%} (100% = uniforme)")
            
            if entropy_ratio > 0.9:
                print(f"  ⚠️  ATTENZIONE QUASI UNIFORME! Il decoder potrebbe ignorare l'encoder!")
            elif entropy_ratio < 0.1:
                print(f"  ✓ Attenzione focalizzata su specifiche posizioni encoder")
            
            # Mostra top-k posizioni encoder per alcuni token decoder
            print(f"\n  Top {top_k} encoder positions per selected decoder tokens:")
            for dec_pos in [0, L//4, L//2, 3*L//4, L-1]:
                if dec_pos >= L:
                    continue
                weights_at_pos = attn_avg[dec_pos]  # (T,)
                top_indices = np.argsort(-weights_at_pos)[:top_k]
                top_values = weights_at_pos[top_indices]
                print(f"    Decoder pos {dec_pos:3d}: ", end="")
                for enc_pos, weight in zip(top_indices, top_values):
                    print(f"[enc[{enc_pos:3d}]={weight:.4f}] ", end="")
                print()


def debug_attention_on_batch(model, batch, tokenizer, device, output_dir=None):
    """
    Funzione di convenienza per visualizzare attenzione su un singolo batch.
    
    Args:
        model: SignLanguageTransformer
        batch: batch dal dataloader
        tokenizer: per etichette
        device: torch device
        output_dir: dove salvare le heatmap (opzionale)
    """
    src = batch["src"].to(device)
    tgt_input = batch["tgt_input"].to(device)
    src_mask = batch["src_key_padding_mask"].to(device) if "src_key_padding_mask" in batch else None
    tgt_mask = model._make_causal_mask(tgt_input.shape[1], device)
    
    visualizer = AttentionVisualizer(model)
    
    # Debug del padding mask
    visualizer.print_padding_mask_info(src_mask)
    
    visualizer.capture_attention_manual(src, tgt_input, src_mask, tgt_mask, device)
    visualizer.print_attention_summary(top_k=5)
    
    if output_dir:
        visualizer.save_heatmaps(Path(output_dir), prefix="batch_debug")
    
    return visualizer


def run_attention_visualization_on_loader(
    model,
    loader,
    tokenizer,
    device,
    output_dir,
    epoch: int = None,
    max_batches: int = 1,
    top_k: int = 5,
    save_heatmaps: bool = True,
):
    """
    Run attention capture on up to `max_batches` from `loader` and save summaries/heatmaps.

    Intended to be called from the training loop (e.g. every N epochs).

    Args:
        model: SignLanguageTransformer instance
        loader: DataLoader yielding batches compatible with `debug_attention_on_batch`
        tokenizer: SentenceTokenizer instance
        device: torch.device or string
        output_dir: path-like where heatmaps will be written
        epoch: optional int epoch number for filename prefixes
        max_batches: how many batches to process (default 1)
        top_k: how many top encoder positions to print in summary
        save_heatmaps: whether to save PNG heatmaps (requires matplotlib)

    Returns:
        List of AttentionVisualizer instances produced for each processed batch
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve training/eval state
    was_training = model.training
    model.eval()

    visualizers = []
    device = torch.device(device if isinstance(device, str) and torch.cuda.is_available() and device == "cuda" else device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            vis = AttentionVisualizer(model)

            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            src_mask = batch.get("src_key_padding_mask", None)
            if src_mask is not None:
                src_mask = src_mask.to(device)

            # Print padding info and capture attention
            vis.print_padding_mask_info(src_mask)
            vis.capture_attention_manual(src, tgt_input, src_mask, None, device)
            vis.print_attention_summary(top_k=top_k)

            if save_heatmaps:
                prefix = f"epoch_{epoch:03d}" if epoch is not None else "epoch_unknown"
                prefix = f"{prefix}_batch_{batch_idx:02d}"
                vis.save_heatmaps(output_dir, prefix=prefix)

            visualizers.append(vis)

    # restore training state
    if was_training:
        model.train()

    return visualizers
