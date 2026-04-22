"""
PHASE 1 TRAINING CONFIGURATION
===============================

File di configurazione dedicato alla Phase 1 di addestramento.
Modifica i parametri qui per personalizzare il training senza
modificare train_phase1.py.

Utilizzo:
    from config.phase1_config import Phase1Config, TrainingHyperparameters
    
    config = Phase1Config()
    hyperparams = TrainingHyperparameters()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch


# =============================================================================
# HYPERPARAMETERS SECTION
# =============================================================================

@dataclass
class TrainingHyperparameters:
    """
    Hyperparameters per Phase 1 Training.
    
    Modifica questi valori per ottimizzare il training.
    """
    
    # ==================== LEARNING ====================
    # Learning rate iniziale
    initial_learning_rate: float = 1e-3
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau" o "step"
    scheduler_factor: float = 0.5  # LR *= factor quando patience esaurito
    scheduler_patience: int = 5  # Epoch senza miglioramento prima di ridurre LR
    scheduler_min_lr: float = 5e-5  # LR minimo
    
    # Optimizer
    optimizer_type: str = "adamw"  # "adamw", "adam", "sgd"
    weight_decay: float = 1e-5  # L2 regularization (AdamW)
    adam_betas: tuple = (0.9, 0.999)  # For AdamW
    adam_epsilon: float = 1e-8  # For AdamW
    
    # ==================== GRADIENT ====================
    # Gradient clipping (stabilità per Transformer)
    gradient_clip_norm: float = 1.0
    gradient_clip_value: Optional[float] = None
    
    # Gradient accumulation (per simulare batch size maggiore)
    use_gradient_accumulation: bool = False
    accumulation_steps: int = 4
    
    # ==================== LOSS FUNCTION ====================
    # Loss function
    loss_type: str = "cross_entropy"  # "cross_entropy", "focal", "label_smoothing"
    loss_ignore_index: int = 0  # Index per padding (ignorato in loss)
    label_smoothing: float = 0.0  # Smooth label (0.1 è ragionevole)
    
    # ==================== REGULARIZATION ====================
    # Dropout
    dropout_rate: float = 0.1
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 15  # Epoch senza miglioramento prima di stop
    
    # Mixup / Cutmix (data augmentation)
    use_mixup: bool = False
    mixup_alpha: float = 0.2


@dataclass
class ModelHyperparameters:
    """
    Hyperparameters per l'architettura del modello.
    """
    
    # ==================== ENCODER - LANDMARKS BRANCH ====================
    # Transformer Encoder (per landmarks)
    landmark_dim: int = 2108  # Dimensione input landmarks
    transformer_encoder_hidden_dim: int = 512  # Dimensione embedding
    transformer_encoder_num_layers: int = 4  # Numero layer
    transformer_encoder_num_heads: int = 8  # Multi-head attention heads
    transformer_encoder_ff_dim: int = 2048  # Dimensione feed-forward
    transformer_encoder_dropout: float = 0.1
    transformer_encoder_activation: str = "relu"
    
    # ==================== ENCODER - VIDEO BRANCH ====================
    # MobileNet (congelato)
    freeze_mobilenet: bool = True  # ← IMPORTANTE per Phase 1!
    mobilenet_pretrained: bool = True
    
    # Conv1D (video processing)
    conv1d_out_channels: int = 512
    conv1d_kernel_size: int = 3
    conv1d_dropout: float = 0.1
    
    # GRU (sequential modeling)
    gru_hidden_dim: int = 512
    gru_num_layers: int = 1
    gru_dropout: float = 0.1
    gru_bidirectional: bool = False  # Unidirezionale per efficienza
    
    # ==================== FUSION MODULE ====================
    fusion_hidden_dim: int = 512  # Dimensione output fusion
    fusion_type: str = "dense"  # "dense", "attention", "weighted"
    
    # ==================== DECODER ====================
    # Transformer Decoder (text generation)
    text_vocab_size: int = 10000  # Dimensione vocabolario (adjust!)
    text_max_len: int = 512  # Massima lunghezza sequenza testo
    transformer_decoder_hidden_dim: int = 512
    transformer_decoder_num_layers: int = 4
    transformer_decoder_num_heads: int = 8
    transformer_decoder_ff_dim: int = 2048
    transformer_decoder_dropout: float = 0.1
    
    # ==================== VIDEO PROCESSING ====================
    video_max_frames: int = 150  # Max numero frame per video
    video_height: int = 224
    video_width: int = 224
    video_frame_resize: tuple = (224, 224)


@dataclass
class DataHyperparameters:
    """
    Hyperparameters per il dataset.
    """
    
    # ==================== DATASET ====================
    # Numero campioni per modalità
    num_samples_rapid: int = 100  # Testing rapido
    num_samples_standard: int = 1000  # Standard Phase 1 ← CONSIGLIATO
    num_samples_extended: int = 5000  # Extended training
    num_samples_full: int = 50000  # Full dataset (per Phase 2+)
    
    # ==================== DATALOADER ====================
    batch_size_rapid: int = 32
    batch_size_standard: int = 32
    batch_size_gpu_limited: int = 16  # Se memoria GPU limitata
    batch_size_extended: int = 64
    
    # Numero worker per parallel loading
    num_workers: int = 4
    
    # Prefetch
    pin_memory: bool = True
    
    # ==================== TRAIN/VAL SPLIT ====================
    # Train/Validation split
    validation_split: float = 0.2  # 80% train, 20% validation
    
    # ==================== DATA AUGMENTATION ====================
    use_augmentation: bool = True
    
    # Augmentation parametri
    augment_landmarks: bool = True
    augment_landmarks_noise_std: float = 0.01
    augment_landmarks_rotation_deg: float = 5.0
    
    augment_video: bool = True
    augment_video_brightness: float = 0.2
    augment_video_contrast: float = 0.2


# =============================================================================
# TRAINING MODES (Preset Configurations)
# =============================================================================

def get_rapid_config() -> tuple:
    """Configurazione per testing rapido (~5 min su GPU)."""
    training = TrainingHyperparameters()
    model = ModelHyperparameters()
    data = DataHyperparameters()
    
    data.num_samples = data.num_samples_rapid
    data.batch_size = data.batch_size_rapid
    
    return training, model, data


def get_standard_config() -> tuple:
    """Configurazione standard Phase 1 (~1-2 ore su GPU). ← CONSIGLIATO"""
    training = TrainingHyperparameters()
    model = ModelHyperparameters()
    data = DataHyperparameters()
    
    data.num_samples = data.num_samples_standard
    data.batch_size = data.batch_size_standard
    
    return training, model, data


def get_extended_config() -> tuple:
    """Configurazione estesa (~4-6 ore su GPU A100)."""
    training = TrainingHyperparameters()
    model = ModelHyperparameters()
    data = DataHyperparameters()
    
    # Aumenta sample e batch
    data.num_samples = data.num_samples_extended
    data.batch_size = data.batch_size_extended
    
    # Optimizer più aggressivo
    training.initial_learning_rate = 2e-3
    
    return training, model, data


def get_limited_gpu_config() -> tuple:
    """Configurazione per GPU limitata (~2-4 GB)."""
    training = TrainingHyperparameters()
    model = ModelHyperparameters()
    data = DataHyperparameters()
    
    # Ridotti batch e model dims
    data.num_samples = data.num_samples_standard
    data.batch_size = data.batch_size_gpu_limited
    
    model.transformer_encoder_hidden_dim = 256
    model.transformer_decoder_hidden_dim = 256
    model.gru_hidden_dim = 256
    
    return training, model, data


# =============================================================================
# PHASE 1 MASTER CONFIGURATION CLASS
# =============================================================================

class Phase1MasterConfig:
    """
    Configurazione master che combina tutti i parametri.
    Usare questo come punto di partenza.
    """
    
    def __init__(
        self,
        mode: str = "standard",  # "rapid", "standard", "extended", "limited_gpu"
        device: Optional[torch.device] = None,
    ):
        """
        Inizializza Phase1Config con preset.
        
        Args:
            mode: Modalità training
            device: Device (auto-detect se None)
        """
        
        # Seleziona config basata su mode
        if mode == "rapid":
            training, model, data = get_rapid_config()
        elif mode == "standard":
            training, model, data = get_standard_config()
        elif mode == "extended":
            training, model, data = get_extended_config()
        elif mode == "limited_gpu":
            training, model, data = get_limited_gpu_config()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.training = training
        self.model = model
        self.data = data
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Directories
        self.checkpoint_dir = Path("checkpoints/phase1")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path("logs/phase1")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Misc
        self.seed = 42
        self.log_frequency = 50  # Log ogni N batch
        self.save_frequency = 1  # Salva checkpoint ogni N epoch
        self.save_best_only = True
        
        self.mode = mode
    
    def print_summary(self):
        """Stampa resoconto configurazione."""
        print("\n" + "="*70)
        print("PHASE 1 CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Mode: {self.mode}")
        print(f"Device: {self.device}")
        print(f"Samples: {self.data.num_samples}")
        print(f"Batch Size: {self.data.batch_size if hasattr(self.data, 'batch_size') else 'N/A'}")
        print(f"Learning Rate: {self.training.initial_learning_rate}")
        print(f"Gradient Clip: {self.training.gradient_clip_norm}")
        print(f"Freeze MobileNet: {self.model.freeze_mobilenet}")
        print("="*70 + "\n")
    
    def to_dict(self) -> dict:
        """Converti a dizionario per logging."""
        return {
            'mode': self.mode,
            'device': str(self.device),
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__ if hasattr(self.data, '__dict__') else {},
        }


# =============================================================================
# PRESETS FOR QUICK USE
# =============================================================================

# Preset rapid per testing
PHASE1_RAPID = Phase1MasterConfig(mode="rapid")

# Preset standard (CONSIGLIATO per Phase 1)
PHASE1_STANDARD = Phase1MasterConfig(mode="standard")

# Preset extended
PHASE1_EXTENDED = Phase1MasterConfig(mode="extended")

# Preset per GPU limitata
PHASE1_LIMITED_GPU = Phase1MasterConfig(mode="limited_gpu")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """Esempio di utilizzo della configurazione."""
    
    print("Loading Phase 1 Configuration Examples...\n")
    
    # Rapid mode
    print("1. RAPID MODE (Testing)")
    config_rapid = Phase1MasterConfig(mode="rapid")
    config_rapid.print_summary()
    
    # Standard mode (consigliato)
    print("2. STANDARD MODE (Recommended for Phase 1)")
    config_standard = Phase1MasterConfig(mode="standard")
    config_standard.print_summary()
    
    # Extended mode
    print("3. EXTENDED MODE (Long Training)")
    config_extended = Phase1MasterConfig(mode="extended")
    config_extended.print_summary()
    
    # Limited GPU mode
    print("4. LIMITED GPU MODE")
    config_limited = Phase1MasterConfig(mode="limited_gpu")
    config_limited.print_summary()
