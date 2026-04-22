"""
PHASE 1 - SETUP VERIFICATION SCRIPT
====================================

Script per verificare che l'ambiente e i modelli siano configurati
correttamente prima di avviare il training Phase 1.

Esecuzione:
  python verify_phase1_setup.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime


class ColorText:
    """Helper per colored terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def success(text):
        print(f"{ColorText.GREEN}✓ {text}{ColorText.RESET}")
    
    @staticmethod
    def error(text):
        print(f"{ColorText.RED}✗ {text}{ColorText.RESET}")
    
    @staticmethod
    def warning(text):
        print(f"{ColorText.YELLOW}⚠ {text}{ColorText.RESET}")
    
    @staticmethod
    def info(text):
        print(f"{ColorText.BLUE}ℹ {text}{ColorText.RESET}")
    
    @staticmethod
    def bold(text):
        print(f"{ColorText.BOLD}{text}{ColorText.RESET}")


class Phase1SetupVerifier:
    """
    Verifica che il setup Phase 1 sia corretto.
    """
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.project_root = Path(__file__).parent
    
    def run_all_checks(self):
        """Esegui tutti i check."""
        ColorText.bold("\n" + "="*70)
        ColorText.bold("PHASE 1 TRAINING - SETUP VERIFICATION")
        ColorText.bold("="*70)
        
        # 1. Python & PyTorch
        self._check_python_version()
        self._check_pytorch()
        self._check_cuda()
        
        # 2. Project Structure
        self._check_project_structure()
        
        # 3. Core Modules
        self._check_core_modules()
        
        # 4. Model Architecture
        self._check_model_architecture()
        
        # 5. Training Script
        self._check_training_script()
        
        # 6. GPU Memory (if CUDA available)
        if torch.cuda.is_available():
            self._check_gpu_memory()
        
        # Report
        self._print_report()
    
    def _check_python_version(self):
        """Verifica Python version."""
        ColorText.info("Checking Python version...")
        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        if version_info.major >= 3 and version_info.minor >= 8:
            ColorText.success(f"Python version: {version_str}")
            self.passed.append("Python >= 3.8")
        else:
            ColorText.error(f"Python version too old: {version_str} (need >= 3.8)")
            self.failed.append("Python version")
    
    def _check_pytorch(self):
        """Verifica PyTorch installation."""
        ColorText.info("Checking PyTorch...")
        
        try:
            pytorch_version = torch.__version__
            ColorText.success(f"PyTorch version: {pytorch_version}")
            self.passed.append(f"PyTorch {pytorch_version}")
            
            # Check tensor operations
            x = torch.randn(10, 10)
            y = x.sum()
            ColorText.success("PyTorch tensor operations: OK")
            self.passed.append("PyTorch operations")
        
        except Exception as e:
            ColorText.error(f"PyTorch error: {e}")
            self.failed.append("PyTorch")
    
    def _check_cuda(self):
        """Verifica CUDA availability."""
        ColorText.info("Checking CUDA...")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            
            ColorText.success(f"CUDA available: {device_name}")
            ColorText.success(f"CUDA version: {cuda_version}")
            ColorText.success(f"cuDNN version: {cudnn_version}")
            
            self.passed.append("CUDA GPU")
            
            # Test GPU tensor
            try:
                x = torch.randn(100, 100, device='cuda')
                y = (x @ x.T).sum()
                ColorText.success("GPU tensor operations: OK")
                self.passed.append("GPU operations")
            except Exception as e:
                ColorText.error(f"GPU operation failed: {e}")
                self.failed.append("GPU operations")
        else:
            ColorText.warning("CUDA not available - using CPU (SLOW)")
            self.warnings.append("No GPU detected")
    
    def _check_project_structure(self):
        """Verifica project directory structure."""
        ColorText.info("Checking project structure...")
        
        required_dirs = [
            "src/models",
            "src/preprocessing",
            "src/data",
            "dataset",
            "config",
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                ColorText.success(f"Directory exists: {dir_name}/")
                self.passed.append(f"Directory: {dir_name}")
            else:
                ColorText.warning(f"Directory missing: {dir_name}/ (may need setup)")
                self.warnings.append(f"Missing: {dir_name}")
        
        # Required files
        required_files = [
            "train_phase1.py",
            "src/models/__init__.py",
            "src/models/transformer_encoder.py",
            "src/models/transformer_decoder.py",
            "src/models/cnn_1d_gru_module.py",
            "src/models/end_to_end_seq2seq.py",
            "config/config.py",
        ]
        
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                ColorText.success(f"File exists: {file_name}")
                self.passed.append(f"File: {file_name}")
            else:
                ColorText.error(f"File missing: {file_name}")
                self.failed.append(f"Missing: {file_name}")
    
    def _check_core_modules(self):
        """Verifica che i moduli core si importino."""
        ColorText.info("Checking core modules...")
        
        modules_to_check = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
        ]
        
        for module_name, friendly_name in modules_to_check:
            try:
                __import__(module_name)
                ColorText.success(f"Module {friendly_name} imported")
                self.passed.append(f"Module: {friendly_name}")
            except ImportError as e:
                ColorText.error(f"Module {friendly_name} import failed: {e}")
                self.failed.append(f"Module: {friendly_name}")
        
        # Project modules
        sys.path.insert(0, str(self.project_root))
        
        project_modules = [
            ("src.models.transformer_encoder", "TransformerEncoder"),
            ("src.models.transformer_decoder", "TransformerDecoder"),
            ("src.models.cnn_1d_gru_module", "CNN1DGRUModule"),
            ("src.models.end_to_end_seq2seq", "SignLanguageTranslationModel"),
        ]
        
        for module_path, class_name in project_modules:
            try:
                module = __import__(module_path, fromlist=[class_name])
                ColorText.success(f"Module {module_path} imported")
                self.passed.append(f"Module: {module_path}")
            except ImportError as e:
                ColorText.error(f"Module {module_path} import failed: {e}")
                self.failed.append(f"Module: {module_path}")
    
    def _check_model_architecture(self):
        """Verifica model instantiation."""
        ColorText.info("Checking model architecture...")
        
        sys.path.insert(0, str(self.project_root))
        
        try:
            from src.models.end_to_end_seq2seq import SignLanguageTranslationModel
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create tiny model for testing
            model = SignLanguageTranslationModel(
                video_hidden_dim=128,  # Small for fast testing
                landmark_dim=2108,
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_heads=4,
                text_vocab_size=5000,
                text_max_len=256,
                device=device,
            )
            
            # Sposta modello e tutti i parametri sul device (GPU/CPU)
            model = model.to(device)
            
            ColorText.success("SignLanguageTranslationModel created successfully")
            self.passed.append("Model instantiation")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            ColorText.success(f"Total parameters: {total_params:,}")
            ColorText.success(f"Trainable parameters: {trainable_params:,}")
            self.passed.append("Model parameters counted")
            
            # Test forward pass (with small tensors)
            ColorText.info("Testing forward pass...")
            batch_size = 2
            seq_len = 10
            
            landmarks = torch.randn(batch_size, seq_len, 2108, device=device)
            video_frames = torch.randn(batch_size, seq_len, 3, 64, 64, device=device)
            target_tokens = torch.randint(0, 5000, (batch_size, 256), device=device)
            
            with torch.no_grad():
                output = model(
                    landmarks=landmarks,
                    video_frames=video_frames,
                    target_tokens=target_tokens,
                )
            
            if output is not None:
                ColorText.success(f"Forward pass successful, output shape: {output.shape}")
                self.passed.append("Forward pass")
            else:
                ColorText.error("Forward pass returned None")
                self.failed.append("Forward pass")
        
        except Exception as e:
            ColorText.error(f"Model instantiation failed: {e}")
            self.failed.append("Model architecture")
            import traceback
            print(traceback.format_exc())
    
    def _check_training_script(self):
        """Verifica training script."""
        ColorText.info("Checking training script...")
        
        train_script = self.project_root / "train_phase1.py"
        
        if train_script.exists():
            ColorText.success("train_phase1.py found")
            self.passed.append("Training script")
            
            # Basic syntax check
            try:
                with open(train_script, 'r') as f:
                    code = f.read()
                
                compile(code, str(train_script), 'exec')
                ColorText.success("train_phase1.py syntax valid")
                self.passed.append("Training script syntax")
            
            except SyntaxError as e:
                ColorText.error(f"Syntax error in train_phase1.py: {e}")
                self.failed.append("Training script syntax")
        else:
            ColorText.error("train_phase1.py not found")
            self.failed.append("Training script")
    
    def _check_gpu_memory(self):
        """Verifica GPU memory availability."""
        ColorText.info("Checking GPU memory...")
        
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            
            ColorText.success(f"GPU memory total: {gpu_memory:.1f} GB")
            ColorText.success(f"GPU memory allocated: {allocated:.1f} GB")
            ColorText.success(f"GPU memory available: {gpu_memory - allocated:.1f} GB")
            
            if gpu_memory >= 6:
                ColorText.success("GPU memory sufficient for Phase 1")
                self.passed.append("GPU memory")
            else:
                ColorText.warning(f"GPU memory low ({gpu_memory:.1f} GB)")
                self.warnings.append("Low GPU memory")
        
        except Exception as e:
            ColorText.error(f"GPU memory check failed: {e}")
    
    def _print_report(self):
        """Stampa report finale."""
        ColorText.bold("\n" + "="*70)
        ColorText.bold("VERIFICATION REPORT")
        ColorText.bold("="*70)
        
        # Passed
        if self.passed:
            ColorText.info(f"✓ Passed checks: {len(self.passed)}")
            for check in self.passed[:3]:
                print(f"  • {check}")
            if len(self.passed) > 3:
                print(f"  ... and {len(self.passed) - 3} more")
        
        # Warnings
        if self.warnings:
            ColorText.warning(f"⚠ Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Failed
        if self.failed:
            ColorText.error(f"✗ Failed checks: {len(self.failed)}")
            for failed in self.failed:
                print(f"  • {failed}")
        
        # Summary
        ColorText.bold("\n" + "="*70)
        if not self.failed:
            ColorText.success("✓ ALL CHECKS PASSED - READY FOR PHASE 1 TRAINING!")
            ColorText.info("Run: python train_phase1.py")
        else:
            ColorText.error("✗ SOME CHECKS FAILED - FIX ERRORS BEFORE TRAINING")
            sys.exit(1)
        
        ColorText.bold("="*70 + "\n")


if __name__ == "__main__":
    verifier = Phase1SetupVerifier()
    verifier.run_all_checks()
