#!/bin/bash

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║     QUICK START SCRIPT - PHASE 1 TRAINING                            ║
# ║     Sign Language Translation Model                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

echo "════════════════════════════════════════════════════════════════════"
echo "  PHASE 1 TRAINING - Sign Language Translation"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# 1. Verifica setup
echo "📋 STEP 1: Verifica Setup..."
python verify_phase1_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Verifiche fallite. Risolvere errori prima di continuare."
    exit 1
fi

echo ""

# 2. Menu opzioni
echo "📝 STEP 2: Seleziona modalità training"
echo ""
echo "Opzioni:"
echo "  1) Training rapido (100 campioni, 10 epochs, testing)"
echo "  2) Training standard (1000 campioni, 100 epochs)"
echo "  3) Training esteso (5000 campioni, 200 epochs)"
echo "  4) Personalizzato (inserisci parametri)"
echo "  5) Esci"
echo ""

read -p "Scegli opzione (1-5): " choice

case $choice in
    1)
        echo "🚀 Avviando training RAPIDO..."
        python train_phase1.py \
            --num_samples 100 \
            --batch_size 32 \
            --num_epochs 10 \
            --learning_rate 1e-3 \
            --device cuda
        ;;
    
    2)
        echo "🚀 Avviando training STANDARD..."
        python train_phase1.py \
            --num_samples 1000 \
            --batch_size 32 \
            --num_epochs 100 \
            --learning_rate 1e-3 \
            --device cuda
        ;;
    
    3)
        echo "🚀 Avviando training ESTESO..."
        python train_phase1.py \
            --num_samples 5000 \
            --batch_size 32 \
            --num_epochs 200 \
            --learning_rate 1e-3 \
            --device cuda
        ;;
    
    4)
        echo "📝 Training personalizzato"
        read -p "Numero campioni (default 1000): " num_samples
        num_samples=${num_samples:-1000}
        
        read -p "Batch size (default 32): " batch_size
        batch_size=${batch_size:-32}
        
        read -p "Numero epochs (default 100): " num_epochs
        num_epochs=${num_epochs:-100}
        
        read -p "Learning rate (default 1e-3): " learning_rate
        learning_rate=${learning_rate:-1e-3}
        
        read -p "Device (cuda/cpu, default cuda): " device
        device=${device:-cuda}
        
        echo "🚀 Avviando con parametri personalizzati..."
        python train_phase1.py \
            --num_samples $num_samples \
            --batch_size $batch_size \
            --num_epochs $num_epochs \
            --learning_rate $learning_rate \
            --device $device
        ;;
    
    5)
        echo "Uscita senza training"
        exit 0
        ;;
    
    *)
        echo "❌ Opzione non valida"
        exit 1
        ;;
esac

echo ""
echo "✅ Training completato!"
echo ""
echo "📊 Risultati disponibili in:"
echo "  • Checkpoints: checkpoints/phase1/"
echo "  • Logs: logs/phase1/"
echo ""
