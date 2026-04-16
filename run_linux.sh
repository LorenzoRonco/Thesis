#!/bin/bash

# Pipeline Landmark - Sign Language Recognition
# Estrazione, Normalizzazione, Augmentation e Training di Landmark
# Versione dedicata a Linux

# Attiva virtual environment se esiste
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo ""
echo "============================================================"
echo "     PIPELINE LANDMARK - SIGN LANGUAGE RECOGNITION"
echo "============================================================"
echo ""
echo "=== SETUP ==="
echo "1. Setup (installa dipendenze)"
echo "2. Diagnosi (verifica setup)"
echo ""
echo "=== VERIFICHE DATASET ==="
echo "3. Verifica Integrita' Video (CSV vs cartella)"
echo ""
echo "=== SEGMENTAZIONE VIDEO ==="
echo "4. Test Segmentazione (2 video)"
echo "5. Segmentazione COMPLETA (tutti i video)"
echo ""
echo "=== ESTRAZIONE LANDMARK ==="
echo "6. Estrazione Landmark (MediaPipe Holistic)"
echo "7. Visualizza Landmark Video"
echo ""
echo "=== PREPROCESSING (NORMALIZZAZIONE) ==="
echo "8. Normalizza Landmark (shoulder-centric + global scale)"
echo "9. Visualizza Effetto Normalizzazione (raw vs normalized)"
echo ""
echo "10. Esci"
echo ""

read -p "Inserisci il numero (1-10): " choice

case $choice in
    1)
        echo ""
        echo "Avvio setup..."
        python3 scripts/setup_env.py
        ;;
    2)
        echo ""
        echo "Avvio diagnostica..."
        python3 scripts/diagnose.py
        ;;
    3)
        echo ""
        echo "Verifica Integrita' Video..."
        echo "Confronto CSV vs cartella dataset"
        echo ""
        python3 scripts/check_video_integrity.py
        ;;
    4)
        echo ""
        echo "Avvio TEST su 2 video..."
        python3 -m src.preprocessing.video_segmentation test
        ;;
    5)
        echo ""
        echo "ATTENZIONE: Questo elaborera' TUTTI i video (~50,000)"
        echo "Puo' richiedere molte ore!"
        echo ""
        read -p "Sei sicuro? (s/n): " confirm
        if [[ "$confirm" == "s" || "$confirm" == "S" ]]; then
            python3 -m src.preprocessing.video_segmentation full
        else
            echo "Annullato"
        fi
        ;;
    6)
        echo ""
        echo "Estrazione landmark dai video segmentati..."
        echo "(MediaPipe Holistic: Pose + Mani + Viso)"
        echo ""
        python3 src/preprocessing/landmark_extraction.py
        ;;
    7)
        echo ""
        echo "Visualizzazione video con landmark MediaPipe..."
        
        # Seleziona un video a caso da dataset/segmented/
        if [ -d "dataset/segmented" ]; then
            RANDOM_VIDEO=$(find dataset/segmented -name "*.mp4" | shuf -n 1)
            if [ -n "$RANDOM_VIDEO" ]; then
                echo "Video selezionato: $RANDOM_VIDEO"
                python3 scripts/visualize_landmarks.py "$RANDOM_VIDEO" --output output_video.mp4
                echo ""
                echo "Video generato: output_video.mp4"
            else
                echo "Errore: nessun video trovato in dataset/segmented/"
            fi
        else
            echo "Errore: cartella dataset/segmented/ non trovata"
        fi
        ;;
    8)
        echo ""
        echo "Normalizzazione Landmark..."
        echo "Metodo: Shoulder-centric + Global Scale"
        echo "Input:  dataset/landmarks/"
        echo "Output: dataset/landmarks_normalized/"
        echo ""
        python3 scripts/normalize_landmarks.py
        ;;
    9)
        echo ""
        echo "Visualizzazione Effetto Normalizzazione..."
        echo "Confronto side-by-side: RAW vs NORMALIZED"
        echo "Output: visualization_output/normalization_check/"
        echo ""
        python3 scripts/visualize_normalization_effect.py
        ;;
    10)
        echo ""
        echo "Fine."
        exit 0
        ;;
    *)
        echo "Scelta non valida"
        ;;
esac

echo ""
