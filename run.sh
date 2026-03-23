#!/bin/bash

# Quick Start Script per Video Segmentation
# Questo file facilita l'esecuzione degli script su Linux/macOS
# Collocato nella ROOT per facile accesso

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║    VIDEO SEGMENTATION - MENU PRINCIPALE (Linux/macOS)     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Scegli un'opzione:"
echo ""
echo "1. Setup (installa dipendenze)"
echo "2. Diagnosi (verifica setup)"
echo "3. Test Segmentazione (2 video)"
echo "4. Segmentazione COMPLETA (tutti i video)"
echo "5. Esci"
echo ""

read -p "Inserisci il numero (1-5): " choice

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
        echo "Avvio TEST su 2 video..."
        python3 -m src.preprocessing.video_segmentation test
        ;;
    4)
        echo ""
        echo "⚠️  ATTENZIONE: Questo elaborerà TUTTI i video (~50,000)"
        echo "    Può richiedere molte ore!"
        echo ""
        read -p "Sei sicuro? (s/n): " confirm
        if [[ $confirm == "s" || $confirm == "S" ]]; then
            python3 -m src.preprocessing.video_segmentation full
        else
            echo "Annullato"
        fi
        ;;
    5)
        echo ""
        echo "Fine."
        exit 0
        ;;
    *)
        echo "Scelta non valida"
        ;;
esac

echo ""
read -p "Premi Enter per tornare al menu o Ctrl+C per uscire..."
exit 0
