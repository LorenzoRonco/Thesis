@echo off
REM Quick Start Batch per Video Segmentation
REM Questo file facilita l'esecuzione degli script su Windows
REM Collocato nella ROOT per facile accesso

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo      VIDEO SEGMENTATION - MENU PRINCIPALE (Windows)
echo ============================================================
echo.
echo Scegli un'opzione:
echo.
echo 1. Setup (installa dipendenze)
echo 2. Diagnosi (verifica setup)
echo 3. Test Segmentazione (2 video)
echo 4. Segmentazione COMPLETA (tutti i video)
echo 5. Estrazione Landmark (MediaPipe Holistic)
echo 6. Esci
echo.

set /p choice="Inserisci il numero (1-6): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto diagnose
if "%choice%"=="3" goto test
if "%choice%"=="4" goto full
if "%choice%"=="5" goto landmarks
if "%choice%"=="6" goto end

echo Scelta non valida
goto end

:setup
echo.
echo Avvio setup...
python scripts\setup_env.py
pause
goto end

:diagnose
echo.
echo Avvio diagnostica...
python scripts\diagnose.py
pause
goto end

:test
echo.
echo Avvio TEST su 2 video...
python -m src.preprocessing.video_segmentation test
pause
goto end

:full
echo.
echo ⚠️  ATTENZIONE: Questo elaborera' TUTTI i video (~50,000)
echo    Puo' richiedere molte ore!
echo.
set /p confirm="Sei sicuro? (s/n): "
if /i "%confirm%"=="s" (
    python -m src.preprocessing.video_segmentation full
) else (
    echo Annullato
)
pause
goto end

:landmarks
echo.
echo Estrazione landmark dai video segmentati...
echo (MediaPipe Holistic: Pose + Mani + Viso)
echo.
python src/preprocessing/landmark_extraction.py
pause
goto end

:end
echo.
echo Fine.
