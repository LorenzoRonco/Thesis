@echo off
REM Pipeline Landmark - Sign Language Recognition
REM Estrazione, Normalizzazione, Augmentation e Training di Landmark
REM Collocato nella ROOT per facile accesso

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo      PIPELINE LANDMARK - SIGN LANGUAGE RECOGNITION
echo ============================================================
echo.
echo === SETUP ===
echo 1. Setup (installa dipendenze)
echo 2. Diagnosi (verifica setup)
echo.
echo === VERIFICHE DATASET ===
echo 3. Verifica Integrita' Video (CSV vs cartella)
echo.
echo === SEGMENTAZIONE VIDEO ===
echo 4. Test Segmentazione (2 video)
echo 5. Segmentazione COMPLETA (tutti i video)
echo.
echo === ESTRAZIONE LANDMARK ===
echo 6. Estrazione Landmark (MediaPipe Holistic)
echo 7. Visualizza Landmark Video
echo.
echo === PREPROCESSING (NORMALIZZAZIONE) ===
echo 8. Normalizza Landmark (shoulder-centric + global scale)
echo 9. Visualizza Effetto Normalizzazione (raw vs normalized)
echo.
echo 10. Esci
echo.

set /p choice="Inserisci il numero (1-10): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto diagnose
if "%choice%"=="3" goto check_integrity
if "%choice%"=="4" goto test
if "%choice%"=="5" goto full
if "%choice%"=="6" goto landmarks
if "%choice%"=="7" goto visualize
if "%choice%"=="8" goto normalize
if "%choice%"=="9" goto visualize_norm
if "%choice%"=="10" goto end

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

:check_integrity
echo.
echo Verifica Integrità Video...
echo Confronto CSV vs cartella dataset
echo.
python scripts\check_video_integrity.py
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

:visualize
echo.
echo Visualizzazione video con landmark MediaPipe...
REM Seleziona un video a caso da dataset/segmented/
for /F "delims=" %%f in ('powershell -Command "Get-ChildItem dataset\segmented\*.mp4 | Get-Random | Select-Object -ExpandProperty Name"') do set RANDOM_VIDEO=%%f

if defined RANDOM_VIDEO (
    echo Video selezionato: !RANDOM_VIDEO!
    python scripts/visualize_landmarks.py dataset\segmented\!RANDOM_VIDEO! --output output_video.mp4
    echo.
    echo Video generato: output_video.mp4
) else (
    echo Errore: nessun video trovato in dataset/segmented/
)
pause
goto end
:normalize
echo.
echo Normalizzazione Landmark...
echo Metodo: Shoulder-centric + Global Scale
echo Input:  dataset\landmarks\
echo Output: dataset\landmarks_normalized\
echo.
python scripts\normalize_landmarks.py
pause
goto end

:visualize_norm
echo.
echo Visualizzazione Effetto Normalizzazione...
echo Confronto side-by-side: RAW vs NORMALIZED
echo Output: visualization_output\normalization_check\
echo.
python scripts\visualize_normalization_effect.py
pause
goto end

:end
echo.
echo Fine.
