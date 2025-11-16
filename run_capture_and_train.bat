@echo off
REM run_capture_and_train.bat - Complete SAE training pipeline

cd /d "%~dp0"

echo ================================================================================
echo NEUROTRACE - SAE TRAINING PIPELINE
echo ================================================================================
echo.
echo Step 1: Capture Layer 0 MLP activations (raw, no compression)
echo Step 2: Train SAE on captured activations
echo Step 3: Validate monosemantic features
echo.
pause

echo.
echo [1/2] Capturing Layer 0 MLP activations...
echo ================================================================================
python capture_ioi_activations.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Activation capture failed!
    pause
    exit /b 1
)

echo.
echo.
echo [2/2] Training SAE...
echo ================================================================================
echo.

REM Find the latest activation directory
for /f "delims=" %%i in ('dir /b /ad /o-d runs\phase1_ioi_activations') do (
    set LATEST_DIR=%%i
    goto :found
)

:found
echo Using activations from: runs\phase1_ioi_activations\%LATEST_DIR%\activations
echo.

set PYTHONPATH=%CD%

python cli\train_sae.py --activations_dir runs\phase1_ioi_activations\%LATEST_DIR%\activations --layer_name layer_0.mlp --model_name gpt2 --output_dir checkpoints\sae --epochs 10 --batch_size 256 --device cuda --dict_mult 4 --sparsity_lambda 1e-3

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: SAE training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SAE TRAINING COMPLETE!
echo ================================================================================
echo.
echo Trained SAE saved to: checkpoints\sae\
echo.
pause
