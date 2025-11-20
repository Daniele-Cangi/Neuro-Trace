@echo off
REM ============================================================================
REM NEUROTRACE - DEEP IOI DATASET CAPTURE (100K+ Examples)
REM ============================================================================
REM
REM Purpose: Capture comprehensive activation dataset for publication-quality SAE
REM Output: ~2-3 GB of raw activations (768-dim) across all layers
REM Time: ~30 minutes for 100K examples
REM
REM ============================================================================

cd /d "%~dp0"
set PYTHONPATH=%CD%

echo ============================================================================
echo NEUROTRACE - DEEP DATASET CAPTURE
echo ============================================================================
echo.
echo This will capture 100,000 IOI examples across ALL 12 layers.
echo.
echo Estimated:
echo   - Time: ~30 minutes
echo   - Disk: ~2-3 GB
echo   - Quality: Publication-ready (80-90%% monosemantic SAE)
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Starting deep capture...
echo.

python capture_deep_dataset.py ^
    --num_examples 100000 ^
    --batch_size 50 ^
    --capture_all_layers ^
    --device cuda ^
    --output_dir runs/deep_ioi_capture ^
    --max_seq_len 30 ^
    --seed 42

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo ERROR: Deep capture failed!
    echo ============================================================================
    echo.
    echo Check the error messages above.
    echo.
    echo Common issues:
    echo   - CUDA out of memory: Try --batch_size 25
    echo   - Missing dependencies: pip install transformers torch
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo DEEP CAPTURE COMPLETE!
echo ============================================================================
echo.
echo Next steps:
echo   1. run_train_enhanced_sae.bat (train SAE on Layer 0)
echo   2. setup_saelens.bat (install SAELens for comparison)
echo   3. run_hybrid_analysis.bat (compare Layer 0 vs Layer 9)
echo.
pause
