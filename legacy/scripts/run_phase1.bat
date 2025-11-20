@echo off
REM run_phase1.bat - Capture activations with PYTHONPATH set

cd /d "%~dp0"
set PYTHONPATH=%CD%

echo ================================================================================
echo NEUROTRACE - Phase 1 Activation Capture
echo ================================================================================
echo.
echo PYTHONPATH: %PYTHONPATH%
echo.

python cli\run_phase1_capture.py --model gpt2 --device cuda --num-examples 100 --batch-size 20 --out-dir runs\phase1_ioi_activations --max-seq-len 30

echo.
echo ================================================================================
echo Phase 1 capture complete
echo ================================================================================
pause
