@echo off
echo ========================================
echo CLEANUP: Removing SAE-dependent work
echo ========================================
echo.

REM Remove SAE-based analysis
del /f /q feature_circuit_discovery.json 2>nul
del /f /q atlas_analysis_report.json 2>nul
del /f /q validation_results.json 2>nul
del /f /q feature_flow_narrative.txt 2>nul

REM Remove Phase 2 files (all based on invalid SAE)
del /f /q PHASE2_*.md 2>nul
del /f /q build_candidate_circuits.py 2>nul
del /f /q validate_candidate_circuits.py 2>nul
del /f /q generate_feature_narrative.py 2>nul
del /f /q sanity_check_*.py 2>nul

REM Remove invalid circuit registry
rmdir /s /q circuits 2>nul

REM Remove invalid SAE checkpoints
rmdir /s /q checkpoints\all_layers_sae 2>nul

echo.
echo KEPT (valid):
echo   - IOI dataset and activations
echo   - Component discovery results (VLO)
echo   - Infrastructure code
echo.
echo REMOVED (SAE-dependent):
echo   - All Atlas SAE checkpoints
echo   - Feature discovery
echo   - Circuit construction
echo   - Phase 2 validation
echo.
echo ========================================
echo Ready for clean SAE re-training
echo ========================================
