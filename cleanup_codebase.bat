@echo off
echo ================================================================================
echo NEUROTRACE CODEBASE CLEANUP
echo ================================================================================
echo.
echo This will archive 14 old/test files to: archive_old_files\
echo.
echo Files to archive:
echo   - 5 test files (old tests)
echo   - 9 obsolete scripts (replaced by newer versions)
echo.
echo IMPORTANT: test_system_diagnostic.py will be KEPT (it's critical!)
echo.
pause

echo.
echo Creating archive directory...
if not exist archive_old_files mkdir archive_old_files
if not exist archive_old_files\test_files mkdir archive_old_files\test_files
if not exist archive_old_files\obsolete_files mkdir archive_old_files\obsolete_files

echo.
echo [1/2] Moving TEST files to archive...
move test_causal_discovery.py archive_old_files\test_files\
move test_control_plane.py archive_old_files\test_files\
move test_neurotrace_pipeline.py archive_old_files\test_files\
move test_sae_training.py archive_old_files\test_files\
move test_visualization.py archive_old_files\test_files\

echo.
echo [2/2] Moving OBSOLETE files to archive...
move train_all_layers_sae.py archive_old_files\obsolete_files\
move train_enhanced_sae.py archive_old_files\obsolete_files\
move setup_saelens.py archive_old_files\obsolete_files\
move monitor_training.py archive_old_files\obsolete_files\
move phase2_verify.py archive_old_files\obsolete_files\
move run_discovery.py archive_old_files\obsolete_files\
move run_discovery_validation.py archive_old_files\obsolete_files\
move compare_discovery_runs.py archive_old_files\obsolete_files\
move complete_validation_analysis.py archive_old_files\obsolete_files\

echo.
echo ================================================================================
echo CLEANUP COMPLETE
echo ================================================================================
echo.
echo Archived:
echo   - 5 test files
echo   - 9 obsolete files
echo.
echo Kept (CRITICAL):
echo   - tests\validation\test_system_diagnostic.py (system validation)
echo   - All neurotrace\ library files
echo   - train_atlas_simple.py (working Atlas trainer)
echo   - validate_atlas.py (Atlas validation)
echo.
echo Files are in: archive_old_files\
echo You can delete archive_old_files\ folder if you don't need them.
echo.
pause
