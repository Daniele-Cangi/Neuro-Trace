# cleanup_old_files.bat
@echo off
echo Creating archive directory...
mkdir archive_old_files
mkdir archive_old_files\test_files
mkdir archive_old_files\obsolete_files

echo.
echo Moving TEST files to archive...
move "test_causal_discovery.py" archive_old_files\test_files\
move "test_control_plane.py" archive_old_files\test_files\
move "test_neurotrace_pipeline.py" archive_old_files\test_files\
move "test_sae_training.py" archive_old_files\test_files\
move "test_visualization.py" archive_old_files\test_files\
move "test_system_diagnostic.py" archive_old_files\test_files\

echo.
echo Moving OBSOLETE files to archive...
move "compare_discovery_runs.py" archive_old_files\obsolete_files\
move "complete_validation_analysis.py" archive_old_files\obsolete_files\
move "monitor_training.py" archive_old_files\obsolete_files\
move "phase2_verify.py" archive_old_files\obsolete_files\
move "run_discovery.py" archive_old_files\obsolete_files\
move "run_discovery_validation.py" archive_old_files\obsolete_files\
move "setup_saelens.py" archive_old_files\obsolete_files\
move "train_all_layers_sae.py" archive_old_files\obsolete_files\
move "train_enhanced_sae.py" archive_old_files\obsolete_files\

echo.
echo Cleanup complete!
echo Files moved to archive_old_files/
echo.
pause
