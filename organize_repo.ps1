# Script di riorganizzazione repository
# Questo script sposta i file nelle cartelle appropriate per mettere ordine

# 1. Crea le directory se non esistono
New-Item -ItemType Directory -Force -Path "phases"
New-Item -ItemType Directory -Force -Path "reports"
New-Item -ItemType Directory -Force -Path "configs"
New-Item -ItemType Directory -Force -Path "legacy"
New-Item -ItemType Directory -Force -Path "legacy/scripts"
New-Item -ItemType Directory -Force -Path "legacy/logs"

# 2. Sposta gli script delle Fasi (phase*.py)
Move-Item -Path "phase*.py" -Destination "phases" -ErrorAction SilentlyContinue

# 3. Sposta i report e i risultati (PHASE*.md, *_results.json)
Move-Item -Path "PHASE*.md" -Destination "reports" -ErrorAction SilentlyContinue
Move-Item -Path "*_results.json" -Destination "reports" -ErrorAction SilentlyContinue
Move-Item -Path "ATLAS_COMPLETE.md" -Destination "reports" -ErrorAction SilentlyContinue
Move-Item -Path "PROJECT_OVERVIEW.md" -Destination "reports" -ErrorAction SilentlyContinue
Move-Item -Path "QUICK_START.md" -Destination "reports" -ErrorAction SilentlyContinue
Move-Item -Path "DOCUMENTATION_SUMMARY.md" -Destination "reports" -ErrorAction SilentlyContinue

# 4. Sposta le configurazioni (*_config.json)
Move-Item -Path "*_config.json" -Destination "configs" -ErrorAction SilentlyContinue
Move-Item -Path "feature_circuit_discovery.json" -Destination "configs" -ErrorAction SilentlyContinue
Move-Item -Path "adversarial_delta_feature_decomposition_layer10.json" -Destination "reports" -ErrorAction SilentlyContinue

# 5. Sposta script legacy e utility vecchie
Move-Item -Path "capture_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "check_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "discover_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "generate_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "layer_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "learn_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "project_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "run_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "train_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "validate_*.py" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "run_*.bat" -Destination "legacy/scripts" -ErrorAction SilentlyContinue
Move-Item -Path "cleanup_*.bat" -Destination "legacy/scripts" -ErrorAction SilentlyContinue

# 6. Sposta log e file temporanei
Move-Item -Path "*.log" -Destination "legacy/logs" -ErrorAction SilentlyContinue
Move-Item -Path "diagnostic_output.txt" -Destination "legacy/logs" -ErrorAction SilentlyContinue
Move-Item -Path "STRUCTURE_SUMMARY.txt" -Destination "legacy/logs" -ErrorAction SilentlyContinue
Move-Item -Path "war_surface.csv" -Destination "reports" -ErrorAction SilentlyContinue

Write-Host "Riorganizzazione completata!"
Write-Host "Struttura aggiornata:"
Get-ChildItem -Directory
