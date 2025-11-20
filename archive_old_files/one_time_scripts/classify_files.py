# classify_files.py
"""
Classifica tutti i file Python in:
- CORE: File essenziali del framework
- SCRIPTS: Script utili da mantenere
- TEST: File di test (candidati per cleanup)
- OBSOLETE: File sostituiti o non più necessari
"""

from pathlib import Path
import os

# Definizioni
CORE_PATTERNS = [
    "neurotrace/",  # Tutto il core library
]

ESSENTIAL_SCRIPTS = [
    "setup.py",
    "train_atlas_simple.py",  # Script che ha funzionato per Atlas
    "validate_atlas.py",  # Validazione Atlas
    "compare_trainings.py",  # Analisi comparativa
]

TEST_PATTERNS = [
    "test_",  # Tutti i file test_*.py
]

OBSOLETE_CANDIDATES = [
    "train_all_layers_sae.py",  # Sostituito da train_atlas_simple.py
    "train_enhanced_sae.py",  # Generico, non usato
    "setup_saelens.py",  # Setup one-time
    "monitor_training.py",  # Monitoring, non essenziale
    "phase2_verify.py",  # Fase specifica
    "run_discovery.py",  # Script vecchio discovery
    "run_discovery_validation.py",  # Validation vecchia
    "compare_discovery_runs.py",  # Discovery runs
    "complete_validation_analysis.py",  # Sostituito da system diagnostic
]

def classify_file(filepath: Path) -> str:
    """Classifica un file Python."""
    rel_path = str(filepath.relative_to(Path.cwd()))

    # Check CORE
    for pattern in CORE_PATTERNS:
        if pattern in rel_path:
            return "CORE"

    # Check ESSENTIAL
    if filepath.name in ESSENTIAL_SCRIPTS:
        return "ESSENTIAL"

    # Check TEST
    for pattern in TEST_PATTERNS:
        if pattern in filepath.name:
            return "TEST"

    # Check OBSOLETE
    if filepath.name in OBSOLETE_CANDIDATES:
        return "OBSOLETE"

    # Check examples/
    if "examples/" in rel_path:
        return "EXAMPLES"

    # Check tests/
    if "tests/" in rel_path:
        return "TESTS"

    # Check cli/
    if "cli/" in rel_path:
        return "CLI"

    # Default: SCRIPT (needs review)
    return "SCRIPT"


def main():
    print("=" * 80)
    print("PYTHON FILE CLASSIFICATION")
    print("=" * 80)
    print()

    # Find all Python files
    py_files = list(Path.cwd().glob("**/*.py"))

    # Classify
    classified = {
        "CORE": [],
        "ESSENTIAL": [],
        "EXAMPLES": [],
        "CLI": [],
        "TESTS": [],
        "TEST": [],
        "OBSOLETE": [],
        "SCRIPT": [],
    }

    for filepath in py_files:
        # Skip venv, __pycache__, etc
        if any(p in str(filepath) for p in [".venv", "venv", "__pycache__", ".git"]):
            continue

        category = classify_file(filepath)
        classified[category].append(filepath)

    # Print results
    for category in ["CORE", "ESSENTIAL", "EXAMPLES", "CLI", "TESTS", "SCRIPT", "TEST", "OBSOLETE"]:
        files = classified[category]
        if not files:
            continue

        print(f"\n{category} ({len(files)} files):")
        print("-" * 80)
        for f in sorted(files):
            rel_path = f.relative_to(Path.cwd())
            size_kb = f.stat().st_size / 1024
            print(f"  {rel_path} ({size_kb:.1f} KB)")

    # Summary
    print("\n" + "=" * 80)
    print("CLEANUP RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Test files
    test_files = classified["TEST"]
    if test_files:
        print(f"TEST FILES ({len(test_files)} files) - Safe to archive/delete:")
        for f in sorted(test_files):
            print(f"  {f.name}")
        print()

    # Obsolete files
    obsolete_files = classified["OBSOLETE"]
    if obsolete_files:
        print(f"OBSOLETE FILES ({len(obsolete_files)} files) - Consider archiving:")
        for f in sorted(obsolete_files):
            print(f"  {f.name}")
        print()

    # Scripts needing review
    script_files = classified["SCRIPT"]
    if script_files:
        print(f"SCRIPTS ({len(script_files)} files) - Manual review needed:")
        for f in sorted(script_files):
            rel_path = f.relative_to(Path.cwd())
            print(f"  {rel_path}")
        print()

    # Summary stats
    total_files = sum(len(files) for files in classified.values())
    core_files = len(classified["CORE"])
    essential_files = len(classified["ESSENTIAL"]) + len(classified["EXAMPLES"]) + len(classified["CLI"]) + len(classified["TESTS"])
    cleanup_files = len(classified["TEST"]) + len(classified["OBSOLETE"])

    print(f"TOTAL: {total_files} Python files")
    print(f"  CORE library: {core_files} files (keep)")
    print(f"  ESSENTIAL scripts/tests: {essential_files} files (keep)")
    print(f"  CANDIDATES for cleanup: {cleanup_files} files")
    print(f"  SCRIPTS needing review: {len(script_files)} files")
    print()

    # Create cleanup script
    print("=" * 80)
    print("GENERATING CLEANUP SCRIPT")
    print("=" * 80)
    print()

    archive_dir = Path("archive_old_files")

    cleanup_script = f"""# cleanup_old_files.bat
@echo off
echo Creating archive directory...
mkdir {archive_dir}
mkdir {archive_dir}\\test_files
mkdir {archive_dir}\\obsolete_files

echo.
echo Moving TEST files to archive...
"""

    for f in sorted(test_files):
        cleanup_script += f'move "{f.name}" {archive_dir}\\test_files\\\n'

    cleanup_script += """
echo.
echo Moving OBSOLETE files to archive...
"""

    for f in sorted(obsolete_files):
        cleanup_script += f'move "{f.name}" {archive_dir}\\obsolete_files\\\n'

    cleanup_script += """
echo.
echo Cleanup complete!
echo Files moved to archive_old_files/
echo.
pause
"""

    # Save cleanup script
    script_path = Path("cleanup_old_files.bat")
    with open(script_path, 'w') as f:
        f.write(cleanup_script)

    print(f"Cleanup script created: {script_path}")
    print()
    print("To cleanup, run:")
    print(f"  {script_path}")
    print()
    print("This will move test and obsolete files to archive_old_files/")
    print()


if __name__ == "__main__":
    main()
