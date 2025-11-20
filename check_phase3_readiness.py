"""
Phase 3 Readiness Check

Verifies that all infrastructure is ready for feature discovery.
"""

import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 80)
print("PHASE 3 READINESS CHECK")
print("=" * 80)
print()

all_checks_passed = True

# Check 1: SAE Checkpoints
print("[1/5] Checking SAE checkpoints...")
atlas_dir = Path("checkpoints/all_layers_sae")
if not atlas_dir.exists():
    print("  ✗ FAIL: Atlas directory not found")
    all_checks_passed = False
else:
    missing_layers = []
    for layer in range(12):
        checkpoint = atlas_dir / f"layer_{layer}" / "final.pt"
        if not checkpoint.exists():
            missing_layers.append(layer)

    if missing_layers:
        print(f"  ✗ FAIL: Missing layers {missing_layers}")
        all_checks_passed = False
    else:
        print("  ✓ PASS: All 12 SAE checkpoints found")
        # Show checkpoint sizes
        total_size_mb = sum(
            (atlas_dir / f"layer_{i}" / "final.pt").stat().st_size
            for i in range(12)
        ) / (1024**2)
        print(f"    Total size: {total_size_mb:.1f} MB")
print()

# Check 2: Discovery Scripts
print("[2/5] Checking discovery scripts...")
required_scripts = [
    "discover_feature_circuits.py",
    "discover_real_circuits.py",
]
missing_scripts = []
for script in required_scripts:
    if not Path(script).exists():
        missing_scripts.append(script)

if missing_scripts:
    print(f"  ✗ FAIL: Missing scripts {missing_scripts}")
    all_checks_passed = False
else:
    print("  ✓ PASS: All discovery scripts found")
    for script in required_scripts:
        size_kb = Path(script).stat().st_size / 1024
        print(f"    - {script} ({size_kb:.1f} KB)")
print()

# Check 3: Infrastructure Classes
print("[3/5] Checking infrastructure classes...")
try:
    from neurotrace.control import EnhancedSAEFeatureStore
    from neurotrace.discovery import FeatureCircuitDiscoverer
    from neurotrace.causal import VLOTester
    from neurotrace.control import CircuitRegistry
    print("  ✓ PASS: All core classes importable")
    print("    - EnhancedSAEFeatureStore")
    print("    - FeatureCircuitDiscoverer")
    print("    - VLOTester")
    print("    - CircuitRegistry")
except ImportError as e:
    print(f"  ✗ FAIL: Import error: {e}")
    all_checks_passed = False
print()

# Check 4: GPU Availability
print("[4/5] Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ✓ PASS: GPU available")
        print(f"    Device: {gpu_name}")
        print(f"    Memory: {gpu_mem:.1f} GB")
    else:
        print("  ⚠ WARNING: No GPU available (will use CPU, slower)")
except Exception as e:
    print(f"  ✗ FAIL: PyTorch error: {e}")
    all_checks_passed = False
print()

# Check 5: Atlas Validation Results
print("[5/5] Checking Atlas validation...")
summary_path = atlas_dir / "training_summary.json"
if not summary_path.exists():
    print("  ✗ FAIL: training_summary.json not found")
    all_checks_passed = False
else:
    import json
    with open(summary_path) as f:
        summary = json.load(f)

    valid_count = sum(
        1 for result in summary["results"]
        if result["validation"]["valid"]
    )

    if valid_count == 12:
        print(f"  ✓ PASS: All 12 layers validated")

        # Show validation stats
        losses = [r["validation"]["loss"] for r in summary["results"]]
        avg_loss = sum(losses) / len(losses)
        print(f"    Average reconstruction loss: {avg_loss:+.1%}")

        # Count improved layers
        improved = sum(1 for loss in losses if loss < 0)
        print(f"    Layers with improved accuracy: {improved}/12")
    else:
        print(f"  ✗ FAIL: Only {valid_count}/12 layers validated")
        all_checks_passed = False
print()

# Final Summary
print("=" * 80)
if all_checks_passed:
    print("✅ ALL CHECKS PASSED - READY FOR PHASE 3")
    print()
    print("Next steps:")
    print("  1. Run: python discover_feature_circuits.py")
    print("  2. Expected runtime: 2-3 minutes")
    print("  3. Output: feature_circuit_discovery.json")
else:
    print("❌ SOME CHECKS FAILED - NOT READY")
    print()
    print("Fix the issues above before proceeding.")
print("=" * 80)
print()

sys.exit(0 if all_checks_passed else 1)
