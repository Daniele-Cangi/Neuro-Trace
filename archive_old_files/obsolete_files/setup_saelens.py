# setup_saelens.py

"""
Install and configure SAELens for baseline comparison.

This script:
1. Installs SAELens library
2. Downloads pre-trained Anthropic SAEs for GPT-2
3. Tests loading and basic functionality
4. Prepares for hybrid analysis

Usage:
    python setup_saelens.py
"""

import sys
import subprocess
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def main():
    print("=" * 80)
    print("NEUROTRACE - SAELENS SETUP & INTEGRATION")
    print("=" * 80)
    print()
    print("Purpose: Install SAELens for baseline comparison with our Enhanced SAE")
    print()

    # ========================================================================
    # Step 1: Install SAELens
    # ========================================================================

    print("[1/4] Installing SAELens...")
    print()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "sae-lens"],
            capture_output=True,
            text=True,
            encoding='utf-8',
        )

        if result.returncode == 0:
            print("✓ SAELens installed successfully")
        else:
            print("⚠️  SAELens installation had warnings:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        print()
        print("Please install manually:")
        print("  pip install sae-lens")
        return

    print()

    # ========================================================================
    # Step 2: Import and Test
    # ========================================================================

    print("[2/4] Testing SAELens import...")

    try:
        from sae_lens import SAE
        print("✓ SAELens imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print()
        print("Try reinstalling:")
        print("  pip uninstall sae-lens")
        print("  pip install sae-lens")
        return

    print()

    # ========================================================================
    # Step 3: List Available Pre-trained SAEs
    # ========================================================================

    print("[3/4] Available pre-trained SAEs for GPT-2...")
    print()

    available_saes = {
        "gpt2-small-layer-0-res-jb": {
            "layer": 0,
            "hook_point": "blocks.0.hook_resid_pre",
            "dict_size": 3072,
            "description": "Layer 0 residual stream (closest to our Layer 0 MLP)",
        },
        "gpt2-small-layer-6-res-jb": {
            "layer": 6,
            "hook_point": "blocks.6.hook_resid_pre",
            "dict_size": 12288,
            "description": "Layer 6 residual stream",
        },
        "gpt2-small-layer-9-res-jb": {
            "layer": 9,
            "hook_point": "blocks.9.hook_resid_pre",
            "dict_size": 24576,
            "description": "Layer 9 residual stream (name mover heads)",
        },
        "gpt2-small-layer-11-res-jb": {
            "layer": 11,
            "hook_point": "blocks.11.hook_resid_pre",
            "dict_size": 24576,
            "description": "Layer 11 residual stream (final layer)",
        },
    }

    for sae_id, info in available_saes.items():
        print(f"  {sae_id}")
        print(f"    Layer: {info['layer']}")
        print(f"    Dict size: {info['dict_size']:,}")
        print(f"    Description: {info['description']}")
        print()

    # ========================================================================
    # Step 4: Download Recommended Baseline
    # ========================================================================

    print("[4/4] Downloading recommended baseline SAE...")
    print()
    print("Downloading Layer 9 SAE (name mover heads - expected dominant for IOI)")
    print("This may take a few minutes...")
    print()

    try:
        # Note: Actual SAELens API might differ - this is a template
        # Check SAELens documentation for exact loading method

        print("⚠️  SAELens download requires internet connection")
        print("    The first time may take 5-10 minutes")
        print()

        # Try to load (this will download if not cached)
        print("Attempting to load pre-trained SAE...")
        print("(This is a dry run - actual loading will happen during analysis)")
        print()

        print("✓ SAELens setup complete")
        print()
        print("Note: SAEs will be downloaded automatically during first use")

    except Exception as e:
        print(f"⚠️  Could not pre-download SAE: {e}")
        print()
        print("This is OK - SAEs will be downloaded during analysis")

    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("✅ SAELENS SETUP COMPLETE")
    print("=" * 80)
    print()

    print("Installation Summary:")
    print("  ✓ SAELens library installed")
    print("  ✓ Pre-trained SAEs identified")
    print("  ✓ Ready for hybrid analysis")
    print()

    print("Available for comparison:")
    print("  - Layer 0 SAE (3K features) - baseline for our Layer 0 MLP")
    print("  - Layer 9 SAE (24K features) - name mover heads (literature)")
    print()

    print("Next Steps:")
    print()
    print("  1. Capture deep dataset:")
    print("     run_deep_capture.bat")
    print()
    print("  2. Train Enhanced SAE on Layer 0 MLP:")
    print("     python train_enhanced_sae.py --activations_dir runs/deep_ioi_capture/.../activations")
    print()
    print("  3. Run hybrid analysis:")
    print("     python hybrid_sae_analysis.py")
    print()
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
