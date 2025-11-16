# monitor_training.py

"""
Monitor Enhanced SAE training progress.

Usage:
    python monitor_training.py
"""

import sys
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def main():
    print("=" * 80)
    print("NEUROTRACE - TRAINING MONITOR")
    print("=" * 80)
    print()

    checkpoint_dir = Path("checkpoints/layer0_sae")

    if not checkpoint_dir.exists():
        print("⚠️  Checkpoint directory not found")
        print("    Training may not have started yet")
        return

    print(f"Monitoring: {checkpoint_dir}")
    print()

    # Check for checkpoints
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))

    if len(checkpoints) == 0:
        print("⏳ Training in progress (no checkpoints yet)")
        print("   First checkpoint will be saved at epoch 2")
    else:
        print(f"✓ Found {len(checkpoints)} checkpoints:")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 ** 2)
            print(f"    {ckpt.name} ({size_mb:.1f} MB)")

    # Check for final checkpoint
    final_ckpt = checkpoint_dir / "final.pt"
    if final_ckpt.exists():
        print()
        print("✅ TRAINING COMPLETE!")
        print(f"   Final checkpoint: {final_ckpt}")
    else:
        print()
        print("⏳ Training still in progress...")
        print("   Run this script again to check status")

    print()
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
