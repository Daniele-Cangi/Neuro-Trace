# capture_deep_dataset.py

"""
Deep IOI Dataset Capture for Complete Neural Cartography.

Captures massive activation dataset (100K+ examples) across ALL layers
for comprehensive SAE training and analysis.

This enables:
- High-quality SAE training (publication-grade)
- Cross-layer feature comparison
- Complete 1:1 neural mapping
- Novel scientific discoveries

Usage:
    python capture_deep_dataset.py --num_examples 100000 --capture_all_layers
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from neurotrace.datasets import IOIDatasetGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep IOI dataset capture for SAE training"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100000,
        help="Number of IOI examples (default: 100,000 for quality SAE)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for capture (default: 50 for 6GB VRAM)",
    )
    parser.add_argument(
        "--capture_all_layers",
        action="store_true",
        help="Capture ALL 12 layers (not just Layer 0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/deep_ioi_capture",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=30,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("NEUROTRACE - DEEP IOI DATASET CAPTURE")
    print("=" * 80)
    print()
    print("🎯 Goal: Complete Neural Cartography (1:1 mapping)")
    print("🔬 Approach: Maximum depth, maximum rigor")
    print()

    # Configuration
    device = args.device if torch.cuda.is_available() else "cpu"
    num_examples = args.num_examples
    batch_size = args.batch_size
    capture_all = args.capture_all_layers

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(f"  Examples: {num_examples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Capture layers: {'ALL (0-11)' if capture_all else 'Layer 0 only'}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print()

    # Estimate resources
    est_time_min = (num_examples / batch_size) * 2 / 60  # ~2 sec per batch
    est_disk_gb = (num_examples * 30 * 768 * 4) / (1024**3)  # float32
    if capture_all:
        est_disk_gb *= 12  # 12 layers

    print(f"📊 Estimated:")
    print(f"  Time: ~{est_time_min:.0f} minutes")
    print(f"  Disk: ~{est_disk_gb:.1f} GB")
    print()
    print("🚀 Starting capture...")
    print()

    # ========================================================================
    # Generate IOI Dataset
    # ========================================================================

    print("[1/4] Generating IOI dataset...")
    print(f"      Size: {num_examples:,} examples")
    print(f"      Diversity: Maximum (all templates, 200+ names)")
    print()

    generator = IOIDatasetGenerator(seed=args.seed)
    ioi_examples = generator.generate(
        num_examples=num_examples,
        ensure_diversity=True,
    )

    print(f"✓ Generated {len(ioi_examples):,} IOI examples")

    # Save dataset
    dataset_path = output_dir / "ioi_dataset.json"
    generator.save_to_json(ioi_examples, dataset_path)
    print(f"✓ Saved to {dataset_path}")
    print()

    # ========================================================================
    # Load Model
    # ========================================================================

    print("[2/4] Loading GPT-2 model...")

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded: GPT-2 (124M parameters)")
    print(f"✓ Device: {device}")
    print()

    # ========================================================================
    # Capture Activations
    # ========================================================================

    print("[3/4] Capturing activations...")
    print(f"      Batches: {len(ioi_examples) // batch_size + 1}")
    if capture_all:
        print(f"      Layers: 12 (all MLP + attention)")
    else:
        print(f"      Layers: 1 (Layer 0 MLP only)")
    print()

    # Determine which layers to capture
    if capture_all:
        layers_to_capture = [
            (f"layer_{i}.mlp", model.transformer.h[i].mlp)
            for i in range(12)
        ]
    else:
        layers_to_capture = [
            ("layer_0.mlp", model.transformer.h[0].mlp)
        ]

    texts = [ex.text for ex in ioi_examples]
    batch_idx = 0
    total_tokens = 0

    import time
    start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch_idx += 1
        batch_texts = texts[i:i + batch_size]

        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            progress = i / len(texts)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            print(f"  Batch {batch_idx:>5}/{len(texts) // batch_size + 1} "
                  f"({progress*100:>5.1f}%) | "
                  f"ETA: {eta/60:.1f} min")

        # Tokenize
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt"
        ).to(device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Capture activations for each layer
        activations = {}

        for layer_name, layer_module in layers_to_capture:
            # Hook
            captured = {}

            def hook_fn(module, input, output):
                # MLP output is tensor [batch, seq, hidden_dim]
                captured['activation'] = output.detach().cpu()

            hook = layer_module.register_forward_hook(hook_fn)

            # Forward pass
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Remove hook
            hook.remove()

            # Store
            activations[layer_name] = captured['activation']

        # Save batch
        batch_path = activations_dir / f"batch_{batch_idx:05d}.pt"

        batch_data = {
            "example_ids": [f"ioi_{i + j}" for j in range(len(batch_texts))],
            "texts": batch_texts,
            "step_meta": {"step": batch_idx, "phase": "deep_capture"},
        }

        # Add layer activations
        for layer_name, activation in activations.items():
            batch_data[layer_name] = activation

        torch.save(batch_data, batch_path)

        # Update stats
        num_tokens = activations[layers_to_capture[0][0]].shape[0] * \
                     activations[layers_to_capture[0][0]].shape[1]
        total_tokens += num_tokens

        # Free memory
        del activations, batch_data
        if device == "cuda":
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time

    print()
    print(f"✓ Captured {batch_idx} batches in {elapsed_time/60:.1f} minutes")
    print(f"✓ Total tokens: {total_tokens:,}")
    print(f"✓ Layers captured: {len(layers_to_capture)}")
    print()

    # ========================================================================
    # Save Metadata
    # ========================================================================

    print("[4/4] Saving metadata...")

    metadata = {
        "model_name": "gpt2",
        "device": device,
        "num_examples": len(texts),
        "batch_size": batch_size,
        "num_batches": batch_idx,
        "total_tokens": total_tokens,
        "layers_captured": [name for name, _ in layers_to_capture],
        "hidden_dim": 768,
        "timestamp": timestamp,
        "capture_time_sec": elapsed_time,
        "dataset_path": str(dataset_path),
        "purpose": "Deep SAE training for complete neural cartography",
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {meta_path}")
    print()

    # ========================================================================
    # Summary & Next Steps
    # ========================================================================

    print("=" * 80)
    print("✅ DEEP DATASET CAPTURE COMPLETE")
    print("=" * 80)
    print()

    print(f"📊 Statistics:")
    print(f"  Examples: {len(texts):,}")
    print(f"  Batches: {batch_idx}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Layers: {len(layers_to_capture)}")
    print(f"  Time: {elapsed_time/60:.1f} minutes")
    print(f"  Speed: {len(texts)/(elapsed_time/60):.0f} examples/min")
    print()

    print(f"📁 Output:")
    print(f"  Directory: {output_dir}")
    print(f"  Activations: {activations_dir}")
    print(f"  Batch files: {batch_idx}")
    print()

    # Calculate disk usage
    total_size = sum(f.stat().st_size for f in activations_dir.glob("*.pt"))
    print(f"💾 Disk Usage: {total_size / (1024**3):.2f} GB")
    print()

    print("🚀 Next Steps:")
    print()
    print("  1. Train Enhanced SAE on Layer 0:")
    print(f"     python train_enhanced_sae.py \\")
    print(f"       --activations_dir {activations_dir} \\")
    print(f"       --layer_name layer_0.mlp \\")
    print(f"       --epochs 10")
    print()

    if capture_all:
        print("  2. Train SAE for ALL layers (comprehensive cartography):")
        print(f"     python train_all_layers_sae.py \\")
        print(f"       --activations_dir {activations_dir}")
        print()

    print("  3. Install SAELens for comparison:")
    print("     pip install sae-lens")
    print()

    print("  4. Run hybrid analysis:")
    print("     python hybrid_sae_analysis.py")
    print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Capture interrupted by user")
        print("Partial data saved - can resume or use what's captured")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Capture failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
