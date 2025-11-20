# capture_ioi_activations.py

"""
Capture activations from IOI examples for SAE training.

This script:
1. Loads IOI dataset (1000 examples)
2. Runs forward passes through GPT-2
3. Captures Layer 0 MLP activations
4. Saves in format compatible with SAE training
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 80)
    print("NEUROTRACE - IOI ACTIVATION CAPTURE")
    print("=" * 80)
    print()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2"
    num_examples = 1000
    batch_size = 50  # 6GB VRAM optimized

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/phase1_ioi_activations/{timestamp}")
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Examples: {num_examples}")
    print(f"Batch size: {batch_size}")
    print()

    # ========================================================================
    # Load IOI Dataset
    # ========================================================================

    print("[1/4] Loading IOI dataset...")

    # Use existing dataset from validation run
    ioi_dataset_path = Path("runs/discovery_validation/20251116_120236/ioi_dataset.json")

    if not ioi_dataset_path.exists():
        print(f"❌ IOI dataset not found: {ioi_dataset_path}")
        print("Please run discovery validation first to generate dataset")
        return

    with open(ioi_dataset_path, 'r') as f:
        ioi_data = json.load(f)

    # Extract texts
    if isinstance(ioi_data, dict) and 'examples' in ioi_data:
        texts = [ex['text'] for ex in ioi_data['examples'][:num_examples]]
    elif isinstance(ioi_data, list):
        texts = [ex['text'] if isinstance(ex, dict) else str(ex) for ex in ioi_data[:num_examples]]
    else:
        print(f"❌ Unexpected dataset format")
        return

    print(f"✓ Loaded {len(texts)} IOI examples")
    print()

    # ========================================================================
    # Load Model
    # ========================================================================

    print("[2/4] Loading GPT-2 model...")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded: {model_name}")
    print()

    # ========================================================================
    # Capture Activations
    # ========================================================================

    print("[3/4] Capturing activations...")
    print(f"      - Batches: {len(texts) // batch_size + 1}")
    print(f"      - Target: Layer 0 MLP (layer.0.mlp)")
    print()

    batch_idx = 0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_idx += 1
        batch_texts = texts[i:i + batch_size]

        print(f"[Batch {batch_idx}/{len(texts) // batch_size + 1}] Processing {len(batch_texts)} examples...")

        # Tokenize
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=30,
            return_tensors="pt"
        ).to(device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Hook to capture Layer 0 MLP output
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                # output[0] is the tensor [batch, seq, hidden_dim]
                activations[name] = output[0].detach().cpu()
            return hook

        # Register hook on Layer 0 MLP
        # GPT-2 structure: transformer.h[0].mlp
        hook = model.transformer.h[0].mlp.register_forward_hook(hook_fn("layer_0.mlp"))

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Remove hook
        hook.remove()

        # Save batch
        batch_path = activations_dir / f"batch_{batch_idx:04d}.pt"

        torch.save({
            "example_ids": [f"ioi_{i + j}" for j in range(len(batch_texts))],
            "texts": batch_texts,
            "layer_0.mlp": activations["layer_0.mlp"],  # [B, S, 768]
            "step_meta": {"step": batch_idx, "phase": "ioi_capture"},
        }, batch_path)

        num_tokens = activations["layer_0.mlp"].shape[0] * activations["layer_0.mlp"].shape[1]
        total_tokens += num_tokens

        print(f"  ✓ Captured: {activations['layer_0.mlp'].shape}")
        print(f"  ✓ Saved to: {batch_path.name}")

        # Free memory
        del activations, outputs, input_ids, attention_mask
        if device == "cuda":
            torch.cuda.empty_cache()

    print()
    print(f"✓ Captured {batch_idx} batches")
    print(f"✓ Total tokens: {total_tokens:,}")
    print()

    # ========================================================================
    # Save Metadata
    # ========================================================================

    print("[4/4] Saving metadata...")

    metadata = {
        "model_name": model_name,
        "device": device,
        "num_examples": len(texts),
        "batch_size": batch_size,
        "num_batches": batch_idx,
        "total_tokens": total_tokens,
        "layer_captured": "layer_0.mlp",
        "hidden_dim": 768,
        "timestamp": timestamp,
        "source_dataset": str(ioi_dataset_path),
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {meta_path}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("✅ ACTIVATION CAPTURE COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Activation files: {activations_dir}")
    print(f"Total batches: {batch_idx}")
    print(f"Total tokens: {total_tokens:,}")
    print()
    print("Next step:")
    print(f"  python cli/train_sae.py \\")
    print(f"    --activations_dir {activations_dir} \\")
    print(f"    --layer_name layer_0.mlp \\")
    print(f"    --model_name gpt2 \\")
    print(f"    --output_dir checkpoints/sae \\")
    print(f"    --epochs 10")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Capture failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
