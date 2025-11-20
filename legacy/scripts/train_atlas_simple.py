# train_atlas_simple.py

"""
Train SAEs for all 12 layers using the working train_layer0_sae.py approach.

Simple sequential training - one layer at a time.
"""

import sys
import json
import torch
import time
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from torch.utils.data import Dataset, DataLoader

from neurotrace.training import (
    create_enhanced_sae,
    EnhancedSAETrainer,
    EnhancedTrainingConfig,
)
from neurotrace.control import EnhancedSAEFeatureStore
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class DeepCaptureDataset(Dataset):
    """Dataset for deep capture activations - loads everything in RAM."""

    def __init__(self, activations_dir: str, layer_name: str, max_examples: int = 100000):
        self.activations_dir = Path(activations_dir)
        self.layer_name = layer_name

        # Find all batch files
        all_batch_files = sorted(self.activations_dir.glob("batch_*.pt"))

        if len(all_batch_files) == 0:
            raise ValueError(f"No batch files found in {activations_dir}")

        # Determine examples_per_batch from first batch
        first_batch = torch.load(all_batch_files[0], map_location='cpu')
        first_layer_acts = first_batch[self.layer_name]

        if len(first_layer_acts.shape) == 3:
            # [batch, seq, hidden] -> count batch * seq
            examples_per_batch = first_layer_acts.shape[0] * first_layer_acts.shape[1]
        else:
            # [tokens, hidden] -> count tokens
            examples_per_batch = first_layer_acts.shape[0]

        max_batches = int(max_examples / examples_per_batch)
        self.batch_files = all_batch_files[:max_batches]

        print(f"Loading activations from {len(self.batch_files)} batches (target: {max_examples:,} examples)...")
        print(f"  Examples per batch: {examples_per_batch}")
        all_activations = []

        for i, batch_file in enumerate(self.batch_files):
            if i % 500 == 0:
                print(f"  Progress: {i+1}/{len(self.batch_files)} batches...")

            batch_data = torch.load(batch_file, map_location='cpu')

            if self.layer_name not in batch_data:
                raise KeyError(f"Layer {self.layer_name} not found in {batch_file}")

            # Get layer activations (could be [B, S, D] or [N, D])
            layer_acts = batch_data[self.layer_name]

            # Flatten if needed
            if len(layer_acts.shape) == 3:
                layer_acts = layer_acts.reshape(-1, layer_acts.shape[-1])

            all_activations.append(layer_acts)

        # Concatenate all
        self.activations = torch.cat(all_activations, dim=0)

        # Trim to exact max_examples if needed
        if len(self.activations) > max_examples:
            self.activations = self.activations[:max_examples]

        print(f"✓ Loaded {self.activations.shape[0]:,} activation vectors")
        print(f"  Shape: {self.activations.shape}")
        mem_gb = self.activations.element_size() * self.activations.nelement() / (1024**3)
        print(f"  Memory: {mem_gb:.2f} GB")

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]


def validate_reconstruction(
    checkpoint_path: Path,
    layer: int,
    ioi_dataset_path: Path,
    device: str,
    reconstruction_threshold: float = 0.10,
    num_samples: int = 100
) -> dict:
    """
    Validate SAE reconstruction quality on IOI task.

    Returns dict with baseline_acc, reconstructed_acc, loss, valid
    """
    print(f"\n[VALIDATION] Testing reconstruction quality...")

    # Load model
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load IOI examples
    with open(ioi_dataset_path) as f:
        ioi_data = json.load(f)
    examples = ioi_data["examples"][:num_samples]

    # Load SAE
    feature_store = EnhancedSAEFeatureStore()
    feature_store.load_sae(
        checkpoint_path=str(checkpoint_path),
        layer=layer,
        device=device
    )
    sae = feature_store.saes[layer]

    def compute_accuracy(use_sae=False):
        correct = 0

        if use_sae:
            def sae_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                else:
                    hidden_states = output
                    rest = ()

                # Encode→Decode using EXACT same pipeline as training
                batch, seq, hidden = hidden_states.shape
                flat = hidden_states.view(-1, hidden)

                with torch.no_grad():
                    # Use the same forward() call that training uses
                    # SAE handles pre_bias centering internally
                    output = sae(flat)
                    reconstructed_flat = output['reconstruction']

                reconstructed = reconstructed_flat.view(batch, seq, hidden)

                if rest:
                    return (reconstructed, *rest)
                else:
                    return reconstructed

            handle = model.transformer.h[layer].mlp.register_forward_hook(sae_hook)

        try:
            for ex in examples:
                text = ex["text"]
                correct_answer = ex["correct_answer"]

                inputs = tokenizer(text, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0, -1, :]

                pred_id = logits.argmax().item()
                pred_token = tokenizer.decode([pred_id]).strip()

                if pred_token.lower() == correct_answer.lower():
                    correct += 1
        finally:
            if use_sae:
                handle.remove()

        return correct / len(examples)

    # Test baseline
    baseline_acc = compute_accuracy(use_sae=False)

    # Test with SAE reconstruction
    reconstructed_acc = compute_accuracy(use_sae=True)

    # Calculate loss
    loss = baseline_acc - reconstructed_acc
    valid = loss < reconstruction_threshold

    print(f"  Baseline accuracy:       {baseline_acc:.1%}")
    print(f"  Reconstructed accuracy:  {reconstructed_acc:.1%}")
    print(f"  Loss:                    {loss:+.1%}")
    print(f"  Valid:                   {'✓ YES' if valid else '✗ NO (exceeds threshold)'}")

    return {
        "baseline_acc": baseline_acc,
        "reconstructed_acc": reconstructed_acc,
        "loss": loss,
        "valid": valid,
        "threshold": reconstruction_threshold
    }


def train_layer(layer_idx: int, activations_dir: Path, output_dir: Path,
                ioi_dataset_path: Path, epochs: int = 10, device: str = "cuda"):
    """Train SAE for a single layer."""

    print("\n" + "=" * 80)
    print(f"TRAINING SAE FOR LAYER {layer_idx}")
    print("=" * 80)
    print()

    layer_name = f"layer_{layer_idx}.mlp"

    # Create dataset
    print(f"[1/4] Loading dataset for {layer_name}...")
    dataset = DeepCaptureDataset(
        activations_dir=str(activations_dir),
        layer_name=layer_name,
    )

    # Create SAE
    print(f"\n[2/4] Creating Enhanced SAE...")
    sae = create_enhanced_sae(
        input_dim=768,
        dict_mult=8,
        k_sparse=128,
        sparsity_lambda=1e-4,
        use_jumprelu=False,
        normalize_decoder=True,
    )
    print(f"✓ SAE created: {sae.dict_size} features")

    # Create output directory
    layer_output_dir = output_dir / f"layer_{layer_idx}"
    layer_output_dir.mkdir(parents=True, exist_ok=True)

    # Training config
    config = EnhancedTrainingConfig(
        input_dim=768,
        dict_mult=8,
        k_sparse=128,
        use_jumprelu=False,
        sparsity_lambda=1e-4,
        ghost_grad_weight=0.1,
        learning_rate=3e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        num_epochs=epochs,
        warmup_steps=500,
        use_cosine_schedule=True,
        batch_size=256,
        device=device,
        checkpoint_dir=str(layer_output_dir),
        save_every_n_epochs=5,
        log_every_n_batches=100,
        evaluate_every_n_epochs=1,
    )

    # Create trainer
    trainer = EnhancedSAETrainer(sae, config)

    # Create dataloader
    print(f"\n[3/4] Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )
    print(f"✓ DataLoader ready: {len(dataloader)} batches per epoch")

    # Train
    print(f"\n[4/4] Training SAE for Layer {layer_idx}...")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print()

    try:
        trainer.train(dataloader)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return None

    # Save final checkpoint
    trainer.save_checkpoint("final")

    # Get metrics
    summary = trainer.get_metrics_summary()

    print(f"\n✓ Layer {layer_idx} SAE training complete:")
    if summary:
        print(f"  MSE: {summary.get('final_mse_loss', 0.0):.4f}")
        print(f"  Dead features: {summary.get('final_dead_fraction', 0.0):.1%}")
    else:
        print(f"  ⚠️  No metrics available")
    print(f"  Checkpoint: {layer_output_dir / 'final.pt'}")

    # VALIDATION: Test reconstruction quality
    validation = validate_reconstruction(
        checkpoint_path=layer_output_dir / "final.pt",
        layer=layer_idx,
        ioi_dataset_path=ioi_dataset_path,
        device=device,
        reconstruction_threshold=0.10,
        num_samples=500  # Increased from 100 for better statistics
    )

    return {
        'layer': layer_idx,
        'checkpoint_path': str(layer_output_dir / 'final.pt'),
        'metrics': summary,
        'validation': validation,
    }


def main():
    print("=" * 80)
    print("NEUROTRACE - ATLAS TRAINING (ALL 12 LAYERS)")
    print("=" * 80)
    print()
    print("Goal: Complete neural cartography (1:1 mapping of all 12 layers)")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    activations_dir = Path("D:/NeuroTrace/20251118_123433/activations")
    ioi_dataset_path = Path("D:/NeuroTrace/20251118_123433/ioi_dataset.json")
    output_dir = Path("checkpoints/all_layers_sae")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layers to train (0-11)
    layers_to_train = list(range(12))

    print(f"Configuration:")
    print(f"  Activations: {activations_dir}")
    print(f"  Layers: {layers_to_train}")
    print(f"  Epochs per layer: 10")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print()

    # Estimate time
    est_time_per_layer = 10 * 5  # 10 epochs × ~5 min/epoch
    est_total_time = len(layers_to_train) * est_time_per_layer
    print(f"📊 Estimated total time: ~{est_total_time} minutes ({est_total_time/60:.1f} hours)")
    print()
    print("🚀 Starting training...")
    print()

    # Train each layer
    results = []
    start_time = datetime.now()

    for layer_idx in layers_to_train:
        result = train_layer(
            layer_idx=layer_idx,
            activations_dir=activations_dir,
            output_dir=output_dir,
            ioi_dataset_path=ioi_dataset_path,
            epochs=10,
            device=device,
        )

        if result is not None:
            results.append(result)

        # Show progress
        print(f"\n📊 Progress: {len(results)}/{len(layers_to_train)} layers complete")
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"   Elapsed time: {elapsed:.1f} minutes")
        if len(results) > 0:
            avg_time = elapsed / len(results)
            remaining = avg_time * (len(layers_to_train) - len(results))
            print(f"   Estimated remaining: {remaining:.1f} minutes")
        print()

        # GPU cooldown between layers (except after last layer)
        if layer_idx < layers_to_train[-1]:
            print("🌡️  GPU cooldown: 3 minutes...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for i in range(3, 0, -1):
                print(f"   {i} minutes remaining...", end='\r')
                time.sleep(60)
            print("   ✓ Cooldown complete     ")
            print()

    # Save summary
    summary_path = output_dir / "training_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'layers_trained': layers_to_train,
        'num_epochs': 10,
        'results': results,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("✅ ATLAS TRAINING COMPLETE")
    print("=" * 80)
    print()

    print(f"Trained SAEs for {len(results)} layers:")

    valid_count = 0
    rejected_count = 0

    for result in results:
        validation = result.get('validation', {})
        is_valid = validation.get('valid', False)

        if is_valid:
            valid_count += 1
            status = "✓ VALID"
        else:
            rejected_count += 1
            status = "✗ REJECTED"

        print(f"\n  Layer {result['layer']}: {status}")
        metrics = result.get('metrics', {})
        if metrics:
            print(f"    MSE: {metrics.get('final_mse_loss', 0.0):.4f}")
            print(f"    Dead features: {metrics.get('final_dead_fraction', 0.0):.1%}")

        if validation:
            loss = validation.get('loss', 0.0)
            print(f"    Reconstruction loss: {loss:+.1%}")

        print(f"    Checkpoint: {result['checkpoint_path']}")

    print(f"\n  Summary: {valid_count} valid, {rejected_count} rejected (threshold: 10%)")

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print()
    print(f"Total training time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"Summary saved to: {summary_path}")
    print()

    print("Next Steps:")
    print("  1. Validate all 12 SAE reconstructions")
    print("  2. Cross-layer feature comparison")
    print("  3. Multi-layer circuit discovery")
    print("  4. Build Neural Atlas Explorer (web UI)")
    print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
