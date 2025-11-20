"""
Test per SAE Training Pipeline.

Verifica:
1. ActivationDataset: caricamento da file .pt
2. SAETrainer: training loop
3. SAECheckpoint: save/load
4. Integration end-to-end
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from neurotrace.state_indexer.sae_feature_extractor import LayerSparseAutoencoder
from neurotrace.training import (
    ActivationDataset,
    LayerActivationDataset,
    SAETrainer,
    TrainingConfig,
    SAECheckpoint,
    CheckpointMetadata,
)


# ============================================================================
# Mock Data Generation
# ============================================================================


def create_mock_activation_batch(output_path: Path, num_examples: int = 4, seq_len: int = 10, hidden_dim: int = 768):
    """
    Crea un file batch_*.pt fittizio con attivazioni mock.
    """
    batch_data = {
        "example_ids": [f"example_{i}" for i in range(num_examples)],
        "step_meta": {"step": 0, "phase": "test"},
    }

    # Simula attivazioni per 12 layer (GPT-2-like)
    for layer_idx in range(12):
        layer_name = f"layer_{layer_idx}.block"
        activations = torch.randn(num_examples, seq_len, hidden_dim)
        batch_data[layer_name] = activations

    torch.save(batch_data, output_path)


def create_mock_activations_dir(num_batches: int = 3) -> Path:
    """
    Crea directory temporanea con batch files mock.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="test_activations_"))

    for i in range(1, num_batches + 1):
        batch_path = tmpdir / f"batch_{i:04d}.pt"
        create_mock_activation_batch(batch_path)

    print(f"✓ Created mock activations dir: {tmpdir} ({num_batches} batches)")
    return tmpdir


# ============================================================================
# Tests
# ============================================================================


def test_activation_dataset():
    """Test ActivationDataset loading."""
    print("\n" + "=" * 70)
    print("TEST 1: ActivationDataset")
    print("=" * 70)

    # Create mock data
    activations_dir = create_mock_activations_dir(num_batches=2)

    try:
        # 1. General dataset (all layers)
        dataset = ActivationDataset(
            activations_dir=str(activations_dir),
            flatten_sequences=True,
            device="cpu",
        )
        print(f"✓ Dataset created: {len(dataset)} batch files")

        # 2. Get layer names
        layer_names = dataset.get_layer_names()
        print(f"✓ Found {len(layer_names)} layers: {layer_names[:3]}...")
        assert len(layer_names) == 12

        # 3. Iterate and check shapes
        count = 0
        for layer_name, activations in dataset:
            print(f"  {layer_name}: {tuple(activations.shape)}")
            assert activations.dim() == 2  # [N, D] (flattened)
            assert activations.shape[1] == 768  # hidden_dim
            count += 1
            if count >= 3:  # Just check first 3
                break

        print(f"✓ Iterated {count} layer batches")

        # 4. Estimate hidden dim
        hidden_dim = ActivationDataset.estimate_hidden_dim(str(activations_dir))
        print(f"✓ Estimated hidden_dim: {hidden_dim}")
        assert hidden_dim == 768

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(activations_dir, ignore_errors=True)

    print("\n✅ ActivationDataset tests PASSED")


def test_layer_activation_dataset():
    """Test LayerActivationDataset (single layer)."""
    print("\n" + "=" * 70)
    print("TEST 2: LayerActivationDataset")
    print("=" * 70)

    activations_dir = create_mock_activations_dir(num_batches=2)

    try:
        # Single layer dataset
        dataset = LayerActivationDataset(
            activations_dir=str(activations_dir),
            layer_name="layer_9.block",
            flatten_sequences=True,
            device="cpu",
        )
        print(f"✓ Layer dataset created for: {dataset.layer_name}")

        # Iterate
        count = 0
        for activations in dataset:
            print(f"  Batch {count + 1}: {tuple(activations.shape)}")
            assert activations.dim() == 2
            assert activations.shape[1] == 768
            count += 1

        print(f"✓ Loaded {count} batches for layer_9.block")

    finally:
        import shutil
        shutil.rmtree(activations_dir, ignore_errors=True)

    print("\n✅ LayerActivationDataset tests PASSED")


def test_sae_trainer():
    """Test SAETrainer training loop."""
    print("\n" + "=" * 70)
    print("TEST 3: SAETrainer")
    print("=" * 70)

    activations_dir = create_mock_activations_dir(num_batches=5)
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="test_checkpoints_"))

    try:
        # 1. Create dataset
        dataset = LayerActivationDataset(
            activations_dir=str(activations_dir),
            layer_name="layer_9.block",
            device="cpu",
        )

        dataloader = DataLoader(dataset, batch_size=32)
        print(f"✓ DataLoader created")

        # 2. Create SAE
        input_dim = 768
        dict_size = input_dim * 4  # 3072
        sae = LayerSparseAutoencoder(
            input_dim=input_dim,
            dict_size=dict_size,
            sparsity_lambda=1e-3,
        )
        print(f"✓ SAE created: {input_dim} → {dict_size}")

        # 3. Create training config
        config = TrainingConfig(
            input_dim=input_dim,
            dict_mult=4,
            sparsity_lambda=1e-3,
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=2,  # Just 2 epochs for testing
            device="cpu",
            checkpoint_dir=str(checkpoint_dir),
            save_every_n_batches=None,  # Disable batch checkpointing
            save_every_n_epochs=1,
            log_every_n_batches=10,
        )

        # 4. Create trainer
        trainer = SAETrainer(sae, config)
        print(f"✓ Trainer created")

        # 5. Train
        print("\n--- Training ---")
        trainer.train(dataloader, num_epochs=2)

        # 6. Check metrics
        summary = trainer.get_metrics_summary()
        print(f"\n✓ Training summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        assert summary["total_steps"] > 0
        assert summary["final_total_loss"] > 0

        # 7. Check checkpoint exists
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        print(f"✓ Created {len(checkpoint_files)} checkpoint(s)")
        assert len(checkpoint_files) >= 2  # epoch_1, epoch_2, final

    finally:
        import shutil
        shutil.rmtree(activations_dir, ignore_errors=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print("\n✅ SAETrainer tests PASSED")


def test_sae_checkpoint():
    """Test SAECheckpoint save/load."""
    print("\n" + "=" * 70)
    print("TEST 4: SAECheckpoint")
    print("=" * 70)

    checkpoint_dir = Path(tempfile.mkdtemp(prefix="test_checkpoints_"))

    try:
        # 1. Create SAE
        sae = LayerSparseAutoencoder(input_dim=768, dict_size=3072, sparsity_lambda=1e-3)
        print(f"✓ Created SAE: {sae.input_dim} → {sae.dict_size}")

        # 2. Create metadata
        metadata = CheckpointMetadata(
            layer_name="layer_9.block",
            model_name="gpt2",
            input_dim=768,
            dict_size=3072,
            sparsity_lambda=1e-3,
            training_steps=1000,
            training_epochs=10,
            final_loss=0.123,
            final_sparsity=42.5,
            created_at="2025-11-15T12:00:00",
            notes="Test checkpoint",
        )

        # 3. Save
        checkpoint_manager = SAECheckpoint(str(checkpoint_dir))
        saved_path = checkpoint_manager.save(sae, metadata, name="test_checkpoint")
        print(f"✓ Saved checkpoint: {saved_path}")

        # 4. List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        print(f"✓ Found {len(checkpoints)} checkpoint(s): {checkpoints}")
        assert "test_checkpoint" in checkpoints

        # 5. Get metadata only
        loaded_metadata = checkpoint_manager.get_metadata("test_checkpoint")
        assert loaded_metadata is not None
        assert loaded_metadata.layer_name == "layer_9.block"
        assert loaded_metadata.training_steps == 1000
        print(f"✓ Loaded metadata: {loaded_metadata.layer_name}, {loaded_metadata.training_epochs} epochs")

        # 6. Load full checkpoint
        loaded_sae, loaded_metadata_full = checkpoint_manager.load("test_checkpoint", device="cpu")
        assert loaded_sae.input_dim == 768
        assert loaded_sae.dict_size == 3072
        print(f"✓ Loaded SAE: {loaded_sae.input_dim} → {loaded_sae.dict_size}")

        # 7. Verify weights match
        original_weight = sae.encoder.weight.data
        loaded_weight = loaded_sae.encoder.weight.data
        assert torch.allclose(original_weight, loaded_weight)
        print(f"✓ Weights match (max diff: {(original_weight - loaded_weight).abs().max():.2e})")

    finally:
        import shutil
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print("\n✅ SAECheckpoint tests PASSED")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("NEUROTRACE SAE TRAINING - INTEGRATION TESTS")
    print("=" * 70)

    try:
        test_activation_dataset()
        test_layer_activation_dataset()
        test_sae_trainer()
        test_sae_checkpoint()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED")
        print("=" * 70)
        print("\nSAE Training Pipeline is ready!")
        print("\nNext steps:")
        print("1. Capture activations with Phase 1 CLI")
        print("2. Train SAE with: python cli/train_sae.py")
        print("3. Load trained SAE into SAEFeatureExtractor")
        print("4. Use with Control Plane for real steering")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
