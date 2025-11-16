# neurotrace/training/__init__.py

"""
NeuroTrace Training - SAE training pipeline.

Questo modulo fornisce:
- ActivationDataset: caricamento attivazioni da Phase 1
- SAETrainer: training loop con MSE + L1 sparsity
- SAECheckpoint: save/load trained models
"""

from .activation_dataset import ActivationDataset, ActivationBatch, LayerActivationDataset
from .sae_trainer import SAETrainer, TrainingConfig, TrainingMetrics
from .sae_checkpoint import SAECheckpoint, CheckpointMetadata
from .enhanced_sae import EnhancedSAE, create_enhanced_sae
from .enhanced_sae_trainer import EnhancedSAETrainer, EnhancedTrainingConfig, EnhancedTrainingMetrics

__all__ = [
    # Dataset
    "ActivationDataset",
    "ActivationBatch",
    "LayerActivationDataset",
    # Trainer
    "SAETrainer",
    "TrainingConfig",
    "TrainingMetrics",
    # Checkpoint
    "SAECheckpoint",
    "CheckpointMetadata",
    # Enhanced SAE
    "EnhancedSAE",
    "create_enhanced_sae",
    "EnhancedSAETrainer",
    "EnhancedTrainingConfig",
    "EnhancedTrainingMetrics",
]
