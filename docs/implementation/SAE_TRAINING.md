# 🧠 SAE Training Pipeline - Phase 2

**Training Sparse Autoencoders for Neural Circuit Control**

---

## 🎯 Overview

Phase 2 fornisce un sistema completo per addestrare **Sparse Autoencoders (SAE)** sulle attivazioni catturate in Phase 1, producendo feature monosemantiche utilizzabili dal Control Plane per steering attivo.

### Pipeline Completa

```
Phase 1 Activations (.pt files)
         ↓
ActivationDataset (PyTorch IterableDataset)
         ↓
SAETrainer (MSE + L1 sparsity loss)
         ↓
Trained SAE Checkpoints
         ↓
SAEFeatureExtractor (load from checkpoint)
         ↓
Control Plane (real steering vectors!)
```

---

## 📦 Componenti

### 1. ActivationDataset

**File**: [neurotrace/training/activation_dataset.py](neurotrace/training/activation_dataset.py)

**Purpose**: Carica batch di attivazioni da file `.pt` salvati da Phase 1.

**Layout atteso**:
```
activations_dir/
    batch_0001.pt
    batch_0002.pt
    ...
```

Ogni `batch_*.pt` contiene:
```python
{
    "example_ids": ["ex1", "ex2", ...],
    "step_meta": {"step": 0, "phase": "capture"},
    "layer_0.block": Tensor[B, S, D],
    "layer_1.block": Tensor[B, S, D],
    ...
}
```

**API**:
```python
from neurotrace.training import ActivationDataset

# Tutti i layer
dataset = ActivationDataset(
    activations_dir="runs/phase1/activations",
    flatten_sequences=True,  # [B, S, D] → [B*S, D]
    device="cpu"
)

# Itera su tutti i layer
for layer_name, activations in dataset:
    # layer_name: "layer_9.block"
    # activations: [N, D]
    process(activations)

# Stima hidden_dim
hidden_dim = ActivationDataset.estimate_hidden_dim(activations_dir)
```

**LayerActivationDataset** (specializzazione per singolo layer):
```python
from neurotrace.training import LayerActivationDataset

dataset = LayerActivationDataset(
    activations_dir="runs/phase1/activations",
    layer_name="layer_9.block",
    flatten_sequences=True
)

# Ritorna solo tensori (no layer_name prefix)
for activations in dataset:
    # activations: [N, D]
    train_sae(activations)
```

---

### 2. SAETrainer

**File**: [neurotrace/training/sae_trainer.py](neurotrace/training/sae_trainer.py)

**Purpose**: Training loop completo con:
- **MSE reconstruction loss**
- **L1 sparsity penalty**
- **Cosine LR scheduling**
- **Gradient clipping**
- **Checkpointing**

**Loss function**:
```python
mse_loss = torch.mean((reconstruction - input) ** 2)
l1_loss = torch.mean(torch.abs(sparse_codes))
total_loss = mse_loss + λ * l1_loss
```

**Configurazione**:
```python
from neurotrace.training import TrainingConfig

config = TrainingConfig(
    # Model
    input_dim=768,
    dict_mult=4,  # dict_size = 4 * 768 = 3072
    sparsity_lambda=1e-3,

    # Optimization
    learning_rate=3e-4,
    weight_decay=1e-5,
    batch_size=256,
    num_epochs=10,
    grad_clip=1.0,

    # LR scheduling
    use_cosine_schedule=True,
    min_lr_factor=0.1,

    # Device
    device="cuda",

    # Checkpointing
    checkpoint_dir="checkpoints/sae",
    save_every_n_batches=1000,  # opzionale
    save_every_n_epochs=1,

    # Logging
    log_every_n_batches=100,
)
```

**Usage**:
```python
from neurotrace.training import SAETrainer
from neurotrace.state_indexer.sae_feature_extractor import LayerSparseAutoencoder
from torch.utils.data import DataLoader

# Create SAE
sae = LayerSparseAutoencoder(
    input_dim=768,
    dict_size=3072,
    sparsity_lambda=1e-3
)

# Create trainer
trainer = SAETrainer(sae, config)

# Train
dataloader = DataLoader(dataset, batch_size=256)
trainer.train(dataloader, num_epochs=10)

# Get summary
summary = trainer.get_metrics_summary()
# {
#     "final_total_loss": 0.145,
#     "final_reconstruction_error": 0.142,
#     "final_sparsity": 32.5,  # avg non-zero activations
#     "total_steps": 50000
# }
```

**Metriche loggabili**:
- `total_loss`: MSE + λ·L1
- `mse_loss`: Reconstruction error
- `l1_loss`: Sparsity penalty
- `reconstruction_error`: MSE (duplicato per chiarezza)
- `sparsity`: Media numero attivazioni non-zero (L0-like)
- `current_lr`: Learning rate corrente

---

### 3. SAECheckpoint

**File**: [neurotrace/training/sae_checkpoint.py](neurotrace/training/sae_checkpoint.py)

**Purpose**: Gestione checkpoint SAE con metadata ricchi.

**Formato checkpoint**:
```python
{
    "state_dict": {...},  # SAE weights
    "config": {
        "input_dim": 768,
        "dict_size": 3072,
        "sparsity_lambda": 1e-3
    },
    "metadata": {
        "layer_name": "layer_9.block",
        "model_name": "gpt2",
        "training_steps": 50000,
        "training_epochs": 10,
        "final_loss": 0.145,
        "final_sparsity": 32.5,
        "created_at": "2025-11-16T00:00:00",
        "notes": "Trained on wikitext-103"
    },
    "optimizer_state": {...}  # opzionale
}
```

**API**:
```python
from neurotrace.training import SAECheckpoint, CheckpointMetadata

checkpoint_manager = SAECheckpoint("checkpoints/sae")

# Save
metadata = CheckpointMetadata(
    layer_name="layer_9.block",
    model_name="gpt2",
    input_dim=768,
    dict_size=3072,
    sparsity_lambda=1e-3,
    training_steps=50000,
    training_epochs=10,
    final_loss=0.145,
    final_sparsity=32.5,
    created_at=datetime.utcnow().isoformat(),
)

checkpoint_manager.save(
    sae=trained_sae,
    metadata=metadata,
    name="layer_9_final"
)

# Load
sae, metadata = checkpoint_manager.load("layer_9_final", device="cuda")

# List checkpoints
checkpoints = checkpoint_manager.list_checkpoints()
# ['layer_9_final', 'layer_10_final', ...]

# Get metadata only (fast)
meta = checkpoint_manager.get_metadata("layer_9_final")
print(f"Loss: {meta.final_loss}, Sparsity: {meta.final_sparsity}")
```

**Utility per caricamento batch**:
```python
from neurotrace.training.sae_checkpoint import load_saes_for_model

# Carica tutti i SAE per un modello
saes = load_saes_for_model(
    checkpoint_dir="checkpoints/sae",
    model_name="gpt2",
    device="cuda"
)
# {"layer_0.block": SAE, "layer_1.block": SAE, ...}
```

---

## 🚀 Usage - CLI

### Training SAE per singolo layer

```bash
python cli/train_sae.py \
    --activations_dir runs/phase1_capture/activations \
    --layer_name layer_9.block \
    --model_name gpt2 \
    --output_dir checkpoints/sae \
    --epochs 10 \
    --batch_size 256 \
    --lr 3e-4 \
    --device cuda
```

**Parametri principali**:
- `--activations_dir`: Directory con `batch_*.pt` files
- `--layer_name`: Layer target (es. `layer_9.block`)
- `--model_name`: Nome modello per metadata
- `--output_dir`: Directory output checkpoints
- `--epochs`: Numero epoche
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--dict_mult`: Dictionary size multiplier (default: 4)
- `--sparsity_lambda`: L1 penalty weight (default: 1e-3)
- `--device`: cuda/cpu/auto

**Parametri avanzati**:
- `--grad_clip`: Gradient clipping (default: 1.0)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--no_cosine_schedule`: Disabilita cosine LR scheduling
- `--save_every_n_batches`: Salva checkpoint ogni N batch
- `--save_every_n_epochs`: Salva checkpoint ogni N epoche
- `--max_batches`: Limita batch per debugging
- `--resume_from`: Riprendi training da checkpoint

### Training loop per tutti i layer

```bash
#!/bin/bash
# train_all_layers.sh

ACTIVATIONS_DIR="runs/phase1_capture/activations"
OUTPUT_DIR="checkpoints/sae"

for LAYER in layer_{0..11}.block; do
    echo "Training SAE for $LAYER..."
    python cli/train_sae.py \
        --activations_dir $ACTIVATIONS_DIR \
        --layer_name $LAYER \
        --model_name gpt2 \
        --output_dir $OUTPUT_DIR \
        --epochs 10 \
        --batch_size 256 \
        --device cuda
done

echo "All SAE trained!"
```

### Monitoring training

Training log viene salvato in `{output_dir}/training.log`:
```
2025-11-16 00:00:00 [INFO] SAETrainer initialized: 768 → 3072
2025-11-16 00:00:01 [INFO] Starting training: 10 epochs, 195 batches/epoch
2025-11-16 00:00:05 [INFO] [Epoch 1 | Batch 100/195] Loss: 0.234 (MSE: 0.231, L1: 0.285) | Recon: 0.231 | Sparsity: 35.2 | LR: 3.00e-04
...
2025-11-16 00:05:00 [INFO] Epoch 1 Summary: Avg Loss=0.198, Avg Sparsity=32.8
...
2025-11-16 00:50:00 [INFO] TRAINING COMPLETE
2025-11-16 00:50:01 [INFO] Final checkpoint: checkpoints/sae/layer_9_final.pt
```

---

## 🧪 Testing

```bash
python test_sae_training.py
```

**Test coverage**:
1. ✅ **ActivationDataset**: caricamento batch files, iteration, hidden_dim estimation
2. ✅ **LayerActivationDataset**: single-layer specialization
3. ✅ **SAETrainer**: training loop, checkpointing, metrics
4. ✅ **SAECheckpoint**: save/load, metadata persistence

**Test results** (2025-11-16):
```
✅ ActivationDataset tests PASSED
✅ LayerActivationDataset tests PASSED
✅ SAETrainer tests PASSED
✅ SAECheckpoint tests PASSED

🎉 ALL TESTS PASSED
```

---

## 🔗 Integration con Control Plane

### Step 1: Train SAE

```bash
python cli/train_sae.py \
    --activations_dir runs/phase1/activations \
    --layer_name layer_9.block \
    --model_name gpt2 \
    --output_dir checkpoints/sae \
    --epochs 10
```

### Step 2: Load trained SAE into FeatureExtractor

```python
from neurotrace.state_indexer.sae_feature_extractor import SAEFeatureExtractor
from neurotrace.training.sae_checkpoint import load_saes_for_model

# Carica tutti i SAE addestrati
trained_saes = load_saes_for_model(
    checkpoint_dir="checkpoints/sae",
    model_name="gpt2",
    device="cuda"
)

# Crea SAEFeatureExtractor con SAE trained
# (Modifica SAEFeatureExtractor per accettare dict di SAE)
extractor = SAEFeatureExtractor(cfg, sae_cfg)
extractor.saes = trained_saes  # override auto-init
```

### Step 3: Use con Control Plane

```python
from neurotrace.control import SAEFeatureStore, SteeringBuilder, CircuitController

# FeatureStore con SAE trained
feature_store = SAEFeatureStore(extractor)

# SteeringBuilder usa le direzioni SAE addestrate
builder = SteeringBuilder(feature_store)

# CircuitController con steering vectors reali!
controller = CircuitController(wrapper, registry, builder)
controller.enable_circuit("circuit_ioi", alpha=0.7)
output = controller.generate("John told Mary that...")
# Steering basato su feature monosemantiche apprese!
```

---

## 📊 Hyperparameter Tuning

### Recommended starting points

**GPT-2 Small (768 hidden dim)**:
```python
dict_mult = 4  # 3072 dictionary
sparsity_lambda = 1e-3
learning_rate = 3e-4
batch_size = 256
epochs = 10
```

**GPT-2 Medium/Large**:
```python
dict_mult = 4
sparsity_lambda = 5e-4  # ridotto per modelli grandi
learning_rate = 1e-4    # LR più basso
batch_size = 512        # batch più grandi
epochs = 5-10
```

### Tuning guidelines

**Sparsity too high** (>50 non-zero activations):
- Aumenta `sparsity_lambda` (1e-3 → 5e-3)
- Target: 20-40 non-zero per token

**Reconstruction error too high** (>0.5):
- Riduci `sparsity_lambda` (1e-3 → 5e-4)
- Aumenta `dict_mult` (4 → 8)
- Più epoche

**Training instability**:
- Riduci `learning_rate` (3e-4 → 1e-4)
- Aumenta `grad_clip` (1.0 → 0.5)
- Usa cosine schedule

**Interpretability check**:
- Dopo training, visualizza top activating examples per feature
- Feature dovrebbero essere monosemantiche (1 concept)
- Se polysemantiche: aumenta dict_size

---

## 🔬 Advanced Usage

### Resume training

```bash
python cli/train_sae.py \
    --resume_from layer_9_final \
    --epochs 5  # additional epochs
```

### Multi-GPU training

```python
# Wrap SAE in DataParallel
sae = nn.DataParallel(LayerSparseAutoencoder(...))
trainer = SAETrainer(sae, config)
```

### Custom loss function

Estendi `SAETrainer`:
```python
class CustomSAETrainer(SAETrainer):
    def compute_loss(self, output, activations):
        # Custom loss: MSE + L1 + ghost grads + ...
        mse = torch.mean((output["reconstruction"] - activations) ** 2)
        l1 = torch.mean(torch.abs(output["codes"]))
        ghost_grad = self.compute_ghost_gradients(output)

        return mse + self.config.sparsity_lambda * l1 + 0.1 * ghost_grad
```

### Activation sampling

Per dataset molto grandi, campiona un subset:
```python
dataset = LayerActivationDataset(
    activations_dir="...",
    layer_name="layer_9.block",
    max_batches=1000  # limita a 1000 batch
)
```

---

## 📈 Expected Results

### Training metrics evolution

**Epoch 1**:
- Total Loss: ~0.5-1.0
- MSE: ~0.5
- Sparsity: ~40-60 (initialization random)

**Epoch 5**:
- Total Loss: ~0.2-0.3
- MSE: ~0.15-0.2
- Sparsity: ~25-35 (convergendo)

**Epoch 10** (converged):
- Total Loss: ~0.15-0.2
- MSE: ~0.1-0.15
- Sparsity: ~20-30 (stable)

### Quality indicators

✅ **Good SAE**:
- Reconstruction error < 0.2
- Sparsity 20-40
- Loss plateaued
- Feature interpretability high

⚠️ **Needs tuning**:
- Reconstruction error > 0.5 → più capacità (dict_mult↑)
- Sparsity > 50 → meno penalità (λ↓)
- Sparsity < 10 → più penalità (λ↑)

---

## 🗂️ Files Created

1. **neurotrace/training/__init__.py** - Package exports
2. **neurotrace/training/activation_dataset.py** - Dataset loader
3. **neurotrace/training/sae_trainer.py** - Training loop
4. **neurotrace/training/sae_checkpoint.py** - Checkpoint management
5. **cli/train_sae.py** - CLI tool
6. **test_sae_training.py** - Integration tests

**Total LOC**: ~1,200

---

## 🎯 Next Steps

### Immediate
1. ✅ **Capture activations** con Phase 1 CLI
2. ✅ **Train SAE** per layer critici (es. layer 9, 10, 11)
3. ✅ **Validate quality** (reconstruction, sparsity)

### Short-term
4. ⏳ **Load trained SAE** in SAEFeatureExtractor
5. ⏳ **Integrate con Control Plane**
6. ⏳ **Test steering** con SAE reali vs mock

### Long-term
7. ⏳ **Feature interpretability** analysis
8. ⏳ **Circuit discovery** con trained SAE
9. ⏳ **Cross-model transfer** learning

---

**Phase 2 Status**: 🟢 **COMPLETE & TESTED**

**Production Ready**: ✅ CLI + Tests + Documentation

**Next**: Integra SAE trained in Control Plane per **real steering vectors**! 🚀

