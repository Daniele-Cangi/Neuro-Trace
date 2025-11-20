# NeuroTrace - Project Overview

**Complete structure and functionality documentation**
Last Updated: 2025-11-17

---

## Project Status

**Infrastructure**: ✅ Complete
**Research**: 🔄 In Progress
**Atlas**: ✅ All 12 layers trained (36,864 features)

---

## Core Components

### 1. Production Scripts (Ready to Use)

#### Training
- **`train_atlas_simple.py`** ⭐ PRIMARY TRAINER
  - Trains SAEs for all 12 layers
  - Usage: `python train_atlas_simple.py --layers all`
  - Output: `checkpoints/all_layers_sae/layer_X/final.pt`
  - Status: ✅ Working (5.3 min/layer on consumer GPU)

#### Validation
- **`validate_atlas.py`** ⭐ ATLAS VALIDATOR
  - Validates all 12 trained SAEs
  - Shows MSE, L0, dead features, overlap
  - Output: Quality report + summary JSON
  - Status: ✅ Working

#### Analysis
- **`run_atlas_analysis.py`** ⭐ COMPLETE PIPELINE
  - Loads all 12 SAEs
  - Cross-layer feature analysis
  - Circuit registry loading
  - 3D visualization generation
  - Output: `atlas_analysis_report.json` + HTML viz
  - Status: ✅ Working

### 2. Library Structure (`neurotrace/`)

```
neurotrace/
├── __init__.py                          # Package initialization
├── config.py                            # Global configuration
│
├── models/                              # Model wrappers
│   ├── __init__.py
│   └── wrapper.py                       # TargetModelWrapper for GPT-2
│
├── instrumentation/                     # Activation capture
│   ├── __init__.py
│   ├── adaptive_hook_manager.py         # Hook registration & management
│   └── adaptive_activations_buffer.py   # Memory-efficient activation storage
│
├── training/                            # SAE training
│   ├── __init__.py
│   ├── enhanced_sae.py                  # ⭐ EnhancedSAE architecture
│   ├── enhanced_sae_trainer.py          # ⭐ Trainer with ghost grads
│   ├── activation_dataset.py            # Dataset loader for activations
│   ├── sae_checkpoint.py                # Checkpoint save/load
│   └── sae_trainer.py                   # Base SAE trainer
│
├── control/                             # ⭐ ACTIVE CONTROL PLANE
│   ├── __init__.py
│   ├── enhanced_sae_feature_store.py    # ⭐ Load all 12 SAEs
│   ├── sae_feature_store.py             # Base feature store
│   ├── circuit_registry.py              # ⭐ Circuit database (SQLite)
│   ├── controller.py                    # Active steering controller
│   └── steering_builder.py              # Build steering vectors from SAEs
│
├── causal/                              # Causal discovery
│   ├── __init__.py
│   ├── vlo_tester.py                    # ⭐ VLO (Value of Learned Org) testing
│   └── circuit_extractor.py             # Extract circuits from VLO results
│
├── discovery/                           # Automated discovery
│   ├── __init__.py
│   ├── exhaustive_scanner.py            # Scan all components systematically
│   └── component_interaction_matrix.py  # Component interaction analysis
│
├── state_indexer/                       # Feature database
│   ├── __init__.py
│   ├── sae_feature_extractor.py         # Extract features from SAEs
│   └── vector_state_db.py               # Vector database for features
│
├── visualization/                       # ⭐ VISUALIZATIONS
│   ├── __init__.py
│   ├── activation_explorer.py           # ⭐ 3D PCA/t-SNE/UMAP plots
│   ├── circuit_graph.py                 # Interactive circuit graphs (pyvis)
│   ├── metrics_plotter.py               # Training metrics plots
│   └── sae_feature_viz.py               # SAE feature visualization
│
├── datasets/                            # Task generators
│   ├── __init__.py
│   ├── ioi_generator.py                 # IOI dataset generation
│   └── task_generator.py                # Generic task generator
│
└── analysis/                            # Geometric analysis
    ├── __init__.py
    └── geometric.py                     # LID, spectral analysis
```

### 3. Utility Scripts (Root Directory)

#### Active Use
- **`setup.py`** - Package installation
- **`capture_deep_dataset.py`** - Capture activations for all 12 layers
- **`capture_ioi_activations.py`** - Capture IOI-specific activations

#### One-Time / Debug (Can Archive)
- `train_layer0_sae.py` - Original Layer 0 trainer (superseded by train_atlas_simple.py)
- `compare_trainings.py` - Compare old vs new training (one-time analysis)
- `classify_files.py` - File classification utility (one-time)
- `hybrid_sae_analysis.py` - SAE comparison analysis (one-time)

### 4. CLI Tools (`cli/`)

- **`train_sae.py`** - CLI wrapper for SAE training
- `run_phase1_capture.py` - Phase 1 capture runner
- `neuro_control_run.py` - Control plane CLI

### 5. Examples (`examples/`)

- **`control_plane_steering_example.py`** - Steering demonstration

### 6. Tests (`tests/`)

- **`validation/test_system_diagnostic.py`** - System health check (81.7% pass)

---

## Data Structure

```
Analisi_Neurale/
│
├── checkpoints/
│   └── all_layers_sae/              # ⭐ THE ATLAS
│       ├── layer_0/final.pt         # 3,072 features
│       ├── layer_1/final.pt
│       ├── ...
│       └── layer_11/final.pt
│       └── training_summary.json    # Training stats
│
├── runs/
│   └── deep_ioi_capture/
│       └── 20251116_171258/
│           ├── activations/         # 100K samples × 12 layers
│           │   ├── batch_00001.pt
│           │   └── ...
│           └── ioi_dataset.json     # IOI examples
│
├── circuits/
│   └── atlas_circuits.db            # ⭐ SQLite registry (3 circuits)
│
├── visualizations/
│   └── layer_features_pca_3d.html   # ⭐ Interactive 3D plot (4.5MB)
│
└── atlas_analysis_report.json       # ⭐ Latest analysis results
```

---

## Workflow: How Everything Connects

### Phase 1: Data Capture
```bash
# Capture activations for all layers
python capture_deep_dataset.py
# Output: runs/deep_ioi_capture/*/activations/
```

### Phase 2: Atlas Training
```bash
# Train all 12 SAEs
python train_atlas_simple.py --layers all
# Output: checkpoints/all_layers_sae/layer_*/final.pt
# Time: ~63 minutes total (5.3 min/layer)
```

### Phase 3: Validation
```bash
# Validate Atlas quality
python validate_atlas.py
# Checks: MSE, sparsity, dead features, overlap
```

### Phase 4: Analysis
```bash
# Run complete analysis pipeline
python run_atlas_analysis.py
# Outputs:
#   - atlas_analysis_report.json
#   - visualizations/layer_features_pca_3d.html
```

### Phase 5: Research (In Progress)
```python
# Load Atlas
from neurotrace.control import EnhancedSAEFeatureStore
store = EnhancedSAEFeatureStore()
for i in range(12):
    store.load_sae(f'checkpoints/all_layers_sae/layer_{i}/final.pt', i)

# Access circuits
from neurotrace.control import CircuitRegistry
registry = CircuitRegistry('circuits/atlas_circuits.db')
circuits = registry.list()

# Visualize
from neurotrace.visualization import CircuitGraphVisualizer
viz = CircuitGraphVisualizer()
viz.visualize_circuit(circuits[0], 'output.html')
```

---

## Key Modules Explained

### EnhancedSAE Architecture
**File**: `neurotrace/training/enhanced_sae.py`

Features:
- **Gated encoder**: Better feature separation
- **Decoder normalization**: Prevents feature suppression
- **Ghost gradients**: Revives dead features
- **Top-k activation**: Exactly 64 active features per input

```python
class EnhancedSAE(nn.Module):
    encoder: nn.Linear        # [hidden, dict_size]
    encoder_gate: nn.Linear   # [hidden, dict_size]
    decoder: nn.Linear        # [dict_size, hidden]
    # Top-k=64, L1 coefficient auto-adjusted
```

### EnhancedSAEFeatureStore
**File**: `neurotrace/control/enhanced_sae_feature_store.py`

Purpose: Load and manage all 12 trained SAEs

```python
store = EnhancedSAEFeatureStore()
store.load_sae('checkpoints/all_layers_sae/layer_0/final.pt', layer=0)
sae = store.saes[0]  # Access Layer 0 SAE
```

### CircuitRegistry
**File**: `neurotrace/control/circuit_registry.py`

Purpose: SQLite database for discovered circuits

```python
registry = CircuitRegistry('circuits/atlas_circuits.db')
circuits = registry.list(task_tag='ioi', min_vlo=0.5)
registry.upsert(new_circuit)
```

### ActivationExplorer
**File**: `neurotrace/visualization/activation_explorer.py`

Purpose: 3D visualization with PCA/t-SNE/UMAP

```python
explorer = ActivationExplorer()
fig = explorer.plot_activations_3d(
    activations,
    labels=layer_names,
    method=DimReductionMethod.PCA,
    output_path='viz.html'
)
```

---

## Files to Archive/Remove

### Can Archive (One-Time Use):
1. `classify_files.py` - File classification (already done)
2. `compare_trainings.py` - Training comparison (already analyzed)
3. `hybrid_sae_analysis.py` - SAE comparison (completed)
4. `train_layer0_sae.py` - Old trainer (superseded)

### Keep (Active Use):
1. `train_atlas_simple.py` ⭐ - Primary trainer
2. `validate_atlas.py` ⭐ - Validation
3. `run_atlas_analysis.py` ⭐ - Analysis pipeline
4. `capture_deep_dataset.py` - Data capture
5. `setup.py` - Installation

---

## Atlas Statistics

**Training**:
- Layers: 12/12 ✅
- Features per layer: 3,072
- Total features: 36,864
- Training time: 63.6 minutes total
- GPU: Consumer (6GB VRAM)

**Quality**:
- Excellent (MSE < 0.01): 5 layers
- Good (0.01-0.02): 3 layers
- Acceptable (0.02-0.05): 2 layers
- Monitor (>= 0.05): 2 layers (10, 11)

**Sparsity**:
- L0: Exactly 64.0 (top-k enforcement)
- Dead features: 0.0% across all layers

**Specialization**:
- Cross-layer overlap: 0%
- Each layer processes distinct information

---

## Next Steps (Research)

1. **Circuit Discovery**: Use VLOTester on full Atlas
2. **Validation**: Causal testing of discovered circuits
3. **Steering**: Multi-layer steering experiments
4. **Analysis**: Cross-layer feature evolution patterns
5. **Publication**: Document and release findings

---

## Dependencies

**Required**:
- torch >= 2.0
- transformers >= 4.30
- numpy, scipy
- scikit-learn

**Optional**:
- plotly (3D visualizations) ✅ Installed
- pyvis (circuit graphs) ❌ Not installed
- umap-learn (UMAP reduction) ❌ Not installed

---

## Quick Reference

**Load Atlas**:
```python
from neurotrace.control import EnhancedSAEFeatureStore
store = EnhancedSAEFeatureStore()
for i in range(12):
    store.load_sae(f'checkpoints/all_layers_sae/layer_{i}/final.pt', i)
```

**Run Analysis**:
```bash
python run_atlas_analysis.py
```

**View Results**:
- Report: `atlas_analysis_report.json`
- 3D viz: `visualizations/layer_features_pca_3d.html`

---

**Infrastructure Complete. Research Ready.** 🚀
