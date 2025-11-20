# NeuroTrace Codebase Inventory

**Total Python Files**: 62
**Last Updated**: 2025-11-17
**Status**: Atlas Training in Progress (1/12 layers complete)

---

## 📁 Core Library (`neurotrace/`)

### Configuration
- **`neurotrace/__init__.py`** - Package initialization, version info
- **`neurotrace/config.py`** - Global configuration settings

---

## 🎯 1. Models & Wrappers

### Main Model Interface
- **`neurotrace/models/__init__.py`** - Models package init
- **`neurotrace/models/wrapper.py`** - Model wrapper for instrumentation (hooks GPT-2)

---

## 🔌 2. Instrumentation (Network Interception)

### Hook Management & Activation Capture
- **`neurotrace/instrumentation/__init__.py`** - Instrumentation package init
- **`neurotrace/instrumentation/adaptive_hook_manager.py`** - Dynamic hook management system
- **`neurotrace/instrumentation/adaptive_activations_buffer.py`** - Efficient activation buffering

**Purpose**: 1:1 deterministic capture of neural network activations

---

## 🧠 3. Training (SAE Training)

### SAE Architecture & Training
- **`neurotrace/training/__init__.py`** - Training package init, exports main classes
- **`neurotrace/training/enhanced_sae.py`** - Enhanced SAE model (SOTA: k-sparse, decoder norm, ghost grads)
- **`neurotrace/training/enhanced_sae_trainer.py`** - Trainer for Enhanced SAE (10 epochs → MSE=0.0118)
- **`neurotrace/training/sae_trainer.py`** - Legacy SAE trainer (deprecated?)
- **`neurotrace/training/activation_dataset.py`** - Dataset loader for activation files
- **`neurotrace/training/sae_checkpoint.py`** - Checkpoint save/load utilities

**Scripts**:
- **`train_enhanced_sae.py`** - Train Enhanced SAE (generic)
- **`train_layer0_sae.py`** - Train SAE specifically for Layer 0 (100K IOI examples)
- **`train_all_layers_sae.py`** - Train SAE for all 12 layers (Atlas completion) ⚡ CURRENTLY RUNNING
- **`monitor_training.py`** - Monitor training progress in real-time

**Status**: Layer 0 complete (3,072 features, MSE=0.0118). Layers 1-11 in progress.

---

## 🗺️ 4. State Indexing (Feature Database)

### Vector Search & Feature Storage
- **`neurotrace/state_indexer/__init__.py`** - State indexer package init
- **`neurotrace/state_indexer/sae_feature_extractor.py`** - Extract SAE features from activations
- **`neurotrace/state_indexer/vector_state_db.py`** - Vector database for feature similarity search

**Purpose**: Fast retrieval of similar activation patterns using FAISS

---

## 🎮 5. Control Plane (Active Steering)

### Circuit-Based Model Steering
- **`neurotrace/control/__init__.py`** - Control package init
- **`neurotrace/control/circuit_registry.py`** - SQLite database for discovered circuits
- **`neurotrace/control/controller.py`** - Main controller for runtime steering
- **`neurotrace/control/steering_builder.py`** - Builds steering vectors from circuits
- **`neurotrace/control/sae_feature_store.py`** - Generic SAE feature store interface
- **`neurotrace/control/enhanced_sae_feature_store.py`** - Adapter for Enhanced SAE integration

**Examples**:
- **`examples/control_plane_steering_example.py`** - Complete end-to-end steering demo

**Tests**:
- **`test_control_plane.py`** - Control plane unit tests

**Status**: ✅ PRODUCTION-READY (100% steering effectiveness on IOI task)

---

## 🔍 6. Causal Discovery (Circuit Finding)

### Circuit Extraction & Validation
- **`neurotrace/causal/__init__.py`** - Causal package init
- **`neurotrace/causal/circuit_extractor.py`** - Extract minimal causal circuits from activations
- **`neurotrace/causal/vlo_tester.py`** - VLO (Value of Learned Organization) metric calculator

**Tests**:
- **`test_causal_discovery.py`** - Causal discovery validation

**Purpose**: Identify minimal sets of features causally responsible for model behavior

---

## 📊 7. Analysis & Metrics

### Geometric & Statistical Analysis
- **`neurotrace/analysis/__init__.py`** - Analysis package init
- **`neurotrace/analysis/geometric.py`** - Geometric analysis of activation spaces (PCA, cosine similarity)

---

## 🎨 8. Visualization

### Charts, Graphs, 3D Visualization
- **`neurotrace/visualization/__init__.py`** - Visualization package init
- **`neurotrace/visualization/circuit_graph.py`** - Circuit graph visualization (NetworkX)
- **`neurotrace/visualization/metrics_plotter.py`** - Training metrics plotting
- **`neurotrace/visualization/sae_feature_viz.py`** - SAE feature activation heatmaps
- **`neurotrace/visualization/activation_explorer.py`** - Interactive activation explorer

**Tests**:
- **`test_visualization.py`** - Visualization tests

**Status**: ⚠️ Basic implementation, needs 3D upgrade for multi-layer Atlas

---

## 🔬 9. Discovery System

### Automated Circuit Search
- **`neurotrace/discovery/__init__.py`** - Discovery package init
- **`neurotrace/discovery/exhaustive_scanner.py`** - Exhaustive circuit search across layers
- **`neurotrace/discovery/component_interaction_matrix.py`** - Component interaction analysis

**Scripts**:
- **`run_discovery.py`** - Run circuit discovery on captured activations
- **`compare_discovery_runs.py`** - Compare multiple discovery runs

**Purpose**: Automated discovery of circuits across all model components

---

## 📚 10. Datasets

### Task Generation
- **`neurotrace/datasets/__init__.py`** - Datasets package init
- **`neurotrace/datasets/ioi_generator.py`** - IOI (Indirect Object Identification) task generator
- **`neurotrace/datasets/task_generator.py`** - Generic task generator interface

**Scripts**:
- **`capture_ioi_activations.py`** - Capture activations for IOI task
- **`capture_deep_dataset.py`** - Deep capture with 100K examples (all 12 layers) ✅ DONE

**Status**: 100K IOI examples captured with 1:1 activations for all 12 layers

---

## 🧪 11. Tests & Validation

### System Tests
- **`tests/validation/test_system_diagnostic.py`** - Comprehensive 5-phase system diagnostic
  - ✅ Activation Capture Fidelity (100% - max_diff=0.0)
  - ✅ SAE Reconstruction Quality (100% - MSE=0.0118)
  - ✅ Steering Causality (100% - all prompts affected)
  - ✅ JSON Analysis (100%)
  - ⚠️ Atlas Coverage (8.3% - 1/12 layers, training in progress)

### Legacy Tests
- **`test_neurotrace_pipeline.py`** - Pipeline integration test
- **`test_sae_training.py`** - SAE training test

---

## 🖥️ 12. CLI Tools

### Command-Line Interfaces
- **`cli/train_sae.py`** - CLI for SAE training
- **`cli/neuro_control_run.py`** - CLI for running control plane
- **`cli/run_phase1_capture.py`** - CLI for Phase 1 activation capture

---

## 🔄 13. Analysis Scripts (Root Level)

### Validation & Comparison
- **`complete_validation_analysis.py`** - Complete validation pipeline
- **`run_discovery_validation.py`** - Discovery system validation
- **`phase2_verify.py`** - Phase 2 verification script
- **`hybrid_sae_analysis.py`** - SAE comparison (Enhanced vs SAELens baseline)

---

## ⚙️ 14. Setup & Configuration

- **`setup.py`** - Package installation setup
- **`setup_saelens.py`** - SAELens baseline installation

---

## 📂 File Organization Summary

```
Total: 62 Python files

neurotrace/ (Core Library)          → 31 files
├── models/                         → 2 files
├── instrumentation/                → 3 files
├── training/                       → 6 files (SAE architecture)
├── state_indexer/                  → 3 files
├── control/                        → 6 files (PRODUCTION-READY)
├── causal/                         → 3 files
├── analysis/                       → 2 files
├── visualization/                  → 5 files
├── discovery/                      → 3 files
└── datasets/                       → 3 files

cli/                                → 3 files (CLI tools)
examples/                           → 1 file (control plane demo)
tests/                              → 1 file (system diagnostic)

Root Scripts                        → 17 files
├── Training scripts                → 4 files
├── Capture scripts                 → 2 files
├── Analysis scripts                → 5 files
├── Validation scripts              → 3 files
├── Test scripts                    → 3 files
└── Setup                           → 2 files
```

---

## 🎯 Current Priority Files (Active Development)

### Training (Atlas Completion)
1. **`train_all_layers_sae.py`** ⚡ RUNNING NOW
2. **`neurotrace/training/enhanced_sae.py`** - Model architecture
3. **`neurotrace/training/enhanced_sae_trainer.py`** - Training loop

### Validation
4. **`tests/validation/test_system_diagnostic.py`** - System health checks

### Control Plane (Production)
5. **`neurotrace/control/controller.py`** - Main steering controller
6. **`neurotrace/control/enhanced_sae_feature_store.py`** - SAE integration
7. **`examples/control_plane_steering_example.py`** - Working demo

---

## 🗑️ Candidate Files for Cleanup/Deprecation

### Potentially Obsolete
- **`neurotrace/training/sae_trainer.py`** - Replaced by `enhanced_sae_trainer.py`?
- **`test_neurotrace_pipeline.py`** - Legacy test?
- **`test_sae_training.py`** - Covered by system diagnostic?
- **`cli/train_sae.py`** - Replaced by root training scripts?

### Scripts that may be one-off
- **`setup_saelens.py`** - One-time setup (can archive)
- **`phase2_verify.py`** - Phase 2 specific (archive after completion?)

**ACTION NEEDED**: Review and either archive or document purpose

---

## 📈 Next Development Priorities

1. **Complete Atlas Training** (layers 1-11) - IN PROGRESS
2. **Build JSON Aggregator** - Systematic analysis of all results
3. **Web Interface** - FastAPI + React for Atlas Explorer
4. **3D Visualization** - Multi-layer feature flow visualization
5. **Cross-Layer Analysis** - Feature evolution tracking
6. **Feature Labeling System** - Automated + manual annotation

---

## 🚨 Critical Notes

- **DO NOT DELETE** activation files in `runs/deep_ioi_capture/20251116_171258/` (100K examples, 2GB)
- **Layer 0 SAE** checkpoint at `checkpoints/layer0_sae/final.pt` is GOLD (MSE=0.0118)
- **Atlas training** will create 11 more checkpoints (layers 1-11) → ~1.5GB total
- **Git LFS** required for model checkpoints and large datasets

---

*This inventory is auto-generated and manually curated. Update when adding new files or discovering deprecated code.*
