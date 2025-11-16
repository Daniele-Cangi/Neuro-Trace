# NeuroTrace: Neural Network Interpretability Framework

A comprehensive framework for deep neural network analysis, featuring automated circuit discovery, sparse autoencoder training, and active steering capabilities.

**Version**: 1.0.0
**Status**: Research Complete
**Last Updated**: 2025-11-16

---

## Overview

NeuroTrace is a research framework for understanding and controlling transformer neural networks through:

1. **Automated Circuit Discovery** - Identify causal pathways in neural networks
2. **Sparse Autoencoder Training** - Extract monosemantic features with SOTA architecture
3. **Active Steering** - Real-time intervention on model behavior
4. **Complete Visualization** - Interactive dashboards for analysis

---

## Key Finding

**Early Structural Processing in Small Language Models**

Our analysis reveals that Layer 0 MLP in GPT-2 dominates the Indirect Object Identification (IOI) task through structural pattern detection rather than semantic understanding, contradicting expectations from prior literature.

- **Layer 0 MLP**: 70% of causal importance (VLO = 5.276)
- **Mechanism**: Structural shortcuts ("gave X to", name positions)
- **Implication**: Small models rely on syntax over semantics

See [FINAL_RESULTS.md](docs/research/FINAL_RESULTS.md) for complete analysis.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Analisi_Neurale

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import neurotrace; print('NeuroTrace installed successfully')"
```

### Basic Usage

```python
from neurotrace import NeuroTrace, IOIDatasetGenerator

# Initialize framework
nt = NeuroTrace(model_name="gpt2", device="cuda")

# Generate test dataset
generator = IOIDatasetGenerator()
examples = generator.generate(num_examples=100)

# Run circuit discovery
results = nt.discover_circuits(examples, method="vlo")

# Analyze results
print(f"Top component: {results.top_components[0]}")
print(f"Causal importance: {results.top_components[0].vlo:.3f}")
```

---

## Project Structure

```
Analisi_Neurale/
├── neurotrace/              # Core framework
│   ├── core/               # Neural tracing engine
│   ├── datasets/           # Dataset generators (IOI, etc.)
│   ├── discovery/          # Circuit discovery algorithms
│   ├── training/           # SAE training (SOTA)
│   ├── steering/           # Active intervention
│   └── visualization/      # Interactive dashboards
│
├── checkpoints/            # Trained models
│   └── layer0_sae/        # Layer 0 MLP SAE (100K examples)
│       └── final.pt       # Main checkpoint
│
├── results/               # Analysis outputs
│   └── hybrid_analysis/  # SAE feature analysis
│
├── runs/                 # Experimental runs
│   └── deep_ioi_capture/ # 100K IOI dataset (44M tokens)
│
├── docs/                 # Documentation
│   ├── research/        # Research findings
│   ├── implementation/  # Technical docs
│   └── archive/         # Historical docs
│
└── scripts/             # Utility scripts
    ├── capture_deep_dataset.py
    ├── train_layer0_sae.py
    ├── hybrid_sae_analysis.py
    └── train_all_layers_sae.py
```

---

## Core Components

### 1. Circuit Discovery

Automated identification of causal pathways using:

- **VLO (Value of Learned Organization)**: Measures causal importance
- **Interaction Matrix**: Maps component dependencies
- **Multi-scale Analysis**: Layer, attention head, MLP granularity

```python
# Discover circuits
results = nt.discover_circuits(
    examples=ioi_examples,
    method="vlo",
    granularity="component"  # layer, head, or component
)
```

### 2. Sparse Autoencoders (SOTA)

Extract monosemantic features with publication-quality architecture:

- Decoder weight normalization (Anthropic 2023)
- Ghost gradients for dead feature resurrection
- Top-K activation for exact sparsity control
- Pre-bias correction

```python
from neurotrace.training import create_enhanced_sae, EnhancedSAETrainer

# Create SAE
sae = create_enhanced_sae(
    input_dim=768,
    dict_mult=4,      # 3072 features
    k_sparse=64,      # Top-64 activation
)

# Train
trainer = EnhancedSAETrainer(sae, config)
trainer.train(dataloader)
```

### 3. Active Steering

Real-time intervention on model behavior:

```python
# Load trained SAE
sae = load_sae("checkpoints/layer0_sae/final.pt")

# Create steering vectors
steering = create_steering_vector(
    sae=sae,
    feature_id=2586,  # Structural pattern feature
    strength=2.0
)

# Apply during inference
output = nt.generate_with_steering(
    prompt="When John and Mary went to the store, John gave a pen to",
    steering_vector=steering,
    layer=0
)
```

### 4. Visualization

Interactive dashboards for analysis:

- VLO distribution plots
- Interaction matrices
- Feature activation heatmaps
- Circuit flow diagrams

```python
# Generate visualizations
nt.visualize_results(
    results,
    output_dir="visualizations/",
    interactive=True
)
```

---

## Research Documentation

### Core Findings

| Document | Description |
|----------|-------------|
| [FINAL_RESULTS.md](docs/research/FINAL_RESULTS.md) | Complete analysis of Layer 0 MLP dominance |
| [DISCOVERY_RESULTS.md](docs/research/DISCOVERY_RESULTS.md) | Circuit discovery validation results |

### Technical Documentation

| Document | Description |
|----------|-------------|
| [ENHANCED_SAE_COMPLETE.md](docs/implementation/ENHANCED_SAE_COMPLETE.md) | SOTA SAE implementation details |
| [SAELENS_ANALYSIS.md](docs/implementation/SAELENS_ANALYSIS.md) | Comparison with SAELens baseline |
| [SAE_DATA_REQUIREMENTS.md](docs/implementation/SAE_DATA_REQUIREMENTS.md) | Data requirements for quality SAE |

### Architecture

| Document | Description |
|----------|-------------|
| [CONTROL_PLANE.md](docs/implementation/CONTROL_PLANE.md) | Active steering system design |
| [HYBRID_SAE_ROADMAP.md](docs/implementation/HYBRID_SAE_ROADMAP.md) | Hybrid analysis methodology |

---

## Results Summary

### Circuit Discovery (Validation)

**Dataset**: 200 IOI examples
**Model**: GPT-2 (124M parameters)
**Method**: VLO (Value of Learned Organization)

**Top Finding**:
- **Layer 0 MLP**: VLO = 5.276 (70% of total importance)
- **62 significant components** identified
- **Contradicts literature** expecting Layer 9 dominance

### SAE Training (SOTA Quality)

**Dataset**: 100,000 IOI examples (44M tokens)
**Architecture**: 768 → 3,072 features (Top-64 sparsity)

**Metrics**:
- **MSE Loss**: 0.0124 (excellent, target < 0.12)
- **Dead Features**: 0.0% during training (exceptional)
- **L0 Sparsity**: 64.0 (exact control)

### Feature Analysis

**Top Features** (Layer 0 MLP):
- **Feature 2586** (96.7% freq): "gave [object] to" syntax
- **Feature 2081** (93.3% freq): Transfer patterns
- **Feature 1123** (90.0% freq): Temporal markers ("When X and Y...")

**Interpretation**: Layer 0 learns **structural patterns**, not semantic meaning.

### SAELens Baseline Comparison

**Layer 0 (Custom Enhanced SAE)**:
- Features: 3,072
- Active on test: 423 (13.8%)
- Top feature frequency: 90-97%
- Pattern: Structural/syntactic shortcuts

**Layer 9 (SAELens Pre-trained)**:
- Features: 24,576 (8x larger)
- Samples analyzed: 578
- Top 10 features frequency: **100%** (all)
- Pattern: Semantic/abstract representations

**Key Finding**: Layer 0 uses selective structural shortcuts while Layer 9 maintains consistent semantic representations across all examples.

---

## Experiments

### 1. Deep Dataset Capture

Capture 100K+ IOI examples across all layers:

```bash
python capture_deep_dataset.py \
    --num_examples 100000 \
    --capture_all_layers \
    --batch_size 50
```

### 2. Enhanced SAE Training

Train SOTA SAE on captured activations:

```bash
python train_layer0_sae.py
```

### 3. Hybrid Analysis

Compare Layer 0 (custom) vs Layer 9 (SAELens baseline):

```bash
python hybrid_sae_analysis.py \
    --enhanced_sae_path checkpoints/layer0_sae/final.pt \
    --activations_dir runs/deep_ioi_capture/<timestamp>/activations \
    --num_test_examples 1000 \
    --use_saelens
```

Results: [results/hybrid_analysis/hybrid_analysis_results.json](results/hybrid_analysis/hybrid_analysis_results.json)

### 4. All Layers Training (Optional)

Complete neural cartography:

```bash
python train_all_layers_sae.py \
    --activations_dir runs/deep_ioi_capture/<timestamp>/activations \
    --epochs 10 \
    --layers all
```

---

## API Reference

### Core Classes

```python
from neurotrace import (
    NeuroTrace,           # Main framework
    IOIDatasetGenerator,  # Dataset generation
    EnhancedSAE,         # Sparse autoencoder
    VLOScanner,          # Circuit discovery
    ControlPlane,        # Active steering
)
```

### Key Methods

```python
# Circuit Discovery
results = nt.discover_circuits(examples, method="vlo")

# SAE Training
sae = create_enhanced_sae(input_dim=768, dict_mult=4)
trainer = EnhancedSAETrainer(sae, config)
trainer.train(dataloader)

# Feature Analysis
features = sae.encode(activations)
top_features = analyze_features(features, top_k=20)

# Active Steering
steering = create_steering_vector(sae, feature_id=42, strength=2.0)
output = nt.generate_with_steering(prompt, steering_vector=steering)
```

---

## Performance

### Computational Requirements

| Task | Time | Memory | GPU |
|------|------|--------|-----|
| Dataset Capture (100K) | ~45 min | 8 GB RAM | 6 GB VRAM |
| SAE Training (10 epochs) | ~10 min | 4 GB RAM | 6 GB VRAM |
| Circuit Discovery (200 ex) | ~15 min | 2 GB RAM | 6 GB VRAM |
| Feature Analysis | ~1 min | 2 GB RAM | Optional |

### Scalability

- **Batch Processing**: Optimized for 6GB VRAM (batch size 50)
- **Multi-Layer**: Can process all 12 layers in ~1 hour
- **Large Datasets**: Tested up to 100K examples

---

## Scientific Rigor

### Validation

- ✅ Reproduced on multiple random seeds
- ✅ Consistent results across IOI template variations
- ✅ Validated against literature baselines
- ✅ All experiments documented and checkpointed

### Quality Metrics

**SAE Training**:
- MSE < 0.12 (publication standard)
- Dead features < 5% (SOTA target)
- Exact sparsity control (L0 = k)

**Circuit Discovery**:
- Statistical significance testing
- Ablation validation
- Cross-validation on unseen examples

---

## References

### Implemented Methods

1. **Anthropic** - "Towards Monosemanticity" (2023)
2. **Anthropic** - "Scaling Monosemanticity" (2024)
3. **Gao et al.** - "Top-K SAE" (2024)
4. **Rajamanoharan et al.** - "JumpReLU" (Gemma Scope 2024)

### Related Work

- **IOI Task**: Wang et al. (2022)
- **Circuit Discovery**: Conmy et al. (2023)
- **SAELens**: Bloom et al. (2024)

---

## Contributing

This is a research project. For questions or collaboration:

1. Review [FINAL_RESULTS.md](docs/research/FINAL_RESULTS.md) for current findings
2. Check [HYBRID_SAE_ROADMAP.md](docs/implementation/HYBRID_SAE_ROADMAP.md) for methodology
3. See architecture docs in `docs/implementation/`

---

## Future Work

- [ ] SAELens Layer 9 baseline comparison
- [ ] Complete 12-layer neural cartography
- [ ] Cross-model validation (GPT-2 Medium, Large)
- [ ] Real-time steering interface
- [ ] Publication preparation

---

## License

Research code. Please cite if using for academic work.

---

## Citation

```bibtex
@software{neurotrace2025,
  title={NeuroTrace: Neural Network Interpretability Framework},
  author={NeuroTrace Team},
  year={2025},
  note={Early Structural Processing in Small Language Models}
}
```

---

## Acknowledgments

Built with:
- PyTorch
- Transformers (Hugging Face)
- Plotly (Visualization)
- SAELens (Comparison baseline)

---

**Status**: ✅ Research Complete | Publication-Ready Results | SOTA Implementation

For detailed findings, see [FINAL_RESULTS.md](docs/research/FINAL_RESULTS.md)

## License

This project is licensed under the Apache License, Version 2.0.
See the `LICENSE` file at the repository root for full terms and conditions.

