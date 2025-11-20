# Quick Start Guide

## Setup
```bash
pip install -r requirements.txt
```

## Complete Workflow

### 1. Capture Data (if needed)
```bash
python capture_deep_dataset.py
# Output: runs/deep_ioi_capture/*/activations/
```

### 2. Train Atlas (DONE ✅)
```bash
python train_atlas_simple.py --layers all
# Output: checkpoints/all_layers_sae/ (12 layers, 36,864 features)
# Time: ~63 minutes
```

### 3. Validate
```bash
python validate_atlas.py
# Shows quality metrics for all 12 layers
```

### 4. Analyze
```bash
python run_atlas_analysis.py
# Outputs:
#   - atlas_analysis_report.json
#   - visualizations/layer_features_pca_3d.html
```

## Use Atlas in Code

```python
# Load all 12 SAEs
from neurotrace.control import EnhancedSAEFeatureStore

store = EnhancedSAEFeatureStore()
for layer in range(12):
    store.load_sae(f'checkpoints/all_layers_sae/layer_{layer}/final.pt', layer)

# Access features
sae_layer_0 = store.saes[0]
decoder_directions = sae_layer_0.decoder.weight  # [768, 3072]
```

## Key Files

- `train_atlas_simple.py` - Train SAEs
- `validate_atlas.py` - Validate quality
- `run_atlas_analysis.py` - Complete analysis
- `PROJECT_OVERVIEW.md` - Full documentation

## Atlas Status

✅ **Complete**: 12/12 layers trained
📊 **Features**: 36,864 total (3,072 per layer)
🎯 **Quality**: 8/12 excellent/good, 0% dead features
🚀 **Ready for research**
