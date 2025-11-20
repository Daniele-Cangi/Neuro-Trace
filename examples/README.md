# NeuroTrace Examples

Production-ready examples demonstrating NeuroTrace capabilities.

## Control Plane Steering Example

**File**: `control_plane_steering_example.py`

Demonstrates complete end-to-end active steering using Enhanced SAE features.

### What It Does

1. Loads GPT-2 model
2. Loads trained Enhanced SAE (Layer 0)
3. Creates circuit from discovered IOI features
4. Builds steering vectors from SAE decoder directions
5. Generates text with/without active steering

### Usage

```bash
python examples/control_plane_steering_example.py
```

### Requirements

- Trained SAE checkpoint at `checkpoints/layer0_sae/final.pt`
- Run `python train_layer0_sae.py` first if checkpoint doesn't exist

### Expected Output

Generates 3 IOI prompts with baseline and steered completions, demonstrating:
- Baseline behavior (without steering)
- Modified behavior (with active circuit steering at alpha=1.0)

### Key Components Demonstrated

- `EnhancedSAEFeatureStore`: Loads trained SAE checkpoints
- `CircuitRegistry`: Manages discovered circuits
- `SteeringBuilder`: Constructs steering vectors from SAE features
- `CircuitController`: Applies real-time steering during generation

### Example Output

```
Prompt: "When John and Mary went to the store, John gave a book to"

Baseline (no steering):
  "Mary and Mary said..."

With steering (alpha=1.0):
  "Mary. She said..."
```

The steering improves pronoun usage by amplifying structural IOI features.
