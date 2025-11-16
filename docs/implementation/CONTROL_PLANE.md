# 🧭 NeuroTrace Control Plane

**From Interpretability to Active Control**

## 🎯 Vision

The NeuroTrace Control Plane transforms discovered neural circuits into **controllable objects** that you can:

- ✅ **Activate / Attenuate**: Scale circuit influence with precise α parameters
- ✅ **Combine**: Compose multiple circuits for complex behaviors
- ✅ **Export**: Transfer circuits across models
- ✅ **Query**: Search and filter circuits by task, metrics, or semantics

This is the leap from **understanding** what a model does to **piloting** how it behaves.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROL PLANE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────┐ │
│  │ CircuitRegistry  │───│ SteeringBuilder  │───│ Controller │ │
│  │                  │   │                  │   │            │ │
│  │ • Persistence    │   │ • SAE→Vector     │   │ • Runtime  │ │
│  │ • Query API      │   │ • Aggregation    │   │ • Multi-   │ │
│  │ • Versioning     │   │ • Normalization  │   │   circuit  │ │
│  └──────────────────┘   └──────────────────┘   └────────────┘ │
│         ▲                       ▲                      ▲       │
└─────────┼───────────────────────┼──────────────────────┼───────┘
          │                       │                      │
          │                       │                      │
┌─────────┴───────────────────────┴──────────────────────┴───────┐
│                    CAUSAL DISCOVERY                             │
│  CriticalPathExtractor → Circuits with VLO, faithfulness       │
└─────────────────────────────────────────────────────────────────┘
          │                       │                      │
          │                       │                      │
┌─────────┴───────────────────────┴──────────────────────┴───────┐
│                 INTERPRETABILITY LAYER                          │
│  SAE Features + VectorDB + Geometric Analysis                  │
└─────────────────────────────────────────────────────────────────┘
          │                       │                      │
          │                       │                      │
┌─────────┴───────────────────────┴──────────────────────┴───────┐
│                   TARGET MODEL WRAPPER                          │
│  TargetModelWrapper + Hooks + Residual Stream                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Components

### 1. CircuitRegistry

**Persistent storage and query interface for discovered circuits.**

**Data Model:**

```python
@dataclass
class CircuitRecord:
    circuit_id: str                        # Unique identifier
    model_name: str                        # "gpt2", "llama-7b", etc.

    # Structural
    components: List[CircuitComponent]     # layer.head.index
    features: CircuitFeatures              # SAE indices, geometric props

    # Causal
    causal_metrics: CircuitCausalMetrics   # VLO, faithfulness, scrubbing

    # Semantic
    semantics: CircuitSemantics            # task_tag, human_label, examples
```

**API:**

```python
registry = CircuitRegistry(db_path="circuits.db")

# Insert/update
registry.upsert(circuit_record)

# Retrieve
circuit = registry.get("circuit_0037_ioi")

# Query
circuits = registry.list(
    task_tag="reasoning",
    min_vlo=1.0,
    min_faithfulness=0.8,
    limit=10
)

# Stream (for bulk operations)
for circuit in registry.iter_all():
    process(circuit)
```

**Storage:**
- **SQLite** for metadata + fast queries
- **JSON blob** for full circuit representation
- **WAL mode** for thread-safe concurrent access

---

### 2. SteeringBuilder

**Converts circuits into steering vectors for residual stream intervention.**

**Pipeline:**

```
CircuitRecord
    ↓
Extract SAE directions per layer (via FeatureStore)
    ↓
Aggregate multi-feature (mean / VLO-weighted / learned)
    ↓
Normalize (L2 norm = 1.0)
    ↓
Apply safety bounds
    ↓
SteeringSpec (ready for runtime application)
```

**API:**

```python
builder = SteeringBuilder(
    feature_store=sae_feature_store,
    default_alpha=0.7,
    alpha_bounds=(-2.0, 2.0),
    device=torch.device("cuda")
)

steering_spec = builder.build_from_circuit(
    record=circuit_record,
    per_layer_scaling={9: 1.5, 10: 0.8}  # optional per-layer weights
)

# steering_spec contains:
#   - layer_vectors: Dict[int, LayerSteeringVector]
#   - Each LayerSteeringVector has:
#       - direction: Tensor[hidden_dim] (normalized)
#       - default_alpha: float
#       - alpha_bounds: Tuple[float, float]
```

**FeatureStore Interface:**

```python
class FeatureStore(Protocol):
    def get_sae_directions(
        self,
        model_name: str,
        layer: int,
        feature_indices: List[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:  # [num_features, hidden_dim]
        ...
```

**Provided Implementations:**
- `SAEFeatureStore`: connects to `SAEFeatureExtractor` from Phase 1
- `MockFeatureStore`: for testing without trained SAE

---

### 3. CircuitController

**Runtime orchestration layer for active steering.**

**Workflow:**

```python
controller = CircuitController(
    model_wrapper=target_model_wrapper,  # TargetModelWrapper instance
    registry=circuit_registry,
    steering_builder=steering_builder,
    residual_position="post_mlp"  # where to inject steering
)

# 1. Discovery
circuits = controller.list_circuits(task_tag="ioi", min_vlo=1.5)

# 2. Activation
controller.enable_circuit(
    "circuit_0037_ioi",
    global_alpha=0.8,
    per_layer_scaling={9: 1.2}  # boost layer 9
)

# 3. Generation with active steering
output = controller.generate(
    prompt="John told Mary that Peter helped her because",
    max_new_tokens=64,
    temperature=0.0  # greedy for deterministic steering
)

# 4. Introspection
trace = controller.last_trace()
print(trace.active_circuits)      # ["circuit_0037_ioi"]
print(trace.layer_alphas)         # {circuit_id: {layer: alpha}}

summary = controller.active_circuits_summary()
# {
#   "count": 1,
#   "circuits": [{
#       "circuit_id": "circuit_0037_ioi",
#       "task_tag": "ioi",
#       "layers": [9, 10],
#       "alpha_per_layer": {9: 0.96, 10: 0.8}
#   }]
# }

# 5. Deactivation
controller.disable_circuit("circuit_0037_ioi")
controller.clear_all()  # disable all
```

**Multi-Circuit Composition:**

```python
# Enable multiple circuits
controller.enable_circuit("reasoning_cot", alpha=0.8)
controller.enable_circuit("factual_recall", alpha=0.6)
controller.disable_circuit("social_bias", alpha=1.0)  # suppression

# Hooks are composed sequentially (order of enable_circuit calls)
# Future: configurable composition modes (additive, max, learned)
```

---

## 🔌 Integration with Existing Pipeline

### Phase 1-7: Discovery
```
Input → Wrapper → Hooks → Compression → SAE → VectorDB
        → CausalImportance → CriticalPath → CircuitRecord
```

### Phase 8: Control (NEW)
```
CircuitRecord → CircuitRegistry (persist)
CircuitRecord → SteeringBuilder → SteeringSpec
SteeringSpec  → CircuitController (apply to model)
```

**Zero Breaking Changes** ✓
All existing functionality preserved. Control Plane is **additive**.

---

## 🚀 Usage Examples

### Example 1: Basic Steering

```python
from neurotrace.config import NeuroTraceConfig
from neurotrace.models.wrapper import TargetModelWrapper
from neurotrace.control import (
    CircuitRegistry,
    SteeringBuilder,
    CircuitController,
    SAEFeatureStore,
)
from neurotrace.state_indexer.sae_feature_extractor import SAEFeatureExtractor

# 1. Load model
cfg = NeuroTraceConfig(model_name_or_path="gpt2", device="cuda")
wrapper = TargetModelWrapper(cfg)

# 2. Setup Control Plane
registry = CircuitRegistry("circuits.db")
sae_extractor = SAEFeatureExtractor(...)  # from Phase 1
feature_store = SAEFeatureStore(sae_extractor)
builder = SteeringBuilder(feature_store)
controller = CircuitController(wrapper, registry, builder)

# 3. Use circuit
controller.enable_circuit("circuit_ioi_name_mover", alpha=0.7)
output = controller.generate("John and Mary went to the store. John gave a drink to")
print(output)  # Should complete with "Mary" (IOI behavior)
```

### Example 2: Circuit Discovery → Control

```python
# After running causal discovery...
from neurotrace.causal.critical_path_extractor import extract_critical_circuits

circuits = extract_critical_circuits(
    importance_results=importance_results,
    causal_test_results=causal_results,
    min_vlo=1.0,
    min_faithfulness=0.7
)

# Register discovered circuits
registry = CircuitRegistry("circuits.db")
for circuit in circuits:
    circuit_record = CircuitRecord(
        circuit_id=f"circuit_{circuit['id']}",
        model_name="gpt2",
        components=circuit['components'],
        features=circuit['features'],
        causal_metrics=circuit['metrics'],
        semantics=CircuitSemantics(
            task_tag=circuit['task'],
            human_label=circuit['label']
        )
    )
    registry.upsert(circuit_record)

# Now use immediately
controller.enable_circuit(f"circuit_{circuit['id']}", alpha=0.8)
```

### Example 3: CLI Usage

```bash
# List available circuits
python -c "
from neurotrace.control import CircuitRegistry
reg = CircuitRegistry('circuits.db')
for c in reg.list(task_tag='ioi'):
    print(f'{c.circuit_id}: VLO={c.causal_metrics.vlo_mean:.2f}')
"

# Run with steering
python cli/neuro_control_run.py \
    --model_name_or_path gpt2 \
    --registry_db circuits.db \
    --circuit_ids circuit_0037_ioi circuit_0042_counting \
    --alpha 0.7 \
    --prompt "John told Mary that Peter helped her because" \
    --max_new_tokens 64

# Output includes:
# - Generated text (stdout)
# - Control summary: active circuits, layers, alphas (stderr)
```

---

## 🧪 Testing

```bash
# Run full integration test suite
python test_control_plane.py

# Tests cover:
# 1. CircuitRegistry CRUD
# 2. SteeringBuilder vector construction
# 3. CircuitController integration with model
# 4. Multi-circuit composition
```

**Test results validate:**
- ✅ Registry persistence (SQLite)
- ✅ Steering vector normalization
- ✅ Hook injection on residual stream
- ✅ Multi-circuit active composition
- ✅ Control trace logging

---

## 🔬 Technical Details

### Residual Stream Hook Positions

**TargetModelWrapper** supports:

```python
wrapper.add_residual_hook(
    layer_idx=9,
    position="post_attn" | "post_mlp",
    hook_fn=lambda t: t + alpha * direction
)
```

**Architecture-specific:**
- **GPT-2**: `block.attn` (post_attn), `block` output (post_mlp)
- **LLaMA**: similar pattern (extensible via `_iter_transformer_blocks`)

**Hook composition:**
- Multiple hooks on same layer → **sequential application**
- Order: determined by `enable_circuit` call order
- Future: configurable composition strategies (additive, max, orthogonal projection)

### Steering Vector Construction

**Default: Mean aggregation**
```python
directions = feature_store.get_sae_directions(layer, indices)  # [N, D]
vec = directions.mean(dim=0)  # [D]
vec = vec / torch.norm(vec)   # normalize
```

**Advanced: VLO-weighted** (TODO)
```python
weights = get_vlo_per_feature(circuit, layer)  # [N]
vec = (directions * weights.unsqueeze(1)).sum(dim=0)  # weighted sum
vec = vec / torch.norm(vec)
```

**Learned: Meta-steering** (TODO)
```python
# Train small network to predict optimal steering from circuit features
meta_model = MetaSteeringNet(...)
vec = meta_model(circuit_embedding)
```

### Safety Bounds

**Calibration** (manual for now):
```python
alpha_bounds = (-2.0, 2.0)  # conservative default
```

**Auto-calibration** (TODO):
```python
def calibrate_bounds(circuit, test_prompts):
    # Binary search for max alpha that doesn't degrade perplexity >threshold
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ppl = measure_perplexity(test_prompts, steering_alpha=alpha)
        if ppl / baseline_ppl > 1.5:
            return (-alpha/2, alpha/2)
```

---

## 📊 Data Schema

### CircuitRegistry SQLite Schema

```sql
CREATE TABLE circuits (
    circuit_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    task_tag   TEXT NOT NULL,
    vlo_mean   REAL,
    faithfulness REAL,
    created_at TEXT NOT NULL,
    blob       TEXT NOT NULL  -- full JSON CircuitRecord
);

CREATE INDEX idx_task ON circuits(task_tag);
CREATE INDEX idx_vlo ON circuits(vlo_mean);
CREATE INDEX idx_faithfulness ON circuits(faithfulness);
```

**Query patterns:**
```sql
-- High-quality circuits for specific task
SELECT * FROM circuits
WHERE task_tag = 'ioi'
  AND vlo_mean > 1.5
  AND faithfulness > 0.8
ORDER BY vlo_mean DESC
LIMIT 10;

-- All circuits for a model
SELECT * FROM circuits
WHERE model_name = 'gpt2'
ORDER BY created_at DESC;
```

---

## 🛠️ Extension Points

### 1. Custom FeatureStore

```python
class CustomFeatureStore(FeatureStore):
    def get_sae_directions(self, model_name, layer, feature_indices, device):
        # Load from your custom SAE implementation
        # Return normalized directions [N, hidden_dim]
        pass
```

### 2. Custom Composition Modes

```python
class CircuitController:
    def set_composition_mode(self, mode: str):
        # "additive" | "max" | "orth_proj" | "learned"
        self._composition_mode = mode

    def _compose_steering_vectors(self, layer, circuits):
        if self._composition_mode == "additive":
            return sum(c.alpha * c.vec for c in circuits)
        elif self._composition_mode == "max":
            return max(circuits, key=lambda c: abs(c.alpha)).vec
        # ...
```

### 3. Circuit Transfer Learning

```python
def transfer_circuit(circuit_id: str, from_model: str, to_model: str):
    """
    Transfer circuit from one model to another via:
    1. Feature alignment (SAE space mapping)
    2. Layer correspondence heuristics
    3. Fine-tuning steering vectors
    """
    pass
```

---

## 🎯 Next Steps

### Immediate (Available Now)
1. ✅ Use `test_control_plane.py` to validate setup
2. ✅ Integrate with your SAE via `SAEFeatureStore`
3. ✅ Populate `CircuitRegistry` from causal discovery results
4. ✅ Experiment with steering on IOI task

### Short-term (Next Phase)
5. ⏳ Train SAE models for production quality
6. ⏳ Implement VLO-weighted aggregation in `SteeringBuilder`
7. ⏳ Add auto-calibration for safety bounds
8. ⏳ Build dashboard for circuit visualization

### Long-term (Research Frontier)
9. ⏳ Meta-learning for steering vector prediction
10. ⏳ Circuit composition algebra (operators on circuits)
11. ⏳ Cross-model circuit transfer
12. ⏳ Learned composition strategies

---

## 📚 Related Files

**Core Implementation:**
- [neurotrace/control/circuit_registry.py](neurotrace/control/circuit_registry.py)
- [neurotrace/control/steering_builder.py](neurotrace/control/steering_builder.py)
- [neurotrace/control/controller.py](neurotrace/control/controller.py)
- [neurotrace/control/sae_feature_store.py](neurotrace/control/sae_feature_store.py)

**Model Integration:**
- [neurotrace/models/wrapper.py](neurotrace/models/wrapper.py) (extended with `add_residual_hook`, `generate`)

**Testing:**
- [test_control_plane.py](test_control_plane.py)

**CLI:**
- [cli/neuro_control_run.py](cli/neuro_control_run.py)

---

## 💡 Why This Is Beyond BigTech

**Current SOTA (Anthropic, OpenAI, DeepMind):**
- SAE for interpretability ✓
- Circuit discovery (papers) ✓
- Ad-hoc steering (demos) ✓

**NeuroTrace Control Plane:**
- ✅ **Persistent circuit catalog** (anatomia computazionale)
- ✅ **Unified steering API** (not one-off hacks)
- ✅ **Multi-circuit composition** (circuit algebra)
- ✅ **Production-ready tooling** (CLI, tests, docs)
- ✅ **Extensible architecture** (Protocol-based, modular)

**This transforms MI from science → engineering.**

From "we understand this circuit" → "we can deploy it as a knob."

---

**Built with NeuroTrace Control Plane**
*Steering neural circuits like you steer a ship* 🚢

