# NeuroTrace Production Roadmap

**Status**: Prototype → Production-Grade System
**Goal**: Transform research prototype into a complete neural network control and visualization platform

---

## Current State Assessment

### What We Have (Proof of Concept)
- ✅ Enhanced SAE training (SOTA quality)
- ✅ Circuit discovery via VLO
- ✅ Basic steering mechanism (works but fragile)
- ✅ SAELens baseline comparison
- ✅ SQLite registry (in-memory only)

### Critical Gaps
- ❌ No robust error handling
- ❌ No logging/monitoring
- ❌ No persistent storage (database is volatile!)
- ❌ No interactive interface
- ❌ No real-time visualization
- ❌ No 3D network graphs (planned but never implemented)
- ❌ No comprehensive testing
- ❌ No performance optimization
- ❌ Fragile API (too many manual parameters)

---

## Production Requirements (Sequential Implementation)

### Phase 1: Foundation - Robustness & Observability
**Priority**: CRITICAL
**Timeline**: Week 1-2

#### 1.1 Error Handling & Logging
**File**: `neurotrace/control/logging_config.py` (new)

- [ ] Structured logging system (JSON logs)
- [ ] Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- [ ] Context managers for operation tracking
- [ ] Automatic error reporting with stack traces
- [ ] Performance metrics logging (latency, memory)

**Implementation**:
```python
import logging
import structlog
from contextlib import contextmanager

# Structured logger with context
logger = structlog.get_logger()

@contextmanager
def operation_context(operation_name: str, **metadata):
    """Track operations with automatic error handling."""
    logger.info(f"{operation_name}.start", **metadata)
    start_time = time.time()
    try:
        yield
        logger.info(f"{operation_name}.success", duration=time.time()-start_time)
    except Exception as e:
        logger.error(f"{operation_name}.failed", error=str(e), duration=time.time()-start_time)
        raise
```

**Apply to**:
- [ ] `EnhancedSAEFeatureStore.load_sae()` - log checkpoint loading
- [ ] `CircuitController.enable_circuit()` - log steering activation
- [ ] `SteeringBuilder.build_from_circuit()` - log vector construction
- [ ] All database operations in `CircuitRegistry`

#### 1.2 Persistent Database with Versioning
**File**: `neurotrace/control/circuit_registry.py` (upgrade)

- [ ] Replace `:memory:` with file-based SQLite (`circuits.db`)
- [ ] Add versioning table for circuit history
- [ ] Add migration system for schema changes
- [ ] Add backup/restore functionality
- [ ] Add circuit changelog (who modified what, when)

**Schema Changes**:
```sql
-- New tables
CREATE TABLE circuit_versions (
    version_id INTEGER PRIMARY KEY,
    circuit_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    changelog TEXT,
    blob BLOB NOT NULL,
    FOREIGN KEY (circuit_id) REFERENCES circuits(circuit_id)
);

CREATE TABLE circuit_metadata (
    circuit_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    last_modified TIMESTAMP,
    modification_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    usage_count INTEGER DEFAULT 0
);
```

#### 1.3 Input Validation Layer
**File**: `neurotrace/control/validators.py` (new)

- [ ] Validate feature indices against dict_size
- [ ] Validate layer compatibility with model
- [ ] Validate SAE checkpoint integrity before loading
- [ ] Validate steering vector dimensions
- [ ] Validate circuit components structure

**Implementation**:
```python
from pydantic import BaseModel, validator

class CircuitValidator(BaseModel):
    circuit_id: str
    model_name: str
    components: List[CircuitComponent]

    @validator('components')
    def validate_components(cls, components, values):
        """Ensure all components are valid for the model."""
        # Check layer indices, feature indices, etc.
        pass
```

---

### Phase 2: User Interface & Interaction
**Priority**: HIGH
**Timeline**: Week 3-4

#### 2.1 Web-Based Control Interface
**Directory**: `neurotrace/interface/` (new)

**Stack**: FastAPI (backend) + React/Plotly Dash (frontend)

**Features**:
- [ ] Live model explorer (browse layers, neurons, features)
- [ ] Circuit builder (drag-and-drop components)
- [ ] Steering controller (sliders for alpha values)
- [ ] Real-time generation with steering toggle
- [ ] Circuit registry browser (search, filter, activate)
- [ ] SAE feature inspector (view top activating examples)

**Files**:
```
neurotrace/interface/
├── api.py              # FastAPI backend
├── routes/
│   ├── circuits.py     # Circuit CRUD
│   ├── steering.py     # Steering control
│   ├── generation.py   # Text generation
│   └── visualization.py # Graph data
├── frontend/
│   ├── App.jsx         # React main app
│   ├── CircuitBuilder.jsx
│   ├── SteeringPanel.jsx
│   └── NetworkGraph.jsx
└── config.yaml         # Interface config
```

**API Endpoints**:
```python
# FastAPI routes
@app.get("/api/circuits")           # List all circuits
@app.post("/api/circuits/create")   # Create new circuit
@app.put("/api/circuits/{id}/enable")  # Enable steering
@app.post("/api/generate")          # Generate with active steering
@app.get("/api/features/{layer}/{idx}")  # Get feature info
```

#### 2.2 Interactive Network Visualization
**File**: `neurotrace/visualization/network_3d.py` (new)

**Technology**: Plotly 3D / Three.js

**Features**:
- [ ] 3D network graph (nodes = neurons, edges = connections)
- [ ] Layer-by-layer visualization
- [ ] Highlight active circuits
- [ ] Show steering vector overlay
- [ ] Animate forward pass with activations
- [ ] Export to interactive HTML

**Implementation**:
```python
import plotly.graph_objects as go
import networkx as nx

class NetworkVisualizer3D:
    def __init__(self, model_wrapper):
        self.model = model_wrapper
        self.graph = self._build_network_graph()

    def visualize_circuit(self, circuit: CircuitRecord) -> go.Figure:
        """Generate 3D visualization of circuit."""
        # Build 3D node positions (layer-based layout)
        # Highlight circuit components
        # Show steering directions as arrows
        pass

    def animate_forward_pass(self, prompt: str, show_steering: bool = False):
        """Animate activations flowing through network."""
        # Capture activations at each layer
        # Generate frame-by-frame animation
        pass
```

#### 2.3 Neural Atlas Interface
**File**: `neurotrace/atlas/explorer.py` (new)

**Purpose**: Interactive exploration of the complete neural cartography

**Features**:
- [ ] Load all 12 layers of trained SAEs
- [ ] Browse features by layer
- [ ] Search features by semantic label
- [ ] Visualize feature activation patterns
- [ ] Show feature co-occurrence matrices
- [ ] Export feature clusters

**Implementation**:
```python
class NeuralAtlas:
    """Complete map of neural network features across all layers."""

    def __init__(self, checkpoint_dir: Path):
        self.saes = {}  # layer -> EnhancedSAE
        self.feature_labels = {}  # (layer, idx) -> label
        self.activation_stats = {}  # (layer, idx) -> stats

    def load_all_layers(self):
        """Load SAEs for all 12 layers."""
        pass

    def search_features(self, query: str) -> List[Tuple[int, int]]:
        """Search features by semantic label."""
        pass

    def get_feature_neighbors(self, layer: int, feature_idx: int, k: int = 10):
        """Find k most similar features (by activation pattern)."""
        pass
```

---

### Phase 3: Performance & Scalability
**Priority**: MEDIUM
**Timeline**: Week 5

#### 3.1 SAE Direction Caching
**File**: `neurotrace/control/enhanced_sae_feature_store.py` (upgrade)

- [ ] Cache decoder directions after first access
- [ ] LRU cache for frequently used features
- [ ] Batch direction extraction (all features at once)
- [ ] Lazy loading (only load SAE when needed)

**Implementation**:
```python
from functools import lru_cache
import hashlib

class EnhancedSAEFeatureStore:
    def __init__(self, cache_size: int = 1000):
        self.direction_cache = {}  # (layer, feature_idx) -> direction
        self._cache_hits = 0
        self._cache_misses = 0

    @lru_cache(maxsize=1000)
    def get_sae_directions_cached(self, layer: int, feature_tuple: tuple):
        """Cached version for repeated queries."""
        return self.get_sae_directions(layer=layer, feature_indices=list(feature_tuple))
```

#### 3.2 Batch Generation
**File**: `neurotrace/control/controller.py` (upgrade)

- [ ] Support batch text generation (multiple prompts)
- [ ] Parallel steering application
- [ ] GPU optimization (minimize transfers)

#### 3.3 Memory Optimization
- [ ] Streaming SAE loading (don't keep all in memory)
- [ ] Gradient checkpointing for large models
- [ ] Mixed precision support (fp16)

---

### Phase 4: Testing & Validation
**Priority**: HIGH
**Timeline**: Week 6

#### 4.1 Unit Tests
**Directory**: `tests/unit/`

```
tests/unit/
├── test_enhanced_sae_feature_store.py
├── test_circuit_registry.py
├── test_steering_builder.py
├── test_controller.py
└── test_validators.py
```

**Coverage Target**: >80%

**Key Tests**:
- [ ] SAE checkpoint loading (valid/invalid)
- [ ] Direction extraction (correctness, shape)
- [ ] Circuit CRUD operations
- [ ] Database versioning
- [ ] Steering vector construction
- [ ] Error handling paths

#### 4.2 Integration Tests
**Directory**: `tests/integration/`

**Tests**:
- [ ] End-to-end steering pipeline
- [ ] Multi-layer circuit activation
- [ ] Concurrent circuit management
- [ ] Database migration
- [ ] API endpoint testing

#### 4.3 Validation Suite
**File**: `tests/validation/test_neural_fidelity.py` (new)

**Critical Question**: Are we intercepting the network 1:1?

**Tests**:
```python
def test_activation_reconstruction_fidelity():
    """Verify SAE reconstructs activations accurately."""
    # Load model and SAE
    # Run forward pass, capture true activations
    # Reconstruct via SAE encode/decode
    # Assert MSE < threshold (e.g., 0.05)
    pass

def test_steering_causality():
    """Verify steering actually changes model behavior."""
    # Generate baseline output
    # Apply steering
    # Generate steered output
    # Assert outputs differ significantly
    # Measure effect size
    pass

def test_circuit_isolation():
    """Verify circuits are truly independent."""
    # Enable circuit A
    # Enable circuit B
    # Verify no interference
    pass

def test_layer_by_layer_fidelity():
    """Test reconstruction quality for all 12 layers."""
    for layer in range(12):
        # Load layer SAE
        # Test reconstruction MSE
        # Assert quality degrades gracefully with depth
        pass
```

---

### Phase 5: Neural Atlas Completion
**Priority**: MEDIUM
**Timeline**: Week 7-8

#### 5.1 All-Layer SAE Training
**Status**: Planned but never executed

**Script**: `scripts/train_all_layers_sae.py` (exists but untested)

**Tasks**:
- [ ] Train SAE for layers 1-11 (Layer 0 done)
- [ ] Validate quality metrics for each layer
- [ ] Document per-layer findings
- [ ] Compare structural vs semantic features across depth

#### 5.2 Feature Labeling
**File**: `neurotrace/atlas/feature_labeler.py` (new)

**Methods**:
- Automated: Top activating examples → GPT-4 labeling
- Manual: Human annotation interface
- Hybrid: Semi-supervised clustering + manual refinement

#### 5.3 Cross-Layer Feature Tracking
**Research Question**: How do features evolve across layers?

**Analysis**:
- [ ] Feature similarity matrices (layer i vs layer j)
- [ ] Trace feature transformations (structural → semantic)
- [ ] Identify feature bottlenecks (where information compresses)

---

### Phase 6: Advanced Steering Capabilities
**Priority**: LOW (research extension)
**Timeline**: Week 9+

#### 6.1 Multi-Circuit Composition
- [ ] Combine multiple circuits with weighted blending
- [ ] Circuit conflict resolution
- [ ] Hierarchical circuit activation

#### 6.2 Adaptive Steering
- [ ] Automatic alpha tuning based on desired effect
- [ ] Reinforcement learning for optimal steering
- [ ] User feedback loop (thumb up/down → adjust alpha)

#### 6.3 Real-Time Editing
- [ ] Modify circuit mid-generation
- [ ] Dynamic feature activation/deactivation
- [ ] Interactive debugging ("why did it generate this?")

---

## Quality Metrics & Acceptance Criteria

### Robustness
- [ ] System handles 100% of invalid inputs gracefully (no crashes)
- [ ] All errors logged with actionable context
- [ ] 99.9% uptime for API endpoints

### Fidelity
- [ ] SAE reconstruction MSE < 0.05 for all layers
- [ ] Steering changes output in 95%+ of cases (measurable effect)
- [ ] Circuit isolation: <5% cross-talk between circuits

### Performance
- [ ] Single steering operation: <100ms
- [ ] Batch generation (10 prompts): <5 seconds
- [ ] Interface response time: <200ms (API calls)

### Usability
- [ ] Non-expert can build and activate circuit in <5 minutes
- [ ] 3D visualization renders in <3 seconds
- [ ] Interface works on mobile devices

### Testing
- [ ] Unit test coverage: >80%
- [ ] Integration tests: 100% of critical paths
- [ ] All validation tests pass

---

## Open Research Questions

### 1. Are we intercepting 1:1?
**Status**: UNTESTED

**Tests Needed**:
- Activation reconstruction fidelity (per layer)
- Gradient flow preservation (does steering break backprop?)
- Information bottleneck analysis (where do we lose fidelity?)

**Expected Outcome**:
- Layer 0-3: High fidelity (>95% reconstruction)
- Layer 4-8: Medium fidelity (80-95%)
- Layer 9-11: Lower fidelity (60-80%, semantic compression)

### 2. Does the Atlas work?
**Status**: NEVER USED

**Why Not Tested**:
- Only Layer 0 SAE trained (11 layers missing)
- No feature labeling system
- No cross-layer comparison tools

**Validation Plan**:
1. Train all 12 layers
2. Label top 100 features per layer (automated + manual)
3. Build cross-layer similarity matrix
4. Verify interpretability (can humans understand features?)

### 3. Can we modify the network?
**Status**: STEERING ONLY (read-only intervention)

**Not Yet Implemented**:
- Permanent weight modification
- Feature ablation (zero out features)
- Feature transplantation (copy features between models)
- Fine-tuning with steering supervision

**Future Extensions**:
```python
class NetworkEditor:
    """Permanent modifications to model weights."""

    def ablate_feature(self, layer: int, feature_idx: int):
        """Zero out SAE feature permanently."""
        pass

    def amplify_feature(self, layer: int, feature_idx: int, factor: float):
        """Scale up feature's decoder direction."""
        pass

    def transplant_circuit(self, source_model, target_model, circuit: CircuitRecord):
        """Copy circuit from one model to another."""
        pass
```

---

## Implementation Priority

**Week 1-2**: Foundation (Error Handling, Logging, Persistent DB, Validation)
**Week 3-4**: Interface (Web UI, 3D Visualization, Neural Atlas Explorer)
**Week 5**: Performance (Caching, Batch Processing, Optimization)
**Week 6**: Testing (Unit, Integration, Validation Suite)
**Week 7-8**: Atlas Completion (All-Layer SAE, Feature Labeling)
**Week 9+**: Advanced Features (Multi-Circuit, Adaptive Steering)

---

## Success Criteria

System is **production-ready** when:

1. ✅ All Phase 1-4 tasks complete
2. ✅ All validation tests pass (1:1 interception confirmed)
3. ✅ Web interface deployed and usable
4. ✅ Neural Atlas functional (all 12 layers)
5. ✅ Documentation complete (API docs, user guide, tutorials)
6. ✅ Performance benchmarks met
7. ✅ External users can use system without assistance

---

## Notes

**Current Status**: We have a **working prototype** that demonstrates the concept. But it's fragile, unoptimized, and incomplete.

**The Gap**: Moving from "it works on my machine" to "production-grade system that anyone can use reliably."

**The Goal**: Transform NeuroTrace into the **first complete neural network control and cartography platform** - where users can:
- Explore the network's internal representations (Atlas)
- Discover causal pathways (Circuit Discovery)
- Modify behavior in real-time (Active Steering)
- Visualize everything in 3D (Interactive Graphs)
- Trust the system to work reliably (Production Quality)

We've **scratched the stone** - now we build the cathedral.
