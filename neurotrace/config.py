from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict


# ---------------------------------------------------------
# Compression sub-config
# ---------------------------------------------------------

@dataclass
class CompressionConfig:
    # Random projection (Johnson–Lindenstrauss)
    use_random_projection: bool = True
    projection_dim: int = 256  # target dim after projection

    # Quantization
    # 32 = no quantization (fp32), 16 = fp16, 8 = int8
    quantization_bits: int = 16

    # Top-k sparsity on projected vectors
    use_topk_sparsity: bool = False
    topk_per_vector: int = 32


# ---------------------------------------------------------
# SAE sub-config
# ---------------------------------------------------------

@dataclass
class SAEConfig:
    # Sparse Autoencoder configuration
    sparsity_lambda: float = 1e-3
    sae_topk_per_token: int = 32
    sae_dict_mult: int = 4  # dict_size = dict_mult * hidden_dim


# ---------------------------------------------------------
# Vector DB sub-config
# ---------------------------------------------------------

@dataclass
class VectorDBConfig:
    # Vector database configuration
    dim: int = 256
    metric: str = "ip"  # "ip" (inner product) or "l2"
    nlist: int = 100
    nprobe: int = 10
    use_ivfpq: bool = True
    sqlite_path: str = "neurotrace_state_meta.sqlite3"


# ---------------------------------------------------------
# Core NeuroTrace config
# ---------------------------------------------------------

@dataclass
class NeuroTraceConfig:
    # Model / runtime
    model_name_or_path: str = "gpt2"
    device: str = "cuda"  # "cuda" or "cpu"
    precision: str = "fp16"  # "fp32" | "fp16" | "bf16"
    max_seq_len: int = 256

    # Logging
    log_level: str = "info"

    # Seeds
    seed: int = 42

    # Phase 1 – compression config
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    # SAE config
    sae: SAEConfig = field(default_factory=SAEConfig)

    # Vector DB config
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)

    # (Hook/Instrumentation configs avanzati potranno essere aggiunti dopo)
    # es: enable_residual_stream, hook_layer_pattern, etc.

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializza in dict (annidando anche la config di compressione).
        Utile per meta.json e log strutturati.
        """
        base = asdict(self)
        # asdict ha già serializzato CompressionConfig in dict
        return base
