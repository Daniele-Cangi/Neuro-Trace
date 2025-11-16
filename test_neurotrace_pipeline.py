"""
Test script per verificare l'intera pipeline NeuroTrace con GPT-2.

Pipeline testata:
  1. TargetModelWrapper (carica GPT-2 + tokenizer)
  2. AdaptiveActivationsBuffer (compressione)
  3. AdaptiveHookManager (cattura attivazioni)
  4. SAEFeatureExtractor (estrazione feature sparse)
  5. VectorStateDB (indicizzazione FAISS + SQLite) - opzionale se faiss installato

Uso:
  python test_neurotrace_pipeline.py
"""

from __future__ import annotations

import logging
import sys

import torch

from neurotrace.config import NeuroTraceConfig
from neurotrace.models.wrapper import TargetModelWrapper
from neurotrace.instrumentation.adaptive_activations_buffer import AdaptiveActivationsBuffer
from neurotrace.instrumentation.adaptive_hook_manager import AdaptiveHookManager, HookManagerConfig
from neurotrace.state_indexer.sae_feature_extractor import SAEFeatureExtractor

# VectorStateDB è opzionale (richiede faiss)
try:
    from neurotrace.state_indexer.vector_state_db import VectorStateDB, VectorStateDBConfig
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS non disponibile, salto test VectorStateDB")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_basic_pipeline():
    """Test della pipeline base: wrapper + hook + buffer."""
    
    logger.info("=" * 60)
    logger.info("Test 1: Pipeline base (wrapper + hook + buffer)")
    logger.info("=" * 60)

    # Config: GPU se disponibile, altrimenti CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if device == "cuda" else "fp32"
    
    cfg = NeuroTraceConfig(
        model_name_or_path="gpt2",
        device=device,
        precision=precision,
        max_seq_len=64,
    )
    
    logger.info(f"Device: {device}, Precision: {precision}")

    # 1) Carica modello
    logger.info("Carico TargetModelWrapper...")
    wrapper = TargetModelWrapper(cfg)
    logger.info(f"✓ Modello caricato: {cfg.model_name_or_path}")

    # 2) Setup buffer + hook manager
    logger.info("Setup buffer e hook manager...")
    buffer = AdaptiveActivationsBuffer(
        cfg,
        auto_flush_threshold_bytes=100 * 1024 * 1024,  # 100 MB
    )
    
    hook_cfg = HookManagerConfig(
        capture_mode="block_output",
        sampling_strategy="none",
    )
    
    hook_mgr = AdaptiveHookManager(
        wrapper.model,
        cfg,
        buffer,
        hook_cfg=hook_cfg,
    )
    
    hook_mgr.register_hooks()
    logger.info("✓ Hook registrati")

    # 3) Forward con cattura attivazioni
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
    ]
    
    example_ids = [f"test_ex_{i}" for i in range(len(texts))]
    
    logger.info(f"Eseguo forward su {len(texts)} esempi...")
    hook_mgr.set_current_batch(
        example_ids=example_ids,
        step_meta={"phase": "test", "epoch": 0, "step": 0},
    )
    
    # Il wrapper non usa capture_activations perché usiamo AdaptiveHookManager
    with torch.no_grad():
        out = wrapper.run_texts(texts, capture_activations=False)
    
    logger.info(f"✓ Forward completato")
    logger.info(f"  Logits shape: {out['logits'].shape}")
    logger.info(f"  Input IDs shape: {out['input_ids'].shape}")

    # 4) Verifica memoria buffer
    mem_stats = buffer.get_memory_stats()
    logger.info(f"✓ Buffer stats:")
    logger.info(f"  Num examples: {mem_stats['num_examples']}")
    logger.info(f"  Approx MB: {mem_stats['approx_megabytes']:.2f}")

    return wrapper, buffer, hook_mgr, example_ids


def test_sae_extraction(buffer, example_ids):
    """Test estrazione feature SAE dai chunk compressi."""
    
    logger.info("=" * 60)
    logger.info("Test 2: Estrazione feature SAE")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NeuroTraceConfig(
        model_name_or_path="gpt2",
        device=device,
    )
    
    sae_extractor = SAEFeatureExtractor(cfg)
    logger.info("✓ SAEFeatureExtractor creato")

    # Flush buffer e estrai feature
    all_features = []
    
    def indexer_fn(ex_id: str, chunks):
        logger.info(f"  Processing example: {ex_id}, chunks: {len(chunks)}")
        features = sae_extractor.extract_features_for_example(ex_id, chunks)
        
        # Log delle feature SAE per layer
        sae_features = features["sae_features"]
        for layer_name, layer_data in sae_features.items():
            indices_shape = layer_data["indices"].shape
            values_shape = layer_data["values"].shape
            logger.info(f"    {layer_name}: indices={indices_shape}, values={values_shape}")
        
        # Build flat vector
        flat_vec = sae_extractor.build_flat_feature_vector(sae_features)
        logger.info(f"    Flat vector dim: {flat_vec.shape[0]}")
        
        all_features.append({
            "example_id": ex_id,
            "features": features,
            "flat_vector": flat_vec,
        })
    
    logger.info("Flush buffer -> SAE extraction...")
    buffer.flush_to_indexer(indexer_fn)
    logger.info(f"✓ Estratte feature per {len(all_features)} esempi")
    
    return all_features


def test_vector_db(all_features):
    """Test indicizzazione con VectorStateDB (richiede faiss)."""
    
    if not FAISS_AVAILABLE:
        logger.info("=" * 60)
        logger.info("Test 3: VectorStateDB - SALTATO (faiss non disponibile)")
        logger.info("=" * 60)
        logger.info("Per installare: pip install faiss-cpu")
        return

    logger.info("=" * 60)
    logger.info("Test 3: VectorStateDB (FAISS + SQLite)")
    logger.info("=" * 60)

    # Determina dimensione vettore dal primo esempio
    vec_dim = all_features[0]["flat_vector"].shape[0]
    
    vecdb_cfg = VectorStateDBConfig(
        dim=vec_dim,
        metric="ip",
        use_ivfpq=False,  # Flat index per semplicità in test
        sqlite_path="test_neurotrace.sqlite3",
    )
    
    vecdb = VectorStateDB(vecdb_cfg)
    logger.info(f"✓ VectorStateDB creato (dim={vec_dim})")

    # Inserisci tutti gli esempi
    logger.info("Inserisco esempi nel vector DB...")
    for item in all_features:
        ex_id = item["example_id"]
        flat_vec = item["flat_vector"]
        
        metadata = {
            "prompt": f"Test prompt for {ex_id}",
            "output": f"Test output for {ex_id}",
            "task_tag": "test_pipeline",
        }
        
        vecdb.insert(ex_id, flat_vec, metadata)
        logger.info(f"  ✓ Inserted: {ex_id}")

    # Query similarity search
    logger.info("\nTest similarity search...")
    query_vec = all_features[0]["flat_vector"]
    results = vecdb.query_similar(query_vec, k=2)
    
    logger.info(f"✓ Found {len(results)} similar examples:")
    for r in results:
        logger.info(f"  - {r['example_id']}: score={r['score']:.4f}")

    # Query by task
    logger.info("\nTest query by task...")
    task_results = vecdb.query_by_task("test_pipeline", limit=10)
    logger.info(f"✓ Found {len(task_results)} examples with task='test_pipeline'")

    logger.info("\n✓ VectorStateDB test completato")


def main():
    logger.info("🚀 Avvio test completo pipeline NeuroTrace")
    logger.info("")

    try:
        # Test 1: Pipeline base
        wrapper, buffer, hook_mgr, example_ids = test_basic_pipeline()
        logger.info("")

        # Test 2: SAE extraction
        all_features = test_sae_extraction(buffer, example_ids)
        logger.info("")

        # Test 3: Vector DB (opzionale)
        test_vector_db(all_features)
        logger.info("")

        logger.info("=" * 60)
        logger.info("✅ TUTTI I TEST COMPLETATI CON SUCCESSO")
        logger.info("=" * 60)
        
        return 0

    except Exception as e:
        logger.error(f"❌ Test fallito: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
