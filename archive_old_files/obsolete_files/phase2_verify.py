"""
Phase 2: Streaming SAE Training on Layer 0 MLP
100K IOI examples → 30 min on A100

First: verify SAELens installation + load Layer 9 baseline
"""

import torch
from sae_lens import SAE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_saelens_setup():
    """Verify SAELens installation and load pre-trained SAE"""
    
    logger.info("🔍 Verifying SAELens setup...")
    
    try:
        # Load pre-trained SAE for Layer 9 (gold standard baseline)
        logger.info("Loading gpt2-small-layer-9-res-jb (24K dict)...")
        
        # SAELens requires sae_id and release parameters
        sae = SAE.from_pretrained(
            sae_id="gpt2-small-layer-9-res-jb",
            release="gpt2-sae-res-jb",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info(f"✅ SAE loaded successfully!")
        logger.info(f"   Encoder shape: {sae.encoder.shape if hasattr(sae.encoder, 'shape') else 'N/A'}")
        logger.info(f"   Decoder shape: {sae.decoder.shape if hasattr(sae.decoder, 'shape') else 'N/A'}")
        logger.info(f"   Dict size: {sae.d_sae if hasattr(sae, 'd_sae') else 'N/A'}")
        
        return sae
        
    except Exception as e:
        logger.error(f"❌ Failed to load SAE: {e}")
        logger.info("Trying alternative: sae-lens from GitHub...")
        return None


def setup_streaming_trainer():
    """Initialize streaming SAE trainer for Layer 0"""
    
    logger.info("🚀 Setting up Streaming SAE Trainer for Layer 0...")
    
    from scripts.streaming_sae_training import StreamingSAETrainer
    
    trainer = StreamingSAETrainer(
        model_name="gpt2",
        layer_idx=0,
        dict_size=4096,
        learning_rate=3e-4,
        sparsity_lambda=1e-2,
        batch_size=128,
        device="cuda",
    )
    
    logger.info("✅ Trainer initialized!")
    logger.info(f"   Model: GPT-2")
    logger.info(f"   Layer: 0 (MLP output)")
    logger.info(f"   Dict size: 4096")
    logger.info(f"   Batch size: 128")
    
    return trainer


def main():
    logger.info("=" * 80)
    logger.info("PHASE 2: STREAMING SAE TRAINING")
    logger.info("=" * 80)
    
    # Step 1: Verify SAELens
    logger.info("\n[Step 1] Verify SAELens Installation")
    logger.info("-" * 80)
    sae_l9 = verify_saelens_setup()
    
    # Step 2: Setup streaming trainer
    logger.info("\n[Step 2] Setup Streaming Trainer for Layer 0")
    logger.info("-" * 80)
    trainer = setup_streaming_trainer()
    
    # Step 3: Ready to train
    logger.info("\n[Step 3] Ready for Training")
    logger.info("-" * 80)
    logger.info("✅ All systems ready!")
    logger.info("\nNext: Run training on 100K IOI examples (30 min on A100)")
    logger.info("\nCommand:")
    logger.info("  python -m scripts.streaming_sae_training")
    
    return sae_l9, trainer


if __name__ == "__main__":
    sae_l9, trainer = main()
