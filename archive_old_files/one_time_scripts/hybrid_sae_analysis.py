# hybrid_sae_analysis.py

"""
Hybrid SAE Analysis: Compare Enhanced SAE (Layer 0) vs SAELens Baseline (Layer 9).

This script performs comprehensive comparison between:
- Our Enhanced SAE trained on Layer 0 MLP (novel discovery)
- Anthropic's pre-trained SAE for Layer 9 (expected baseline)

Goal: Explain WHY Layer 0 MLP dominates IOI task despite Layer 9 being
      expected dominant according to literature.

Usage:
    python hybrid_sae_analysis.py \
        --enhanced_sae_path checkpoints/enhanced_sae/final.pt \
        --activations_dir runs/deep_ioi_capture/.../activations \
        --output_dir results/hybrid_analysis
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer

from neurotrace.training import EnhancedSAE
from neurotrace.datasets import IOIDatasetGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid SAE Analysis: Enhanced SAE vs SAELens Baseline"
    )
    parser.add_argument(
        "--enhanced_sae_path",
        type=str,
        required=True,
        help="Path to trained Enhanced SAE checkpoint",
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Path to captured activations directory",
    )
    parser.add_argument(
        "--num_test_examples",
        type=int,
        default=1000,
        help="Number of IOI examples to test on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/hybrid_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--use_saelens",
        action="store_true",
        help="Enable SAELens comparison (requires sae-lens installed)",
    )
    return parser.parse_args()


def load_enhanced_sae(checkpoint_path: str, device: str) -> EnhancedSAE:
    """Load trained Enhanced SAE from checkpoint."""
    print(f"Loading Enhanced SAE from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 768)
    dict_size = config.get('dict_size', 3072)

    # Create SAE
    sae = EnhancedSAE(
        input_dim=input_dim,
        dict_size=dict_size,
        k_sparse=config.get('k_sparse', 64),
        sparsity_lambda=config.get('sparsity_lambda', 1e-3),
        use_jumprelu=config.get('use_jumprelu', False),
        ghost_threshold=config.get('ghost_threshold', 1e-5),
        normalize_decoder=config.get('normalize_decoder', True),
    )

    # Load weights
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()

    print(f"✓ Enhanced SAE loaded:")
    print(f"  Input dim: {input_dim}")
    print(f"  Dict size: {dict_size}")

    final_mse = checkpoint.get('final_mse_loss', None)
    if final_mse is not None:
        print(f"  Training metrics: MSE={final_mse:.4f}")
    else:
        print(f"  Training metrics: Not available")

    return sae


def analyze_enhanced_sae_features(
    sae: EnhancedSAE,
    activations: torch.Tensor,
    texts: List[str],
    top_k: int = 10,
) -> Dict:
    """Analyze Enhanced SAE features on IOI data."""
    print("\nAnalyzing Enhanced SAE features...")

    with torch.no_grad():
        # Encode
        encode_output = sae.encode(activations)  # Returns dict
        codes = encode_output['codes']  # [num_examples, dict_size]

        # Feature statistics
        feature_activation_freq = (codes > 0).float().mean(dim=0)  # [dict_size]
        feature_mean_activation = codes.mean(dim=0)  # [dict_size]
        feature_max_activation = codes.max(dim=0)[0]  # [dict_size]

        # Top activating features
        top_features = feature_activation_freq.topk(top_k)

        # For each top feature, find top activating examples
        feature_examples = {}
        for feat_idx in top_features.indices.tolist():
            # Get activation values for this feature across all examples
            feat_activations = codes[:, feat_idx]  # [num_examples]

            # Top 5 activating examples
            top_ex_indices = feat_activations.topk(5).indices.tolist()
            top_ex_values = feat_activations.topk(5).values.tolist()

            feature_examples[feat_idx] = {
                'activation_freq': feature_activation_freq[feat_idx].item(),
                'mean_activation': feature_mean_activation[feat_idx].item(),
                'max_activation': feature_max_activation[feat_idx].item(),
                'top_examples': [
                    {
                        'text': texts[idx],
                        'activation': val,
                    }
                    for idx, val in zip(top_ex_indices, top_ex_values)
                ],
            }

    results = {
        'num_features': sae.dict_size,
        'num_examples': len(texts),
        'feature_activation_freq': feature_activation_freq.cpu().numpy(),
        'feature_mean_activation': feature_mean_activation.cpu().numpy(),
        'dead_features': (feature_activation_freq == 0).sum().item(),
        'dead_fraction': (feature_activation_freq == 0).float().mean().item(),
        'top_features': feature_examples,
    }

    print(f"✓ Enhanced SAE analysis complete:")
    print(f"  Active features: {sae.dict_size - results['dead_features']}/{sae.dict_size}")
    print(f"  Dead features: {results['dead_features']} ({results['dead_fraction']:.1%})")

    return results


def compare_with_saelens_baseline(
    enhanced_results: Dict,
    ioi_examples: List,
    device: str,
    activations_dir: str,
    num_test_examples: int = 1000,
) -> Dict:
    """Compare Enhanced SAE with SAELens pre-trained baseline."""
    print("\n" + "=" * 80)
    print("SAELENS BASELINE COMPARISON")
    print("=" * 80)

    try:
        from sae_lens import SAE
        print("✓ SAELens imported")
    except ImportError:
        print("⚠️  SAELens not installed - skipping baseline comparison")
        print("   Install with: pip install sae-lens")
        return None

    print("\nLoading SAELens pre-trained SAE for Layer 9...")

    try:
        # Load Layer 9 residual stream SAE (contains name mover heads information)
        # Using correct API: from_pretrained(release, sae_id, device)
        # Note: Using residual stream since MLP-specific SAEs may not be available
        sae_baseline = SAE.from_pretrained(
            release="gpt2-small-res-jb",  # Release name
            sae_id="blocks.9.hook_resid_pre",  # Layer 9 residual stream (pre)
            device=device
        )
        print(f"✓ SAELens Layer 9 Residual SAE loaded")
        print(f"  Config: {sae_baseline.cfg.d_in} -> {sae_baseline.cfg.d_sae} features")
    except Exception as e:
        print(f"⚠️  Could not load SAELens baseline: {e}")
        print("   Skipping baseline comparison")
        return None

    # Load Layer 9 activations from the deep dataset
    print("\nLoading Layer 9 activations from dataset...")
    layer9_activations = []
    layer9_indices = []

    # The deep dataset has activations stored as "layer_X.mlp" keys
    # Batch files start at 1, not 0
    for batch_idx in range(1, min(num_test_examples // 50, 100) + 1):  # 50 examples per batch
        batch_file = os.path.join(activations_dir, f"batch_{batch_idx:05d}.pt")
        if not os.path.exists(batch_file):
            continue

        batch_data = torch.load(batch_file, map_location='cpu', weights_only=False)

        # Get Layer 9 MLP output activations
        # NOTE: Activations are already flattened: [batch, d_model] or [batch*seq, d_model]
        if 'layer_9.mlp' in batch_data:
            mlp_acts = batch_data['layer_9.mlp']  # Already [N, d_model]
            layer9_activations.append(mlp_acts)

            # Track which examples these are
            start_idx = (batch_idx - 1) * mlp_acts.shape[0]  # Based on actual batch size
            layer9_indices.extend(range(start_idx, start_idx + mlp_acts.shape[0]))

    if not layer9_activations:
        print("⚠️  No Layer 9 activations found in dataset")
        return None

    layer9_activations = torch.cat(layer9_activations, dim=0).to(device)
    print(f"✓ Loaded {layer9_activations.shape[0]} Layer 9 activation samples")

    # Run SAELens SAE on Layer 9 activations
    print("\nAnalyzing Layer 9 features with SAELens SAE...")

    with torch.no_grad():
        # Encode Layer 9 activations
        sae_output = sae_baseline.encode(layer9_activations)

        # Handle different output formats
        if isinstance(sae_output, dict):
            layer9_codes = sae_output['latent_acts']  # SAELens format
        else:
            layer9_codes = sae_output

    # Analyze Layer 9 feature statistics
    feature_activations = (layer9_codes > 0).float().cpu().numpy()  # [n_samples, n_features]
    feature_frequencies = feature_activations.mean(axis=0)  # Activation frequency per feature
    feature_max_acts = layer9_codes.max(dim=0)[0].cpu().numpy()  # Max activation per feature

    # Get top Layer 9 features by frequency
    top_l9_indices = np.argsort(feature_frequencies)[::-1][:10]

    print(f"\n{'='*60}")
    print("LAYER 9 (SAELENS) - TOP FEATURES")
    print(f"{'='*60}")

    for rank, feat_idx in enumerate(top_l9_indices, 1):
        freq = feature_frequencies[feat_idx]
        max_act = feature_max_acts[feat_idx]
        print(f"{rank}. Feature {feat_idx:4d}: Freq={freq:5.1%}, Max={max_act:6.2f}")

    return {
        # Don't include SAE object (not JSON serializable)
        'num_features': sae_baseline.cfg.d_sae,
        'num_samples': layer9_activations.shape[0],
        'top_features': {
            str(feat_idx): {
                'activation_freq': float(feature_frequencies[feat_idx]),
                'max_activation': float(feature_max_acts[feat_idx]),
            }
            for feat_idx in top_l9_indices[:10]
        },
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("NEUROTRACE - HYBRID SAE ANALYSIS")
    print("=" * 80)
    print()
    print("Goal: Compare Enhanced SAE (Layer 0) vs SAELens Baseline (Layer 9)")
    print("Question: WHY does Layer 0 MLP dominate IOI despite Layer 9 expected?")
    print()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Load Enhanced SAE
    # ========================================================================

    print("[1/5] Loading Enhanced SAE...")
    enhanced_sae = load_enhanced_sae(args.enhanced_sae_path, device)
    print()

    # ========================================================================
    # Generate Test IOI Dataset
    # ========================================================================

    print("[2/5] Generating IOI test dataset...")
    generator = IOIDatasetGenerator(seed=42)
    ioi_examples = generator.generate(
        num_examples=args.num_test_examples,
        ensure_diversity=True,
    )
    print(f"✓ Generated {len(ioi_examples)} IOI test examples")
    print()

    # ========================================================================
    # Capture Layer 0 MLP Activations
    # ========================================================================

    print("[3/5] Capturing Layer 0 MLP activations...")

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    texts = [ex.text for ex in ioi_examples]

    # Tokenize
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=30,
        return_tensors="pt"
    ).to(device)

    # Hook to capture Layer 0 MLP output
    layer0_activations = []

    def hook_fn(module, input, output):
        # output[0] is [batch, seq, hidden_dim]
        layer0_activations.append(output[0].detach().cpu())

    hook = model.transformer.h[0].mlp.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(**encoding)

    hook.remove()

    # Flatten sequences: [batch, seq, 768] -> [batch*seq, 768]
    activations_flat = torch.cat(layer0_activations, dim=0)  # [batch, seq, 768]
    activations_flat = activations_flat.reshape(-1, activations_flat.shape[-1])  # [batch*seq, 768]

    print(f"✓ Captured Layer 0 MLP activations: {activations_flat.shape}")
    print()

    # ========================================================================
    # Analyze Enhanced SAE Features
    # ========================================================================

    print("[4/5] Analyzing Enhanced SAE features on IOI...")
    enhanced_results = analyze_enhanced_sae_features(
        sae=enhanced_sae,
        activations=activations_flat.to(device),
        texts=texts,
        top_k=20,
    )
    print()

    # ========================================================================
    # Compare with SAELens (if enabled)
    # ========================================================================

    print("[5/5] Comparing with SAELens baseline...")
    baseline_results = None
    if args.use_saelens:
        baseline_results = compare_with_saelens_baseline(
            enhanced_results=enhanced_results,
            ioi_examples=ioi_examples,
            device=device,
            activations_dir=args.activations_dir,
            num_test_examples=args.num_test_examples,
        )
    else:
        print("⚠️  SAELens comparison disabled (use --use_saelens to enable)")

    print()

    # ========================================================================
    # Save Results
    # ========================================================================

    print("Saving results...")

    # Save feature statistics
    results_path = output_dir / "hybrid_analysis_results.json"
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'enhanced_sae': {
            'checkpoint': args.enhanced_sae_path,
            'num_features': enhanced_results['num_features'],
            'num_examples': enhanced_results['num_examples'],
            'dead_features': int(enhanced_results['dead_features']),
            'dead_fraction': float(enhanced_results['dead_fraction']),
            'top_features': {
                str(k): {
                    'activation_freq': float(v['activation_freq']),
                    'mean_activation': float(v['mean_activation']),
                    'max_activation': float(v['max_activation']),
                    'top_examples': v['top_examples'],
                }
                for k, v in enhanced_results['top_features'].items()
            },
        },
        'saelens_baseline': baseline_results,
    }

    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"✓ Results saved to: {results_path}")
    print()

    # Save feature activation distributions
    np.save(
        output_dir / "enhanced_sae_feature_activations.npy",
        enhanced_results['feature_activation_freq'],
    )

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("✅ HYBRID SAE ANALYSIS COMPLETE")
    print("=" * 80)
    print()

    print("Enhanced SAE (Layer 0 MLP) Results:")
    print(f"  Features: {enhanced_results['num_features']}")
    print(f"  Active: {enhanced_results['num_features'] - enhanced_results['dead_features']}")
    print(f"  Dead: {enhanced_results['dead_features']} ({enhanced_results['dead_fraction']:.1%})")
    print()

    print("Top 5 Most Frequent Features:")
    sorted_features = sorted(
        enhanced_results['top_features'].items(),
        key=lambda x: x[1]['activation_freq'],
        reverse=True,
    )[:5]

    for feat_id, feat_info in sorted_features:
        print(f"\n  Feature {feat_id}:")
        print(f"    Activation freq: {feat_info['activation_freq']:.1%}")
        print(f"    Mean activation: {feat_info['mean_activation']:.3f}")
        print(f"    Top example: \"{feat_info['top_examples'][0]['text']}\"")

    print()
    print(f"Results saved to: {output_dir}")
    print()

    print("Next Steps:")
    print("  1. Analyze feature monosemanticity (manual interpretation)")
    print("  2. Compare Layer 0 vs Layer 9 feature types")
    print("  3. Identify WHY Layer 0 dominates IOI task")
    print("  4. Write publication/report")
    print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
