# neurotrace/discovery/feature_circuit_discoverer.py

"""
Feature-Based Automatic Circuit Discovery

Discovers circuits by analyzing activation patterns across 73,728 Atlas features.
Uses SAE sparse codes to identify which feature combinations drive specific behaviors.

Strategy:
1. Forward examples through model + all 12 SAEs
2. Collect sparse codes (which features activate per layer)
3. Identify features that co-activate on task-relevant examples
4. Test causal importance via feature ablation
5. Extract multi-layer circuits from validated feature sets
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from neurotrace.control import EnhancedSAEFeatureStore
from neurotrace.datasets import IOIExample


@dataclass
class FeatureActivation:
    """Records when a feature activates."""
    layer: int
    feature_idx: int
    activation_strength: float
    example_idx: int
    is_correct: bool  # Did model predict correctly on this example?


@dataclass
class FeatureImportance:
    """Causal importance of a feature."""
    layer: int
    feature_idx: int
    mean_activation: float
    activation_frequency: float  # % examples where it activates
    correlation_with_success: float  # How much it predicts correct answers
    ablation_effect: float  # VLO when ablated (higher = more important)


class FeatureCircuitDiscoverer:
    """
    Discovers circuits by analyzing which SAE features are causally important.

    Unlike component-level discovery (layer_0.mlp), this finds SPECIFIC features
    within each layer that drive behavior.

    Example:
        discoverer = FeatureCircuitDiscoverer(feature_store, model, tokenizer)

        # Discover which features drive IOI
        features = discoverer.discover_from_examples(
            ioi_examples,
            top_k_per_layer=10,
            min_correlation=0.3
        )

        # Returns: [(layer, feature_idx, importance_score), ...]
    """

    def __init__(
        self,
        feature_store: EnhancedSAEFeatureStore,
        model,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Args:
            feature_store: Loaded with all 12 SAE layers
            model: Target transformer model
            tokenizer: Tokenizer for model
            device: Device for computation
        """
        self.feature_store = feature_store
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Check which layers are available
        self.available_layers = sorted(feature_store.saes.keys())
        if not self.available_layers:
            raise ValueError("No SAEs loaded in feature_store!")

        print(f"FeatureCircuitDiscoverer initialized")
        print(f"  Available layers: {self.available_layers}")
        print(f"  Total features: {len(self.available_layers) * 6144}")

    def collect_activations(
        self,
        examples: List[IOIExample],
        verbose: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Forward examples through model and collect SAE activations.

        Args:
            examples: IOI examples to analyze
            verbose: Show progress bar

        Returns:
            Dict mapping layer -> [num_examples, dict_size] activation tensor
        """
        # Tokenize all examples
        texts = [ex.text for ex in examples]
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Get target positions (last token for IOI)
        target_positions = (attention_mask.sum(dim=1) - 1).tolist()

        # Storage for SAE codes
        layer_activations = {}

        # Forward through model with hooks to capture activations
        with torch.no_grad():
            # Get MLP activations for all layers
            mlp_acts = {}  # layer -> [batch, seq, hidden_dim]

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # MLP output in GPT-2 is directly the tensor, not a tuple
                    if isinstance(output, tuple):
                        mlp_acts[layer_idx] = output[0].detach()
                    else:
                        mlp_acts[layer_idx] = output.detach()
                return hook

            # Register hooks
            handles = []
            for layer_idx in self.available_layers:
                module = self.model.transformer.h[layer_idx].mlp
                handle = module.register_forward_hook(make_hook(layer_idx))
                handles.append(handle)

            # Forward pass
            _ = self.model(input_ids, attention_mask=attention_mask)

            # Remove hooks
            for handle in handles:
                handle.remove()

        # Extract activations at target positions and encode with SAE
        for layer_idx in tqdm(self.available_layers, desc="Encoding SAE features", disable=not verbose):
            sae = self.feature_store.saes[layer_idx]

            # Get MLP activations at target positions
            # Shape: [batch, seq, hidden_dim]
            layer_mlp = mlp_acts[layer_idx]

            # Extract target position activations
            batch_size = layer_mlp.size(0)
            target_acts = torch.stack([
                layer_mlp[i, target_positions[i], :]
                for i in range(batch_size)
            ])  # [batch, hidden_dim]

            # Encode with SAE
            sae_output = sae.forward(target_acts)
            codes = sae_output['codes']  # [batch, dict_size]

            layer_activations[layer_idx] = codes.cpu()

        return layer_activations

    def analyze_feature_importance(
        self,
        activations: Dict[int, torch.Tensor],
        examples: List[IOIExample],
        top_k_per_layer: int = 20,
        min_activation_threshold: float = 0.1
    ) -> List[FeatureImportance]:
        """
        Analyze which features are important based on activation patterns.

        Args:
            activations: Dict from collect_activations()
            examples: Original examples
            top_k_per_layer: How many top features to keep per layer
            min_activation_threshold: Minimum activation to count as "active"

        Returns:
            List of FeatureImportance objects, sorted by correlation
        """
        # Check which examples model got correct
        # (For IOI: check if IO token probability > S token probability)
        model_correct = self._check_correctness(examples)

        feature_stats = []

        for layer_idx, codes in activations.items():
            # codes: [num_examples, dict_size]
            num_examples, dict_size = codes.shape

            for feature_idx in range(dict_size):
                feature_acts = codes[:, feature_idx]  # [num_examples]

                # Mean activation strength
                mean_act = feature_acts.mean().item()

                # Skip very rare features
                if mean_act < 0.01:
                    continue

                # Activation frequency
                active_mask = feature_acts > min_activation_threshold
                freq = active_mask.float().mean().item()

                # Correlation with model correctness
                if freq > 0 and freq < 1:  # Skip always-on or always-off
                    # Point-biserial correlation
                    correct_tensor = torch.tensor(model_correct, dtype=torch.float32)
                    correlation = torch.corrcoef(torch.stack([
                        feature_acts,
                        correct_tensor
                    ]))[0, 1].item()

                    if torch.isnan(torch.tensor(correlation)):
                        correlation = 0.0
                else:
                    correlation = 0.0

                feature_stats.append(
                    FeatureImportance(
                        layer=layer_idx,
                        feature_idx=feature_idx,
                        mean_activation=mean_act,
                        activation_frequency=freq,
                        correlation_with_success=correlation,
                        ablation_effect=0.0,  # Filled in later if needed
                    )
                )

        # Sort by correlation with success
        feature_stats.sort(key=lambda x: abs(x.correlation_with_success), reverse=True)

        # Keep top K per layer
        layer_counts = {layer: 0 for layer in self.available_layers}
        filtered_stats = []

        for stat in feature_stats:
            if layer_counts[stat.layer] < top_k_per_layer:
                filtered_stats.append(stat)
                layer_counts[stat.layer] += 1

        return filtered_stats

    def _check_correctness(self, examples: List[IOIExample]) -> List[bool]:
        """
        Check if model predicts correct answer for each example.

        Returns:
            List of booleans (True = correct prediction)
        """
        texts = [ex.text for ex in examples]
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        target_positions = (attention_mask.sum(dim=1) - 1).tolist()

        # Get correct/incorrect token IDs
        correct_ids = [
            self.tokenizer.encode(" " + ex.correct_answer, add_special_tokens=False)[0]
            for ex in examples
        ]
        incorrect_ids = [
            self.tokenizer.encode(" " + ex.incorrect_answer, add_special_tokens=False)[0]
            for ex in examples
        ]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Check predictions
        correct_mask = []
        for i in range(len(examples)):
            pos = target_positions[i]
            logit_correct = logits[i, pos, correct_ids[i]].item()
            logit_incorrect = logits[i, pos, incorrect_ids[i]].item()
            correct_mask.append(logit_correct > logit_incorrect)

        return correct_mask

    def discover_from_examples(
        self,
        examples: List[IOIExample],
        top_k_per_layer: int = 20,
        min_correlation: float = 0.2,
        verbose: bool = True
    ) -> List[FeatureImportance]:
        """
        Complete discovery pipeline: collect activations -> analyze importance.

        Args:
            examples: Task examples to analyze
            top_k_per_layer: Max features per layer to return
            min_correlation: Minimum correlation to include
            verbose: Show progress

        Returns:
            List of important features sorted by correlation
        """
        if verbose:
            print("=" * 80)
            print("FEATURE-BASED CIRCUIT DISCOVERY")
            print("=" * 80)
            print(f"Examples: {len(examples)}")
            print(f"Layers: {len(self.available_layers)}")
            print(f"Total features: {len(self.available_layers) * 6144}")
            print()

        # Step 1: Collect activations
        if verbose:
            print("Step 1: Collecting SAE activations...")
        activations = self.collect_activations(examples, verbose=verbose)

        if verbose:
            print(f"  Collected activations for {len(activations)} layers")
            print()

        # Step 2: Analyze importance
        if verbose:
            print("Step 2: Analyzing feature importance...")

        important_features = self.analyze_feature_importance(
            activations,
            examples,
            top_k_per_layer=top_k_per_layer,
            min_activation_threshold=0.1
        )

        # Filter by correlation threshold
        important_features = [
            f for f in important_features
            if abs(f.correlation_with_success) >= min_correlation
        ]

        if verbose:
            print(f"  Found {len(important_features)} important features")
            print()
            print("Top 20 features:")
            for i, feat in enumerate(important_features[:20], 1):
                print(f"  {i:2d}. Layer {feat.layer:2d} Feature {feat.feature_idx:4d}  "
                      f"Corr={feat.correlation_with_success:+.3f}  "
                      f"MeanAct={feat.mean_activation:.3f}  "
                      f"Freq={feat.activation_frequency*100:.1f}%")
            print()

        return important_features
