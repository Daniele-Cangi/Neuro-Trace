# neurotrace/control/hierarchical_steering.py

"""
Hierarchical Multi-Layer Steering

Coordinates interventions across multiple layers simultaneously.
Uses discovered circuits to perform orchestrated steering.

Example:
    # Load circuit
    circuit = registry.get("atlas_vlo_validated_20251117")

    # Create hierarchical steering
    steerer = HierarchicalSteering(model, feature_store)

    # Steer using circuit (activates Layer 0 MLP features)
    steered_output = steerer.steer_with_circuit(
        circuit=circuit,
        input_text="When Alice and Bob...",
        strength=2.0
    )
"""

from __future__ import annotations

import torch
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from neurotrace.control import EnhancedSAEFeatureStore
from neurotrace.control.circuit_registry import CircuitRecord


@dataclass
class SteeringConfig:
    """Configuration for hierarchical steering."""
    layer: int
    feature_indices: List[int]
    strength: float  # Multiplier for feature activation
    mode: str = "add"  # "add", "set", "scale"


class HierarchicalSteering:
    """
    Multi-layer coordinated steering using discovered circuits.

    This enables:
    1. Circuit-based steering (activate all features in a circuit)
    2. Coordinated multi-layer interventions
    3. Fine-grained control over feature activation strengths
    """

    def __init__(
        self,
        model,
        feature_store: EnhancedSAEFeatureStore,
        device: str = "cuda"
    ):
        """
        Args:
            model: Target transformer model
            feature_store: Loaded with SAEs for layers you want to steer
            device: Device for computation
        """
        self.model = model
        self.feature_store = feature_store
        self.device = device

        # Track active hooks
        self._active_hooks = []

    def steer_with_circuit(
        self,
        circuit: CircuitRecord,
        input_text: str,
        tokenizer,
        strength: float = 1.0,
        top_k_features_per_layer: int = 10
    ) -> Dict:
        """
        Steer model using a discovered circuit.

        Args:
            circuit: Circuit object with components and SAE features
            input_text: Input text to generate from
            tokenizer: Tokenizer for model
            strength: Overall strength multiplier
            top_k_features_per_layer: How many features to activate per layer

        Returns:
            Dict with:
                - output_text: Generated text
                - logits: Output logits
                - interventions: List of applied interventions
        """
        # Build steering configs from circuit
        configs = []

        for layer_idx, feature_indices in circuit.features.sae_indices.items():
            if layer_idx not in self.feature_store.saes:
                print(f"Warning: Layer {layer_idx} SAE not loaded, skipping")
                continue

            # Take top K features by activation/importance
            # (for now just take first K, later could sort by importance)
            top_features = feature_indices[:top_k_features_per_layer]

            if top_features:
                configs.append(
                    SteeringConfig(
                        layer=layer_idx,
                        feature_indices=top_features,
                        strength=strength,
                        mode="add"
                    )
                )

        # Steer with these configs
        return self.steer_with_configs(
            configs=configs,
            input_text=input_text,
            tokenizer=tokenizer
        )

    def steer_with_configs(
        self,
        configs: List[SteeringConfig],
        input_text: str,
        tokenizer,
        max_new_tokens: int = 20
    ) -> Dict:
        """
        Steer model with explicit steering configurations.

        Args:
            configs: List of SteeringConfig objects
            input_text: Input text
            tokenizer: Tokenizer
            max_new_tokens: Max tokens to generate

        Returns:
            Dict with output_text, logits, interventions
        """
        # Tokenize input
        encoding = tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"]

        # Register steering hooks
        self._register_steering_hooks(configs)

        try:
            # Generate with steering active
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "output_text": output_text,
                "input_text": input_text,
                "interventions": [
                    {
                        "layer": cfg.layer,
                        "num_features": len(cfg.feature_indices),
                        "strength": cfg.strength,
                        "mode": cfg.mode
                    }
                    for cfg in configs
                ]
            }

        finally:
            # Always clean up hooks
            self._remove_all_hooks()

    def _register_steering_hooks(self, configs: List[SteeringConfig]):
        """Register forward hooks to apply steering."""
        self._remove_all_hooks()  # Clean slate

        for config in configs:
            sae = self.feature_store.saes.get(config.layer)
            if sae is None:
                print(f"Warning: No SAE for layer {config.layer}, skipping")
                continue

            # Get decoder directions for features
            directions = self.feature_store.get_sae_directions(
                model_name="gpt2",
                layer=config.layer,
                feature_indices=config.feature_indices,
                device=self.device
            )  # [num_features, hidden_dim]

            # Create hook for this layer
            def make_hook(layer_idx, steering_directions, strength, mode):
                def hook(module, input, output):
                    """
                    Apply steering by modifying MLP output.

                    mode:
                        - "add": output += strength * direction
                        - "set": output = strength * direction (dangerous!)
                        - "scale": output *= (1 + strength)
                    """
                    # Handle both tuple and direct tensor outputs
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        rest = output[1:]
                    else:
                        hidden_states = output
                        rest = ()

                    # Apply steering to all positions
                    # hidden_states: [batch, seq, hidden_dim]
                    # steering_directions: [num_features, hidden_dim]

                    if mode == "add":
                        # Add steering vectors (broadcast across seq positions)
                        steering_vector = (steering_directions.sum(dim=0) * strength)  # [hidden_dim]
                        modified = hidden_states + steering_vector.unsqueeze(0).unsqueeze(0)

                    elif mode == "scale":
                        # Scale existing activations
                        modified = hidden_states * (1.0 + strength)

                    elif mode == "set":
                        # Replace with steering direction (VERY strong intervention)
                        steering_vector = (steering_directions.sum(dim=0) * strength)
                        modified = steering_vector.unsqueeze(0).unsqueeze(0).expand_as(hidden_states)

                    else:
                        raise ValueError(f"Unknown steering mode: {mode}")

                    # Return in original format
                    if rest:
                        return (modified, *rest)
                    else:
                        return modified

                return hook

            # Register hook
            module = self.model.transformer.h[config.layer].mlp
            handle = module.register_forward_hook(
                make_hook(config.layer, directions, config.strength, config.mode)
            )
            self._active_hooks.append(handle)

    def _remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self._active_hooks:
            handle.remove()
        self._active_hooks = []

    def compare_with_without_steering(
        self,
        configs: List[SteeringConfig],
        input_text: str,
        tokenizer,
        max_new_tokens: int = 20
    ) -> Dict:
        """
        Generate both with and without steering for comparison.

        Returns:
            Dict with baseline_output, steered_output, diff
        """
        # Baseline (no steering)
        encoding = tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"]

        with torch.no_grad():
            baseline_outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for comparison
                pad_token_id=tokenizer.eos_token_id
            )

        baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)

        # Steered
        steered_result = self.steer_with_configs(
            configs=configs,
            input_text=input_text,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens
        )

        return {
            "input_text": input_text,
            "baseline_output": baseline_text,
            "steered_output": steered_result["output_text"],
            "interventions": steered_result["interventions"]
        }
