# neurotrace/control/sae_feature_store.py

from __future__ import annotations

from typing import List

import torch

from neurotrace.state_indexer.sae_feature_extractor import SAEFeatureExtractor


class SAEFeatureStore:
    """
    Adapter che collega SteeringBuilder al tuo SAEFeatureExtractor esistente.

    Questo implementa il Protocol FeatureStore richiesto da steering_builder.py.
    """

    def __init__(self, sae_extractor: SAEFeatureExtractor) -> None:
        """
        Args:
            sae_extractor: Istanza di SAEFeatureExtractor già configurata
                           con SAE caricati/addestrati.
        """
        self.sae_extractor = sae_extractor

    def get_sae_directions(
        self,
        model_name: str,
        layer: int,
        feature_indices: List[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Recupera i vettori di direzione SAE per feature specifiche in un layer.

        Args:
            model_name: Nome del modello (usato per validazione, opzionale)
            layer: Indice del layer transformer
            feature_indices: Lista di indici di feature SAE da recuperare
            device: Device target (default: usa device del SAE)

        Returns:
            Tensor [len(feature_indices), hidden_dim] con le direzioni SAE.
            Ogni riga è la direzione del decoder SAE per quella feature.
        """
        device = device or torch.device("cpu")

        # Validazione layer
        layer_key = f"layer_{layer}"
        if layer_key not in self.sae_extractor.saes:
            raise ValueError(
                f"Layer {layer} not found in SAEFeatureExtractor. "
                f"Available: {list(self.sae_extractor.saes.keys())}"
            )

        sae = self.sae_extractor.saes[layer_key]

        # Validazione feature indices
        dict_size = sae.dict_size
        for idx in feature_indices:
            if idx < 0 or idx >= dict_size:
                raise ValueError(
                    f"Feature index {idx} out of bounds for layer {layer} "
                    f"(dict_size={dict_size})"
                )

        # Estrazione direzioni dal decoder SAE
        # Il decoder di LayerSparseAutoencoder ha shape [dict_size, input_dim]
        # Ogni riga è la direzione di ricostruzione per quella feature
        decoder_weights = sae.decoder.weight.data  # [dict_size, input_dim]

        # Seleziona solo le feature richieste
        directions = decoder_weights[feature_indices, :]  # [num_features, input_dim]

        # Normalizza ogni direzione (opzionale ma raccomandato)
        # Questo garantisce che l'alpha scaling sia interpretabile
        norms = torch.norm(directions, dim=1, keepdim=True)
        directions = directions / (norms + 1e-8)

        return directions.to(device)

    def get_layer_sae_summary(self, layer: int) -> dict:
        """
        Utility: ottieni metadata sul SAE di un layer.

        Returns:
            Dict con input_dim, dict_size, sparsity_lambda, etc.
        """
        layer_key = f"layer_{layer}"
        if layer_key not in self.sae_extractor.saes:
            raise ValueError(f"Layer {layer} not in SAE extractor")

        sae = self.sae_extractor.saes[layer_key]
        return {
            "layer": layer,
            "input_dim": sae.input_dim,
            "dict_size": sae.dict_size,
            "sparsity_lambda": sae.sparsity_lambda,
            "device": str(sae.decoder.weight.device),
        }
