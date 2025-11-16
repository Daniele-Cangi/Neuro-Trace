# neurotrace/visualization/activation_explorer.py

"""
Esplorazione interattiva di attivazioni con dimensionality reduction.

Supporta:
- PCA: Principal Component Analysis
- t-SNE: t-Distributed Stochastic Neighbor Embedding
- UMAP: Uniform Manifold Approximation and Projection (opzionale)
- 2D/3D scatter plots interattivi con Plotly
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimReductionMethod(Enum):
    """Metodi di dimensionality reduction disponibili."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


class ActivationExplorer:
    """
    Esplora attivazioni con dimensionality reduction e plot interattivi.

    Usa sklearn per PCA/t-SNE, opzionalmente UMAP,
    e Plotly per visualizzazioni 2D/3D.
    """

    def __init__(
        self,
        template: str = "plotly_dark",
        width: int = 1000,
        height: int = 800,
    ):
        """
        Args:
            template: Plotly template
            width: Larghezza plot
            height: Altezza plot
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly not installed. Install with: pip install plotly"
            )

        self.template = template
        self.width = width
        self.height = height

    def reduce_dimensions(
        self,
        activations: torch.Tensor | np.ndarray,
        method: DimReductionMethod = DimReductionMethod.PCA,
        n_components: int = 3,
        **kwargs,
    ) -> np.ndarray:
        """
        Riduce dimensionalità di attivazioni.

        Args:
            activations: [N, D] tensor o array
            method: Metodo reduction (PCA, t-SNE, UMAP)
            n_components: Numero componenti output (2 o 3)
            **kwargs: Parametri per algoritmo (e.g., perplexity per t-SNE)

        Returns:
            [N, n_components] array ridotto
        """
        # Converti a numpy
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()

        # Normalizza (opzionale ma raccomandato per t-SNE/UMAP)
        if method in [DimReductionMethod.TSNE, DimReductionMethod.UMAP]:
            from sklearn.preprocessing import StandardScaler
            activations = StandardScaler().fit_transform(activations)

        # Applica algoritmo
        if method == DimReductionMethod.PCA:
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == DimReductionMethod.TSNE:
            # Default perplexity=30, max_iter=1000
            default_kwargs = {"perplexity": 30, "max_iter": 1000, "random_state": 42}
            default_kwargs.update(kwargs)
            reducer = TSNE(n_components=n_components, **default_kwargs)
        elif method == DimReductionMethod.UMAP:
            if not UMAP_AVAILABLE:
                raise ImportError(
                    "umap-learn not installed. Install with: pip install umap-learn"
                )
            # Default n_neighbors=15, min_dist=0.1
            default_kwargs = {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42}
            default_kwargs.update(kwargs)
            reducer = umap.UMAP(n_components=n_components, **default_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        reduced = reducer.fit_transform(activations)
        return reduced

    def plot_activations_2d(
        self,
        activations: torch.Tensor | np.ndarray,
        labels: Optional[List[str | int]] = None,
        method: DimReductionMethod = DimReductionMethod.PCA,
        output_path: Optional[str | Path] = None,
        show: bool = False,
        title: Optional[str] = None,
        **reduction_kwargs,
    ) -> go.Figure:
        """
        Plot 2D di attivazioni ridotte.

        Args:
            activations: [N, D] attivazioni
            labels: Etichette per colore (opzionale)
            method: Metodo reduction
            output_path: Path HTML
            show: Mostra in browser
            title: Titolo plot
            **reduction_kwargs: Parametri per algoritmo reduction

        Returns:
            Figure
        """
        # Riduzione
        reduced = self.reduce_dimensions(
            activations, method=method, n_components=2, **reduction_kwargs
        )

        # Crea scatter plot
        if labels is not None:
            # Con colori per label
            fig = px.scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                color=labels,
                labels={"x": f"{method.value.upper()} 1", "y": f"{method.value.upper()} 2"},
                template=self.template,
            )
        else:
            # Senza colori
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode="markers",
                    marker=dict(size=8, opacity=0.7),
                )
            )
            fig.update_xaxes(title=f"{method.value.upper()} 1")
            fig.update_yaxes(title=f"{method.value.upper()} 2")

        # Layout
        if title is None:
            title = f"Activations 2D ({method.value.upper()})"

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title=title,
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_activations_3d(
        self,
        activations: torch.Tensor | np.ndarray,
        labels: Optional[List[str | int]] = None,
        method: DimReductionMethod = DimReductionMethod.PCA,
        output_path: Optional[str | Path] = None,
        show: bool = False,
        title: Optional[str] = None,
        **reduction_kwargs,
    ) -> go.Figure:
        """
        Plot 3D di attivazioni ridotte (interattivo, rotazione).

        Args:
            activations: [N, D] attivazioni
            labels: Etichette per colore (opzionale)
            method: Metodo reduction
            output_path: Path HTML
            show: Mostra in browser
            title: Titolo plot
            **reduction_kwargs: Parametri algoritmo

        Returns:
            Figure
        """
        # Riduzione
        reduced = self.reduce_dimensions(
            activations, method=method, n_components=3, **reduction_kwargs
        )

        # Crea scatter 3D
        if labels is not None:
            fig = px.scatter_3d(
                x=reduced[:, 0],
                y=reduced[:, 1],
                z=reduced[:, 2],
                color=labels,
                labels={
                    "x": f"{method.value.upper()} 1",
                    "y": f"{method.value.upper()} 2",
                    "z": f"{method.value.upper()} 3",
                },
                template=self.template,
            )
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    marker=dict(size=5, opacity=0.7),
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title=f"{method.value.upper()} 1",
                    yaxis_title=f"{method.value.upper()} 2",
                    zaxis_title=f"{method.value.upper()} 3",
                )
            )

        # Layout
        if title is None:
            title = f"Activations 3D ({method.value.upper()})"

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title=title,
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_variance_explained(
        self,
        activations: torch.Tensor | np.ndarray,
        max_components: int = 50,
        output_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Plot varianza spiegata da PCA components.

        Args:
            activations: [N, D] attivazioni
            max_components: Numero massimo componenti da testare
            output_path: Path HTML
            show: Mostra in browser

        Returns:
            Figure
        """
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()

        # PCA con max_components
        pca = PCA(n_components=min(max_components, activations.shape[1]))
        pca.fit(activations)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # Plot
        fig = go.Figure()

        # Varianza per componente
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(explained_variance) + 1)),
                y=explained_variance,
                name="Per Component",
                marker=dict(color="blue", opacity=0.7),
            )
        )

        # Varianza cumulativa
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_variance) + 1)),
                y=cumulative_variance,
                name="Cumulative",
                mode="lines+markers",
                line=dict(color="red", width=3),
                yaxis="y2",
            )
        )

        # Linea a 95%
        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="green",
            annotation_text="95%",
            yref="y2",
        )

        # Layout con doppio asse Y
        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title="PCA Variance Explained",
            xaxis_title="Principal Component",
            yaxis=dict(title="Variance Ratio (per component)"),
            yaxis2=dict(
                title="Cumulative Variance Ratio",
                overlaying="y",
                side="right",
                range=[0, 1],
            ),
            legend=dict(x=0.7, y=0.2),
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig

    def plot_activation_heatmap(
        self,
        activations: torch.Tensor | np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        output_path: Optional[str | Path] = None,
        show: bool = False,
        title: str = "Activation Heatmap",
    ) -> go.Figure:
        """
        Heatmap di attivazioni.

        Args:
            activations: [N, D] attivazioni
            row_labels: Etichette righe (esempi)
            col_labels: Etichette colonne (features)
            output_path: Path HTML
            show: Mostra in browser
            title: Titolo

        Returns:
            Figure
        """
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()

        # Limita dimensioni per visualizzazione
        max_rows, max_cols = 100, 100
        if activations.shape[0] > max_rows:
            # Subsample righe
            indices = np.linspace(0, activations.shape[0] - 1, max_rows, dtype=int)
            activations = activations[indices]
            if row_labels:
                row_labels = [row_labels[i] for i in indices]

        if activations.shape[1] > max_cols:
            # Subsample colonne
            indices = np.linspace(0, activations.shape[1] - 1, max_cols, dtype=int)
            activations = activations[:, indices]
            if col_labels:
                col_labels = [col_labels[i] for i in indices]

        # Crea heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=activations,
                x=col_labels if col_labels else None,
                y=row_labels if row_labels else None,
                colorscale="Viridis",
                colorbar=dict(title="Activation"),
            )
        )

        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title=title,
            xaxis_title="Features",
            yaxis_title="Examples",
        )

        if output_path:
            fig.write_html(str(output_path))
        if show:
            fig.show()

        return fig
