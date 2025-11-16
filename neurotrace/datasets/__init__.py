# neurotrace/datasets/__init__.py

"""
NeuroTrace Datasets - Production-grade task generators.

Questo modulo fornisce:
- IOIDatasetGenerator: IOI task con variability estrema
- UniversalTaskGenerator: 50+ task templates
- TaskExample: formato unificato per tutti i task
"""

from .ioi_generator import IOIDatasetGenerator, IOIExample, IOITemplate
from .task_generator import UniversalTaskGenerator, TaskExample, TaskCategory

__all__ = [
    "IOIDatasetGenerator",
    "IOIExample",
    "IOITemplate",
    "UniversalTaskGenerator",
    "TaskExample",
    "TaskCategory",
]
