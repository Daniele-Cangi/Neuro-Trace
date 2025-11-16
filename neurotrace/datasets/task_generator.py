# neurotrace/datasets/task_generator.py

"""
Universal Task Generator - 50+ task templates (stub for now).

Will be expanded in future iterations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class TaskCategory(Enum):
    """Task categories."""
    IOI = "ioi"
    FACTUAL = "factual"
    ARITHMETIC = "arithmetic"
    SYNTAX = "syntax"


@dataclass
class TaskExample:
    """Generic task example."""
    text: str
    correct_answer: str
    incorrect_answer: str
    category: TaskCategory


class UniversalTaskGenerator:
    """Universal task generator (placeholder)."""

    def __init__(self):
        pass

    def generate(self, num_examples: int = 100) -> List[TaskExample]:
        """Generate task examples."""
        return []
