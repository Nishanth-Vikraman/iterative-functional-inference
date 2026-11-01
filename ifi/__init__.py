"""Iterative Functional Inference (IFI)."""

from .config import ExperimentConfig
from .functions import ComposedFunction, FunctionSystem
from .operators import FunctionalOperator

__all__ = ["ComposedFunction", "ExperimentConfig", "FunctionalOperator", "FunctionSystem"]
__version__ = "0.1.0"

