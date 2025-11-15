"""
SPROUT: Self-organizing Progressive Routing with Organic Unified Trees

A dynamic neural network architecture that grows tree-like structures
based on input compatibility.
"""

from .model import SPROUT
from .node import Node
from .router import Router
from .language_model import SproutLanguageModel

__version__ = "0.1.0"

__all__ = ["SPROUT", "Node", "Router", "SproutLanguageModel"]
