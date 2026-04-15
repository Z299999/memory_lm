"""Simplex Memory Network implementation.

This package provides modules for building and training simplex memory networks:
- lattice: V_{n,m} lattice generation
- potential: Potential function H(α) and edge orientation
- graph: SimplexMemoryGraph class
- config: Configuration dataclass
- model: PyTorch nn.Module
- train: Training loop
- data: Target functions
- plot: Visualization
"""

from src.lattice import generate_lattice, cardinality
from src.potential import vertex_potentials, node_potential
from src.graph import SimplexMemoryGraph
from src.config import Config, load_config_from_yaml
from src.model import SMNNetwork

__all__ = [
    "generate_lattice",
    "cardinality",
    "vertex_potentials",
    "node_potential",
    "SimplexMemoryGraph",
    "Config",
    "load_config_from_yaml",
    "SMNNetwork",
]
