"""
Alkahest — type dissolution primitives and the Otter engine.

Three states of matter:
  Volatile (gas)  — always re-precipitates
  Fluid (liquid)  — stable until broken
  Salt (solid)    — consumed, becomes concrete
"""

from alkahest.phases import Volatile, Fluid, Salt
from alkahest.engine import otter_step
from alkahest.state import Item, Edge, Clause, OtterState
from alkahest.bridge import Graph, load_state, save_derived
from alkahest.solvers import (
    CausalDAG, CausalEvent, SymbolicEncoding, Frame,
    subdag, euler_product_complex, interference, ConvergentProof,
)
from alkahest.confidence import Confidence, BoundedPredictionSet, TARGET

__all__ = [
    "Volatile", "Fluid", "Salt",
    "Item", "Edge", "Clause", "OtterState",
    "otter_step",
    "Graph", "load_state", "save_derived",
    "CausalDAG", "CausalEvent", "SymbolicEncoding", "Frame",
    "subdag", "euler_product_complex", "interference", "ConvergentProof",
    "Confidence", "BoundedPredictionSet", "TARGET",
]
