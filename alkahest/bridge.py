"""
The graph bridge.

Defines what it means to be an otter-lineage graph.
Implement Graph to participate in the ecosystem.

bro-engine is the reference implementation.
"""

from typing import Protocol, runtime_checkable
from .state import Edge, OtterState


@runtime_checkable
class Graph(Protocol):
    """
    An otter-lineage persistent graph.

    Implement to_otter_edges and from_otter_edges to participate
    in the otter loop. The loop doesn't know about your storage,
    your schema, or your lifecycle — just edges and confidence.
    """

    def to_otter_edges(self, **kwargs) -> list[Edge]:
        """
        Load edges from the graph in otter-legible form.

        kwargs are graph-specific query parameters (confidence threshold,
        provenance filter, limit, etc.). The caller doesn't need to know
        the schema — just what it wants back.
        """
        ...

    def from_otter_edges(self, edges: list[Edge], via: str) -> None:
        """
        Write derived edges back to the graph.

        via: provenance — what session or process derived these edges.
        The graph decides how to store them (phase, confidence cap, etc.).
        """
        ...


def load_state(graph: Graph, **kwargs) -> OtterState:
    """
    Load a graph's edges into an OtterState.

    All edges go into set_of_support — the frontier the otter loop
    will work through. Pass query kwargs to filter what loads.
    """
    state = OtterState()
    for edge in graph.to_otter_edges(**kwargs):
        state.set_of_support.append(edge)
    return state


def save_derived(state: OtterState, graph: Graph, via: str) -> None:
    """
    Write derived edges from a completed otter run back to the graph.

    Reads from state.usable — edges that have been focused on and
    produced new items. Only writes Edge instances, not Clauses or Items.
    """
    derived = [e for e in state.usable if isinstance(e, Edge)]
    graph.from_otter_edges(derived, via)
