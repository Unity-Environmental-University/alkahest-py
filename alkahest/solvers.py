"""
Solvers for large-scale causal graphs.

Three strategies, chosen by scale:

  Prime encoding   — exact, integer arithmetic, breaks down when Gödel numbers
                     grow too large for the problem.

  Symbolic         — gauge-deferred. Computes causal geometry (Gram matrix,
                     path counts, overlaps) without ever assigning primes.
                     Frame-invariant. Scales to arbitrary DAG size.

  Complex amplitude — when primes are too large, switch to wave mechanics.
                     Each prime becomes a frequency. Causality becomes
                     interference. The Euler product on the critical line
                     s = 1/2 + it gives amplitude where integer divisibility
                     would overflow.

Reference frames
----------------
A Frame is a gauge choice: the same causal structure seen through a specific
prime assignment. The Gram matrix (causal geometry) is invariant across frames.
Only the wave mechanics (amplitudes, Born probabilities) are frame-dependent.

  sym = SymbolicEncoding(dag)        # frame-independent
  frame_a = Frame(sym)               # default gauge
  frame_b = Frame(sym, primes={...}) # different observer, same geometry

Extracted from otter-centaur (the prototype). Proof machinery omitted.
"""

import math
import cmath
from dataclasses import dataclass, field
from typing import Optional


# =====================================================================
# Causal DAG
# =====================================================================

@dataclass
class CausalEvent:
    """An event in a causal DAG."""
    name: str
    causes: frozenset = field(default_factory=frozenset)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, CausalEvent) and self.name == other.name

    def __repr__(self):
        if self.causes:
            return f"CausalEvent({self.name!r}, causes={set(self.causes)})"
        return f"CausalEvent({self.name!r})"


@dataclass
class CausalDAG:
    """A directed acyclic graph of causal events."""
    events: dict = field(default_factory=dict)

    def add(self, name: str, causes: list = None) -> 'CausalDAG':
        self.events[name] = CausalEvent(name=name, causes=frozenset(causes or []))
        return self

    def roots(self) -> list:
        return [e for e in self.events.values() if not e.causes]

    def children(self, name: str) -> list:
        return [e for e in self.events.values() if name in e.causes]

    def ancestors(self, name: str) -> set:
        result = set()
        frontier = set(self.events[name].causes)
        while frontier:
            n = frontier.pop()
            if n not in result:
                result.add(n)
                frontier |= self.events[n].causes
        return result

    def topological_order(self) -> list:
        """Kahn's algorithm."""
        in_degree = {name: len(e.causes) for name, e in self.events.items()}
        queue = [name for name, d in in_degree.items() if d == 0]
        order = []
        while queue:
            name = queue.pop(0)
            order.append(name)
            for child in self.children(name):
                in_degree[child.name] -= 1
                if in_degree[child.name] == 0:
                    queue.append(child.name)
        return order


def subdag(dag: CausalDAG, event_names: set) -> CausalDAG:
    """Extract a sub-DAG containing only the specified events."""
    sub = CausalDAG()
    for name in event_names:
        if name in dag.events:
            causes = dag.events[name].causes & event_names
            sub.events[name] = CausalEvent(name=name, causes=frozenset(causes))
    return sub


# =====================================================================
# Symbolic encoding — frame-independent
# =====================================================================

class SymbolicEncoding:
    """
    Causal geometry without prime assignment.

    Computes path counts, Gram matrix, and overlaps from DAG structure
    alone. No primes needed. Frame-invariant: different gauge choices
    produce the same geometry.

    Fix the gauge (via Frame) only when you need wave mechanics.
    """

    def __init__(self, dag: CausalDAG):
        self.dag = dag
        self._path_cache: dict = {}
        self._gram_cache: Optional[dict] = None

    def path_count(self, ancestor: str, descendant: str) -> int:
        """Count directed paths from ancestor to descendant."""
        key = (ancestor, descendant)
        if key not in self._path_cache:
            if ancestor == descendant:
                self._path_cache[key] = 1
            else:
                total = 0
                for child in self.dag.children(ancestor):
                    total += self.path_count(child.name, descendant)
                self._path_cache[key] = total
        return self._path_cache[key]

    def gram_matrix(self) -> dict:
        """
        G(X, Y) = Σ_E paths(E→X) · paths(E→Y)

        Frame-invariant. The same regardless of prime assignment.
        """
        if self._gram_cache is not None:
            return self._gram_cache

        names = self.dag.topological_order()
        matrix = {}
        for x in names:
            for y in names:
                g = sum(
                    self.path_count(e, x) * self.path_count(e, y)
                    for e in names
                )
                matrix[(x, y)] = g

        self._gram_cache = {
            'matrix': matrix,
            'names': names,
            'gleason_applies': len(names) >= 3,
        }
        return self._gram_cache

    def overlap(self, a: str, b: str) -> float:
        """Causal overlap: G(a,b) / (||a|| · ||b||). Frame-invariant."""
        G = self.gram_matrix()['matrix']
        ip = G[(a, b)]
        na = math.sqrt(G[(a, a)])
        nb = math.sqrt(G[(b, b)])
        if na == 0 or nb == 0:
            return 0.0
        return ip / (na * nb)


# =====================================================================
# Reference frames
# =====================================================================

class Frame:
    """
    A reference frame: a specific gauge choice over a symbolic encoding.

    The causal geometry (overlap, Gram matrix) is the same in all frames.
    The wave mechanics (amplitude, Born probability) are frame-dependent.

    Usage:
        sym = SymbolicEncoding(dag)
        frame = Frame(sym)                        # default primes
        frame = Frame(sym, primes={"a": 2, ...})  # custom gauge
        amp = frame.amplitude("a", t=14.134)      # complex amplitude
    """

    def __init__(self, encoding: SymbolicEncoding, primes: Optional[dict] = None):
        self.encoding = encoding
        self.primes = primes or self._default_primes()
        self._gn: dict = {}
        self._build()

    def _default_primes(self) -> dict:
        order = self.encoding.dag.topological_order()
        p_list = _first_n_primes(len(order))
        return {name: p_list[i] for i, name in enumerate(order)}

    def _build(self):
        for name in self.encoding.dag.topological_order():
            gn = self.primes[name]
            for cause in self.encoding.dag.events[name].causes:
                gn *= self._gn[cause]
            self._gn[name] = gn

    def godel_number(self, name: str) -> int:
        return self._gn[name]

    def causes(self, a: str, b: str) -> bool:
        """A causes B iff gn(A) divides gn(B)."""
        return self._gn[b] % self._gn[a] == 0

    def amplitude(self, name: str, t: float) -> complex:
        """
        Complex amplitude for this event at parameter t.

        ψ(E, t) = p_E^{-(1/2 + it)} = p_E^{-1/2} · e^{-it·ln(p_E)}

        The fresh prime p_E contributes one oscillator at frequency ln(p_E).
        Interference between amplitudes is where the quantum solve lives.
        """
        p = self.primes[name]
        return p ** (-(0.5 + 1j * t))

    def born_probability(self, name: str, t: float, names: list) -> float:
        """
        Born probability for this event relative to all events at t.

        P(E) = |ψ(E,t)|² / Σ_F |ψ(F,t)|²
        """
        amp = self.amplitude(name, t)
        total = sum(abs(self.amplitude(n, t)) ** 2 for n in names)
        if total == 0:
            return 0.0
        return abs(amp) ** 2 / total


# =====================================================================
# Complex amplitude solver — for when primes are too large
# =====================================================================

def euler_product_complex(s: complex, num_primes: int = 100) -> complex:
    """
    ∏_{p≤P} 1/(1 - p^{-s}) for complex s.

    On the critical line s = 1/2 + it, approximates ζ(s).
    Use when prime factorization overflows: treat causality as
    wave interference rather than divisibility.

    Dips in |euler_product_complex(0.5 + it)| mark nodes where
    causal threads destructively interfere.
    """
    primes = _first_n_primes(num_primes)
    product = complex(1.0, 0.0)
    for p in primes:
        product *= 1.0 / (1.0 - p ** (-s))
    return product


def interference(frame_a: Frame, frame_b: Frame, name: str, t: float) -> float:
    """
    Interference between two frames' amplitudes for the same event.

    Re(ψ_A · conj(ψ_B)) — positive means frames agree, negative means
    they destructively interfere. Useful for detecting when two reference
    frames give inconsistent readings of the same causal structure.
    """
    amp_a = frame_a.amplitude(name, t)
    amp_b = frame_b.amplitude(name, t)
    return (amp_a * amp_b.conjugate()).real


# =====================================================================
# Convergent proof — confidence as a converging sequence
# =====================================================================

@dataclass
class ConvergentProof:
    """
    A sequence of confidence values over an expanding evidence set.

    Each step adds one more piece of evidence and records the resulting
    confidence. The sequence traces the curve of approach toward truth.

    Use to know when a solve has settled — is_cauchy() is the certificate.
    """
    conclusion: str
    steps: list  # list of (evidence_label, confidence | None)

    @property
    def confidences(self) -> list:
        return [(label, c) for label, c in self.steps if c is not None]

    @property
    def limit(self) -> Optional[float]:
        vals = [c for _, c in self.confidences]
        if len(vals) < 2:
            return vals[-1] if vals else None
        deltas = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        if len(deltas) < 2 or deltas[-2] == 0:
            return vals[-1]
        ratio = deltas[-1] / deltas[-2]
        if abs(ratio) >= 1.0:
            return vals[-1]
        remaining = deltas[-1] * ratio / (1.0 - ratio)
        return min(1.0, vals[-1] + remaining)

    def is_cauchy(self, epsilon: float = 0.01) -> bool:
        """Practical convergence certificate. Do not claim more than the evidence supports."""
        vals = [c for _, c in self.confidences]
        if len(vals) < 4:
            return False
        tail = vals[len(vals) // 2:]
        return max(tail) - min(tail) < epsilon

    def __repr__(self):
        lim = self.limit
        lim_str = f"{lim:.4f}" if lim is not None else "?"
        n = len(self.confidences)
        return f"ConvergentProof({self.conclusion!r}, {n} steps, limit≈{lim_str})"


# =====================================================================
# Utilities
# =====================================================================

def _first_n_primes(n: int) -> list:
    """Sieve of Eratosthenes for the first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes
