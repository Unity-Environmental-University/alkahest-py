"""
Confidence accumulation — Bayesian product of independent observations.

Each observation nudges confidence toward 1.0 but never reaches it.
Confidence = 1 - Π(1 - cᵢ) where cᵢ is each observation's weight.

A node is "collapsed" when confidence exceeds a threshold —
the definite-state fast path. No need to recompute, just return
the cached prediction.
"""

import math

# Never quite reach 1.0
CONFIDENCE_CEILING = 0.99

# 1 - 1/e: the universal thermostat
TARGET = 1 - 1 / math.e


class Confidence:
    """Running confidence accumulator."""
    __slots__ = ['doubt', 'observations']

    def __init__(self):
        self.doubt = 1.0  # Π(1 - cᵢ), starts at total uncertainty
        self.observations = 0

    @property
    def value(self) -> float:
        return min(1.0 - self.doubt, CONFIDENCE_CEILING)

    @property
    def collapsed(self) -> bool:
        return self.value >= TARGET

    def observe(self, weight: float = 0.1) -> float:
        """Accumulate one observation. Returns new confidence."""
        self.doubt *= (1.0 - min(weight, CONFIDENCE_CEILING))
        self.observations += 1
        return self.value

    def decay(self, factor: float = 0.01):
        """Thermal decay — inject a little doubt back in."""
        self.doubt = min(1.0, self.doubt + factor * (1.0 - self.doubt))

    def __repr__(self):
        return f"Confidence({self.value:.4f}, n={self.observations})"


class BoundedPredictionSet:
    """
    Fixed-size prediction set that packs preferring higher-information entries.

    Entries compete for slots. An entry's "information" is its length
    (longer chunks = more bits per slot) weighted by confidence.

    When a new entry wants in and the set is full, it displaces the
    lowest-scoring entry — but only if it scores higher.
    """
    __slots__ = ['capacity', 'entries', '_scores']

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: dict[str, float] = {}  # symbol -> score
        self._scores: list[tuple[float, str]] = []  # sorted ascending

    def score(self, symbol: str, confidence: float) -> float:
        """Score = len * confidence. Longer + more confident = better."""
        return len(symbol) * max(confidence, 0.01)

    def offer(self, symbol: str, confidence: float) -> bool:
        """Offer a symbol for inclusion. Returns True if accepted."""
        sc = self.score(symbol, confidence)

        if symbol in self.entries:
            # Update existing
            self.entries[symbol] = sc
            self._rebuild_scores()
            return True

        if len(self.entries) < self.capacity:
            # Room available
            self.entries[symbol] = sc
            self._rebuild_scores()
            return True

        # Full — check if we beat the worst
        if self._scores and sc > self._scores[0][0]:
            # Displace the weakest
            _, weakest = self._scores[0]
            del self.entries[weakest]
            self.entries[symbol] = sc
            self._rebuild_scores()
            return True

        return False

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def _rebuild_scores(self):
        self._scores = sorted((sc, sym) for sym, sc in self.entries.items())
