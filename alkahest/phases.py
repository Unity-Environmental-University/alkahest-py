"""
Three states of matter for type dissolution.

Volatile (gas)  — always re-precipitates. Decohere overwrites every run.
Fluid (liquid)  — precipitates once, re-flows only if type-checking breaks.
Salt (solid)    — precipitates once, replaces itself in source. Consumed.

Use as base classes. Pinned fields survive dissolution:

    class CourseData(Fluid):
        id: int
        name: str

    class ParseResult(Volatile):
        pass  # total dissolution — no pins

    class CaseRecord(Salt):
        id: str
        status: str
"""

from typing import Any, ClassVar


class _PhaseMeta(type):
    """Metaclass that marks classes with their dissolution phase."""

    _phase: str

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: Any):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        for base in bases:
            if hasattr(base, "__phase__"):
                cls.__phase__ = base.__phase__  # type: ignore
                break
        return cls


class Volatile(metaclass=_PhaseMeta):
    """Gas — always re-precipitates."""
    __phase__: ClassVar[str] = "volatile"


class Fluid(metaclass=_PhaseMeta):
    """Liquid — stable until broken."""
    __phase__: ClassVar[str] = "fluid"


class Salt(metaclass=_PhaseMeta):
    """Solid — consumed in the reaction."""
    __phase__: ClassVar[str] = "salt"
