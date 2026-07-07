"""Execution layer: one interface, two implementations.

PaperExecutor simulates fills locally with durable state — the default and the
only mode that runs without explicit multi-step opt-in. BybitExecutor routes
real orders (testnet by default) behind the safety stack documented in
bybit.py. Both enforce the same portfolio caps and circuit breakers, so weeks
of paper behavior transfer to live behavior unchanged.
"""
from .base import CircuitBreakers, Executor, PositionState  # noqa: F401
from .paper import PaperExecutor  # noqa: F401
