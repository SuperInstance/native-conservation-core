"""
native-conservation-core — Python bindings package.

A ctypes-based binding layer for the native conservation law C library.
Provides conservation metrics (γ + η = C), Monte Carlo verification,
and a lock-free ring buffer with a Pythonic, type-safe API.

Quick start
-----------
    from conservation import ConservationCore

    core = ConservationCore()
    print(core.efficiency(50))        # ~0.863
    print(core.delta(100))            # ~0.085
    cancellation = core.monte_carlo(50, 10_000)
"""

from .conservation import (
    ConservationCore,
    ConservationError,
    ConservationState,
    SignalRingBuffer,
    TernarySignal,
    benchmark,
    format_benchmark_results,
    load_library,
)

__version__ = "1.0.0"
__author__ = "SuperInstance"
__license__ = "MIT"

__all__ = [
    "ConservationCore",
    "ConservationError",
    "ConservationState",
    "SignalRingBuffer",
    "TernarySignal",
    "benchmark",
    "format_benchmark_results",
    "load_library",
]
