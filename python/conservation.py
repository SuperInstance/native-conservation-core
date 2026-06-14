"""
conservation.py — Python ctypes bindings for native-conservation-core.

Provides a production-quality, type-safe interface to the C conservation
law library with NumPy array compatibility, dataclass results, a context-
managed ring buffer, and a built-in benchmark harness.

Usage
-----
    from conservation import (
        ConservationCore,
        SignalRingBuffer,
        TernarySignal,
        ConservationState,
    )

    core = ConservationCore()
    state = core.compute(signals)
    print(state.gamma, state.eta, state.C)

    with SignalRingBuffer(core, capacity=1024) as rb:
        rb.push(TernarySignal(valence=1, magnitude=0.5))
        sig = rb.pop()

    cancellation = core.monte_carlo(fleet_size=50, n_trials=10_000)

Author: SuperInstance
License: MIT
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    c_bool,
    c_double,
    c_int32,
    c_uint32,
    c_uint64,
    c_size_t,
    byref,
    cast,
    pointer,
    sizeof,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = [
    "ConservationState",
    "TernarySignal",
    "ConservationCore",
    "SignalRingBuffer",
    "ConservationError",
    "load_library",
]

# ─── Errors ──────────────────────────────────────────────────────────


class ConservationError(RuntimeError):
    """Raised when the native library encounters an error."""


# ─── ctypes Structure Definitions ────────────────────────────────────


class _CConservationState(Structure):
    """Mirror of C ``conservation_state`` — three doubles, 24 bytes."""

    _fields_ = [
        ("gamma", c_double),
        ("eta", c_double),
        ("C", c_double),
    ]
    _pack_ = 1  # tight packing: all doubles, naturally aligned


class _CTernarySignal(Structure):
    """Mirror of C ``ternary_signal`` — 32 bytes with alignment padding.

    C layout (System V x86-64):
        offset  0: int32_t  valence    (4 bytes)
        offset  4: <4 bytes padding to align double>
        offset  8: double   magnitude  (8 bytes)
        offset 16: uint64_t agent_id   (8 bytes)
        offset 24: uint64_t timestamp  (8 bytes)
        total   32 bytes
    """

    _fields_ = [
        ("valence", c_int32),
        ("_pad0", c_int32),  # explicit alignment padding
        ("magnitude", c_double),
        ("agent_id", c_uint64),
        ("timestamp", c_uint64),
    ]


# ─── Pythonic Dataclasses ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ConservationState:
    """Conservation law metrics: γ (coupling), η (value), C (capacity).

    The identity γ + η = C must hold (Shannon chain rule).
    """

    gamma: float
    eta: float
    C: float

    @property
    def residual(self) -> float:
        """Absolute deviation from the conservation identity (should be ~0)."""

        return abs(self.gamma + self.eta - self.C)

    @classmethod
    def _from_c(cls, cs: _CConservationState) -> ConservationState:
        return cls(gamma=cs.gamma, eta=cs.eta, C=cs.C)


@dataclass(slots=True)
class TernarySignal:
    """A single ternary contribution signal: valence ∈ {-1, 0, +1}.

    Attributes
    ----------
    valence : int
        Direction: -1 (adversarial), 0 (neutral), +1 (cooperative).
    magnitude : float
        Signal strength (default 1.0).
    agent_id : int
        Source agent identifier (default 0).
    timestamp : int
        Epoch-microsecond stamp (default 0).
    """

    valence: int = 0
    magnitude: float = 1.0
    agent_id: int = 0
    timestamp: int = 0

    def _to_c(self) -> _CTernarySignal:
        return _CTernarySignal(
            valence=self.valence,
            _pad0=0,
            magnitude=self.magnitude,
            agent_id=self.agent_id,
            timestamp=self.timestamp,
        )

    @classmethod
    def _from_c(cls, cs: _CTernarySignal) -> TernarySignal:
        return cls(
            valence=cs.valence,
            magnitude=cs.magnitude,
            agent_id=cs.agent_id,
            timestamp=cs.timestamp,
        )


# ─── Library Loader ──────────────────────────────────────────────────


def load_library(lib_path: Optional[Union[str, Path]] = None) -> CDLL:
    """Locate and load ``libconservation.so``.

    Search order when *lib_path* is ``None``:
      1. Directory containing this file (``python/``).
      2. Parent of this file (repo root).
      3. ``LD_LIBRARY_PATH`` / system loader.

    Parameters
    ----------
    lib_path
        Explicit path to the shared library. If provided, used directly.

    Returns
    -------
    CDLL
        Loaded shared library handle.

    Raises
    ------
    ConservationError
        If the library cannot be found or loaded.
    """

    here = Path(__file__).resolve().parent
    candidates: list[Path] = []

    if lib_path is not None:
        candidates.append(Path(lib_path))
    else:
        candidates.append(here / "libconservation.so")
        candidates.append(here.parent / "libconservation.so")

    tried: list[str] = []
    for c in candidates:
        tried.append(str(c))
        if c.is_file():
            try:
                lib = ctypes.CDLL(str(c))
                _configure_prototypes(lib)
                return lib
            except OSError as exc:
                raise ConservationError(
                    f"Found {c} but failed to load: {exc}"
                ) from exc

    # Fall back to system loader (may pick up via LD_LIBRARY_PATH).
    try:
        lib = ctypes.CDLL("libconservation.so")
        _configure_prototypes(lib)
        return lib
    except OSError:
        raise ConservationError(
            "libconservation.so not found. Tried: "
            + ", ".join(tried)
            + " (and system loader)"
        )


def _configure_prototypes(lib: CDLL) -> None:
    """Set argtypes/restype for every exported function."""

    # conservation_compute(const ternary_signal*, size_t) → conservation_state
    lib.conservation_compute.argtypes = [POINTER(_CTernarySignal), c_size_t]
    lib.conservation_compute.restype = _CConservationState

    # conservation_delta(size_t) → double
    lib.conservation_delta.argtypes = [c_size_t]
    lib.conservation_delta.restype = c_double

    # conservation_efficiency(size_t) → double
    lib.conservation_efficiency.argtypes = [c_size_t]
    lib.conservation_efficiency.restype = c_double

    # conservation_adversarial_threshold(size_t) → double
    lib.conservation_adversarial_threshold.argtypes = [c_size_t]
    lib.conservation_adversarial_threshold.restype = c_double

    # conservation_audit(const ternary_signal*, size_t, conservation_state*, double) → bool
    lib.conservation_audit.argtypes = [
        POINTER(_CTernarySignal),
        c_size_t,
        POINTER(_CConservationState),
        c_double,
    ]
    lib.conservation_audit.restype = c_bool

    # conservation_monte_carlo(size_t, size_t) → double
    lib.conservation_monte_carlo.argtypes = [c_size_t, c_size_t]
    lib.conservation_monte_carlo.restype = c_double

    # Ring buffer: all use opaque pointer
    rb_ptr = type(lib.ringbuf_create.restype)  # shorthand
    lib.ringbuf_create.argtypes = [c_uint32]
    lib.ringbuf_create.restype = ctypes.c_void_p

    lib.ringbuf_destroy.argtypes = [ctypes.c_void_p]
    lib.ringbuf_destroy.restype = None

    lib.ringbuf_push.argtypes = [ctypes.c_void_p, POINTER(_CTernarySignal)]
    lib.ringbuf_push.restype = c_bool

    lib.ringbuf_pop.argtypes = [ctypes.c_void_p, POINTER(_CTernarySignal)]
    lib.ringbuf_pop.restype = c_bool

    lib.ringbuf_count.argtypes = [ctypes.c_void_p]
    lib.ringbuf_count.restype = c_size_t


# ─── Signal Array Helpers ────────────────────────────────────────────


def _build_signal_array(
    signals: Union[
        Sequence[TernarySignal],
        "np.ndarray",
        Sequence[int],
    ],
) -> ctypes.Array:
    """Convert various input types to a ctypes array of _CTernarySignal.

    Accepted inputs:
      - List of TernarySignal objects
      - NumPy int8/int32 array (valences; magnitude defaults to 1.0)
      - List of ints (valences; magnitude defaults to 1.0)
    """

    n = len(signals)
    arr = (_CTernarySignal * n)()

    if _HAS_NUMPY and isinstance(signals, np.ndarray):
        valences = np.asarray(signals, dtype=np.int32)
        for i in range(n):
            arr[i] = _CTernarySignal(
                valence=int(valences[i]),
                _pad0=0,
                magnitude=1.0,
                agent_id=i,
                timestamp=0,
            )
    elif n > 0 and isinstance(signals[0], TernarySignal):
        for i, sig in enumerate(signals):
            arr[i] = sig._to_c()
    else:
        # Assume sequence of ints (valences).
        for i, v in enumerate(signals):
            arr[i] = _CTernarySignal(
                valence=int(v),
                _pad0=0,
                magnitude=1.0,
                agent_id=i,
                timestamp=0,
            )

    return arr


# ─── Main API ────────────────────────────────────────────────────────


class ConservationCore:
    """High-level Python interface to the native conservation library.

    Parameters
    ----------
    lib_path
        Optional explicit path to ``libconservation.so``.

    Examples
    --------
    >>> core = ConservationCore()
    >>> core.efficiency(50)
    0.863...
    >>> core.delta(100)
    0.085...
    """

    def __init__(self, lib_path: Optional[Union[str, Path]] = None) -> None:
        self._lib: CDLL = load_library(lib_path)
        self._closed = False

    # ── Conservation Law ──────────────────────────────────────────

    def compute(
        self,
        signals: Union[Sequence[TernarySignal], "np.ndarray", Sequence[int]],
    ) -> ConservationState:
        """Compute conservation metrics (γ, η, C) from an array of signals.

        Parameters
        ----------
        signals
            Signal collection: ``TernarySignal`` list, NumPy int array
            (valences), or plain int list.

        Returns
        -------
        ConservationState
            Frozen dataclass with ``gamma``, ``eta``, ``C``.
        """

        n = len(signals)
        if n == 0:
            return ConservationState(gamma=0.0, eta=1.0, C=1.0)
        arr = _build_signal_array(signals)
        cs: _CConservationState = self._lib.conservation_compute(arr, c_size_t(n))
        return ConservationState._from_c(cs)

    def audit(
        self,
        signals: Union[Sequence[TernarySignal], "np.ndarray"],
        epsilon: float = 1e-10,
    ) -> tuple[ConservationState, bool]:
        """Run a conservation audit: compute + identity check.

        Returns
        -------
        (state, passed)
            The computed ``ConservationState`` and whether
            ``|γ + η − C| < epsilon``.
        """

        n = len(signals)
        if n == 0:
            state = ConservationState(gamma=0.0, eta=1.0, C=1.0)
            return state, True
        arr = _build_signal_array(signals)
        c_state = _CConservationState()
        passed = self._lib.conservation_audit(
            arr, c_size_t(n), byref(c_state), c_double(epsilon)
        )
        return ConservationState._from_c(c_state), bool(passed)

    def delta(self, n: int) -> float:
        """Theoretical cancellation factor δ(n) = (1/√n)(1 − 3/(2n))."""

        if n < 2:
            return 1.0
        return float(self._lib.conservation_delta(c_size_t(n)))

    def efficiency(self, n: int) -> float:
        """Predicted fleet efficiency: 1 − δ(n).

        At n=50 this returns ≈0.863 (86.3 % cancellation).
        """

        if n < 2:
            return 0.0
        return float(self._lib.conservation_efficiency(c_size_t(n)))

    def adversarial_threshold(self, n: int) -> float:
        """Adversarial tolerance before the conservation law breaks."""

        return float(self._lib.conservation_adversarial_threshold(c_size_t(n)))

    def monte_carlo(self, fleet_size: int, n_trials: int) -> float:
        """Monte Carlo cancellation measurement (OpenMP-parallelised).

        Parameters
        ----------
        fleet_size
            Number of agents per trial.
        n_trials
            Number of simulation trials.

        Returns
        -------
        float
            Mean cancellation factor across all trials.
        """

        if fleet_size <= 0 or n_trials <= 0:
            return 0.0
        return float(
            self._lib.conservation_monte_carlo(
                c_size_t(fleet_size), c_size_t(n_trials)
            )
        )

    # ── Ring Buffer Factory ──────────────────────────────────────

    def ringbuf(self, capacity: int) -> SignalRingBuffer:
        """Create a lock-free ring buffer.

        ``capacity`` must be a power of two.
        """

        return SignalRingBuffer(self._lib, capacity)

    # ── Lifecycle ────────────────────────────────────────────────

    def close(self) -> None:
        """Release the library handle (the .so stays in process memory)."""

        self._closed = True

    def __repr__(self) -> str:
        return f"<ConservationCore closed={self._closed}>"


# ─── Ring Buffer (Context Manager) ───────────────────────────────────


class SignalRingBuffer:
    """Lock-free SPSC ring buffer for ternary signals.

    Use as a context manager for automatic cleanup:

        with core.ringbuf(1024) as rb:
            rb.push(TernarySignal(valence=1))
            sig = rb.pop()

    Parameters
    ----------
    lib
        Loaded CDLL handle.
    capacity
        Buffer capacity (must be power of 2).
    """

    def __init__(self, lib: CDLL, capacity: int) -> None:
        self._lib = lib
        self._capacity = capacity

        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError(
                f"capacity must be a positive power of 2, got {capacity}"
            )

        ptr = lib.ringbuf_create(c_uint32(capacity))
        if not ptr:
            raise ConservationError(
                f"ringbuf_create returned NULL for capacity={capacity}"
            )
        self._ptr: int = ptr  # store as int (void_p)

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, signal: TernarySignal) -> bool:
        """Push a signal. Returns ``False`` if the buffer is full."""

        c_sig = signal._to_c()
        return bool(self._lib.ringbuf_push(self._ptr, byref(c_sig)))

    def pop(self) -> Optional[TernarySignal]:
        """Pop a signal. Returns ``None`` if the buffer is empty."""

        c_sig = _CTernarySignal()
        ok = bool(self._lib.ringbuf_pop(self._ptr, byref(c_sig)))
        if not ok:
            return None
        return TernarySignal._from_c(c_sig)

    def count(self) -> int:
        """Approximate number of elements currently in the buffer."""

        return int(self._lib.ringbuf_count(self._ptr))

    def is_empty(self) -> bool:
        return self.count() == 0

    def is_full(self) -> bool:
        return self.count() >= self._capacity

    def __len__(self) -> int:
        return self.count()

    def __enter__(self) -> SignalRingBuffer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Free the underlying native buffer. Safe to call once."""

        ptr = getattr(self, "_ptr", None)
        if ptr is not None and ptr != 0:
            self._lib.ringbuf_destroy(ptr)
            self._ptr = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"<SignalRingBuffer capacity={self._capacity} "
            f"count={self.count()} closed={self._ptr == 0}>"
        )


# ─── Benchmark Harness ───────────────────────────────────────────────


def _pure_python_monte_carlo(fleet_size: int, n_trials: int) -> float:
    """Reference Monte Carlo in pure Python for benchmarking.

    Mirrors the C implementation's logic: random ternary valences,
    measures mean cancellation = 1 − |Σ valence| / fleet_size.
    """

    import random

    total = 0.0
    for _ in range(n_trials):
        raw_sum = sum(random.choice((-1, 0, 1)) for _ in range(fleet_size))
        cancellation = 1.0 - abs(raw_sum) / fleet_size
        total += cancellation
    return total / n_trials if n_trials > 0 else 0.0


def benchmark(
    fleet_sizes: Sequence[int] = (10, 50, 100, 500, 1000),
    n_trials: int = 5000,
    core: Optional[ConservationCore] = None,
) -> dict[str, list[Any]]:
    """Compare C vs. pure-Python Monte Carlo and check agreement.

    Returns a dict with columns suitable for printing or DataFrame use.
    """

    if core is None:
        core = ConservationCore()

    results: dict[str, list[Any]] = {
        "fleet_size": [],
        "c_cancel": [],
        "py_cancel": [],
        "theory": [],
        "error_pct": [],
        "c_ms": [],
        "py_ms": [],
        "speedup": [],
    }

    for n in fleet_sizes:
        # C version
        t0 = time.perf_counter()
        c_result = core.monte_carlo(n, n_trials)
        t1 = time.perf_counter()
        c_ms = (t1 - t0) * 1000.0

        # Python version
        t0 = time.perf_counter()
        py_result = _pure_python_monte_carlo(n, n_trials)
        t1 = time.perf_counter()
        py_ms = (t1 - t0) * 1000.0

        theory = core.efficiency(n)
        err = abs(c_result - theory) / theory * 100.0 if theory > 0 else 0.0

        results["fleet_size"].append(n)
        results["c_cancel"].append(c_result)
        results["py_cancel"].append(py_result)
        results["theory"].append(theory)
        results["error_pct"].append(err)
        results["c_ms"].append(c_ms)
        results["py_ms"].append(py_ms)
        results["speedup"].append(py_ms / c_ms if c_ms > 0 else float("inf"))

    return results


def format_benchmark_results(results: dict[str, list[Any]]) -> str:
    """Pretty-print benchmark results as a table."""

    header = (
        f"{'Fleet':>6}  {'C Cancel':>9}  {'Py Cancel':>9}  "
        f"{'Theory':>9}  {'Err%':>7}  {'C ms':>8}  {'Py ms':>8}  {'Speedup':>7}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for i in range(len(results["fleet_size"])):
        lines.append(
            f"{results['fleet_size'][i]:>6}  "
            f"{results['c_cancel'][i]:>9.4f}  "
            f"{results['py_cancel'][i]:>9.4f}  "
            f"{results['theory'][i]:>9.4f}  "
            f"{results['error_pct'][i]:>7.3f}  "
            f"{results['c_ms'][i]:>8.1f}  "
            f"{results['py_ms'][i]:>8.1f}  "
            f"{results['speedup'][i]:>7.1f}x"
        )
    return "\n".join(lines)


# ─── CLI entry ───────────────────────────────────────────────────────


def _main() -> None:
    core = ConservationCore()

    print("=== Conservation Core — Python Bindings ===\n")

    # Verify 86.3% at n=50
    eff50 = core.efficiency(50)
    print(f"Efficiency at n=50: {eff50:.4f} ({eff50*100:.1f}% cancellation)")
    assert abs(eff50 - 0.863) < 0.001, f"Expected ~0.863, got {eff50}"
    print("  ✓ Matches expected 86.3% cancellation\n")

    # Benchmark
    results = benchmark(core=core)
    print(format_benchmark_results(results))


if __name__ == "__main__":
    _main()
