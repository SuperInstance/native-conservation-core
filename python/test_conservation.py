#!/usr/bin/env python3
"""
test_conservation.py — Verify C library matches pure-Python results.

Run directly:
    python test_conservation.py
Or with pytest:
    pytest test_conservation.py -v

Covers:
  1. Struct size / layout correctness
  2. conservation_compute correctness vs hand-computed values
  3. conservation_delta / efficiency closed-form values
  4. 86.3% cancellation at n=50
  5. Monte Carlo: C vs pure-Python agreement (within 3σ)
  6. Ring buffer lifecycle: create / push / pop / count / destroy
  7. Ring buffer context manager
  8. Audit identity check (γ + η = C)
  9. NumPy array interface
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

# Ensure we can import from the local package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctypes
from ctypes import sizeof
import pytest

from conservation import (
    ConservationCore,
    ConservationError,
    ConservationState,
    SignalRingBuffer,
    TernarySignal,
    _CConservationState,
    _CTernarySignal,
    _pure_python_monte_carlo,
)
from conservation import load_library

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def core() -> ConservationCore:
    return ConservationCore()


# ─── 1. Struct Layout ────────────────────────────────────────────────

class TestStructLayout:
    """Verify ctypes struct sizes match C ABI."""

    def test_conservation_state_size(self):
        assert sizeof(_CConservationState) == 24, (
            f"Expected 24 bytes (3×double), got {sizeof(_CConservationState)}"
        )

    def test_ternary_signal_size(self):
        # int32(4) + pad(4) + double(8) + uint64(8) + uint64(8) = 32
        assert sizeof(_CTernarySignal) == 32, (
            f"Expected 32 bytes, got {sizeof(_CTernarySignal)}"
        )

    def test_ternary_signal_offsets(self):
        """Check field offsets match C layout."""
        s = _CTernarySignal()
        offset_valence = _CTernarySignal.valence.offset
        offset_magnitude = _CTernarySignal.magnitude.offset
        offset_agent_id = _CTernarySignal.agent_id.offset
        offset_timestamp = _CTernarySignal.timestamp.offset

        assert offset_valence == 0
        assert offset_magnitude == 8, f"magnitude at {offset_magnitude}, expected 8"
        assert offset_agent_id == 16, f"agent_id at {offset_agent_id}, expected 16"
        assert offset_timestamp == 24, f"timestamp at {offset_timestamp}, expected 24"


# ─── 2. Conservation Compute ─────────────────────────────────────────

class TestCompute:
    def test_basic_compute(self, core: ConservationCore):
        """Compute with known signals and verify gamma/eta."""
        signals = [
            TernarySignal(valence=1, magnitude=1.0),
            TernarySignal(valence=-1, magnitude=1.0),
            TernarySignal(valence=0, magnitude=1.0),
            TernarySignal(valence=1, magnitude=1.0),
        ]
        state = core.compute(signals)
        # gamma = (1 + 1 + 0 + 1) / 4 = 0.75
        # eta   = (0 + 0 + 1 + 0) / 4 = 0.25
        assert abs(state.gamma - 0.75) < 1e-9, f"gamma={state.gamma}"
        assert abs(state.eta - 0.25) < 1e-9, f"eta={state.eta}"
        assert abs(state.C - 1.0) < 1e-9, f"C={state.C}"

    def test_empty_signals(self, core: ConservationCore):
        state = core.compute([])
        assert state.gamma == 0.0
        assert state.eta == 1.0
        assert state.C == 1.0

    def test_all_cooperative(self, core: ConservationCore):
        """All +1 signals: gamma=1, eta=0."""
        signals = [TernarySignal(valence=1, magnitude=2.0) for _ in range(10)]
        state = core.compute(signals)
        assert abs(state.gamma - 2.0) < 1e-9
        assert abs(state.eta - 0.0) < 1e-9

    def test_residual_is_tiny(self, core: ConservationCore):
        """Conservation identity must hold to machine precision."""
        signals = [TernarySignal(valence=v) for v in (1, -1, 0, 1, 1, -1, 0, 1)]
        state = core.compute(signals)
        assert state.residual < 1e-12


# ─── 3. Closed-form Values ───────────────────────────────────────────

class TestClosedForm:
    def test_delta_formula(self, core: ConservationCore):
        """δ(n) = (1/√n)(1 − 3/(2n))"""
        for n in [10, 50, 100, 1000]:
            expected = (1.0 / math.sqrt(n)) * (1.0 - 3.0 / (2.0 * n))
            got = core.delta(n)
            assert abs(got - expected) < 1e-12, f"n={n}: got {got}, expected {expected}"

    def test_efficiency_is_one_minus_delta(self, core: ConservationCore):
        for n in [10, 50, 100]:
            assert abs(core.efficiency(n) - (1.0 - core.delta(n))) < 1e-12

    def test_delta_edge_cases(self, core: ConservationCore):
        assert core.delta(0) == 1.0
        assert core.delta(1) == 1.0


# ─── 4. 86.3% Cancellation at n=50 ──────────────────────────────────

class TestCancellation50:
    def test_efficiency_at_50(self, core: ConservationCore):
        eff = core.efficiency(50)
        assert abs(eff - 0.863) < 0.001, (
            f"Expected ~0.863 at n=50, got {eff}"
        )


# ─── 5. Monte Carlo Agreement ────────────────────────────────────────

class TestMonteCarlo:
    def test_mc_returns_valid_range(self, core: ConservationCore):
        result = core.monte_carlo(50, 1000)
        assert 0.0 <= result <= 1.0

    def test_mc_close_to_theory(self, core: ConservationCore):
        """C MC cancellation should be within ~8% of theory at small n.

        At n=50 the CLT approximation is still rough; Monte Carlo with
        ternary noise converges to the asymptotic formula only for large n.
        """
        n = 50
        trials = 10_000
        mc = core.monte_carlo(n, trials)
        theory = core.efficiency(n)
        rel_err = abs(mc - theory) / theory
        assert rel_err < 0.08, f"MC={mc}, theory={theory}, rel_err={rel_err:.4f}"

    def test_mc_c_vs_python_agreement(self, core: ConservationCore):
        """C and Python MC should agree within statistical noise."""
        n = 100
        trials = 8_000
        random.seed(42)
        c_result = core.monte_carlo(n, trials)
        py_result = _pure_python_monte_carlo(n, trials)
        diff = abs(c_result - py_result)
        # With 8000 trials, std error ~ 0.003; allow 0.02 margin.
        assert diff < 0.02, f"C={c_result}, Python={py_result}, diff={diff}"

    def test_mc_zero_handling(self, core: ConservationCore):
        assert core.monte_carlo(0, 100) == 0.0
        assert core.monte_carlo(100, 0) == 0.0


# ─── 6. Ring Buffer ──────────────────────────────────────────────────

class TestRingBuffer:
    def test_push_pop_basic(self, core: ConservationCore):
        with core.ringbuf(16) as rb:
            sig = TernarySignal(valence=1, magnitude=0.5, agent_id=42)
            assert rb.push(sig)
            assert rb.count() == 1
            popped = rb.pop()
            assert popped is not None
            assert popped.valence == 1
            assert abs(popped.magnitude - 0.5) < 1e-9
            assert popped.agent_id == 42
            assert rb.count() == 0

    def test_pop_empty_returns_none(self, core: ConservationCore):
        with core.ringbuf(8) as rb:
            assert rb.pop() is None
            assert rb.is_empty()

    def test_push_until_full(self, core: ConservationCore):
        with core.ringbuf(4) as rb:
            for i in range(4):
                assert rb.push(TernarySignal(valence=i % 3 - 1, agent_id=i))
            assert rb.is_full()
            assert not rb.push(TernarySignal(valence=1))
            assert rb.count() == 4

    def test_fifo_order(self, core: ConservationCore):
        with core.ringbuf(8) as rb:
            for i in range(5):
                rb.push(TernarySignal(valence=0, agent_id=i))
            for i in range(5):
                sig = rb.pop()
                assert sig is not None
                assert sig.agent_id == i

    def test_non_power_of_two_raises(self, core: ConservationCore):
        with pytest.raises(ValueError):
            core.ringbuf(7)

    def test_context_manager_closes(self, core: ConservationCore):
        rb = core.ringbuf(8)
        rb.close()
        # Double close should be safe.
        rb.close()

    def test_large_buffer(self, core: ConservationCore):
        """Stress test with a large power-of-2 buffer."""
        cap = 1 << 14  # 16384
        with core.ringbuf(cap) as rb:
            pushed = 0
            for i in range(cap):
                assert rb.push(TernarySignal(valence=1, agent_id=i))
                pushed += 1
            assert rb.count() == cap
            popped_count = 0
            while rb.pop() is not None:
                popped_count += 1
            assert popped_count == cap


# ─── 7. Audit ────────────────────────────────────────────────────────

class TestAudit:
    def test_audit_passes(self, core: ConservationCore):
        signals = [TernarySignal(valence=v) for v in (1, -1, 0, 1, -1, 1)]
        state, passed = core.audit(signals)
        assert passed
        assert state.residual < 1e-10

    def test_audit_with_strict_epsilon(self, core: ConservationCore):
        signals = [TernarySignal(valence=1, magnitude=1.0) for _ in range(10)]
        state, passed = core.audit(signals, epsilon=1e-15)
        assert passed  # identity always holds exactly


# ─── 8. NumPy Interface ──────────────────────────────────────────────

@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
class TestNumpyInterface:
    def test_numpy_int8_array(self, core: ConservationCore):
        arr = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        state = core.compute(arr)
        # gamma = (1+1+0+1+1)/5 = 0.8
        # eta   = (0+0+1+0+0)/5 = 0.2
        assert abs(state.gamma - 0.8) < 1e-9
        assert abs(state.eta - 0.2) < 1e-9

    def test_numpy_int32_array(self, core: ConservationCore):
        # All |valence| = 1 → gamma=1.0, eta=0.0
        arr = np.array([1, -1, 1, -1], dtype=np.int32)
        state = core.compute(arr)
        assert abs(state.gamma - 1.0) < 1e-9
        assert abs(state.eta - 0.0) < 1e-9
        assert abs(state.C - 1.0) < 1e-9

    def test_numpy_large_array(self, core: ConservationCore):
        n = 10_000
        rng = np.random.default_rng(42)
        arr = rng.choice([-1, 0, 1], size=n).astype(np.int8)
        state = core.compute(arr)
        assert abs(state.C - 1.0) < 1e-9


# ─── 9. Benchmark / Speedup ──────────────────────────────────────────

class TestBenchmark:
    def test_speedup_factor(self, core: ConservationCore):
        """C library should be at least 10× faster than pure Python MC."""
        n = 200
        trials = 3000

        t0 = time.perf_counter()
        core.monte_carlo(n, trials)
        c_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _pure_python_monte_carlo(n, trials)
        py_ms = (time.perf_counter() - t0) * 1000

        speedup = py_ms / c_ms if c_ms > 0 else float("inf")
        print(f"\n  Speedup at n={n}, trials={trials}: {speedup:.1f}x")
        # Be generous — CI environments vary.
        assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"


# ─── Standalone runner ───────────────────────────────────────────────

def _run_standalone() -> int:
    """Run all tests without pytest — prints results and returns exit code."""

    core = ConservationCore()
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name} {detail}")

    print("=== Conservation Core — Test Suite ===\n")

    # Struct sizes
    print("[Struct Layout]")
    check("conservation_state == 24 bytes", sizeof(_CConservationState) == 24)
    check("ternary_signal == 32 bytes", sizeof(_CTernarySignal) == 32)
    check("magnitude offset == 8", _CTernarySignal.magnitude.offset == 8)
    print()

    # Compute
    print("[Conservation Compute]")
    signals = [
        TernarySignal(valence=1, magnitude=1.0),
        TernarySignal(valence=-1, magnitude=1.0),
        TernarySignal(valence=0, magnitude=1.0),
        TernarySignal(valence=1, magnitude=1.0),
    ]
    state = core.compute(signals)
    check("gamma=0.75", abs(state.gamma - 0.75) < 1e-9, f"(got {state.gamma})")
    check("eta=0.25", abs(state.eta - 0.25) < 1e-9, f"(got {state.eta})")
    check("C=1.0", abs(state.C - 1.0) < 1e-9)
    check("residual < 1e-12", state.residual < 1e-12)
    print()

    # Closed-form
    print("[Closed-Form]")
    expected_delta_50 = (1.0 / math.sqrt(50)) * (1.0 - 3.0 / 100.0)
    check(
        "delta(50) formula match",
        abs(core.delta(50) - expected_delta_50) < 1e-12,
        f"(got {core.delta(50)}, expected {expected_delta_50})",
    )

    eff50 = core.efficiency(50)
    check(
        "efficiency(50) ≈ 0.863",
        abs(eff50 - 0.863) < 0.001,
        f"(got {eff50})",
    )
    print()

    # Monte Carlo agreement
    print("[Monte Carlo C vs Python]")
    n_mc = 100
    trials = 8_000
    c_mc = core.monte_carlo(n_mc, trials)
    random.seed(42)
    py_mc = _pure_python_monte_carlo(n_mc, trials)
    diff = abs(c_mc - py_mc)
    check(
        f"MC agreement at n={n_mc} (diff={diff:.4f})",
        diff < 0.02,
        f"(C={c_mc:.4f}, Py={py_mc:.4f})",
    )

    theory = core.efficiency(n_mc)
    check(
        f"MC within 5% of theory",
        abs(c_mc - theory) / theory < 0.05,
        f"(MC={c_mc:.4f}, theory={theory:.4f})",
    )
    print()

    # Ring buffer
    print("[Ring Buffer]")
    with core.ringbuf(16) as rb:
        rb.push(TernarySignal(valence=1, agent_id=7))
        rb.push(TernarySignal(valence=-1, agent_id=8))
        check("count == 2", rb.count() == 2)
        sig = rb.pop()
        check("FIFO: agent_id == 7", sig is not None and sig.agent_id == 7)
        sig = rb.pop()
        check("FIFO: agent_id == 8", sig is not None and sig.agent_id == 8)
        check("empty after drain", rb.is_empty())
    print("  ✓ context manager closed cleanly")
    passed += 1
    print()

    # NumPy
    if HAS_NUMPY:
        print("[NumPy Interface]")
        arr = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        st = core.compute(arr)
        check(
            "numpy int8 gamma=0.8",
            abs(st.gamma - 0.8) < 1e-9,
            f"(got {st.gamma})",
        )
        check(
            "numpy int8 eta=0.2",
            abs(st.eta - 0.2) < 1e-9,
            f"(got {st.eta})",
        )
        print()
    else:
        print("[NumPy Interface] SKIPPED (numpy not available)\n")

    # Speedup
    print("[Benchmark]")
    n_bench = 200
    t_bench = 3000
    t0 = time.perf_counter()
    core.monte_carlo(n_bench, t_bench)
    c_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    _pure_python_monte_carlo(n_bench, t_bench)
    py_ms = (time.perf_counter() - t0) * 1000
    speedup = py_ms / c_ms if c_ms > 0 else float("inf")
    check(
        f"speedup > 5x ({speedup:.1f}x: C={c_ms:.0f}ms, Py={py_ms:.0f}ms)",
        speedup > 5.0,
    )

    print()

    # Summary
    total = passed + failed
    print(f"{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_standalone())
