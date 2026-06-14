/*
 * conservation_core.h — Bulletproof conservation law core
 *
 * γ + η = C (Shannon chain rule: H(X) = I(X;G) + H(X|G))
 *
 * Thread-safe, cache-aligned, SIMD-ready C implementation.
 * Designed for embeddable use in fleet governor, edge workers,
 * and embedded shells (ESP32, Pi, Jetson).
 *
 * License: MIT
 * Author: SuperInstance
 */

#ifndef CONSERVATION_CORE_H
#define CONSERVATION_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Types ────────────────────────────────────────────────────────── */

typedef struct {
    double gamma;    /* coupling cost: I(X;G) */
    double eta;      /* value delivered: H(X|G) */
    double C;        /* capacity: H(X) = γ + η */
} conservation_state;

typedef struct {
    int32_t valence;   /* -1, 0, +1 */
    double magnitude;  /* |contribution| */
    uint64_t agent_id; /* source agent */
    uint64_t timestamp;
} ternary_signal;

/* Cache-aligned ring buffer for concurrent signal processing */
typedef struct {
    uint32_t capacity;
    uint32_t mask;           /* capacity - 1, for fast modulo */
    uint32_t _pad0[2];       /* cache line padding */
    _Atomic uint64_t head;
    uint64_t _pad1[7];       /* prevent false sharing */
    _Atomic uint64_t tail;
    uint64_t _pad2[7];
    ternary_signal *buffer;  /* capacity entries */
} signal_ringbuf;

/* ─── Conservation Law ─────────────────────────────────────────────── */

/*
 * Compute conservation metrics from n signals.
 *
 * γ = sum of |valence| * magnitude / n   (mean coupling)
 * η = 1 - γ                               (normalized residual)
 * C = γ + η                               (should equal 1.0)
 *
 * With CLT convergence: δ(n) = (1/√n)(1 - 3/(2n))
 * Cancellation = 1 - δ(n)
 *
 * Thread-safe. Uses OpenMP for n > 1024.
 */
conservation_state conservation_compute(const ternary_signal *signals, size_t n);

/*
 * Theoretical cancellation factor at fleet size n.
 * Returns δ(n) = (1/√n)(1 - 3/(2n))
 */
double conservation_delta(size_t n);

/*
 * Predicted fleet efficiency: 1 - δ(n)
 * At n=50: ~0.863 (86.3% cancellation)
 * At n=10000: ~0.990 (99.0% cancellation)
 */
double conservation_efficiency(size_t n);

/*
 * Adversarial tolerance: fraction of fleet that can be adversarial
 * before conservation law breaks.
 * Returns ~0.108 (10.8%) for large n.
 */
double conservation_adversarial_threshold(size_t n);

/* ─── Signal Ring Buffer (Lock-Free SPSC) ──────────────────────────── */

/*
 * Create a lock-free single-producer single-consumer ring buffer.
 * Capacity must be power of 2. Returns NULL on failure.
 */
signal_ringbuf *ringbuf_create(uint32_t capacity);
void ringbuf_destroy(signal_ringbuf *rb);

/*
 * Push a signal. Returns true on success, false if full.
 * Producer-only. Not thread-safe for multiple producers.
 */
bool ringbuf_push(signal_ringbuf *rb, const ternary_signal *sig);

/*
 * Pop a signal. Returns true on success, false if empty.
 * Consumer-only. Not thread-safe for multiple consumers.
 */
bool ringbuf_pop(signal_ringbuf *rb, ternary_signal *sig);

/*
 * Current count. Approximate under concurrency.
 */
size_t ringbuf_count(const signal_ringbuf *rb);

/* ─── Batch Conservation Audit ─────────────────────────────────────── */

/*
 * Process a batch of signals through the conservation law.
 * Writes results to `state`. Returns audit pass boolean.
 *
 * Audit passes when: |γ + η - C| < epsilon
 * Default epsilon = 1e-10 (information-theoretic identity)
 */
bool conservation_audit(
    const ternary_signal *signals,
    size_t n,
    conservation_state *state,
    double epsilon
);

/* ─── Monte Carlo Verification ─────────────────────────────────────── */

/*
 * Run n_trials Monte Carlo simulations at fleet_size.
 * Returns measured cancellation factor.
 * Uses all available CPU cores via OpenMP.
 */
double conservation_monte_carlo(
    size_t fleet_size,
    size_t n_trials
);

#ifdef __cplusplus
}
#endif

#endif /* CONSERVATION_CORE_H */
