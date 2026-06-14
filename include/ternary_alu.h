/*
 * ternary_alu.h — Ternary Arithmetic Logic Unit
 *
 * Hardware-inspired ternary operations for {-1, 0, +1} algebra.
 * All operations are branchless where possible (use bit tricks).
 * Designed for SIMD vectorization and FPGA/GPU porting.
 *
 * License: MIT
 */

#ifndef TERNARY_ALU_H
#define TERNARY_ALU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Core Types ───────────────────────────────────────────────────── */

typedef int8_t trit_t;  /* -1, 0, +1 */

/* Balanced ternary: {-1, 0, +1} maps to {T, 0, 1} */
#define TRIT_NEG  (-1)
#define TRIT_ZERO ( 0)
#define TRIT_POS  ( 1)

/* ─── Single-Trit Operations (Branchless) ──────────────────────────── */

/* Ternary AND (min): a ∧ b */
static inline trit_t trit_and(trit_t a, trit_t b) {
    /* min(a, b) — use comparison-free trick */
    return a < b ? a : b;
}

/* Ternary OR (max): a ∨ b */
static inline trit_t trit_or(trit_t a, trit_t b) {
    return a > b ? a : b;
}

/* Ternary NOT (negation): ¬a = -a */
static inline trit_t trit_not(trit_t a) {
    return -a;
}

/* Ternary consensus: returns the consensus of three trits */
static inline trit_t trit_consensus(trit_t a, trit_t b, trit_t c) {
    /* Majority vote with ternary semantics */
    if (a == b) return a;
    if (a == c) return a;
    if (b == c) return b;
    return TRIT_ZERO;
}

/* ─── Vectorized Operations ────────────────────────────────────────── */

/*
 * Ternary dot product: Σ aᵢ × bᵢ
 * For n elements. Returns integer result.
 *
 * Complexity: O(n) with 2 ops per element (compare + add)
 * vs float dot product: O(n) with 3 ops (mul + add + rounding)
 *
 * Theoretical speedup: 1.5× per op, plus no rounding error.
 */
int64_t ternary_dotproduct(const trit_t *a, const trit_t *b, size_t n);

/*
 * Ternary convolution: (a ★ b)[k] = Σ a[i] × b[k-i]
 * Output buffer must be (len_a + len_b - 1) elements.
 * Used for wavelet decomposition in conservation law verification.
 */
void ternary_convolve(
    const trit_t *a, size_t len_a,
    const trit_t *b, size_t len_b,
    int64_t *output
);

/*
 * Ternary matmul: C = A × B
 * A is M×K ternary, B is K×N ternary, C is M×N int64.
 * Cache-blocked for L1/L2 locality.
 *
 * Block sizes tuned for 32KB L1 data cache:
 * - MC=64, KC=256, NC=64 (fits in L1)
 */
void ternary_matmul(
    const trit_t *A, size_t M, size_t K,
    const trit_t *B, size_t K_b, size_t N,
    int64_t *C
);

/* ─── Conservation Operations ──────────────────────────────────────── */

/*
 * Fleet cancellation: compute |Σ signals| / n
 * Returns the aggregate coupling magnitude.
 * Uses SIMD-friendly accumulation.
 */
double fleet_cancellation_factor(const trit_t *signals, size_t n);

/*
 * Conservation entropy: H(X) for ternary distribution
 * H(X) = -Σ p(x) log₂(x) for x ∈ {-1, 0, +1}
 * Maximum: log₂(3) ≈ 1.585 bits (uniform distribution)
 */
double ternary_entropy(const trit_t *signals, size_t n);

/*
 * Mutual information: I(X; G) = H(X) - H(X|G)
 * γ = I(X;G) coupling cost
 * η = H(X|G) residual value
 */
typedef struct {
    double gamma;  /* I(X;G) = H(X) - H(X|G) */
    double eta;    /* H(X|G) */
    double C;      /* H(X) = γ + η */
    double H_max;  /* log₂(3) ≈ 1.585 */
} ternary_conservation;

ternary_conservation ternary_conservation_analyze(
    const trit_t *signals_X,
    const trit_t *signals_G,
    size_t n
);

/* ─── Wavelet Decomposition (Haar on Ternary) ──────────────────────── */

/*
 * Single-level Haar wavelet decomposition on ternary signal.
 * Approximation = low-pass (average)
 * Detail = high-pass (difference)
 *
 * For conservation law: approximation = η, detail = γ
 *
 * output_approx and output_detail must each be n/2 elements.
 * In-place if output buffers alias input (requires n power of 2).
 */
void ternary_haar_decompose(
    const trit_t *input, size_t n,
    double *output_approx,
    double *output_detail
);

/*
 * Full multi-level decomposition (n must be power of 2).
 * Output: log₂(n) detail levels + 1 final approximation.
 * Returns array of [level][n/2^level] detail coefficients.
 */
double *ternary_haar_full_decompose(
    const trit_t *input, size_t n,
    size_t *n_levels
);

/* ─── SIMD Batch Operations ────────────────────────────────────────── */

/*
 * Batch dot product: compute dot products for m pairs of n-length vectors.
 * Uses OpenMP SIMD if available. Output buffer must be m elements.
 */
void ternary_batch_dotproduct(
    const trit_t *A,  /* m × n row-major */
    const trit_t *B,  /* m × n row-major */
    size_t m, size_t n,
    int64_t *output   /* m results */
);

/*
 * Batch conservation audit for m independent fleets.
 * Each fleet has fleet_size agents.
 * Returns cancellation factor per fleet.
 */
void ternary_batch_cancellation(
    const trit_t *signals,  /* m × fleet_size */
    size_t m, size_t fleet_size,
    double *output          /* m cancellation factors */
);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_ALU_H */
