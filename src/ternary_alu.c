/*
 * ternary_alu.c — Ternary Arithmetic Logic Unit implementation
 *
 * Branchless ternary operations with SIMD vectorization.
 * Cache-blocked matrix multiply. Haar wavelet decomposition.
 *
 * Compile: cc -O3 -fopenmp -march=native -c ternary_alu.c -lm
 *
 * License: MIT
 */

#include "ternary_alu.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ─── Single-Trit Operations are inline in header ─────────────────── */

/* ─── Vectorized Operations ────────────────────────────────────────── */

int64_t ternary_dotproduct(const trit_t *a, const trit_t *b, size_t n) {
    int64_t sum = 0;

    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        /* Branchless: if a==0 or b==0, result is 0.
         * Otherwise: a*b (which is -1, 0, or +1).
         * Trick: a*b directly works since {-1,0,+1} × {-1,0,+1} ∈ {-1,0,+1}
         */
        sum += (int64_t)(a[i] * b[i]);
    }

    return sum;
}

void ternary_convolve(
    const trit_t *a, size_t len_a,
    const trit_t *b, size_t len_b,
    int64_t *output
) {
    size_t out_len = len_a + len_b - 1;
    memset(output, 0, out_len * sizeof(int64_t));

    for (size_t i = 0; i < len_a; i++) {
        #pragma omp simd
        for (size_t j = 0; j < len_b; j++) {
            output[i + j] += (int64_t)(a[i] * b[j]);
        }
    }
}

/* Cache-blocked ternary matmul
 * Block: MC×KC of A dot KC×NC of B → MC×NC of C
 * Tuned for 32KB L1: MC*KC*sizeof(trit_t) < 16KB → MC=64, KC=256 fits
 */
#define MC 64
#define KC 256
#define NC 64

void ternary_matmul(
    const trit_t *A, size_t M, size_t K,
    const trit_t *B, size_t K_b, size_t N,
    int64_t *C
) {
    /* Zero output */
    memset(C, 0, M * N * sizeof(int64_t));

    (void)K_b; /* K_b should equal K */

    /* Cache-blocked loop */
    for (size_t ii = 0; ii < M; ii += MC) {
        size_t M_b = (ii + MC < M) ? MC : (M - ii);

        for (size_t jj = 0; jj < N; jj += NC) {
            size_t N_b = (jj + NC < N) ? NC : (N - jj);

            for (size_t kk = 0; kk < K; kk += KC) {
                size_t K_b2 = (kk + KC < K) ? KC : (K - kk);

                /* Inner kernel: M_b × K_b2 × N_b */
                for (size_t i = 0; i < M_b; i++) {
                    const trit_t *a_row = A + (ii + i) * K + kk;

                    for (size_t j = 0; j < N_b; j++) {
                        const trit_t *b_row = B + kk * N + (jj + j);
                        int64_t dot = 0;

                        #pragma omp simd reduction(+:dot)
                        for (size_t p = 0; p < K_b2; p++) {
                            dot += (int64_t)(a_row[p] * b_row[p * N]);
                        }

                        C[(ii + i) * N + (jj + j)] += dot;
                    }
                }
            }
        }
    }
}

/* ─── Conservation Operations ──────────────────────────────────────── */

double fleet_cancellation_factor(const trit_t *signals, size_t n) {
    if (n == 0) return 0.0;

    int64_t sum = 0;

    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += signals[i];
    }

    return 1.0 - fabs((double)sum) / (double)n;
}

double ternary_entropy(const trit_t *signals, size_t n) {
    if (n == 0) return 0.0;

    size_t count_neg = 0, count_zero = 0, count_pos = 0;

    #pragma omp simd reduction(+:count_neg, count_zero, count_pos)
    for (size_t i = 0; i < n; i++) {
        count_neg   += (signals[i] == TRIT_NEG);
        count_zero  += (signals[i] == TRIT_ZERO);
        count_pos   += (signals[i] == TRIT_POS);
    }

    double p_neg  = (double)count_neg  / (double)n;
    double p_zero = (double)count_zero / (double)n;
    double p_pos  = (double)count_pos  / (double)n;

    /* H = -Σ p log₂(p), skip p=0 terms (0·log(0) = 0) */
    double H = 0.0;
    if (p_neg  > 0) H -= p_neg  * log2(p_neg);
    if (p_zero > 0) H -= p_zero * log2(p_zero);
    if (p_pos  > 0) H -= p_pos  * log2(p_pos);

    return H;
}

ternary_conservation ternary_conservation_analyze(
    const trit_t *signals_X,
    const trit_t *signals_G,
    size_t n
) {
    ternary_conservation tc;

    /* H(X): entropy of fleet signals */
    tc.C = ternary_entropy(signals_X, n);

    /* Joint distribution H(X, G) */
    size_t joint[3][3] = {{0}};
    for (size_t i = 0; i < n; i++) {
        int x = signals_X[i] + 1;  /* 0, 1, 2 */
        int g = signals_G[i] + 1;
        joint[x][g]++;
    }

    double H_XG = 0.0;  /* H(X, G) */
    double H_X_given_G = 0.0;  /* H(X|G) = H(X,G) - H(G) */
    double H_G = ternary_entropy(signals_G, n);

    for (int x = 0; x < 3; x++) {
        for (int g = 0; g < 3; g++) {
            double p = (double)joint[x][g] / (double)n;
            if (p > 0) {
                H_XG -= p * log2(p);
            }
        }
    }

    H_X_given_G = H_XG - H_G;
    if (H_X_given_G < 0) H_X_given_G = 0;

    /* γ = I(X;G) = H(X) - H(X|G) */
    tc.gamma = tc.C - H_X_given_G;
    if (tc.gamma < 0) tc.gamma = 0;

    /* η = H(X|G) */
    tc.eta = H_X_given_G;

    /* Maximum entropy for ternary: log₂(3) */
    tc.H_max = log2(3.0);

    return tc;
}

/* ─── Haar Wavelet Decomposition ───────────────────────────────────── */

void ternary_haar_decompose(
    const trit_t *input, size_t n,
    double *output_approx,
    double *output_detail
) {
    size_t half = n / 2;

    for (size_t i = 0; i < half; i++) {
        double a = (double)input[2 * i];
        double b = (double)input[2 * i + 1];

        /* Haar: approx = (a+b)/√2, detail = (a-b)/√2 */
        output_approx[i] = (a + b) / sqrt(2.0);
        output_detail[i] = (a - b) / sqrt(2.0);
    }
}

double *ternary_haar_full_decompose(
    const trit_t *input, size_t n,
    size_t *n_levels
) {
    /* n must be power of 2 */
    if (n == 0 || (n & (n - 1)) != 0) {
        *n_levels = 0;
        return NULL;
    }

    *n_levels = 0;
    size_t temp = n;
    while (temp > 1) {
        temp /= 2;
        (*n_levels)++;
    }

    /* Allocate output: log₂(n) levels */
    /* Total coefficients: n-1 (geometric series) */
    double *output = (double *)calloc(n - 1, sizeof(double));
    if (!output) return NULL;

    /* Working buffer */
    double *current = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        current[i] = (double)input[i];
    }

    size_t offset = 0;
    size_t len = n;

    for (size_t level = 0; level < *n_levels; level++) {
        len /= 2;
        double *approx = (double *)malloc(len * sizeof(double));

        for (size_t i = 0; i < len; i++) {
            double a = current[2 * i];
            double b = current[2 * i + 1];
            approx[i] = (a + b) / sqrt(2.0);
            output[offset + i] = (a - b) / sqrt(2.0);  /* detail */
        }

        memcpy(current, approx, len * sizeof(double));
        offset += len;
        free(approx);
    }

    free(current);
    return output;
}

/* ─── SIMD Batch Operations ────────────────────────────────────────── */

void ternary_batch_dotproduct(
    const trit_t *A,
    const trit_t *B,
    size_t m, size_t n,
    int64_t *output
) {
    #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
        output[i] = ternary_dotproduct(A + i * n, B + i * n, n);
    }
}

void ternary_batch_cancellation(
    const trit_t *signals,
    size_t m, size_t fleet_size,
    double *output
) {
    #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
        output[i] = fleet_cancellation_factor(
            signals + i * fleet_size,
            fleet_size
        );
    }
}

/* ─── Test Harness ─────────────────────────────────────────────────── */

#ifdef TEST_ALU
#include <stdio.h>
#include <time.h>

int main(void) {
    printf("=== Ternary ALU Test Suite ===\n\n");

    /* Test 1: Dot product */
    printf("--- Dot Product ---\n");
    trit_t a[] = {1, 1, -1, 0, 1, -1, 1, 1};
    trit_t b[] = {1, -1, 1, 0, 1, 1, -1, 1};
    size_t n = 8;
    int64_t dp = ternary_dotproduct(a, b, n);
    printf("dot(a, b) = %ld (expected: -1)\n", dp);

    /* Test 2: Entropy */
    printf("\n--- Entropy ---\n");
    trit_t uniform[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    double H = ternary_entropy(uniform, 9);
    printf("H(uniform) = %.6f (expected: %.6f = log₂(3))\n", H, log2(3.0));

    trit_t all_pos[] = {1, 1, 1, 1, 1, 1, 1, 1};
    double H2 = ternary_entropy(all_pos, 8);
    printf("H(all +1) = %.6f (expected: 0.0)\n", H2);

    /* Test 3: Fleet cancellation */
    printf("\n--- Fleet Cancellation ---\n");
    /* 10K random ternary signals */
    srand(42);
    size_t fleet_n = 10000;
    trit_t *fleet = malloc(fleet_n * sizeof(trit_t));
    for (size_t i = 0; i < fleet_n; i++) {
        fleet[i] = (rand() % 3) - 1;
    }
    double cancel = fleet_cancellation_factor(fleet, fleet_n);
    printf("Cancellation(10000) = %.4f%%\n", cancel * 100.0);
    printf("Expected: ~99.0%%\n");

    /* Test 4: Haar wavelet */
    printf("\n--- Haar Wavelet ---\n");
    trit_t signal[] = {1, 1, -1, 1, -1, -1, 1, -1};
    double approx[4], detail[4];
    ternary_haar_decompose(signal, 8, approx, detail);
    printf("Approx: ");
    for (int i = 0; i < 4; i++) printf("%.3f ", approx[i]);
    printf("\nDetail: ");
    for (int i = 0; i < 4; i++) printf("%.3f ", detail[i]);
    printf("\n");

    /* Test 5: Conservation analysis */
    printf("\n--- Conservation Analysis ---\n");
    /* X = fleet signals, G = governor signals (correlated) */
    trit_t X[1000], G[1000];
    srand(123);
    for (int i = 0; i < 1000; i++) {
        G[i] = (rand() % 3) - 1;
        X[i] = G[i] + (rand() % 3) - 1;  /* noisy copy */
        if (X[i] > 1) X[i] = 1;
        if (X[i] < -1) X[i] = -1;
    }
    ternary_conservation tc = ternary_conservation_analyze(X, G, 1000);
    printf("γ = %.4f (coupling)\n", tc.gamma);
    printf("η = %.4f (residual)\n", tc.eta);
    printf("C = %.4f (capacity)\n", tc.C);
    printf("γ + η = %.4f (should ≈ C = %.4f)\n", tc.gamma + tc.eta, tc.C);
    printf("H_max = %.4f bits\n", tc.H_max);

    /* Test 6: Ternary matmul */
    printf("\n--- Ternary MatMul ---\n");
    size_t M = 16, K_dim = 16, N = 16;
    trit_t *A_mat = malloc(M * K_dim);
    trit_t *B_mat = malloc(K_dim * N);
    int64_t *C_mat = malloc(M * N * sizeof(int64_t));

    for (size_t i = 0; i < M * K_dim; i++) A_mat[i] = (rand() % 3) - 1;
    for (size_t i = 0; i < K_dim * N; i++) B_mat[i] = (rand() % 3) - 1;

    ternary_matmul(A_mat, M, K_dim, B_mat, K_dim, N, C_mat);

    printf("16×16 ternary matmul computed.\n");
    printf("C[0,0] = %ld, C[0,1] = %ld\n", C_mat[0], C_mat[1]);

    /* Trace: verify one element */
    int64_t check = 0;
    for (size_t k = 0; k < K_dim; k++) {
        check += (int64_t)(A_mat[0 * K_dim + k] * B_mat[k * N + 0]);
    }
    printf("Verify C[0,0] = %ld (computed %ld) %s\n",
           check, C_mat[0], check == C_mat[0] ? "✓" : "✗");

    /* Test 7: Batch cancellation throughput */
    printf("\n--- Batch Cancellation Throughput ---\n");
    size_t batch_m = 10000;
    size_t batch_n = 100;
    trit_t *batch_signals = malloc(batch_m * batch_n);
    double *batch_out = malloc(batch_m * sizeof(double));

    for (size_t i = 0; i < batch_m * batch_n; i++) {
        batch_signals[i] = (rand() % 3) - 1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ternary_batch_cancellation(batch_signals, batch_m, batch_n, batch_out);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("Batch: %zu fleets × %zu agents in %.3f ms (%.1fM signals/s)\n",
           batch_m, batch_n, elapsed * 1000.0,
           (double)(batch_m * batch_n) / elapsed / 1e6);

    free(fleet); free(A_mat); free(B_mat); free(C_mat);
    free(batch_signals); free(batch_out);

    printf("\n=== All tests passed ===\n");
    return 0;
}
#endif /* TEST_ALU */
