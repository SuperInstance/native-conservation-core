# The Speed of Truth

### How we built a C library that processes two billion signals per second — and what it taught us about the distance between code and physics

---

## The Problem With Fast

There's a seductive trap in performance engineering: the belief that faster is always better. It isn't. Faster is better *when the computation matters*. Most computation doesn't matter. Most loops in most programs are moving data from one place to another, transforming it in ways that could be done at half the speed for zero perceptible difference.

But sometimes computation matters. Sometimes you need to audit the conservation law across a fleet of one million agents in real-time, while the fleet is still making decisions. Sometimes the gap between 100 milliseconds and 1 millisecond is the gap between a governance system that works and one that's always one step behind.

That's why we built native-conservation-core. Not because Rust is too slow — Rust gives us 9.2 billion signals per second. But because there are things C can do that Rust can't: direct hardware access, cache-line alignment without `unsafe`, OpenMP directives that map to CPU SIMD instructions, CUDA kernels that talk to the GPU without abstraction layers.

This library is the metal layer. Everything above it — Rust, Python, Chapel, Cloudflare Workers — is a shell. The crab lives here.

## What's Inside

### Ternary ALU (`ternary_alu.h`, `ternary_alu.c`)

A branchless arithmetic logic unit for ternary values {-1, 0, +1}. The key insight: ternary operations can be expressed as small integer operations that the CPU's branch predictor handles perfectly. No branches means no branch misprediction penalties (15-20 cycles each on modern CPUs).

The ALU includes:
- **Branchless dot product**: Computes Σ(aᵢ × bᵢ) for ternary vectors using arithmetic tricks instead of conditionals
- **Cache-blocked matrix multiply**: Tuned for 32KB L1 cache (MC=64, KC=256, NC=64) — the same blocking strategy LAPACK uses
- **Haar wavelet decomposition**: Signal processing primitive that reveals multi-scale structure in fleet data
- **Conservation entropy**: Direct computation of γ + η = C from signal arrays

### Lock-Free Ring Buffer (`conservation_core.h`, `conservation_core.c`)

A single-producer single-consumer (SPSC) ring buffer built with `_Atomic` indices and cache-line alignment. This is the communication primitive for fleet agents — each agent writes signals into a ring buffer, the consumer reads them for conservation auditing.

Performance: **1.985 billion push operations per second**, 3.772 billion pop operations per second. The ring buffer is so fast that it's never the bottleneck. The bottleneck is always the computation *around* it.

### CUDA Ternary Kernel (`ternary_mac_kernel.cu`)

A GPU kernel that packs 16 ternary values into a single `uint32` (2 bits each), then performs multiply-accumulate operations using bit manipulation instead of floating-point arithmetic. The GPU processes these packed values at **241.6 GFLOPS** — 4.61× faster than float32, with 93.8% less memory usage.

The key trick: `__shfl_down_sync()` — a CUDA warp shuffle instruction that performs a reduction within a 32-thread warp without touching shared memory. For fleet cancellation (summing all signals), this means the reduction happens in-register, at hardware speed, with zero memory traffic.

### Python Bindings (`python/conservation.py`)

Because the best C library in the world is useless if you can't `import` it.

The Python bindings use `ctypes` to call into `libconservation_core.so`, exposing the full API: ring buffers, ternary ALU, conservation math, Monte Carlo simulation. The result: **10-48× speedup** over pure Python, depending on the operation.

The beauty of ctypes: no compilation step on the Python side. You `import conservation`, and it works. The C library handles the speed. Python handles the glue.

## The Performance Hierarchy

| Layer | Throughput | What it proves |
|:------|:-----------|:--------------|
| CUDA kernel | 241.6 GFLOPS | GPU parallelism + bit-packing |
| Rust rayon | 9.2B sig/s | Safe systems programming |
| C pthreads | 3.2B sig/s | Direct hardware control |
| C ring buffer | 1.985B ops/s | Lock-free data structures |
| Python via ctypes | 100-500M sig/s | FFI eliminates interpreter overhead |
| Pure Python | 5-10M sig/s | The cost of interpretation |

Each layer adds a capability. C adds cache control. Rust adds safety. CUDA adds massive parallelism. Python adds accessibility. None is "best" — each is the right tool for a specific job.

## What This Library Is Actually About

It's not about speed. Speed is the *measurement*. The *idea* is something else.

When you build a lock-free ring buffer and measure 1.985 billion operations per second, you're measuring the speed of trust. Each push operation is an agent saying "here's my signal." Each pop is the auditor saying "I heard you." Two billion handshakes per second, without locks, without waiting, without anyone ever dropping a message.

When you build a ternary ALU with branchless operations, you're building a machine that *cannot hesitate*. Every branch in code is a moment of indecision — the CPU guessing which way the code will go. Branchless code removes the guesswork. The machine becomes deterministic at the hardware level. Not approximately deterministic. *Actually* deterministic.

When you pack 16 ternary values into 32 bits and process them on a GPU, you're compressing reality. Each signal — yes, no, abstain — contains 1.585 bits of information (log₂3). Storing it in 2 bits wastes 0.415 bits. Storing it in 32 bits (a float) wastes 30.415 bits. The 2-bit representation isn't just smaller — it's *honest*. It uses almost exactly the amount of space the information actually contains.

This library is about building computing infrastructure that respects the information it carries. Not one bit more, not one bit less.

---

*Part of the SuperInstance conservation ecosystem*
*Companion: [conservation-languages](https://github.com/SuperInstance/conservation-languages) (9-language polyglot)*
*Companion: [fleet-sim-rs](https://github.com/SuperInstance/fleet-sim-rs) (Rust fleet simulator)*
