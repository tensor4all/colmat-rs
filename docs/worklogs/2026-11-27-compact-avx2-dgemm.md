# Compact column-major AVX2 DGEMM around order 512

## Decisions

- Route a sequential f64 overwrite with unit product scale to the full-depth kernel for orders 511, 512 and 513. Eight-row A panels and six columns of B fit twelve vector accumulators. Interleaving the next A panel's packing with the current tile's computation reduces separate copy work. Each output consumes the original depth order.
- Keep general GEMM and caller-scratch GEMM on their existing blocking schedule. The companion kernel has an explicit entry point so the dispatch choice belongs to colmat.
- In a seven-set comparison, a separate column-width experiment achieved 1.0424× at 512 and 1.0411× at 513. Neither reached the 1.05× target. The candidate brought here measured 1.0434× and 1.0429× respectively. This comparison does not establish a significant difference between the two candidates.

## Verification conclusions and constraints

- The public matmul benchmark compared colmat main 475e2842 and the prototype to oneMKL 2026.2 through fbench. The 511 result reached 1.0507× on this Intel host. The 512 and 513 measurements demonstrated roughly a 4.3% throughput advantage. All three timings use one P-core, with observers on E-cores sharing its last-level cache. No upstream faer timing is claimed. Per-shape times and the harness settings are in the accompanying PR.
- The kernel's output matched a depth-ordered scalar FMA reference on independent inputs and on cancellation inputs with signed zeros and subnormals. Output guards survived each tested alignment.
- Colmat selects this route only on the AVX2/FMA branch. CPUs exposing AVX-512F take the existing AVX-512 branch. Ryzen devices on the AVX2/FMA branch are eligible, and performance there remains unmeasured.
- The draft pins the companion change by Git revision. A published dependency and the wider acceptance grid remain merge prerequisites.
