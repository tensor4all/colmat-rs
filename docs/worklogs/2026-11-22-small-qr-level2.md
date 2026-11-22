# Small QR: one reflector (pair) at a time below the blocking sizes

Issue: [colmat-rs#269](https://github.com/tensor4all/colmat-rs/issues/269).

## Where the time was

The issue's premise was the small-GEMM finding of #262: a ragged team grid in the products the
blocked QR issues. That is not what limits small QR. At `n = 32` the f64 `qr_factor` cell is
16.8 us at one thread and 17.5 us at four and eight, against AOCL's 7.4 us at every count: the
gap is serial, and the team paths run these sizes on the calling thread already.

Temporary cycle counters (removed) on a `32 x 32` f64 factorization at block width 4 (the shape's
`recommended_block_size`): the trailing block applies took 49% of the call, the 4-column panels
21%, the block-factor merges (`upgrade_householder_factor`) 20%. Every one of those is a product of
a few hundred to a few thousand multiply-adds: `nano_gemm` runs `4 x 28 x 28` (`V^H B`) in
0.50 us and `28 x 28 x 4` in 0.38 us (6-8 GMAC/s on Zen 5, AVX2 kernels: the fork builds without
`x86-v4`), and a unit-triangular `4 x 4` times `4 x 28` costs 0.26 us. The colmat wrapper adds
15-25 ns to each; the kernels themselves are the cost.

## Decisions

- **Not bitwise-preserving, by the maintainer's decision.** The issue asked to keep the
  reduction order bitwise unchanged. No change that leaves the bits alone closes this gap: the
  only route-level lever that keeps them (`W^T = B^T V` instead of `W = V^H B`, bit-identical
  because each output element is the same sequential FMA chain) is slower (0.77 us against 0.50,
  `nano_gemm` then packs a row-major `28 x 28`), the AVX-512 `private_gemm_x86` kernel is slower
  on these shapes (0.70/0.44 us) and differs in 15% of the output bits, and a whole unblocked
  factorization through the existing per-column loop is 14.2 us. Asked with that evidence, the
  maintainer chose a new small-matrix path whose last bits differ from the previous release. It is
  deterministic at a fixed thread count, which is what [reproducibility](../reproducibility.md)
  promises; it does not depend on the participant count at all (the path is sequential).
- **The small path is LAPACK's `geqr2` / `org2r` shape with a SIMD rank-one kernel.** Each column
  is reduced by the existing `make_qr_householder` and its reflector is applied to all the
  trailing columns in one SIMD dispatch: per column an inner product and an FMA update, four
  target columns at a time so that four reduction chains share every load of the reflector
  (`householder::apply_reflector_small`). The existing unblocked panel makes one `inner_prod`
  call and one element-wise pass per column pair; the dispatch per pair, not the arithmetic, was
  its cost.
- **Reflectors go in pairs in the full-rank step.** The first reflector is applied to the next
  column, which makes the second, and the pair then sweeps the rest once:
  `a = v_1^H c`, `b = v_2^H c` reduced together, `k_1 = -a / tau_1`, `k_2 = -(b + s k_1) / tau_2`
  with `s = v_2^H v_1`, `c += v_1 k_1 + v_2 k_2` (`householder::apply_reflector_pair_small`; the
  `REVERSE` order serves `Q`). This halves the passes over the trailing matrix and doubles the
  independent chains. A column that adds no rank falls back to the single-reflector step, so the
  rank handling (`row < col`, the reflector stored at column `row`) is the unblocked loop's.
- **The block factors are formed afterwards, directly.** The caller owns the width of `H`, so the
  path cannot pick its own: for each block of `block_size` reflectors it writes `tau` on the
  diagonal and `v_i^H v_j` above it, one inner product per pair, which is the factor
  `upgrade_householder_factor` produces and every blocked consumer reads.
- **Three gates, each set by its own crossover** (f64, one thread, evo):
  - factorization (`factor::small_path`): column-major, a SIMD dtype, `m n <= 128^2` (a
    complex dtype on a team counts each element four times, the cost model's complex/real GEMM
    rate ratio) and `n <= 32 m`. Without the weight, c64 `128^2` at four and eight participants
    took 500 us against 433 and 375 on the blocked route, and c64/c32 `96^2` broke even; with it
    those cells keep the blocked route and c64 `64^2` still goes from 117-120 to 76 us. Square: the small path wins at every size up to 192 at one thread (192:
    527 us against 656), but the blocked route's team products win from 192 at four and eight
    participants (373 us), so the area stops at `128^2`, where the small path still wins at every
    count (159 us against 195-253). Tall shapes win throughout (`2048 x 8` 54 against 74 us,
    `256 x 64` 122 against 180). Very wide ones lose: `8 x 512` 19.5 against 18.4, `8 x 1024`
    37.5 against 33.0, `16 x 1024` 81.2 against 76.4, while `16 x 512` (the aspect bound) wins
    40.7 against 42.8.
  - `compute_thin_Q` (`unitary::apply_to_identity`): `m q <= 96^2`. It meets only the trailing
    columns (`org2r`): `64^2` 34.0 against 39.6 us, `96^2` 77.8 against 81.0, `128^2` 171 against
    162.
  - the public sequence apply (`apply_block_householder_sequence_on_the_left_in_place_with_conj`,
    `Conj::No` or a real dtype): target `m n <= 32^2`. It must apply every reflector to every
    column, so it gives the saving back sooner: on an identity, `24^2` 5.1 against 6.6 us,
    `32^2` 8.8 against 10.9, `40^2` 13.8 against 13.1, `64^2` 39.5 against 35.1. This is the
    route that forms `Q` in the `qr` benchmark family, and it is shared, so it also reaches the
    small `Q` formations of other decompositions (see below). On a team it is taken only for
    block factors of at most 8 columns: a single 32-reflector block on a `32^2` target is one
    product the team splits (6.1 us at eight participants against 8.4 at one; 8.1 on the small
    path), while the 4- and 8-wide blocks of a small QR are products too small to split.
- **Rejected: masked tails.** Running the vector body unmasked and the last `len % lanes` rows in
  scalar code was 8.5 against 9.0 us at 32 and 40.1 against 38.7 at 64: not a consistent gain,
  dropped. **Rejected: eight target columns per step**: 8.6 against 9.0 at 32, 40.5 against
  38.7 at 64.
- The yardstick (AOCL) was timed through its public API only. The algorithm is LAPACK's
  reference `dgeqr2`/`dorg2r` (BSD) structure; the pairing is the standard two-reflector
  aggregation of a compact WY update of width two.

## Verification conclusions and constraints

- Paired A/B, evo (Ryzen AI 9 HX 470, boost off), fbench AOCL arm (AOCL 6.0.0), base
  `origin/main` 33d34668 against this change's code commit, median [Q1, Q3] of 5, one process
  per thread count, measured CPUs pinned (4 / 4-7 / 4-11) with the observer outside them. Raw
  rows, tables and scripts: [evidence](evidence/2026-11-22-small-qr-269.tar.gz). f64, us per call,
  before -> after (after/AOCL):

  | cell | 1T | 4T | 8T |
  |---|---|---|---|
  | `qr` 32 | 28.2 -> 16.3 (1.02) | 29.3 -> 16.4 (1.02) | 29.3 -> 16.4 (1.03) |
  | `qr` 64 | 104.2 -> 67.9 (0.68) | 95.4 -> 57.3 (0.59) | 94.6 -> 57.2 (0.58) |
  | `qr` 128 | 450 -> 358 (0.32) | 300 -> 273 (0.55) | 275 -> 238 (0.49) |
  | `qr_factor` 32 | 16.7 -> 8.2 (1.09) | 17.2 -> 8.0 (1.08) | 17.4 -> 8.1 (1.13) |
  | `qr_factor` 48 | 38.7 -> 17.3 (0.88) | 40.5 -> 17.3 (0.87) | 40.3 -> 17.5 (0.89) |
  | `qr_factor` 64 | 67.9 -> 30.2 (0.74) | 67.7 -> 32.0 (0.79) | 67.7 -> 30.4 (0.74) |
  | `qr_factor` 96 | 146 -> 77 (0.56) | 133 -> 76 (0.56) | 130 -> 77 (0.56) |
  | `qr_factor` 128 | 253 -> 158 (0.26) | 205 -> 158 (0.59) | 193 -> 161 (0.65) |

  The issue's headline cell (`qr`, factor plus thin `Q`, n = 32) goes from 0.54-0.57x of AOCL's
  rate to 0.97-0.98x; `qr_factor` at 32 from 0.41-0.45x to 0.88-0.93x. At 48 and above both families are
  ahead of AOCL. c64 `qr_factor` 32: 32.2 -> 14.5 us at one thread (AOCL 20.4), 31.8 -> 14.7 at
  four; f32 `qr_factor` 32/64/128: 15.3 -> 8.3, 61.8 -> 27.5, 222 -> 119 at one thread.
- Not regressed: 160 and 192 (above the gates) at 0.99-1.02 of before at every count; 256, 512
  and 1024 at 1.00-1.01 at one, four and eight participants (a first 8T pass read 0.85-0.96 with
  the base arm measured first; repeated with the candidate first it reads 1.00-1.01, so that was
  drift). Complex on a team at 96 and 128 now 0.98-1.01 (unchanged route). `householder_apply`
  (one block as wide as the basis) at eight participants 0.96-1.04 over three repeats; at one
  thread 0.43-0.89 below `32^2`. `eigh` and `svd_thin` at 16-32 move 0.85-1.02 (their small
  `Q` formations), at 48 0.97-1.03; `svd_values` 0.98-1.02. `lu`, `cholesky` and `solve`, which do not reach the
  changed code, spread 0.87-1.16 between repeats on their 1-12 us cells in both directions;
  that is the host's run-to-run spread at that size, not this change.
- The `qr_factor` 32 cell moved between 7.7 and 8.6 us across builds whose differences could not
  matter to it (moving the kernels between modules, a `const` generic in place of a `bool`):
  code placement, not the algorithm. The table is the committed code's.

- Correctness: `fbench verify` at 1/4/8 participants, f64/c64/f32/c32, shapes on both sides of
  every gate (including `16 x 512`/`16 x 513`, `128^2`/`129 x 128`, `1 x 3`, `3 x 1`), for
  `qr_factor`, `qr`, `solve`, `svd_values`, `svd_thin`, `eigh`, `householder_apply`, `lu`,
  `cholesky`: all within tolerance. `cargo test -p colmat --lib` green. The focused test
  `qr::no_pivoting::factor::tests::small_path_matches_the_blocked_route` crosses the gates, low
  rank and a skipped column, and checks that the block factors are the ones a blocked consumer
  (the transpose sequence, which reads every entry of `H`) expects. `no_step_alloc` gains the
  small shapes (factor, thin `Q`, sequence apply; sequential and on a held team): no allocation.
- Only the last bits change below the gates; above them every route is the previous one. The
  row-major (`c` order) `qr_factor` at one participant keeps its transpose-dual route, which the
  small path does not serve (its columns are not contiguous); with a team it copies to column
  major and takes the small path. LQ is unchanged (its view is row-major).
- Not measured here: a 64-core host (the small path is sequential, so participant count cannot
  move it, but the team route it replaces up to `128^2` is faster on wider hosts), and an AVX-512
  (`x86-v4`) build, where `nano_gemm` and the SIMD kernels both double their width and the
  crossovers may move. The participant gates (complex weight, team block width) are set on evo's
  8 participants.
