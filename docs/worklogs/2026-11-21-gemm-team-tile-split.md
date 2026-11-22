# Small products on a team: balanced tile split instead of the kernel's job grid

tensor4all/colmat-rs#262: at `n = 32..192` with 4 and 8 participants the general product is about
`0.48x` AOCL, while it is at parity sequentially and at parity or better from `256^3` up. Measured
from `origin/main` (`b376b3ce`), evo host (Ryzen AI 9 HX 470, 12C/24T), pinned CPU sets, median of
5 reps, both orders.

## Decisions

- **The gap is the kernel's job decomposition, not the kernel or the section count.**
  `t4a-private-gemm-x86` 0.1.28 (`MilliPar::call`) cuts a product into
  `ceil(m / (mf * mr)) * ceil(n / (nf * nr))` jobs with `mf = clamp(ceil(m / mr) / (2 nt), 2, 4)`,
  `mr = 48`, `nf * nr = 32` (f64 AVX-512). At `n = 64` that is **2 jobs for 8 members**, at
  `96` 3, at `128` 8 ragged jobs (row bands `96 + 32`) and at `192` 12 jobs, i.e. 1.5 waves; the
  measured `64^3` figure (9.7 us at 8 participants) is `17.2 / 2` plus about 1 us of dispatch. Only
  one `fork.run` happens at these sizes (no packing pass), so the section count contributes nothing.
- **Balanced tiles of the output, each a sequential kernel call, on one section.** Every member
  takes a tile of the `tile_grid` partition of the destination and calls the same kernel
  sequentially on it; `k` is never split, so there is no reduction and no partial buffer. This is
  the route `gemm_tiled` already ran on targets without the x86 kernel, now shared by both
  (`tile_split` + `run_tiles` in `utils::team`), so the fork sees the same panel packing as before
  and the tile route needs no job tables at all.
- **One bound in serial time, not per dtype or per width.** `TILED_CEILING` is 8 ms of estimated
  serial work (MACs at the dtype's rate, complex MAC counted as 4.5 real ones). Below it the split
  wins at every size; above it the kernel's grid wins because a tile packs the shared panels once
  per tile and the whole-product route packs them once.
- **The ceiling is 8 ms, set by a cache-aliasing cliff, not by a smooth crossover.** With the
  ceiling at 10 ms (which still wins at `256^3`: `0.82x`) `512^3` f64 (`8.6 ms`) is **1.17x slower**
  than the whole-product route in `f` order and 0.95x in `c` order, reproducibly (two runs,
  1.15 and 1.17); `480^3` (`7.0 ms`) is still 0.91x/0.96x and `544^3` (above the ceiling) is 1.00x.
  A 512-wide destination row stride is a power of two (4 KiB), which lands one L1 set, and the
  per-tile traversal is what makes it visible. At 8 ms the last f64 size that splits is `480^3`
  and `512^3` keeps the whole-product route.
- **Rejected: widening `MilliPar`'s grid in `t4a-private-gemm-x86`.** It would need a cross-repo pin
  bump in this repository for a behaviour the fork can already correct above the kernel, and the
  grid cannot be balanced at every size without changing that crate's `Fork` contract (one `run`
  for packing plus one for the jobs). Recorded for the maintainer, not opened upstream.
- **Rejected: a k reduction across members.** At `64^3` the reduction and the padded partial
  buffer cost more than the imbalance saves, and the flat `k` keeps the result bit-identical to
  the sequential call.
- **Rejected: reading AOCL's small-product structure.** The sources are reference only (no code
  copied), and nothing in them was needed once the decomposition was the whole gap.

## Verification conclusions and constraints

- Paired comparison, same harness, `origin/main` binary against this change, identical pinned CPU
  sets and rotation, 8 participants, per call in microseconds (`cand/base`, `cand/aocl`):

  | shape (f64) | base | change | ratio | vs AOCL |
  |---|---|---|---|---|
  | `32^3` | 2.39 | 1.61 | 0.68 | 0.71 |
  | `64^3` | 9.72 | 3.68 | 0.38 | 0.81 |
  | `96^3` | 19.94 | 8.83 | 0.44 | 0.93 |
  | `128^3` | 32.50 | 18.38 | 0.57 | 0.86 |
  | `160^3` | 57.91 | 34.16 | 0.59 | 0.88 |
  | `192^3` | 86.18 | 57.93 | 0.67 | 0.99 |
  | `256^3` | 165.0 | 135.4 | 0.82 | 0.94 |
  | `480^3` | 968.4 | 883.4 | 0.91 | 1.00 |
  | `512^3` | 1131.9 | 1129.4 | 1.00 | 1.00 |
  | `544^3` | 1368.9 | 1372.8 | 1.00 | 1.03 |
  | `1024^3` | 8957 | 8969 | 1.00 | 0.97 |

  At 4 participants the same shapes move to 0.75/0.49/0.67/0.77/0.85/0.95 in `c` order and
  0.72/0.49/0.65/0.77/0.85/0.96 in `f` order; at 1 participant every cell is 1.00 (no route change below two members).
  `c64` at 8 participants: `32^3` 9.82 to 2.49 (0.25), `64^3` 26.66 to 10.05 (0.38), `96^3` 39.55 to
  30.06 (0.76), `128^3` 102.7 to 68.0 (0.66), `192^3` 235.5 to 225.6 (0.96), `256^3` 580.9 to 540.4
  (0.93); `f32` `128^3` 0.45 and `192^3` 0.60; `c32` `128^3` 0.64 and `192^3` 0.73.
- **Factorizations improve, none regress**, because their trailing products are the same size
  range: f64 8 participants, `cand/base`: LU `128^3` 0.91/0.95 and `256^3` 0.87/0.90, Cholesky
  `128^3` 0.95/0.97 and `256^3` 0.97/0.97, QR `256^3` 0.98, eigh `256^3` 0.99, SVD values 1.00.
  QR/LU/Cholesky/eigh at `32^3` and `64^3` are within 1.00-1.01 of the previous route (the only
  cells that moved outside it: `cholesky/32^3` read 1.10 in one scan and 1.00 on the repeat, which
  is cross-run drift on a 2.8 us cell, not the route).
- **`f32`/`c32` at `32^3` stay behind AOCL (`0.51x`, `0.45x`) and this change does not address
  that**: the serial rate of the `f32`/`c32` kernel is the limit there (`2.1 us` and `8.4 us` of
  serial work against AOCL's `0.69 us` and `0.95 us`), not the decomposition. Unverified lead for
  a separate experiment.
- **Unchanged where it must be**: no route change at one participant, no allocation on either
  route (`tests/no_step_alloc.rs`, `tests/fx128_matmul.rs`, the row-major no-copy tests pass), the
  result is the same product on every dtype and order (`fbench verify` at 1, 4 and 8 participants
  for `gemm` over `f64/c64/f32/c32` and both orders, and for `qr_factor/lu/cholesky/eigh` at
  `linalg-small`/`linalg-mid`), and `cargo test -p colmat --lib` is green.
- **Not verified here**: a 64-core or multi-CCD host, where `tile_split`'s width and the ceiling
  would need their own scan; and the aarch64/`wasm` path, whose `gemm_tiled` route is unchanged
  in behaviour (the same gate, grid and `PAYS` arithmetic, moved into shared helpers).
