# The tiled layout copy answers to the source's row stride (#323)

## Decisions

- `#323` asked for a working-set-size predicate at the tiled copy's decision point. That half had already
  landed as `#330` (`TILE_COPY_CACHE_BYTES` = both buffers of a `512 x 512` f64 copy, 4 MiB) plus `#352`'s
  power-of-two clause, so this change starts from what measurement said was still wrong, not from the issue's
  text.
- Measured first, and the measurement falsified the issue's remaining item. **Raggedness is not a cost.** On
  this host, `Llt::new` on a row-major f64 matrix at 1T: `511`, `513` and `520` (ragged, `n % 16 != 0`) lose
  to the elementwise walk by 8.3-9.1%, and `528`, `1008`, `1040` (`n % 16 == 0`) lose by 5.8-8.1%; `512` and
  `1024` win for the tiles (23.6% / 15.0%) because their row stride is a power of two, not because they are
  aligned.
- A 2x2 factorial on the copy alone isolates what does decide. `MatMut::copy_from_triangular_lower` at
  `511 x 511`, with the destination's column stride and the source's row stride controlled independently
  through views (us, 21 reps, one process per cell, 1T):

  | source row stride | destination column stride | tiles | walk |
  | --- | --- | --- | --- |
  | 4088 B | 4096 B | 402 | 120 |
  | 4088 B | 4160 B | 402 | 117 |
  | 4096 B | 4096 B | 393 | 1474 |
  | 4096 B | 4160 B | 398 | 1490 |
  | 4160 B | 4096 B | 381 | 115 |
  | 4160 B | 4160 B | 381 | 111 |

  The tiles cost 380-400 us whatever the strides are. The walk costs 111-120 us unless the **source's row
  stride** is a power of two, where it costs 1.5 ms: the walk reads the source with that stride
  (`PERFORMANCE_TIPS.md`, "A leading dimension that is a multiple of 1 KiB aliases L1 sets"). The
  destination's column stride moves nothing measurable (2% across the four cells).
- Chosen: `tile_copy_pays` keeps its destination clause and additionally requires a power-of-two source row
  stride. The source side gets no floor of its own, because the destination's floor already gates the clause
  and a 2048-byte source floor would silently revert `#355`'s `2048 x 64` cell (source row stride 512 B).
- What the source side was hiding: an owned f64 matrix rounds its column stride up to a multiple of 8 rows
  (`matown::align_for`), so a `505 x 505` destination has a 512-column = 4096-byte column stride, a power of
  two, while its source row stride is 4088 bytes and not one. `505`..`511` took the tiles on the destination
  stride alone. `#290`'s `511` cell (+4.7% against the tiles on evo2) is this cell, not a ragged edge.
- Not chosen: replacing the destination clause with the source clause. No measurement shows the tiles paying
  when the destination stride is not a power of two, and replacing would revert the four QR/LQ guard cells
  `#355` flipped without measuring them.
- Not done, deliberately: the band **above** the byte count still takes the tiles where the walk is cheaper on
  a host whose L3 does not hold the pair (`513`..~`1280` here, 5.8-10.3%). The byte constant is pinned to the
  cache of the host `#290` measured, and moving it is a change that needs its own measurement on each host
  class. Recorded in the `TILE_COPY_CACHE_BYTES` doc comment as the open half of `#323`.

## Verification conclusions and constraints

Host and protocol: `broccoli`, AMD EPYC 7702P (Zen2), 64 CPUs online with the agent confined to CPUs 60-63 =
one 16 MiB L3 domain (L2 512 KiB/core), `boost` 0 (disabled), governor `performance`, 1796 MHz. Foreign load
on the slot throughout: 2-3% of each CPU (other agent sessions). Release build (`--release`). One thread
(`Par::Seq`), pinned to cpu63; 5 reps after a 250 ms prime, median per process, one process per cell,
interleaved process sets for the arms. **This is not an adoption-protocol acceptance**: the observing side
cannot be pinned outside the measured L3 domain on a 4-CPU slot, and the arms of the "before" column were
selected by a temporary env switch in the same binary (removed before the commit). It is a decision
experiment, and `#290` already records that this harness class cannot resolve a few percent below
`n = 256`.

`Llt::new` on a row-major f64 matrix, before = the gate as it stood, after = this change, 1T, median ms:

| n | band | before | after | change |
| --- | --- | --- | --- | --- |
| 250 | 249..255 | 0.529 | 0.443 | -16.2% |
| 255 | 249..255 | 0.520 | 0.474 | -8.9% |
| 505 | 505..511 | 3.630 | 3.323 | -8.5% |
| 508 | 505..511 | 3.759 | 3.322 | -11.6% |
| 511 | 505..511 | 3.764 | 3.466 | -7.9% |
| 512 | unchanged | 3.598 | 3.598 | 0.0% |
| 256 | unchanged | 0.529 | 0.504 | -4.8% (same dispatch; drift) |
| 1024 | unchanged | 25.197 | 25.229 | +0.1% |
| 1536 | unchanged | 84.218 | 83.984 | -0.3% |
The other scalars, same shape of measurement (before / after, median ms): c64 126 0.371 / 0.340, 127 0.268 /
0.246, 254 1.568 / 1.516, 255 1.624 / 1.560, and 256 1.609 / 1.622 (unchanged dispatch); f32 500 1.795 /
1.638, 505 1.840 / 1.761, 511 1.897 / 1.681, and 512 1.873 / 1.888 (unchanged); c32 250 1.155 / 1.075, 255
1.181 / 1.103. Each band fixes the same thing: the destination column stride is a power of two (the owned
destination rounds its rows up to 8 for f64 and c32, 4 for c64, 16 for f32), the source row stride is not.

Every "after" cell equals the walk arm measured in the same session on the same binary, so the change is the
dispatch flip and not a speed-up of either path. The cells whose dispatch the change does not touch (512, 256,
1024, 1536, c64 256, f32 512) reproduce within 1% at the large sizes; the 256 cells move by up to 5%, which is
this host's resolution at single-digit-microsecond cells.

Blast radius: the clause is reached through `copy_from_triangular_lower`, `copy_from_triangular_upper` (both
views transposed, so the argument is the stride the walk reads with) and so the QR/LQ `copy_r`/`copy_l` routes,
`Llt`/`Ldlt`/`Lblt`, the triangular solves in `solvers.rs`, `eigh`'s tridiagonal copy, the operator `lsmr`
residual and the sparse Cholesky extraction. Only the `Llt::new` cells above are measured end to end. The rest
are argued from the stride each one passes, which is the walk's own read stride:

- A **column-major** source has `row_stride() == 1`, and `copy_lower_across_layouts` returns before it asks the
  gate at all, so `Llt::new(A, Side::Lower)` and everything downstream of `to_owned()` are untouched.
- The `Side::Upper` / adjoint arms (`solvers.rs:989`, `1037`, `1093`, the sparse extractions at
  `sparse/linalg/cholesky.rs:5328` and its neighbours) hand the copy `A.adjoint()` of a column-major matrix,
  whose row stride is the owner's padded column stride -- a power of two whenever the byte count lets the gate
  fire -- so they keep the tiles exactly as before, and correctly: that is the stride the walk would alias on.
- The QR/LQ packed factors are row-major by construction (`qr/.../factor.rs`'s `copy_r`), which is the case the
  change was measured on, and their dispatch tests read `tiled_copies()`: `2048 x 256`, `64 x 2048`,
  `2048 x 64` and `256 x 2048` still take the path `#355` recorded. Their end-to-end times are not usable
  evidence here: `Qr::new` at `2048 x 256`
moves by 33.17 vs 33.07 ms between arms while single samples scatter over 17-49 ms, because the copy is a
fraction of a millisecond next to the factorisation. The four route tests assert the dispatch instead, which is
the contract.

## Tests

- `linalg::tests::tile_copy_threshold_is_the_measured_crossover` now asks the gate with the padded destination
  stride a real copy has and pins the affected bands (`505`, `508`, `511` and `250`, `255` walk; `256`, `512`
  tiles) for f64, c32 `255`/`256`, c64 `126`/`255`/`256` (both c64 bands) and f32 `505`/`512`.
- `linalg::tests::tile_copy_pays_on_a_power_of_two_stride_pair` pins that the destination stride alone is not
  enough.
- `linalg::tests::tile_copy_is_chosen_on_the_working_set_not_on_the_layout` and
  `tiled_triangular_copy_writes_what_the_elementwise_one_does` ask the predicate with `out.col_stride()` and
  `src.row_stride()` instead of `m as isize`; asking it with the side is how the padded-stride band stayed
  invisible.
- The QR/LQ route tests drop the re-derived predicate assert and keep the `tiled_copies()` dispatch assert.

Local gate: `cargo test -p t4a-faer --lib` 819 passed, 162 ignored; `cargo clippy -p t4a-faer-traits
--all-targets -- -D warnings` clean. `cargo fmt --all -- --check` reports the whole tree under stable
rustfmt (the config needs nightly), pre-existing and unrelated to this diff; CI's formatting step is
`continue-on-error`.
