# Index soundness: `One`-shaped permutations and `Array` (colmat-rs#256)

## Decisions

- `Idx<One, I>` is the zero-sized `Zero`, so a `[Idx<One, I>]` is never a
  `[I]` in memory. Every reinterpretation between index slices is now taken
  only when both sides are stored as `I`, which `idx_has_index_layout::<N, I>()`
  decides as `size_of::<Idx<N, I>>() == size_of::<I>()`. `Shape` is sealed and
  its index types are `I`, the `#[repr(transparent)]` `bound::Idx<'n, I>`, or
  `Zero`, so the size comparison is exact and constant-folds.
- When a side is `One`, the dimension is one and the only permutation is the
  identity `[0]`. `Perm::into_shape` then drops the input boxes with their own
  layout and allocates the result as `Box<[M::Idx<I>]>` with `M`'s layout: one
  layout per allocation, no reinterpretation. The borrowed forms
  (`cast_idx_slice`, `PermRef::as_shape`) have no `I` in memory to borrow, so
  they return a slice of a static, 8-aligned zero (`zero_idx_slice`). It holds
  one entry, which is the length of every `One`-shaped index slice that a
  permutation carries; `cast_idx_slice` panics on a longer `[Zero]` instead of
  reading past it. Rejected: panicking for every `One` input (the colmat-next-rs
  choice), because `PermRef::<_, One>::new_checked(&[Zero], &[Zero], One)` is a
  valid permutation and should keep working.
- `Array::as_mut` returns a slice that borrows `&mut self`. This is the one
  signature change; no in-crate caller relied on the unbounded lifetime.
- `Array`'s range `Index`/`IndexMut` assert `start <= end` before slicing.
  `IdxInc<'n>` bounds each end by the length, not their order. Only the release
  arm (`get_unchecked`) needed it, since the debug arm panicked through the
  checked slice, but the assertion runs in both profiles. The reversed-range
  tests pin its message, so the debug-profile Miri lane notices if it is removed;
  without it they fail there with the slice's message.
- `IdxIncOne` and `MaybeIdxOne` gain `#[repr(transparent)]`. `cast_idx_inc_slice`
  reinterprets `[IdxIncOne<I>]` as `[I]`, which the default representation does
  not guarantee. It has no in-crate caller, so this is defensive, not a
  reproduced defect.

## Verification conclusions and constraints

- Miri (x86_64 host, nightly 2026-11-13, `--release` so the range arms are the
  unchecked ones): before the fix the issue's four reproducers, and two more for
  `PermRef::as_shape` and `Perm::into_shape` from `One` to `usize`, report
  out-of-bounds reads, the two 4-byte leaks, a Stacked Borrows violation, an
  `unchecked_sub` overflow and a dangling `Box`. After it, the perm reproducers
  are clean, the range reproducer panics on its assertion, and the aliasing one
  no longer compiles. Of the ten in-crate tests in `perm::tests` and
  `utils::bound::tests`, the six that reach the defects report UB under Miri on
  the unfixed tree (the other four are controls: the `usize`/`Dim` cast path,
  two-entry rejection, sequential `as_mut`, ordered ranges); all ten pass on the
  fix in both profiles. They are added to the weekly aarch64 Miri lane; they were
  run under Miri on x86_64 only, and they do not reach `pulp`'s SIMD.
- Sites checked for the same pattern: the sparse `row_idx_of_col_raw` /
  `col_idx_of_row_raw` reinterpret `[I]` as `[Idx<_, I>]`, which for `One` is a
  `[Zero]` (zero-sized, any aligned pointer is valid), so they are sound.
  `PermRef::canonicalized` / `uncanonicalized` keep the shape and change only
  `I` to an index of the same size, so both sides have one layout.
- Upstream: all four sites are inherited from faer `04ef63628` (2024-12-04), and
  upstream `main` at `0ecd846d` (2026-10-17) still has the same code. That is a
  reading of the source; upstream was not run under Miri. Feedback is deferred.
- No performance claim. The layout test is a compile-time constant, the
  `start <= end` comparison is one branch per range index, and the `One` paths
  are reached only at dimension one.
