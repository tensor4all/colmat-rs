<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tensor4all/tensor4all-meta/main/branding/colmat/svg/colmat-wordmark-dark.svg">
  <img alt="colmat" src="https://raw.githubusercontent.com/tensor4all/tensor4all-meta/main/branding/colmat/svg/colmat-wordmark.svg" height="64">
</picture>

[![documentation](https://docs.rs/colmat/badge.svg)](https://docs.rs/colmat)
[![crate](https://img.shields.io/crates/v/colmat.svg)](https://crates.io/crates/colmat)
[![ci](https://github.com/tensor4all/colmat-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tensor4all/colmat-rs/actions/workflows/ci.yml)
[![code quality](https://github.com/tensor4all/colmat-rs/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/tensor4all/colmat-rs/actions/workflows/code-quality.yml)

`colmat` is a pure-Rust dense and sparse linear algebra library for scientific and HPC work. It
began as the linear algebra backend of [tenferro](https://github.com/tensor4all/tenferro-rs) and the
other [tensor4all](https://github.com/tensor4all) libraries, and is now developed as a
general-purpose library.

Our goal is not to beat vendor LAPACK, but to build an equally performant linear algebra library
with precise thread control. The BLAS/LAPACK interface is old and essentially Fortran: no function
takes a thread team as an argument, and thread control lives outside the standard, in global or
per-thread settings of each vendor library. Scientific applications already have their own
scheduler (MPI ranks, then parallel tasks inside a rank), so they need nested parallelism: an
outer loop and the linear algebra it calls must share the same cores without oversubscription.
This is one real benefit of rebuilding the ecosystem in a modern language like Rust.

- **Precise thread control.** Parallel work runs on a `threadteam` team that the caller owns and
  passes as `Par::Team` (or `Par::Seq` for the calling thread). A team splits into sub-teams to any
  depth, so a batch of small matrices can run one item per thread, and a large factorization can
  use the whole team, with the same mechanism. There is no implicit global thread pool, and each
  parallel region runs at the widest width that pays for its dispatch cost.
- **Performance on par with vendor libraries.** The
  [benchmarks](https://tensor4all.org/colmat-benchmark/) compare it with OpenBLAS and MKL, with
  their measurement conditions.
- **Correct results.** Its decompositions are validated against LAPACK's own test suites, and the
  defects found on the way that also affect the published upstream crates are kept as reproducers in
  [`upstream-reports/`](upstream-reports/README.md).

It changes upstream faer's API where these goals need a different shape. Published versions follow
semver, and APIs can change in incompatible releases.

**Renamed from `t4a-faer`.** The crate was published as `t4a-faer` up to 0.33.0; 0.34.0 is the
same library under the name `colmat` (`use colmat::...`). See the
[rename note](CHANGELOG.md#breaking-the-crate-is-renamed-to-colmat). Development moved from
`tensor4all/faer` to this repository; the old repository, with its issues and pull requests, is
archived as [tensor4all/t4a-faer](https://github.com/tensor4all/t4a-faer), and each old issue has
a copy here that links back to it.

## Links

- Crate and API docs: [crates.io/crates/colmat](https://crates.io/crates/colmat), [docs.rs/colmat](https://docs.rs/colmat)
- Benchmarks, with their measurement conditions: [tensor4all.org/colmat-benchmark](https://tensor4all.org/colmat-benchmark/)
- Migration guides: [from upstream faer, nalgebra and ndarray](docs/migration-guide.md), and
  [from Eigen, BLAS/LAPACK, Julia and NumPy/SciPy](docs/migration/README.md)
- Issues: [github.com/tensor4all/colmat-rs/issues](https://github.com/tensor4all/colmat-rs/issues)
  (read the [review rules](docs/review-rules.md) first)
- Community: [#tensor4all:matrix.org](https://matrix.to/#/#tensor4all:matrix.org)

## Contributing

Read the [review rules](docs/review-rules.md) first. We are looking for:

- **Measurements on hardware we do not have.** Run the harness in
  [tensor4all/colmat-benchmark](https://github.com/tensor4all/colmat-benchmark)
  (it lives there, not in this repository) on other CPUs (ARM servers, recent
  Intel and AMD parts). Numbers must follow its
  [timing protocol](https://github.com/tensor4all/colmat-benchmark/blob/main/docs/protocol-v2-baseline.md):
  pinned CPU sets, the idle guard, and discarded attempts re-run rather than
  selected by their numbers; the commands are in its
  [rules and runbook](https://github.com/tensor4all/colmat-benchmark/blob/main/docs/rules.md).
  Post the result as a pull request to that repository. The
  [published charts](https://tensor4all.org/colmat-benchmark/) show what is
  already measured.
- **Thread width and placement.** Choosing the fastest width rather than the
  widest that pays ([#280](https://github.com/tensor4all/colmat-rs/issues/280)),
  and placement under threadteam.
- **Double-double linear algebra**, following the
  [design under review](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/next-generation-dense-linear-algebra.md).
- **Validation**: more of LAPACK's test suites, and reproducers for defects found
  on the way.

### Software paper: join as a co-author

colmat is part of the tenferro stack, and we plan to submit a software paper on
tenferro at the end of January 2027 (the journal is not fixed and acceptance is
not guaranteed). People who make a substantial contribution to colmat or the
other crates of the stack can join the author list. The criteria and how to
apply are in [tensor4all/tenferro-paper](https://github.com/tensor4all/tenferro-paper)
([AUTHORS-NOTES.md](https://github.com/tensor4all/tenferro-paper/blob/main/AUTHORS-NOTES.md));
contributors coordinate in the tenferro Matrix room
[#tenferro-tensor4all:matrix.org](https://matrix.to/#/#tenferro-tensor4all:matrix.org).
This paper does not replace upstream faer's own paper
([JOSS, doi:10.21105/joss.06099](https://doi.org/10.21105/joss.06099)), which
colmat users should keep citing.

## Origin and acknowledgment

colmat is derived from [faer](https://codeberg.org/sarah-quinones/faer) by Sarah Quiñones
(GitHub mirror: [sarah-quinones/faer-rs](https://github.com/sarah-quinones/faer-rs)). Its design,
kernels and most of its code are faer's, and we thank Sarah Quiñones and the upstream faer
contributors for them. [`PATCHSET.md`](PATCHSET.md) records how colmat differs from upstream, and
[`upstream-reports/`](upstream-reports/README.md) holds reproducers for defects found in the published
upstream crates. colmat is not upstream faer, and questions about it belong here rather than with
the upstream project.

colmat is released under the MIT license ([`LICENSE`](LICENSE)), which carries the upstream copyright,
Copyright (c) 2026 sarah quiñones el kazdadi. Changes made in this repository are Copyright (c)
tensor4all contributors under the same license. Code ported from other projects keeps its own terms;
[`NOTICE`](NOTICE) lists the attributions and the `COPYING.*` files that ship with the crate.
