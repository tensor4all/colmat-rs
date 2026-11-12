#!/usr/bin/env bash
# Table D: the same `lu` cells under four glibc growth policies, one `fbench` process per cell.
#
#   run_alloc_diag.sh <out-dir> <fbench-binary>
#
# These are diagnostics, not campaign cells: the record uses them only to attribute the minor page
# faults table C measures in the owning route. `MALLOC_*` is passed through the harness untouched
# (it strips RAYON/OMP/MKL/GOMP only), and every cell still runs inside `pinned.sh`.
set -euo pipefail
out=${1:?usage: run_alloc_diag.sh <out-dir> <fbench>}
bin=${2:?}
bench=${BENCH_REPO:?set BENCH_REPO to the faer-benchmark checkout}
mkdir -p "$out"
run() { # name, extra env, rows, dtype, family (default both)
    local name=$1 envs=$2 rows=$3 dt=$4 fam=${5:-lu,solve}
    echo "== $name $rows $dt"
    env $envs taskset -c 8-15,24-31 "$bench/benchmarks/scripts/pinned.sh" 0-7 -- \
        "$bin" run --threads 8 --n 512 --dtype "$dt" --family "$fam" --regime linalg-mid \
        --rows "$rows" --reps 5 --prime-ms 250 --statistic median \
        --csv "$out/$name-$dt.csv" 2>&1 | sed 's/^/   /' | grep -v '^   CELL' || true
}
for dt in f64 c64; do
    run alloc-policy-base   ""                               faer-fork,lapack-mkl "$dt"
    run alloc-policy-mmap64 "MALLOC_MMAP_THRESHOLD_=67108864" faer-fork,lapack-mkl "$dt"
    run alloc-policy-toppad "MALLOC_TOP_PAD_=33554432"        faer-fork,lapack-mkl "$dt"
    run alloc-policy-trim   "MALLOC_TRIM_THRESHOLD_=67108864" faer-fork,lapack-mkl "$dt"
    run toppad-faer-upstream "MALLOC_TOP_PAD_=33554432"       faer-upstream        "$dt"
    # The upstream control row is `lu` only: the point is its own owning-against-in-place gap in
    # one process, and `lu` is where the fork's gap is largest.
    for set in 1 2 3; do
        run row-isolated-set${set}-faer-fork     "" faer-fork     "$dt" lu
        run row-isolated-set${set}-faer-upstream "" faer-upstream "$dt" lu
    done
done
