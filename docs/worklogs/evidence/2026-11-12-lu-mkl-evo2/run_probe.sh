#!/usr/bin/env bash
# Phase decomposition of the owning `partial_piv_lu` route, one cell per process.
#
#   run_probe.sh <out-dir> [threads...]
#
# CPU sets follow the evo2 record (`docs/worklogs/2026-11-12-eigh-vs-mkl-evo2.md`): 1T = 0,
# 4T = 0-3, 8T = 0-7, one L3 domain, the observer on 8-15,24-31. Every cell goes through
# `benchmarks/scripts/pinned.sh` of `tensor4all/faer-benchmark`, so a cell that met foreign load
# is retried and recorded, never silently kept.
set -euo pipefail
out=${1:?usage: run_probe.sh <out-dir> [threads...]}
shift || true
threads_list=("$@")
[[ ${#threads_list[@]} -gt 0 ]] || threads_list=(1 4 8)
here=$(cd "$(dirname "$0")" && pwd)
bench=${BENCH_REPO:?set BENCH_REPO to the faer-benchmark checkout}
bin=${PROBE_BIN:-$here/probe/target/release/lu123}
mkdir -p "$out"
for t in "${threads_list[@]}"; do
    case "$t" in
        1) cpus=0 ;;
        4) cpus=0-3 ;;
        8) cpus=0-7 ;;
        *) echo "no CPU set declared for ${t}T" >&2; exit 2 ;;
    esac
    for dtype in f64 c64; do
        for order in c f; do
            for n in 128 256 512 1024; do
                csv="$out/probe-${t}t-${dtype}-${order}-n${n}.csv"
                echo "== cell ${t}T $dtype $order n=$n"
                taskset -c 8-15,24-31 "$bench/benchmarks/scripts/pinned.sh" "$cpus" -- \
                    "$bin" --threads "$t" --dtype "$dtype" --order "$order" --n "$n" \
                    --reps 5 --prime-ms 250 --csv "$csv" 2>&1 | sed 's/^/   /' \
                    | grep -v '^   CELL' || true
            done
        done
    done
done
