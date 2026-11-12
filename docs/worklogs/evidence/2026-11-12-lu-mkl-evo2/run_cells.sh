#!/usr/bin/env bash
# The campaign-protocol cell measurement on evo2 (the 32-logical Zen5 host, which no declared
# `faer-benchmark` profile describes, so `record_run.py` refuses it): `fbench`, MKL arm, three
# engines in one binary, one process per (thread budget, size), three process sets per cell.
#
#   run_cells.sh <out-dir> <fbench-binary> [set-numbers...]
#
# CPU sets: 1T = 0, 4T = 0-3, 8T = 0-7 (one L3 domain, CPUs 0-7 + siblings 16-23), the observing
# side on 8-15,24-31, exactly as `docs/worklogs/2026-11-12-eigh-vs-mkl-evo2.md` records for this
# host. Every cell goes through `benchmarks/scripts/pinned.sh`, strict (3 s, 5 % busy).
set -euo pipefail
out=${1:?usage: run_cells.sh <out-dir> <fbench> [sets...]}
bin=${2:?}
shift 2
sets=("$@")
[[ ${#sets[@]} -gt 0 ]] || sets=(1 2 3)
here=$(cd "$(dirname "$0")" && pwd)
bench=${BENCH_REPO:?set BENCH_REPO to the faer-benchmark checkout}
mkdir -p "$out"
for set in "${sets[@]}"; do
    for t in 1 4 8; do
        case "$t" in
            1) cpus=0 ;;
            4) cpus=0-3 ;;
            8) cpus=0-7 ;;
        esac
        for n in 128 256 512 1024; do
            csv="$out/run${set}-${t}t-n${n}.csv"
            echo "== set $set ${t}T n=$n"
            taskset -c 8-15,24-31 "$bench/benchmarks/scripts/pinned.sh" "$cpus" -- \
                "$bin" run --threads "$t" --n "$n" --dtype f64,c64 \
                --family lu,solve --regime linalg-mid \
                --rows faer-fork,faer-upstream,lapack-mkl \
                --reps 5 --prime-ms 250 --rotate-rows 3 --statistic median \
                --csv "$csv" 2>&1 | sed 's/^/   /' | grep -v '^   CELL' || true
        done
    done
done
