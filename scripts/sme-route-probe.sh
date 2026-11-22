#!/usr/bin/env bash
# Which t4a-gemm kernel does colmat's `matmul` reach on an Apple SME host? (tensor4all/colmat-rs#180)
#
# Runs `faer/examples/sme_route.rs` under lldb once per (dtype, shape, threads) and counts the hits
# on the SME `gemm_basic` of that element type and on its NEON fallback (`neon` for f32/f64,
# `neonfcma` for c32/c64). The counts are calls, not time; the example's own timing under the
# debugger is meaningless and is dropped. Time the same shapes without lldb:
#
#   target/release/examples/sme_route <dtype> <m> <n> <k> <threads> [samples]
#
# Usage: scripts/sme-route-probe.sh [threads...]     (default: 1 6)
# Output: one TSV row per run on stdout: dtype m n k threads sme_calls fallback_calls expected_seq route
# where `expected_seq` is what t4a-gemm-common's `sme_thresholds` says for the whole (sequential) shape.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]]; then
	echo "sme-route-probe: Apple Silicon only" >&2
	exit 2
fi
cargo build --release -p colmat --example sme_route >&2
bin="$PWD/target/release/examples/sme_route"
if (($#)); then threads_list=("$@"); else threads_list=(1 6); fi

# The gate of t4a-gemm-common 0.20.4 `sme_thresholds.rs`: k >= 64, m*n >= 2048, and
# m*n*k >= 131072 (f64/c64) or 524288 (f32/c32).
expected() {
	local dtype=$1 m=$2 n=$3 k=$4 work
	case $dtype in f64 | c64) work=131072 ;; *) work=524288 ;; esac
	if ((k >= 64 && m * n >= 2048 && m * n * k >= work)); then echo sme; else echo fallback; fi
}

# Crossover shapes named in sme_thresholds.rs, then the #87 size.
shapes=(
	"16 64 128" "32 64 64" "256 256 32" "64 128 64" "64 64 64" "64 64 128" "128 128 64"
	"256 256 256" "512 512 512" "2048 2048 2048"
)

script=$(mktemp)
trap 'rm -f "$script"' EXIT
printf 'dtype\tm\tn\tk\tthreads\tsme_calls\tfallback_calls\texpected_seq\troute\n'
for dtype in f32 f64 c32 c64; do
	case $dtype in
	f32) crate=gemm_f32 mod=f32 fb=neon ;;
	f64) crate=gemm_f64 mod=f64 fb=neon ;;
	c32) crate=gemm_c32 mod=f32 fb=neonfcma ;;
	c64) crate=gemm_c64 mod=f64 fb=neonfcma ;;
	esac
	cat >"$script" <<EOF
breakpoint set -r "^${crate}::gemm::${mod}::sme::gemm_basic" -G true
breakpoint set -r "^${crate}::gemm::${mod}::${fb}::gemm_basic" -G true
run
breakpoint list
quit
EOF
	for t in "${threads_list[@]}"; do
		for s in "${shapes[@]}"; do
			read -r m n k <<<"$s"
			out=$(lldb -b -s "$script" -- "$bin" "$dtype" "$m" "$n" "$k" "$t" 1 2>&1)
			if ! grep -q "^dtype=" <<<"$out"; then
				echo "sme-route-probe: run failed for $dtype $m $n $k $t" >&2
				echo "$out" >&2
				exit 1
			fi
			sme=$(grep -E '^1: ' <<<"$out" | sed -E 's/.*hit count = ([0-9]+).*/\1/')
			fbc=$(grep -E '^2: ' <<<"$out" | sed -E 's/.*hit count = ([0-9]+).*/\1/')
			if ((sme > 0 && fbc == 0)); then route=sme; elif ((sme == 0 && fbc > 0)); then route=fallback; else route=mixed; fi
			printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$dtype" "$m" "$n" "$k" "$t" "$sme" "$fbc" \
				"$(expected "$dtype" "$m" "$n" "$k")" "$route"
		done
	done
done
