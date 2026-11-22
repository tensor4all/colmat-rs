#!/usr/bin/env bash
# Which t4a-gemm kernel does colmat's `matmul` reach on an Apple SME host? (tensor4all/colmat-rs#180)
#
# Runs `faer/examples/sme_route.rs` under lldb once per (dtype, shape, threads) and counts the hits
# on the SME `gemm_basic` of that element type and on its NEON fallback (`neon` for f32/f64,
# `neonfcma` for c32/c64). The counts are kernel calls, not time; the example's own timing under
# the debugger is meaningless and is dropped. Time the same shapes without lldb:
#
#   target/release/examples/sme_route <dtype> <m> <n> <k> <threads> [samples]
#
# Usage: scripts/sme-route-probe.sh [threads...]     (default: 1 6)
# Output: one TSV row per run on stdout:
#   dtype m n k threads matmul_calls sme_calls fallback_calls gate_seq route
# `matmul_calls` is how many `matmul` calls the example made, so a kernel count above it means the
# team cut each product into several sequential kernel calls (tiles). `gate_seq` is the linked
# `t4a-gemm-common`'s `sme_thresholds` verdict for the whole shape as one sequential call, printed
# by the example itself. `route` is `sme`, `fallback` or `mixed` (both kernels ran). A breakpoint
# that does not resolve to exactly one location (a renamed kernel after a `t4a-gemm` bump), or a
# run where neither kernel fired, stops the script with an error instead of printing a row.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]]; then
	echo "sme-route-probe: Apple Silicon only" >&2
	exit 2
fi
cargo build --release -p colmat --example sme_route >&2
# The example prints `gate_seq` from its own `t4a-gemm-common` dev-dependency. That is the gate the
# kernels use only while the lock holds one copy of the crate: a `t4a-gemm` bump to another
# semver range would leave the dev-dependency on the old copy.
copies=$(grep -c '^name = "t4a-gemm-common"$' Cargo.lock || true)
if ((copies != 1)); then
	echo "sme-route-probe: Cargo.lock has $copies copies of t4a-gemm-common; bump the dev-dependency in faer/Cargo.toml with t4a-gemm" >&2
	exit 1
fi
bin="${CARGO_TARGET_DIR:-$PWD/target}/release/examples/sme_route"
if (($#)); then threads_list=("$@"); else threads_list=(1 6); fi

# Crossover shapes named in t4a-gemm-common's sme_thresholds.rs, then the #87 size.
shapes=(
	"16 64 128" "32 64 64" "256 256 32" "64 128 64" "64 64 64" "64 64 128" "128 128 64"
	"256 256 256" "512 512 512" "2048 2048 2048"
)

field() { # field <name> <text>: the value of `name=value` in the example's result line
	grep -E '^dtype=' <<<"$2" | tr ' ' '\n' | sed -n "s/^$1=//p"
}

script=$(mktemp)
trap 'rm -f "$script"' EXIT
printf 'dtype\tm\tn\tk\tthreads\tmatmul_calls\tsme_calls\tfallback_calls\tgate_seq\troute\n'
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
			fail() {
				echo "sme-route-probe: $1 ($dtype $m $n $k threads=$t)" >&2
				echo "$out" >&2
				exit 1
			}
			grep -q "^dtype=" <<<"$out" || fail "the example did not finish"
			for id in 1 2; do
				grep -qE "^${id}: .*locations = 1, resolved = 1," <<<"$out" ||
					fail "breakpoint $id does not resolve to exactly one kernel symbol"
			done
			sme=$(grep -E '^1: ' <<<"$out" | sed -E 's/.*hit count = ([0-9]+).*/\1/')
			fbc=$(grep -E '^2: ' <<<"$out" | sed -E 's/.*hit count = ([0-9]+).*/\1/')
			[[ $sme =~ ^[0-9]+$ && $fbc =~ ^[0-9]+$ ]] || fail "no hit count in lldb's breakpoint list"
			if ((sme > 0 && fbc == 0)); then
				route=sme
			elif ((sme == 0 && fbc > 0)); then
				route=fallback
			elif ((sme > 0 && fbc > 0)); then
				route=mixed
			else
				fail "neither kernel was called"
			fi
			printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$dtype" "$m" "$n" "$k" "$t" \
				"$(field calls "$out")" "$sme" "$fbc" "$(field gate_seq "$out")" "$route"
		done
	done
done
