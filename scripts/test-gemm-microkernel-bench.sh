#!/usr/bin/env bash
# Tests of scripts/gemm-microkernel-bench.sh and of the ISA guard of the gemm_microkernel probe
# (tensor4all/colmat-rs#253, review of #278).
#
# Usage: scripts/test-gemm-microkernel-bench.sh [PROBE_BINARY]
#
# The wrapper tests run a stub probe (no build): the CPU is chosen in one place, a saved binary is
# rerun only with its build record, and meta.txt keeps the build and the invoking checkout apart.
# With PROBE_BINARY (a built `gemm_microkernel` example, debug or release) it also disassembles
# `fma_zmm` and fails if a ZMM instruction there is not AVX-512F (the probe checks only avx512f).
set -euo pipefail

cd "$(dirname "$0")/.."
wrapper=$PWD/scripts/gemm-microkernel-bench.sh
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fails=0
pass() { echo "ok   $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

# a stub probe: prints the header line the wrapper cross-checks, pinned to its --cpu argument
# unless STUB_PINNED overrides it
stub=$work/stub
cat >"$stub" <<'SH'
#!/bin/sh
cpu=
while [ $# -gt 1 ]; do
	[ "$1" = --cpu ] && cpu=$2
	shift 2
done
echo "# pinned_cpu: ${STUB_PINNED:-$cpu} (smt siblings stub)"
SH
chmod +x "$stub"
sha=$(sha256sum "$stub" | cut -d' ' -f1)
build_head=0123456789abcdef0123456789abcdef01234567
record() { # sha -> a build record for the stub, claiming a build at $build_head
	printf 'binary_sha256: %s\nbuild_date: 2026-01-01T00:00:00Z\nbuild_git_head: %s\nbuild_git_dirty: 0 tracked file(s) modified\nbuild_rustc: rustc stub\nbuild_lock_private_gemm_x86: stub\n' "$1" "$build_head" >"$stub.build"
}
run() { # name args... -> exit status in $st, output in $work/name.log
	local name=$1
	shift
	st=0
	"$wrapper" "$work/$name" "$@" >"$work/$name.log" 2>&1 || st=$?
}
meta() { grep -c "^$2" "$work/$1/meta.txt" || true; }
cpu=0
other=1
[[ -d /sys/devices/system/cpu/cpu$other ]] || other=

# 1. the CPU is chosen only by the wrapper: a forwarded --cpu is refused before anything runs
record "$sha"
for extra in "--cpu 3" "--cpu=3"; do
	# shellcheck disable=SC2086
	run dup --cpu "$cpu" --bin "$stub" -- $extra
	if ((st == 2)) && [[ ! -e $work/dup ]] && grep -q 'CPU is set by this script' "$work/dup.log"; then
		pass "forwarded '$extra' is refused"
	else
		fail "forwarded '$extra': exit $st, out dir $([[ -e $work/dup ]] && echo created || echo absent)"
	fi
	rm -rf "$work/dup"
done

# 2. the probe's pinned CPU is cross-checked against the watched one
if [[ -n $other ]]; then
	STUB_PINNED=$other run mismatch --cpu "$cpu" --bin "$stub"
	if ((st == 3)) && grep -q '^verdict: DISCARDED' "$work/mismatch/meta.txt" &&
		grep -q "^probe_pinned_cpu: $other$" "$work/mismatch/meta.txt"; then
		pass "a probe pinned elsewhere discards the run"
	else
		fail "probe pinned to $other while $cpu watched: exit $st"
	fi
fi

# 3. a saved binary is rerun only with its own build record
rm -f "$stub.build"
run norecord --cpu "$cpu" --bin "$stub"
if ((st == 2)) && [[ ! -e $work/norecord ]]; then pass "--bin without a build record is refused"; else fail "--bin without a record: exit $st"; fi
record "$(printf '%064d' 0)"
run badsha --cpu "$cpu" --bin "$stub"
if ((st == 2)) && grep -q 'does not match its build record' "$work/badsha.log"; then pass "--bin with another binary's record is refused"; else fail "--bin with a wrong record: exit $st"; fi

# 4. meta.txt names the build from the record and the invoking checkout separately
record "$sha"
run rerun --cpu "$cpu" --bin "$stub" -- --samples 5
m=$work/rerun/meta.txt
saved=$work/rerun/bin/gemm_microkernel-${sha:0:16}
if ((st == 0 || st == 3)) &&
	[[ $(meta rerun "build_git_head: $build_head$") == 1 ]] &&
	[[ $(meta rerun "invoker_git_head: $(git rev-parse HEAD)$") == 1 ]] &&
	[[ $(meta rerun "git_head:") == 0 ]] &&
	[[ $(meta rerun "binary_sha256: $sha$") == 1 ]] &&
	[[ $(meta rerun "build_rustc: rustc stub$") == 1 ]] &&
	[[ $(meta rerun "rustc:") == 0 ]] &&
	[[ $(meta rerun "args: --cpu $cpu --bin $stub -- --samples 5$") == 1 ]] &&
	[[ $(meta rerun "probe_pinned_cpu: $cpu$") == 1 ]] &&
	cmp -s "$stub.build" "$saved.build"; then
	pass "a rerun reports the build's head, rustc and lock beside the invoker's head"
else
	fail "rerun provenance: exit $st"
	cat "$m" 2>/dev/null || cat "$work/rerun.log"
fi
# and the copied record makes the saved copy itself rerunnable
run rerun2 --cpu "$cpu" --bin "$saved"
if ((st == 0 || st == 3)) && [[ $(meta rerun2 "build_git_head: $build_head$") == 1 ]]; then pass "the saved copy reruns with its record"; else fail "rerun of the saved copy: exit $st"; fi

# 5. the ZMM ceiling probe uses AVX-512F instructions only
if (($#)); then
	dis=$(objdump -d --demangle --no-show-raw-insn -M intel "$1" |
		awk '/^[0-9a-f]+ <.*imp::fma_zmm[>:]/ { f = 1; next } f && /^$/ { f = 0 } f')
	# mnemonics with a ZMM operand (not a jump whose target label says fma_zmm)
	zmm=$(awk '$3 ~ /zmm/ { print $2 }' <<<"$dis" | sort -u | tr '\n' ' ')
	bad=$(awk '$3 ~ /zmm/ { print $2 }' <<<"$dis" | grep -vxE 'vfmadd231pd|vpxord|vpxorq' | sort -u | tr '\n' ' ' || true)
	if [[ -z $zmm ]]; then
		fail "fma_zmm not found in $1"
	elif [[ -n $bad ]]; then
		fail "fma_zmm uses ZMM instructions beyond AVX-512F: $bad"
	else
		pass "fma_zmm ZMM instructions are AVX-512F: $zmm"
	fi
fi

((fails == 0)) || { echo "$fails failure(s)"; exit 1; }
echo "all passed"
