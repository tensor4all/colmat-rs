#!/usr/bin/env bash
# Microkernel-level benchmark of the x86 f64 GEMM kernels, with its pins recorded (tensor4all/colmat-rs#253, E0).
#
# Builds `faer/examples/gemm_microkernel.rs` (or takes a saved binary with --bin), keeps a copy of
# the binary named by its sha256 under OUT/bin so a later experiment can be paired against the very
# same baseline binary, records the build and host pins, and runs the suite pinned to one CPU.
#
# Usage: scripts/gemm-microkernel-bench.sh OUT [--cpu N] [--bin PATH] [-- <example args>]
#   OUT        new directory for meta.txt, results.tsv, stderr.txt and bin/
#   --cpu N    measured CPU (default 2); its SMT siblings must stay idle. This is the only place
#              the CPU is chosen: a --cpu among the example args is refused
#   --bin PATH run this binary instead of building one (for a paired re-run of a saved baseline);
#              it needs the build record PATH.build that a build run writes next to the saved copy
#
# Provenance: a build run writes OUT/bin/gemm_microkernel-<sha16>.build, the build's git head,
# dirty count, rustc and lock pin keyed by the binary's full sha256. meta.txt takes the build_*
# lines from that record (a --bin run checks the sha256 and copies it), and names the checkout
# that invoked the run separately (invoker_git_*).
#
# Validity gate: the busy share of every *other* logical CPU of the measured core (its SMT
# siblings) over the whole run, from /proc/stat, must stay under 2%; otherwise the run is marked
# DISCARDED in meta.txt and the script exits 3. Pin your own agent and any other busy process off
# the measured core before starting (`taskset -apc <other cpus> <pid>`).
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
	echo "gemm-microkernel-bench: x86_64 Linux only" >&2
	exit 2
fi
(($#)) || { sed -n '2,/^set /{/^set /!p}' "$0" >&2; exit 2; }
out=$1
shift
all_args="$*"
cpu=2
bin=
while (($#)); do
	case $1 in
	--cpu) cpu=$2; shift 2 ;;
	--bin) bin=$2; shift 2 ;;
	--) shift; break ;;
	*) echo "unknown argument $1" >&2; exit 2 ;;
	esac
done
die() { echo "gemm-microkernel-bench: $*" >&2; exit 2; }
[[ $cpu =~ ^[0-9]+$ ]] || die "--cpu $cpu is not a CPU number"
# The probe pins itself to the CPU it is given; a second --cpu would make it run on a core other
# than the one whose siblings this script watches.
for a in "$@"; do
	[[ $a == --cpu || $a == --cpu=* ]] && die "--cpu among the example args: the CPU is set by this script's --cpu only"
done
[[ -e $out ]] && die "$out exists; a run goes to a new directory"

if [[ -n $bin ]]; then
	[[ -f $bin ]] || die "--bin $bin: no such file"
	record_src="$bin.build"
	[[ -f $record_src ]] || die "--bin $bin: no build record $record_src (a saved binary is rerun with the record its build wrote)"
	sha=$(sha256sum "$bin" | cut -d' ' -f1)
	rec_sha=$(awk '$1 == "binary_sha256:" { print $2 }' "$record_src")
	[[ $rec_sha == "$sha" ]] || die "--bin $bin: sha256 $sha does not match its build record ($rec_sha)"
	source="saved binary $bin"
fi
mkdir -p "$out/bin"

if [[ -z $bin ]]; then
	cargo build --release -p colmat --example gemm_microkernel >&2
	bin="${CARGO_TARGET_DIR:-$PWD/target}/release/examples/gemm_microkernel"
	sha=$(sha256sum "$bin" | cut -d' ' -f1)
	record_src="$out/bin/build-record.tmp"
	{
		echo "binary_sha256: $sha"
		echo "build_date: $(date -u +%FT%TZ)" # when this record was written, right after cargo build
		echo "build_git_head: $(git rev-parse HEAD)"
		echo "build_git_dirty: $(git status --porcelain --untracked-files=no | wc -l) tracked file(s) modified"
		echo "build_rustc: $(rustc -V)"
		echo "build_lock_private_gemm_x86: $(awk '/^name = "t4a-private-gemm-x86"$/{f=1} f&&/^(version|source|checksum)/{printf "%s ", $0} f&&/^$/{f=0}' Cargo.lock)"
	} >"$record_src"
	source=built
fi
saved="$out/bin/gemm_microkernel-${sha:0:16}"
cp "$bin" "$saved"
cp "$record_src" "$saved.build"
[[ $record_src == "$out/bin/build-record.tmp" ]] && rm "$record_src"

siblings=$(cat "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list")
# expand "2,14" / "2-3" into the list of the other logical CPUs of the core
others=()
IFS=, read -ra parts <<<"$siblings"
for p in "${parts[@]}"; do
	if [[ $p == *-* ]]; then
		for ((c = ${p%-*}; c <= ${p#*-}; c++)); do ((c == cpu)) || others+=("$c"); done
	else
		((p == cpu)) || others+=("$p")
	fi
done

busy_ticks() { # cpu -> "busy total" jiffies from /proc/stat
	awk -v c="cpu$1" '$1 == c { t = 0; for (i = 2; i <= NF; i++) t += $i; print t - $5 - $6, t }' /proc/stat
}

{
	echo "date: $(date -u +%FT%TZ)"
	echo "host: $(hostname)"
	echo "kernel: $(uname -r)"
	echo "cpu_model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
	echo "measured_cpu: $cpu (smt siblings $siblings)"
	echo "governor: $(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null || echo ?)"
	echo "boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo ?)"
	echo "scaling_max_khz: $(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq 2>/dev/null || echo ?)"
	echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
	cat "$saved.build"
	echo "binary: $saved"
	echo "binary_source: $source"
	echo "invoker_git_head: $(git rev-parse HEAD)"
	echo "invoker_git_dirty: $(git status --porcelain --untracked-files=no | wc -l) tracked file(s) modified"
	echo "args: $all_args"
} >"$out/meta.txt"

declare -A b0 t0
for c in "${others[@]}"; do read -r b0[$c] t0[$c] < <(busy_ticks "$c"); done

status=0
taskset -c "$cpu" "$saved" --cpu "$cpu" "$@" >"$out/results.tsv" 2>"$out/stderr.txt" || status=$?

verdict=VALID
# the CPU the probe reports having pinned must be the one whose siblings were watched
probe_cpu=$(awk '$1 == "#" && $2 == "pinned_cpu:" { print $3; exit }' "$out/results.tsv")
echo "probe_pinned_cpu: ${probe_cpu:-?}" >>"$out/meta.txt"
[[ $probe_cpu == "$cpu" ]] || verdict=DISCARDED
for c in "${others[@]}"; do
	read -r b1 t1 < <(busy_ticks "$c")
	db=$((b1 - b0[$c])) dt=$((t1 - t0[$c]))
	pct=$(awk -v b="$db" -v t="$dt" 'BEGIN { printf "%.2f", (t > 0 ? 100 * b / t : 0) }')
	echo "sibling_cpu$c busy: $pct%" >>"$out/meta.txt"
	# fail closed: an empty or non-numeric reading discards the run too
	if ! [[ $pct =~ ^[0-9]+\.[0-9]+$ ]] || awk -v p="$pct" 'BEGIN { exit !(p >= 2) }'; then verdict=DISCARDED; fi
done
echo "bench_exit: $status" >>"$out/meta.txt"
echo "verdict: $verdict" >>"$out/meta.txt"
cat "$out/meta.txt" >&2
if ((status != 0)); then exit "$status"; fi
if [[ $verdict != VALID ]]; then exit 3; fi
