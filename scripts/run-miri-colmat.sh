#!/usr/bin/env bash
# Run the main crate's selected Miri test set (`scripts/miri-colmat-tests.txt`, see `docs/miri.md`).
#
# One `cargo miri test` invocation with every name as an `--exact` filter, so the crate is built once.
# The run must execute exactly the listed tests: a renamed or deleted test would otherwise drop out of
# the lane silently (libtest runs nothing for a filter that matches nothing), and a test that gained
# `#[ignore]` would be reported as ignored instead of run. Either case fails the script.
#
# Usage: scripts/run-miri-colmat.sh [extra cargo args...]
# Needs a nightly toolchain with `miri` and `rust-src` (e.g. `RUSTUP_TOOLCHAIN=nightly`).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
list="$root/scripts/miri-colmat-tests.txt"

names=()
while IFS= read -r line || [ -n "$line" ]; do
	line="${line%%#*}"
	line="${line//[[:space:]]/}"
	[ -n "$line" ] && names+=("$line")
done < "$list"
expected=${#names[@]}
if [ "$expected" -eq 0 ]; then
	echo "error: $list lists no tests" >&2
	exit 1
fi
dupes=$(printf '%s\n' "${names[@]}" | sort | uniq -d)
if [ -n "$dupes" ]; then
	echo "error: duplicate names in $list:" >&2
	echo "$dupes" >&2
	exit 1
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
cd "$root"
set +e
cargo miri test -p colmat --lib "$@" -- --exact "${names[@]}" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ]; then
	echo "error: cargo miri test exited with $status" >&2
	exit "$status"
fi

# `test result: ok. 52 passed; 0 failed; 0 ignored; 0 measured; 948 filtered out; ...`
result=$(grep -E '^test result:' "$log" | tail -n 1)
passed=$(sed -E 's/.* ([0-9]+) passed.*/\1/' <<<"$result")
ignored=$(sed -E 's/.* ([0-9]+) ignored.*/\1/' <<<"$result")
if [ "$passed" != "$expected" ] || [ "$ignored" != "0" ]; then
	echo "error: $list lists $expected tests, but the run passed $passed and ignored $ignored." >&2
	echo "A listed test was renamed, removed or marked #[ignore]; update the list." >&2
	exit 1
fi
echo "Miri: all $expected listed tests passed."
