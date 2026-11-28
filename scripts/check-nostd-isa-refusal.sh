#!/usr/bin/env bash
# colmat-rs#320: a no-std x86_64 build above the baseline must be REFUSED, and the opt-in feature
# must silence the refusal. Asserts both; a guard that is not exercised is a comment.
set -uo pipefail
cd "$(dirname "$0")/.."
build() { env -u CARGO_ENCODED_RUSTFLAGS RUSTFLAGS="$1" cargo build --profile ci -p colmat --no-default-features ${2:+--features "$2"} 2>&1; }
OPT=assume-target-is-the-build-host
for flags in "-C target-feature=+avx2,+fma" "-C target-feature=+avx512f" "-C target-cpu=x86-64-v3"; do
  echo "== [$flags] without std, without opt-in: must FAIL with the message"
  out=$(build "$flags"); rc=$?
  if [ $rc -eq 0 ]; then echo "$out" | tail -5; echo "FAIL: build succeeded, the guard did not fire"; exit 1; fi
  echo "$out" | grep -q "$OPT" || { echo "$out" | tail -20; echo "FAIL: error does not name the opt-in feature"; exit 1; }
  echo "$out" | grep -m1 "^error: colmat without"
  echo "== [$flags] with --features $OPT: must SUCCEED"
  out=$(build "$flags" "$OPT"); rc=$?
  [ $rc -eq 0 ] || { echo "$out" | tail -20; echo "FAIL: opt-in did not silence the guard"; exit 1; }
done
echo "== baseline (no flags), no opt-in: must SUCCEED"
build "" >/dev/null || { echo "FAIL: baseline no-std build broke"; exit 1; }
echo "PASS"
