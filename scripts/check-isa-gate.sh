#!/usr/bin/env bash
# ISA gate (colmat-rs#320). Builds the probe at the BASELINE target (RUSTFLAGS unset, the same
# habit as run_acceptance.sh) with symbols kept, then fails if an AVX2-tier symbol contains
# AVX-512. Symbol choice and rationale: scripts/isa_gate.py. With an argument, checks that
# binary instead of building (used to demonstrate the gate failing on a widened build).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -ge 1 ]; then bin=$1; else
  env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_TARGET \
    CARGO_PROFILE_CI_STRIP=none \
    cargo build --profile ci -p colmat --example isa_gate_probe
  bin=${CARGO_TARGET_DIR:-target}/ci/examples/isa_gate_probe
fi
exec python3 -I scripts/isa_gate.py "$bin"
