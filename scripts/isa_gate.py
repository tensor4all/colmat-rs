#!/usr/bin/env python3
"""ISA gate (colmat-rs#320): the AVX2-tier kernels must not contain AVX-512 instructions.

Why per symbol: a whole-binary grep for `zmm` fails on the AVX-512 tier that is supposed to be
there (the `*_simd512*` kernels, pulp's 512-bit mask helpers), entered only after run time
`avx512f` detection. The property is about the kernels entered after an
`avx2`-and-not-`avx512f` detection, so only those symbols are inspected.

Usage: isa_gate.py BINARY   (a non-stripped executable built at the baseline target)

AVX2-tier symbols chosen (the name carries the tier, no guessing by address):
  gemm      t4a_gemm_v0_*_simd256*, t4a_gemm_v0_*_simd128*   (t4a-gemm: 256-bit / 128-bit tier)
  pulp      pulp::x86::v3::V3::*, gemm_common::simd::x86::V3::*  (V3 = AVX2+FMA; colmat's own
            `simd.mul_add` loops are compiled into these `vectorize::imp` bodies)
  packed    private_gemm_x86::packed_avx2*                    (the AVX2 packed product)
  nano      nano_gemm_*::x86::*::avx::*                       (the AVX/AVX2 small kernels)
Fails on, inside those symbols: any %zmm register, %xmm/%ymm16..31, any %k0-7 mask register, or an
EVEX-encoded instruction (first opcode byte 0x62, always EVEX in 64-bit mode).

Tripwires so the check cannot pass vacuously: every family above must match >= MIN symbols, and
the AVX-512 tier (`*_simd512*`) must be present AND contain zmm (proves the scan sees AVX-512).
"""
import re, subprocess, sys

FAMILIES = {
    "gemm-simd256/128": (re.compile(r"^t4a_gemm_v0_.*_simd(256|128)"), 100),
    "pulp-V3":          (re.compile(r"^(pulp::x86::v3::V3::|gemm_common::simd::x86::V3::)"), 20),
    "packed-avx2":      (re.compile(r"^private_gemm_x86::packed_avx2"), 3),
    "nano-avx":         (re.compile(r"^nano_gemm_\w+::x86::\w+::avx::"), 10),
}
AVX512_TIER = re.compile(r"_simd512")
WIDE = re.compile(r"%zmm|%[xy]mm(1[6-9]|2[0-9]|3[01])\b|%k[0-7]\b")
HDR = re.compile(r"^[0-9a-f]+ <(.*)>:$")
INS = re.compile(r"^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)\s*(.*)$")

def main(path):
    out = subprocess.run(["objdump", "-d", "-C", "-w", path], capture_output=True, text=True, check=True).stdout
    sym = None
    seen = {k: [] for k in FAMILIES}
    bad = {}          # symbol -> list of offending lines
    n512 = n512_wide = 0
    wide512 = False
    cur_fam = None
    for line in out.splitlines():
        m = HDR.match(line)
        if m:
            sym = line.split(" ", 1)[0] + " " + m.group(1)   # address keeps same-named instances apart
            cur_fam = next((k for k, (rx, _) in FAMILIES.items() if rx.search(sym.split(" ", 1)[1])), None)
            if cur_fam: seen[cur_fam].append(sym)
            is512 = bool(AVX512_TIER.search(sym))
            if is512: n512 += 1
            has512 = False
            continue
        m = INS.match(line)
        if not m or sym is None: continue
        raw, text = m.group(1).split(), m.group(2)
        evex = raw[0] == "62"
        if AVX512_TIER.search(sym) and WIDE.search(text) and not has512:
            has512 = True; n512_wide += 1
        if cur_fam and (evex or WIDE.search(text)):
            bad.setdefault(sym, []).append(("EVEX " if evex else "") + text.strip())
    print("ISA gate: AVX2-tier symbols inspected")
    ok = True
    for k, (_, mn) in FAMILIES.items():
        print(f"  {k:18s} {len(seen[k]):5d} symbols (min {mn})")
        if len(seen[k]) < mn:
            print(f"  FAIL: family {k} matched too few symbols; the gate would be vacuous"); ok = False
    print(f"  AVX-512 tier (*_simd512*): {n512} symbols, {n512_wide} contain zmm/EVEX (expected: nearly all)")
    if n512 == 0 or n512_wide * 10 < n512 * 9:
        print("  FAIL: no AVX-512 tier symbol with zmm found; the scan is not seeing AVX-512"); ok = False
    if bad:
        ok = False
        print(f"FAIL: {len(bad)} AVX2-tier symbols contain AVX-512 instructions:")
        for s, l in sorted(bad.items())[:20]:
            print(f"  {s[:110]}  ({len(l)} insns, e.g. {l[0]})")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
