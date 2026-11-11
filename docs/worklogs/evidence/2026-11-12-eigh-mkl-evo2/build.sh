#!/usr/bin/env python3
"""Build the #391 increment 4 probe against one checkout.

usage: build.sh <checkout> <outdir> <tag> [--instr]

The probe lives in this evidence directory, inside a checkout's tree, so it is built from a copy
in the scratch `outdir` with its `faer` path rewritten to `<checkout>`. `--instr` turns on the
per-phase feature, which only the instrumented copy of the checkout has (`instrumentation.patch`).
"""
import os, shutil, subprocess, sys, pathlib

here = pathlib.Path(__file__).resolve().parent
checkout, outdir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
instr = '--instr' in sys.argv[4:]
src = pathlib.Path(outdir) / f'probe-{tag}'
if src.exists():
    shutil.rmtree(src)
shutil.copytree(here / 'probe', src)
man = src / 'Cargo.toml'
man.write_text(man.read_text().replace('../../../../../faer', str(pathlib.Path(checkout) / 'faer')))
cmd = ['cargo', 'build', '--release', '--target-dir', str(pathlib.Path(outdir) / 'target')]
if instr:
    cmd += ['--features', 'instr']
env = dict(os.environ, CARGO_BUILD_JOBS='4')
subprocess.run(cmd, cwd=src, env=env, check=True)
bin_src = pathlib.Path(outdir) / 'target' / 'release' / 'u391'
bin_dst = pathlib.Path(outdir) / f'u391-{tag}'
shutil.copy2(bin_src, bin_dst)
print(bin_dst)
