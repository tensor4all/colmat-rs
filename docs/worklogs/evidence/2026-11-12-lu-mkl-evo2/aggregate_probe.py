#!/usr/bin/env python3
"""Print the phase decomposition of the owning route from one `run_probe.sh` run directory.

    python3 aggregate_probe.py <probe-runs-dir> [--threads 8] [--dtype f64] [--order f] [--n 512]

Phases are printed in the order they are defined in `probe/src/main.rs`. `faults` is the minor
page faults of one call, which is the quantity that explains the split phases on this host.
"""
import argparse
import csv
import glob
import os
import sys

ORDER = [
    "owning",
    "owning_no_split",
    "owning_u_first",
    "factor_only",
    "clone_only",
    "alloc_zeros",
    "clone_and_alloc_only",
    "alloc_then_clone",
    "split_colslices",
    "split_colwise",
    "split_rowwise",
    "split_upper_only",
    "split_zero_only",
    "split_std_threads",
    "zero_upper_slices_only",
    "copy_upper_slices_only",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--dtype")
    ap.add_argument("--order")
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "probe-*.csv")))
    if not files:
        sys.exit(f"no probe-*.csv under {a.dir}")

    def banner(cell, rows):
        t, dt, order, n = cell
        print(f"\n{t}T {dt} {order}-order n={n}")
        print(f"  {'phase':24s} {'ms':>8s} {'faults':>7s}  note")
        base = rows.get("owning", (float("nan"), 0))[0]
        split = rows.get("owning_no_split", (float("nan"), 0))[0]
        for name in ORDER:
            if name not in rows:
                continue
            ms, f = rows[name]
            note = ""
            if name == "owning":
                note = "the campaign's `lu` row"
            elif name == "owning_no_split":
                note = "owning minus the split"
            elif name == "factor_only":
                note = "factorization alone, MKL row is 0.739 at this cell"
            print(f"  {name:24s} {ms:8.4f} {f:7d}  {note}")
        if base == base and split == split:  # both present
            print(f"  {'owning - owning_no_split':24s} {base - split:8.4f}")

    cells = {}
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                cell = (int(r["threads"]), r["dtype"], r["order"], int(r["n"]))
                if a.threads is not None and cell[0] != a.threads:
                    continue
                if a.dtype and cell[1] != a.dtype:
                    continue
                if a.order and cell[2] != a.order:
                    continue
                if a.n is not None and cell[3] != a.n:
                    continue
                cells.setdefault(cell, {})[r["phase"]] = (
                    float(r["median_ms"]),
                    int(r["median_minor_faults"]),
                )
    for cell in sorted(cells):
        banner(cell, cells[cell])


if __name__ == "__main__":
    main()
