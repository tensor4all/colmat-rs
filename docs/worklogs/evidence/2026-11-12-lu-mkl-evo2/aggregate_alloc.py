#!/usr/bin/env python3
"""Table D of the record: one line per file in `alloc/`, one median per (family, dtype, order,
row, variant).

    python3 aggregate_alloc.py alloc/

Each file is a separate `fbench` process with its own environment, so the files must never be
pooled: the point of the table is that the *same* binary moves when glibc's growth policy changes.
"""
import csv
import glob
import os
import statistics
import sys


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "alloc"
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        sys.exit(f"no csv under {d}")
    for path in files:
        cells = {}
        for r in csv.DictReader(open(path, newline="")):
            if r.get("status") != "ok" or r["order"] != "f":
                continue
            key = (r.get("run", ""), r["family"], r["dtype"], r["row"], r["note"] or "owning")
            cells.setdefault(key, []).append(float(r["total_ms"]))
        print(f"\n{os.path.basename(path)}")
        for key in sorted(cells):
            v = cells[key]
            print(f"  {key[0]:34s} {key[1]:5s} {key[2]} {key[3]:14s} {key[4]:9s} "
                  f"{statistics.median(v):8.4f} ms  ({len(v)} row(s), spread "
                  f"{(max(v) - min(v)) / statistics.median(v) * 100:.1f}%)")


if __name__ == "__main__":
    main()
