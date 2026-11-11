#!/usr/bin/env python3
"""Aggregate the evo2 `lu`/`solve` cells from the raw `fbench` CSVs of one run directory.

    python3 aggregate_lu123.py <cells-dir> [--csv out.csv]

One CSV row per process; this script prints, per cell, the median over the process medians and
the ratio of each faer row against MKL, and never interpolates a missing cell. It also prints the
per-variant split (the owning route, note empty, against `in_place`) because that is the whole
point of the record: the campaign's published `lu`/`solve` rows are the owning route only.

Cells with fewer than `--min-sets` (default 3) processes are reported as `missing` and are not
averaged.
"""
import argparse
import collections
import csv
import glob
import os
import statistics
import sys


def rows(path):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            yield r


def key(r):
    return (
        r["family"],
        r["dtype"],
        r["order"],
        int(r["m"]),
        int(r["threads"]),
        r["row"],
        "in_place" if "in_place" in (r["note"] or "") else "owning",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("--pattern", default="*.csv",
                    help="which CSVs of the directory to read (default *.csv; `cells/` holds "
                         "one merged file, `cells.csv`)")
    ap.add_argument("--min-sets", type=int, default=3,
                    help="processes a cell needs before it is averaged (default 3; the "
                         "single-process diagnostics in alloc/ need --min-sets 1)")
    ap.add_argument("--csv")
    a = ap.parse_args()
    cells = collections.defaultdict(list)
    files = sorted(glob.glob(os.path.join(a.cells, a.pattern)))
    if not files:
        sys.exit(f"no {a.pattern} under {a.cells}")
    for path in files:
        for r in rows(path):
            if r["family"] not in ("lu", "solve"):
                continue
            if not r["total_ms"]:
                continue
            cells[key(r)].append(float(r["total_ms"]))

    def get(fam, dt, order, n, t, row, variant):
        v = cells.get((fam, dt, order, n, t, row, variant))
        if not v or len(v) < a.min_sets:
            return None, 0 if not v else len(v)
        return statistics.median(v), len(v)

    out = []
    print(
        f"{'family':6s} {'dt':4s} {'ord':4s} {'n':>5s} {'T':>2s} {'variant':9s} "
        f"{'faer':>9s} {'upstream':>9s} {'mkl':>9s} {'faer/mkl':>9s} {'up/mkl':>7s} {'sets':>5s}"
    )
    for fam in ("lu", "solve"):
        for dt in ("f64", "c64"):
            for order in ("c", "f"):
                for n in (128, 256, 512, 1024):
                    for t in (1, 4, 8):
                        for variant in ("owning", "in_place"):
                            faer, nf = get(fam, dt, order, n, t, "faer-fork", variant)
                            up, _ = get(fam, dt, order, n, t, "faer-upstream", variant)
                            # MKL has one row per cell (its `dgetrf`/`zgetrf` factor in place);
                            # for the `in_place` variant it is that same row, which is the
                            # like-for-like comparison the record needs
                            mkl, _ = get(fam, dt, order, n, t, "lapack-mkl", variant)
                            if mkl is None:
                                mkl, _ = get(fam, dt, order, n, t, "lapack-mkl", "owning")
                            if faer is None or mkl is None:
                                print(
                                    f"{fam:6s} {dt:4s} {order:4s} {n:5d} {t:2d} {variant:9s} "
                                    f"missing ({nf} sets)"
                                )
                                continue
                            line = dict(
                                family=fam, dtype=dt, order=order, n=n, threads=t,
                                variant=variant, faer_ms=faer, upstream_ms=up, mkl_ms=mkl,
                                faer_over_mkl=faer / mkl,
                                upstream_over_mkl=(up / mkl) if up else "",
                                sets=nf,
                            )
                            out.append(line)
                            print(
                                f"{fam:6s} {dt:4s} {order:4s} {n:5d} {t:2d} {variant:9s} "
                                f"{faer:9.4f} {up if up else float('nan'):9.4f} {mkl:9.4f} "
                                f"{faer / mkl:9.2f} {(up / mkl) if up else float('nan'):7.2f} "
                                f"{nf:5d}"
                            )
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["family", "dtype", "order", "n", "threads", "variant",
                            "faer_ms", "upstream_ms", "mkl_ms", "faer_over_mkl",
                            "upstream_over_mkl", "sets"],
            )
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {a.csv}")

    # the two summaries the record quotes
    own = [r for r in out if r["variant"] == "owning"]
    inplace = [r for r in out if r["variant"] == "in_place"]
    for name, group in (("owning", own), ("in_place", inplace)):
        worst = sorted(group, key=lambda r: -r["faer_over_mkl"])[:5]
        print(f"\n{name}: {len(group)} cells, median ratio "
              f"{statistics.median(r['faer_over_mkl'] for r in group):.2f} against MKL")
        for r in worst:
            print(
                f"  worst {r['family']}/{r['dtype']}/{r['order']}/n{r['n']}/{r['threads']}T "
                f"{r['faer_ms']:.4f} ms against MKL {r['mkl_ms']:.4f} = x{r['faer_over_mkl']:.2f}"
            )


if __name__ == "__main__":
    main()
