#!/usr/bin/env python3
"""tensor4all/faer#391 increment 4 aggregator: every table of the record from the raw CSVs.

usage: aggregate391.py call  <run.csv>...                     # table A: whole-call times
       aggregate391.py stage <run.csv>...                     # table B: wall-time stages
       aggregate391.py phase <run.csv> [--level=0] [--by-level]  # table C: phase attribution
       aggregate391.py mstage <run.csv>...                    # table D: MKL ?sytrd / ?stedc
       aggregate391.py spread <run.csv>...                    # per-cell process counts and spread
       aggregate391.py pairs <run.csv> <armA> <armB>          # two arms of one CSV: whole call and
                                                             # the level-0 secular phase

A cell is (variant, dtype, n, T). Its value is the median over the processes of that cell of the
process's own per-call mean, in us (the CSVs are ns); `spread` is (max - min) / median over the same processes. The
number of processes kept per cell is printed, never assumed.
"""
import collections, csv, statistics, sys

VARIANTS = ['forkdef', 'fork16', 'mkl', 'forkdefi', 'fork16i']


def load(paths):
    """{(variant, dtype, n, T): {key: [values per process]}}, plus the key order per cell"""
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in paths:
        for r in csv.DictReader(open(path)):
            if not r['variant'] or r['variant'].startswith('#'):
                continue
            label = r['variant'].split(':')[0]
            if r['key'] in ('ns', 'nodes', 'allocs', 'stat'):
                key = f"{r['key']}@{r['level']}.{r['phase']}"
            elif r['key'] in ('stage_mean_ns', 'stage_median_ns'):
                key = f"{r['key']}.{r['phase']}"
            else:
                key = r['key']
            cell = (label, r['dtype'], int(r['n']), int(r['T']))
            try:
                rows[cell][key].append(float(r['value']))
            except ValueError:
                pass
    return rows


def med(v):
    return statistics.median(v) if v else float('nan')


def cell_of(rows, variant, dtype, n, t):
    return rows.get((variant, dtype, n, t), {})


def table_call(rows, paths):
    print('| dtype | n | T | fork default | fork thr=16 | MKL ?syevd | thr16 / MKL | default / MKL | procs | spread (fork thr=16) |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for dtype in ('f64', 'c64'):
        for n in (64, 96, 128):
            for t in (1, 4, 8):
                def get(var):
                    v = med(cell_of(rows, var, dtype, n, t).get('call_mean_ns', []))
                    return v / 1e3
                d, s, m = get('forkdef'), get('fork16'), get('mkl')
                xs = cell_of(rows, 'fork16', dtype, n, t).get('call_mean_ns', [])
                spread = (max(xs) - min(xs)) / med(xs) * 100 if xs else float('nan')
                print(f'| {dtype} | {n} | {t} | {d:.3f} | {s:.3f} | {m:.3f} | '
                      f'{s / m:.2f} | {d / m:.2f} | {len(xs)} | {spread:.1f}% |')


STAGES = ['stage_tridiag', 'stage_evd', 'stage_apply']


def table_stage(rows, paths):
    print('| dtype | n | T | variant | tridiagonalize | tridiag EVD | apply | sum | call |')
    print('| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |')
    for dtype in ('f64', 'c64'):
        for n in (64, 96, 128):
            for t in (1, 4, 8):
                for var in ('fork16i', 'forkdefi'):
                    c = cell_of(rows, var, dtype, n, t)
                    if not c:
                        continue
                    st = [med(c.get(f'ns@0.{p}', [])) / 1e3 for p in STAGES]
                    call = med(c.get('call_mean_ns', [])) / 1e3
                    print(f'| {dtype} | {n} | {t} | {var} | ' +
                          ' | '.join(f'{x:.3f}' for x in st) +
                          f' | {sum(st):.3f} | {call:.3f} |')


# the groups of docs/worklogs/2026-11-11-eigh-dc-merge-phases.md, unchanged, so this record's
# tables read against #393's and #397's: (a) secular root finder, (b) z build + deflation, (c)
# Lowener + eigenvector columns + Householder apply, (d) the two matmuls, (d') copy-out, (e) leaf QR
PHASE_GROUPS = {
    '(a)': ['secular'],
    '(b)': ['z_build', 'merge_perm', 'deflate', 'sort_perm'],
    '(c)': ['lowner', 'vectors', 'hh_apply'],
    '(d)': ['gemm'],
    "(d')": ['copy_out'],
    '(e)': ['leaf'],
    'other': ['split', 'outer_pre', 'dc_wrap', 'outer_post'],
}


def grp(c, l, ps):
    """sum of a level's phases of one group, in us, or None when the level has none of them"""
    vs = [med(c.get(f'ns@{l}.{p}', [])) for p in ps]
    vs = [v for v in vs if v == v]
    return sum(vs) / 1e3 if vs else None


def table_phase(rows, paths, level=None, by_level=False):
    heads = list(PHASE_GROUPS)
    if by_level:
        print('| dtype | n | T | variant | level | ' + ' | '.join(heads) + ' | nodes |')
        print('| --- | ---: | ---: | --- | ---: | ' + ' | '.join(['---:'] * len(heads)) + ' | ---: |')
    else:
        print('| dtype | n | T | variant | ' + ' | '.join(heads) + ' | identity | margins |')
        print('| --- | ---: | ---: | --- | ' + ' | '.join(['---:'] * len(heads)) + ' | ---: | ---: |')
    for dtype in ('f64', 'c64'):
        for n in (64, 96, 128):
            for t in (1, 4, 8):
                for var in ('fork16i', 'forkdefi'):
                    c = cell_of(rows, var, dtype, n, t)
                    if not c:
                        continue
                    levels = range(0, 4) if by_level or level is None else [level]
                    if by_level:
                        for l in levels:
                            print(f'| {dtype} | {n} | {t} | {var} | {l} | ' +
                                  ' | '.join(
                                      ('-' if grp(c, l, ps) is None else f'{grp(c, l, ps):.3f}')
                                      for ps in PHASE_GROUPS.values()) +
                                  (' | nan |' if med(c.get(f'nodes@{l}.split', [])) != med(c.get(f'nodes@{l}.split', []))
                                   else f' | {med(c.get(f"nodes@{l}.split", [])):.2f} |'))
                        continue
                    vals = {}
                    for g, ps in PHASE_GROUPS.items():
                        vals[g] = sum(grp(c, l, ps) or 0 for l in levels)
                    # the identity: the phase sum of every level against the call
                    ident = sum(vals.values()) / (med(c.get('call_mean_ns', [])) / 1e3)
                    print(f'| {dtype} | {n} | {t} | {var} | ' +
                          ' | '.join(('-' if not v else f'{v:.3f}') for v in vals.values()) +
                          f' | {ident:.2f} |')


def table_mstage(rows, paths):
    print('| n | T | MKL ?sytrd | MKL ?stedc | fork stage 1 | fork stage 2 | fork apply | MKL ?syevd | fork call |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for n in (64, 96, 128):
        for t in (1, 4, 8):
            c = cell_of(rows, 'mkl', 'f64', n, t)
            f = cell_of(rows, 'fork16i', 'f64', n, t)
            g = cell_of(rows, 'fork16', 'f64', n, t)
            print(f'| {n} | {t} | ' +
                  ' | '.join(f'{med(c.get(f"stage_mean_ns.{k}", [])) / 1e3:.3f}' for k in ('sytrd', 'stedc')) +
                  ' | ' + ' | '.join(f'{med(f.get(f"ns@0.{p}", [])) / 1e3:.3f}' for p in STAGES) +
                  f' | {med(c.get("call_mean_ns", [])) / 1e3:.3f}' +
                  f' | {med(g.get("call_mean_ns", [])) / 1e3:.3f} |')


def table_spread(rows, paths):
    print('| dtype | n | T | variant | processes | min | median | max | spread | busy_after max |')
    print('| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for dtype in ('f64', 'c64'):
        for n in (64, 96, 128):
            for t in (1, 4, 8):
                for var in VARIANTS:
                    c = cell_of(rows, var, dtype, n, t)
                    xs = c.get('call_mean_ns', [])
                    if not xs:
                        continue
                    print(f'| {dtype} | {n} | {t} | {var} | {len(xs)} | {min(xs) / 1e3:.3f} | '
                          f'{med(xs) / 1e3:.3f} | {max(xs) / 1e3:.3f} | '
                          f'{(max(xs) - min(xs)) / med(xs) * 100:.1f}% | '
                          f'{max(c.get("busy_after", [0])):.3f} |')


def table_pairs(rows, paths):
    a, b = sys.argv[3], sys.argv[4]
    print(f'| dtype | n | T | {a} call | {b} call | {b} / {a} | {a} secular | {b} secular | procs |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for dtype in ('f64', 'c64'):
        for n in (64, 96, 128):
            for t in (1, 2, 3, 4, 8):
                ca, cb = cell_of(rows, a, dtype, n, t), cell_of(rows, b, dtype, n, t)
                xa, xb = ca.get('call_mean_ns', []), cb.get('call_mean_ns', [])
                if not xa or not xb:
                    continue
                sa = med(ca.get('ns@0.secular', [])) / 1e3
                sb = med(cb.get('ns@0.secular', [])) / 1e3
                print(f'| {dtype} | {n} | {t} | {med(xa) / 1e3:.3f} | {med(xb) / 1e3:.3f} | '
                      f'{med(xb) / med(xa):.3f} | {sa:.1f} | {sb:.1f} | {len(xa)} |')


def main():
    what = sys.argv[1]
    tail = sys.argv[2:]
    arms = [a for a in tail if not a.startswith('--') and not a.endswith('.csv')]
    paths = [a for a in tail if a.endswith('.csv')]
    rows = load(paths)
    if what == 'call':
        table_call(rows, paths)
    elif what == 'stage':
        table_stage(rows, paths)
    elif what == 'phase':
        table_phase(rows, paths, by_level='--by-level' in sys.argv)
    elif what == 'mstage':
        table_mstage(rows, paths)
    elif what == 'spread':
        table_spread(rows, paths)
    elif what == 'pairs':
        table_pairs(rows, paths)
    else:
        raise SystemExit(__doc__)


main()
