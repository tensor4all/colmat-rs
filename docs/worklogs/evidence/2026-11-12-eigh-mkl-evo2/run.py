#!/usr/bin/env python3
"""tensor4all/faer#391 increment 4 runner: one cell per process, arms interleaved.

usage: run.py <out.csv> <name[:thr]=binary>... [--reps=5] [--mode=call|mstage] [--prime=250] [--window=1000]

Every (dtype, n, T) cell runs each variant `--reps` times, the variants interleaved per rep
(A B C C B A ...), each in its own process. 1T on CPU 0, 4T on CPUs 0-3, 8T on CPUs 0-7 -- the
physical cores of this host's first L3 domain (0-7, siblings 16-23). Outside that domain the
runner itself runs on 8-15,24-31, and the gate requires CPUs 0-7 and 16-23 idle (under 3% busy
over 300 ms) before every process; waits are logged, never a result retry.
"""
import os, subprocess, sys, time, itertools

MEASURED = list(range(0, 8)) + list(range(16, 24))
OBSERVER = list(range(8, 16)) + list(range(24, 32))
CPU_SET = {1: '0', 4: '0-3', 8: '0-7'}


def busy(dt=0.3):
    def snap():
        out = {}
        for line in open('/proc/stat'):
            if line.startswith('cpu') and line[3].isdigit():
                f = line.split()
                v = list(map(int, f[1:]))
                out[int(f[0][3:])] = (sum(v), v[3] + v[4])
        return out
    a = snap()
    time.sleep(dt)
    b = snap()
    worst = 0.0
    for c in MEASURED:
        tot = b[c][0] - a[c][0]
        if tot > 0:
            worst = max(worst, 1 - (b[c][1] - a[c][1]) / tot)
    return worst


def gate(log):
    while True:
        w = busy()
        if w < 0.03:
            return
        log.write(f'# gate: busy {w:.2f}, waiting\n')
        log.flush()
        time.sleep(2)


def main():
    out = sys.argv[1]
    reps, mode, prime, window = 5, 'call', 250, 1000
    variants = []
    for a in sys.argv[2:]:
        if a.startswith('--reps='):
            reps = int(a.split('=')[1])
        elif a.startswith('--mode='):
            mode = a.split('=')[1]
        elif a.startswith('--prime='):
            prime = a.split('=')[1]
        elif a.startswith('--window='):
            window = a.split('=')[1]
        else:
            k, v = a.split('=', 1)
            variants.append((k, v))
    os.sched_setaffinity(0, OBSERVER)
    env0 = {k: v for k, v in os.environ.items()
            if not (k.startswith('RAYON_') or k.startswith('OMP_') or k.startswith('MKL_')
                    or k.startswith('GOMP_'))}
    dtypes = ['f64'] if mode == 'mstage' else ['f64', 'c64']
    cells = list(itertools.product(dtypes, [64, 96, 128], [1, 4, 8]))
    with open(out, 'w') as log:
        log.write('variant,dtype,n,T,rep,key,level,phase,value\n')
        for (dtype, n, t) in cells:
            for rep in range(reps):
                order = variants if rep % 2 == 0 else variants[::-1]
                for (name, binary) in order:
                    label, thr = (name.split(':', 1) + ['-'])[:2]
                    gate(log)
                    env = dict(env0, OMP_NUM_THREADS=str(t))
                    r = subprocess.run(['taskset', '-c', CPU_SET[t], binary,
                                        'fork' if label.startswith('fork') else 'mkl',
                                        mode, dtype, str(n), str(t), thr,
                                        str(prime), str(window)],
                                       capture_output=True, text=True, env=env)
                    if r.returncode != 0:
                        raise SystemExit(f'{name} {dtype} {n} {t}: {r.stderr[-2000:]}')
                    log.write(f'{name},{dtype},{n},{t},{rep},busy_after,,,{busy():.3f}\n')
                    for line in r.stdout.splitlines():
                        log.write(f'{name},{dtype},{n},{t},{rep},{line}\n')
                    log.flush()
            print(dtype, n, t, 'done', flush=True)


main()
