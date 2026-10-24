#!/usr/bin/env python3
"""The per-rank collective timeline (2026-09-06): reads every rank's 'graph
window timeline / wait / peers' lines from a log dir (the launcher's `down`
fetches them; boot with DGPP_BUS_TIMELINE=1 so they are written at INFO —
a debug-level run's per-tick logging perturbs the skew it measures) and
prints, per rank, the per-collective budget (copy + handshake + skew +
fold), the wait histogram, each peer's lag, and the gaps between consecutive
claims (2026-09-19: gaps pinned at one claim round's length are the kernel
gating co-resident peers one after another, not the peers' arrival spread).
Usage: scripts/bus_window_skew.py [LOG_DIR]  (default ~/dgpp/log)"""
import re, sys, glob, statistics, collections
import os
logdir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dgpp/log")
for path in sorted(glob.glob(f"{logdir}/serve_r[0-9].log")):
    rank = path[-5]
    tl = []; waits = []; peers = collections.defaultdict(lambda: [0.0, 0, 0])
    gaps = collections.defaultdict(lambda: [0.0, 0, [0] * 7])
    for line in open(path, errors="replace"):
        m = re.search(r"graph window timeline: rank (\d+) variant (\d+) gens (\d+) avg us: total ([\d.]+) = copy ([\d.]+) \+ handshake ([\d.]+) \+ skew ([\d.]+) \+ fold ([\d.]+); engine post ([\d.]+); max handshake ([\d.]+) skew ([\d.]+) total ([\d.]+).*compute between gens ([\d.]+)", line)
        if m:
            tl.append([float(x) for x in m.groups()[3:]] + [int(m.group(3))]); continue
        m = re.search(r"graph window wait: rank \d+ hist\(<20 <50 <100 <200 <500 >=500 us\) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+); even gens (\d+) avg ([\d.]+) us, odd gens (\d+) avg ([\d.]+) us", line)
        if m:
            waits.append([float(x) for x in m.groups()]); continue
        if "graph window claims" in line:
            for gm in re.finditer(r"claim (\d)->(\d) avg ([\d.]+) us hist ((?:\d+ ?){7})", line):
                hist = [int(x) for x in gm.group(4).split()]
                g = gaps[f"{gm.group(1)}->{gm.group(2)}"]
                g[0] += float(gm.group(3)) * sum(hist); g[1] += sum(hist)
                g[2] = [a + b for a, b in zip(g[2], hist)]
            continue
        for pm in re.finditer(r"rank(\d+) lag (\d+)us(?: \(landed (\d+)us\))? last (\d+)x", line):
            p = peers[pm.group(1)]; p[0] += int(pm.group(2)); p[1] += int(pm.group(4)); p[2] += 1
            if pm.group(3) is not None: p.extend([0] * (4 - len(p))); p[3] += int(pm.group(3))
    if not tl: print(f"rank {rank}: no timeline lines"); continue
    # only full windows (94 gens) after warmup
    full = [t for t in tl if t[-1] >= 90]
    if not full: full = tl
    col = lambda i: statistics.median(t[i] for t in full)
    print(f"rank {rank}: {len(full)} windows | per collective (median of window avgs, us): total {col(0):.1f} = copy {col(1):.1f} + handshake {col(2):.1f} + skew {col(3):.1f} + fold {col(4):.1f}; engine post {col(5):.1f}; max skew {col(7):.0f}; compute between gens {col(9):.1f}")
    if waits:
        w = [sum(x[i] for x in waits) for i in range(6)]
        ev = statistics.median(x[7] for x in waits); od = statistics.median(x[9] for x in waits)
        print(f"        wait hist <20/<50/<100/<200/<500/>=500 us: {w}; even (attention-side) {ev:.1f} us, odd (FFN-side) {od:.1f} us")
    for p, v in sorted(peers.items()):
        lag, last, n = v[0], v[1], v[2]
        landed = f", landed {v[3]/max(n,1):.0f} us" if len(v) > 3 else ""
        print(f"        peer rank{p}: lag {lag/max(n,1):.0f} us avg{landed}, arrived last {last} times")
    for k, (total, n, hist) in sorted(gaps.items()):
        pct = [round(100 * h / max(n, 1)) for h in hist]
        print(f"        claim {k}: gap {total/max(n,1):.1f} us avg; % <3/<5/<7/<10/<15/<25/>=25 us: {pct}")
