"""Reduce concurrency_bench.sh raw logs to the interference table.

    python3 analyze_conc_bench.py /tmp/conc_bench [steps_per_run]

For concurrent flood logs, only runs that start after llama_start and finish
before llama_end count as "under CPU load"; edge runs are discarded.
"""

import re
import statistics
import sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/conc_bench"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

TIME = re.compile(r"time=([0-9.]+)s")
STAMP = re.compile(r"^t=([0-9.]+)")


def flood_runs(path):
    """Yield (start_stamp, seconds) per completed run."""
    runs, start = [], None
    for line in open(path):
        m = STAMP.match(line)
        if m:
            start = float(m.group(1))
            continue
        m = TIME.search(line)
        if m and start is not None:
            runs.append((start, float(m.group(1))))
            start = None
    return runs


def llama_window(path):
    s = open(path).read()
    a = re.search(r"llama_start=([0-9.]+)", s)
    b = re.search(r"llama_end=([0-9.]+)", s)
    return (float(a.group(1)), float(b.group(1))) if a and b else (None, None)


def llama_tps(path):
    m = re.search(r"tg\d+\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)", open(path).read())
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def sps(times):
    vals = [STEPS / t for t in times]
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, sd, len(vals)


def report(label, mean, sd, n, unit):
    print(f"{label:42s} {mean:8.1f} ± {sd:5.1f} {unit}  (n={n})")


for kernel in ("opt", "orig"):
    alone = flood_runs(f"{DIR}/flood_{kernel}_alone.log")
    report(f"flood {kernel} alone", *sps([t for _, t in alone]), "steps/s")

    conc_path = f"{DIR}/flood_{kernel}_conc.log"
    a, b = llama_window(conc_path)
    inside = [t for s, t in flood_runs(conc_path) if a and s >= a and s + t <= b]
    if inside:
        report(f"flood {kernel} under CPU decode", *sps(inside), "steps/s")

for name, path in (
    ("llama alone", f"{DIR}/llama_alone.log"),
    ("llama vs flood opt", f"{DIR}/llama_vs_opt.log"),
    ("llama vs flood orig", f"{DIR}/llama_vs_orig.log"),
):
    mean, sd = llama_tps(path)
    if mean is not None:
        report(name, mean, sd, 5, "t/s   ")

for name in ("route_alone", "route_conc"):
    try:
        m = re.search(r"real\s+(\d+)m([0-9.]+)s", open(f"{DIR}/{name}.log").read())
        if m:
            print(f"{name:42s} {int(m.group(1)) * 60 + float(m.group(2)):8.2f} s")
    except FileNotFoundError:
        pass

print("\nthermal validity (max temp / samples with throttle ACTIVE):")
import glob  # noqa: E402

for path in sorted(glob.glob(f"{DIR}/thermal_*.log")):
    temps, active = [], 0
    for line in open(path):
        mt = re.search(r"temp=([0-9.]+)'C", line)
        mh = re.search(r"throttled=(0x[0-9a-fA-F]+)", line)
        if mt:
            temps.append(float(mt.group(1)))
        if mh and int(mh.group(1), 16) & 0xF:
            active += 1
    if temps:
        name = path.split("/")[-1]
        verdict = "VALID" if active == 0 else f"TAINTED ({active} throttled samples)"
        print(f"  {name:34s} max {max(temps):5.1f} C  {verdict}")
