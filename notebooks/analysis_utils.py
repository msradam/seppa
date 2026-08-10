"""Shared parsers for the benchmark artifacts under docs/paper/artifacts/.

Raw logs are plain text. Flood runs print one `time=<seconds>s` line per
run, preceded by a `t=<epoch>` stamp; concurrent phases carry
`llama_start=`/`llama_end=` (or `window_start=`/`window_end=`) markers so
a run counts only if it began and finished inside the measured window.
llama-bench prints one `tg64 | <mean> ± <sd>` row per invocation.
"""

import re
import statistics
from pathlib import Path


def sps(path, steps=4000):
    """All flood runs in a log, as steps/second."""
    text = Path(path).read_text()
    return [steps / float(t) for t in re.findall(r"time=([0-9.]+)s", text)]


def windowed(path, start="llama_start", end="llama_end", steps=4000):
    """Flood runs that started and finished inside the marked window."""
    text = Path(path).read_text()
    ws = float(re.search(start + r"=([0-9.]+)", text).group(1))
    we = float(re.search(end + r"=([0-9.]+)", text).group(1))
    runs = []
    for m in re.finditer(r"t=([0-9.]+)\n(?:.*?\n)*?.*?time=([0-9.]+)s", text):
        t0, dt = float(m.group(1)), float(m.group(2))
        if t0 >= ws and t0 + dt <= we:
            runs.append(steps / dt)
    return runs


def fmt(values):
    """mean ± sample stdev (n=...) for a list of throughputs."""
    if len(values) > 1:
        return f"{statistics.mean(values):8.1f} ± {statistics.stdev(values):6.1f}  (n={len(values)})"
    return f"{values[0]:8.1f}"


def tps(path):
    """The last llama-bench tg64 'mean ± sd' string in a log."""
    m = re.findall(r"tg64\s*\|\s*([0-9.]+) ± ([0-9.]+)", Path(path).read_text())
    return " ± ".join(m[-1]) if m else "?"


def environment():
    """Print the interpreter and key package versions (reproducibility record)."""
    import sys

    print("python", sys.version.split()[0])
    for mod in ("matplotlib", "nbformat"):
        try:
            print(mod, __import__(mod).__version__)
        except ImportError:
            print(mod, "not installed")
