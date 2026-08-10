"""Repeatability campaign for the flood2 MCP reproduction: run
drive_flood2_mcp.py k times and score pass^k.

    python3 passk_flood2.py [k] [mcp-url]

A run passes when all of the following hold:
  - baseline gates green, baseline in [1300, 1400] steps/s
  - fused strip-2 verdict keep, speedup over baseline >= 1.5x
  - the mass-violating kernel fails verify, the server refuses its
    benchmark, and it is ledgered as revert with a null steps_per_sec

Each run starts only when the SoC is below 55 C (same cooldown gate as
the benchmarks). Transcripts and a summary land in
docs/paper/artifacts/passk_<date>/.
"""

import json
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
URL = sys.argv[2] if len(sys.argv) > 2 else "http://pi.local:8000/mcp"
PI = "root@pi.local"
HERE = Path(__file__).resolve().parent
OUT = HERE / "docs" / "paper" / "artifacts" / f"passk_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)


def soc_temp() -> float:
    p = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", PI, "vcgencmd measure_temp"],
        capture_output=True, text=True, timeout=15,
    )
    return float(p.stdout.split("=")[1].split("'")[0])


def cool_start(limit: float = 55.0) -> float:
    while True:
        t = soc_temp()
        if t < limit:
            return t
        time.sleep(10)


def score(lines: list[dict]) -> dict:
    r = {"baseline": None, "kept": None, "verdicts": [], "refused": False, "broken_null": False}
    for ev in lines:
        res = ev.get("result", {}) if isinstance(ev.get("result"), dict) else {}
        inner = res.get("result", {}) if isinstance(res.get("result"), dict) else {}
        if ev.get("action") == "baseline":
            r["baseline"] = inner.get("baseline_steps_per_sec")
            r["baseline_gate"] = inner.get("gate_ok")
        if ev.get("action") == "log_variant" and isinstance(inner.get("logged"), dict):
            e = inner["logged"]
            r["verdicts"].append(e.get("verdict"))
            if e.get("verdict") == "keep":
                r["kept"] = e.get("steps_per_sec")
            if e.get("verify_ok") is False:
                r["broken_null"] = e.get("steps_per_sec") is None
        if ev.get("gate_demo"):
            resp = ev.get("response", {})
            r["refused"] = bool(ev.get("server_refused")) or (
                isinstance(resp, dict) and resp.get("error") == "invalid_transition"
            )
    ok = (
        r.get("baseline_gate") is True
        and r["baseline"] is not None and 1300 <= r["baseline"] <= 1400
        and r["kept"] is not None and r["kept"] / r["baseline"] >= 1.5
        and r["verdicts"][:2] == ["keep", "revert"]
        and r["refused"] and r["broken_null"]
    )
    r["pass"] = ok
    return r


runs = []
for i in range(1, K + 1):
    t = cool_start()
    p = subprocess.run(
        [str(HERE / ".venv" / "bin" / "python"), str(HERE / "drive_flood2_mcp.py"), URL],
        capture_output=True, text=True, timeout=600,
    )
    (OUT / f"run{i}.jsonl").write_text(p.stdout)
    if p.returncode != 0:
        (OUT / f"run{i}.stderr").write_text(p.stderr)
    lines = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
    r = score(lines)
    r.update(run=i, start_temp_c=t, rc=p.returncode)
    runs.append(r)
    print(json.dumps(r))

baselines = [r["baseline"] for r in runs if r["baseline"]]
kepts = [r["kept"] for r in runs if r["kept"]]
summary = {
    "k": K,
    "passes": sum(r["pass"] for r in runs),
    "pass_k": all(r["pass"] for r in runs),
    "baseline_mean": round(statistics.mean(baselines), 1),
    "baseline_stdev": round(statistics.stdev(baselines), 1) if len(baselines) > 1 else 0.0,
    "kept_mean": round(statistics.mean(kepts), 1),
    "kept_stdev": round(statistics.stdev(kepts), 1) if len(kepts) > 1 else 0.0,
    "speedups": [round(r["kept"] / r["baseline"], 3) for r in runs if r["kept"] and r["baseline"]],
}
(OUT / "summary.json").write_text(json.dumps({"runs": runs, "summary": summary}, indent=1))
print(json.dumps(summary))
