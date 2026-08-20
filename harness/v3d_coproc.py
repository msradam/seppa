"""Reproducible CPU|GPU co-processing characterization on the Pi 5 (Burr FSM).

Same correctness-gated philosophy as the kernel-optimization FSM, applied to
measurement validity: a benchmark trial taken while the Pi is thermally
throttling is untrustworthy, so a THERMAL GUARD discards it (analogous to the
NMSE gate discarding an incorrect kernel). Only thermally-valid trials are
recorded; the FSM keeps sampling until it has N valid trials, then aggregates.

    CHARACTERIZE -> BASELINE -> COOLDOWN -> TRIAL -> VALIDATE
      VALIDATE --(valid)--> RECORD ; --(throttled)--> COOLDOWN (discard, retry)
      RECORD   --(enough)-> AGGREGATE -> STOP ; --(more)--> COOLDOWN

Each trial runs CPU LLM decode (memory-bound) concurrently with GPU GEMM
(compute-bound) via coproc_trial.sh on the Pi, capturing both rates + temp +
throttle flags. That trial script lived on the board and is not archived here;
no paper campaign depends on it. Run: python v3d_coproc.py [n_trials]
"""

from __future__ import annotations

import re
import subprocess
import sys

from burr.core import State, action, expr
from burr.core.application import ApplicationBuilder

PI_HOST = "root@pi.local"
RES = "/root/v3d-research"
MODEL = f"{RES}/models/tinyllama-1.1b-chat.Q4_0.gguf"
LLAMA_BENCH = f"{RES}/llama.cpp/build-vulkan/bin/llama-bench"

TEMP_THROTTLE_C = 80.0  # Pi 5 soft-throttles ~80-85C; trials above this are invalid
COOL_START_C = 68.0  # don't start a trial until the SoC is below this


def _ssh(cmd: str, timeout: int = 180) -> str:
    p = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", PI_HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p.stdout + p.stderr


def read_temp() -> float:
    m = re.search(r"[0-9.]+", _ssh("vcgencmd measure_temp"))
    return float(m.group()) if m else 0.0


def measure_cpu_baseline() -> float:
    out = _ssh(f"{LLAMA_BENCH} -m {MODEL} -ngl 0 -p 0 -n 48 -r 2 2>/dev/null | grep tg48")
    m = re.search(r"\|\s*([0-9.]+)\s*±", out)
    return float(m.group(1)) if m else 0.0


def measure_gpu_baseline() -> float:
    out = _ssh(f"cd {RES} && ./vkgemm 2>/dev/null | grep SZ=1024")
    m = re.search(r"SZ=1024\s+([0-9.]+)", out)
    return float(m.group(1)) if m else 0.0


def run_trial() -> dict:
    out = _ssh(f"bash {RES}/coproc_trial.sh")
    kv = dict(re.findall(r"(\w+)=(\S+)", out))
    return {
        "cpu_tps": float(kv.get("cpu_tps", 0) or 0),
        "gpu_gflops": float(kv.get("gpu_gflops", 0) or 0),
        "temp0": float(kv.get("temp0", 0) or 0),
        "temp1": float(kv.get("temp1", 0) or 0),
        "thr0": kv.get("thr0", "0x0"),
        "thr1": kv.get("thr1", "0x0"),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
@action(reads=[], writes=["temp_start"])
def characterize(state: State) -> tuple[dict, State]:
    t = read_temp()
    return {"temp_start": t}, state.update(temp_start=t)


@action(reads=[], writes=["cpu_baseline", "gpu_baseline"])
def baseline(state: State) -> tuple[dict, State]:
    cpu = measure_cpu_baseline()
    gpu = measure_gpu_baseline()
    return {"cpu_baseline": cpu, "gpu_baseline": gpu}, state.update(
        cpu_baseline=cpu, gpu_baseline=gpu
    )


@action(reads=[], writes=["too_hot", "temp_now"])
def cooldown(state: State) -> tuple[dict, State]:
    t = read_temp()
    too_hot = t >= COOL_START_C
    if too_hot:
        _ssh("sleep 15")  # let it cool before the next trial
    return {"temp_now": t, "too_hot": too_hot}, state.update(too_hot=too_hot, temp_now=t)


@action(reads=[], writes=["current_trial"])
def trial(state: State) -> tuple[dict, State]:
    res = run_trial()
    return {"trial": res}, state.update(current_trial=res)


@action(reads=["current_trial"], writes=["trial_valid"])
def validate(state: State) -> tuple[dict, State]:
    r = state["current_trial"]
    # THERMAL GUARD: throttle bits clear AND temp below the throttle point AND
    # the measurement actually produced numbers.
    throttled = (int(r["thr1"], 16) & 0xF) != 0
    valid = (
        not throttled and r["temp1"] < TEMP_THROTTLE_C and r["cpu_tps"] > 0 and r["gpu_gflops"] > 0
    )
    return {
        "trial_valid": valid,
        "throttled": throttled,
        "temp1": r["temp1"],
    }, state.update(trial_valid=valid)


@action(
    reads=["current_trial", "valid_trials", "n_target"],
    writes=["valid_trials", "enough_trials"],
)
def record(state: State) -> tuple[dict, State]:
    trials = [*state["valid_trials"], state["current_trial"]]
    enough = len(trials) >= state["n_target"]
    return {"n_valid": len(trials), "enough": enough}, state.update(
        valid_trials=trials, enough_trials=enough
    )


@action(reads=["valid_trials", "cpu_baseline", "gpu_baseline"], writes=["summary"])
def aggregate(state: State) -> tuple[dict, State]:
    trials = state["valid_trials"]
    cpu = [t["cpu_tps"] for t in trials]
    gpu = [t["gpu_gflops"] for t in trials]
    base = state["cpu_baseline"]
    summary = {
        "n_valid": len(trials),
        "cpu_baseline_tps": base,
        "gpu_baseline_gflops": state["gpu_baseline"],
        "cpu_concurrent_tps_mean": round(_mean(cpu), 3),
        "cpu_concurrent_tps_std": round(_std(cpu), 3),
        "gpu_concurrent_gflops_mean": round(_mean(gpu), 3),
        "gpu_concurrent_gflops_std": round(_std(gpu), 3),
        "cpu_degradation_pct": round((base - _mean(cpu)) / base * 100, 2) if base else 0,
        "peak_temp_c": max((t["temp1"] for t in trials), default=0),
    }
    return {"summary": summary}, state.update(summary=summary)


@action(reads=["summary"], writes=[])
def stop(state: State) -> tuple[dict, State]:
    return {"done": True}, state


def build_coproc_app(n_target: int = 5, max_discards: int = 8):
    init = {
        "temp_start": 0.0,
        "cpu_baseline": 0.0,
        "gpu_baseline": 0.0,
        "too_hot": False,
        "temp_now": 0.0,
        "current_trial": {},
        "trial_valid": False,
        "valid_trials": [],
        "enough_trials": False,
        "n_target": n_target,
        "summary": {},
    }
    return (
        ApplicationBuilder()
        .with_actions(
            characterize=characterize,
            baseline=baseline,
            cooldown=cooldown,
            trial=trial,
            validate=validate,
            record=record,
            aggregate=aggregate,
            stop=stop,
        )
        .with_transitions(
            ("characterize", "baseline"),
            ("baseline", "cooldown"),
            ("cooldown", "cooldown", expr("too_hot")),  # self-loop until cool
            ("cooldown", "trial", expr("not too_hot")),
            ("trial", "validate"),
            ("validate", "record", expr("trial_valid")),  # THE GUARD
            ("validate", "cooldown", expr("not trial_valid")),  # discard, retry
            ("record", "aggregate", expr("enough_trials")),
            ("record", "cooldown", expr("not enough_trials")),
            ("aggregate", "stop"),
        )
        .with_entrypoint("characterize")
        .with_state(**init)
        .build()
    )


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"[co-processing characterization: {n} thermally-valid trials]\n")
    app = build_coproc_app(n_target=n)
    for step in app.iterate(halt_after=["stop"]):
        act, res, _ = step
        name = act.name
        if name == "baseline":
            print(
                f"baseline: CPU {res['cpu_baseline']:.2f} t/s | GPU {res['gpu_baseline']:.2f} GFLOP/s"
            )
        elif name == "trial":
            t = res["trial"]
            print(
                f"  trial: CPU={t['cpu_tps']:.2f} t/s  GPU={t['gpu_gflops']:.2f} GFLOP/s  temp {t['temp0']:.0f}->{t['temp1']:.0f}C  thr={t['thr1']}"
            )
        elif name == "validate":
            print(f"    -> {'VALID' if res['trial_valid'] else 'DISCARDED (throttled/temp)'}")
        elif name == "aggregate":
            s = res["summary"]
            print("\n=== RESULT ===")
            print(f"  valid trials: {s['n_valid']}")
            print(f"  CPU alone:      {s['cpu_baseline_tps']:.2f} t/s")
            print(
                f"  CPU concurrent: {s['cpu_concurrent_tps_mean']:.2f} ± {s['cpu_concurrent_tps_std']:.2f} t/s  ({s['cpu_degradation_pct']:+.1f}%)"
            )
            print(
                f"  GPU concurrent: {s['gpu_concurrent_gflops_mean']:.2f} ± {s['gpu_concurrent_gflops_std']:.2f} GFLOP/s (harvested)"
            )
            print(f"  peak temp: {s['peak_temp_c']:.0f}C")


if __name__ == "__main__":
    main()
