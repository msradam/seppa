"""Faithful AutoKernel-style exploration loop for a real V3D shader.

Unlike v3d_fsm.py (which walks a fixed variant table), the agent invents each
edit: it rewrites gemm.comp (a tiled SGEMM for the Pi 5 V3D), which is compiled
with glslangValidator and run by the self-contained vkgemm harness (correctness
plus GFLOP/s), then kept or reverted on the measured number. The shipped
gemm.comp does about 7 GFLOP/s against a 43.7 GFLOP/s roofline; the agent
explores that gap (bigger tiles, wider loads, more register blocking) while
staying correct and inside the 16 KB / 256-invocation envelope. The vkgemm check
uses random inputs and a double-precision CPU-reference NMSE, so a kernel that
writes a constant or only part of its output fails the gate.
"""

from __future__ import annotations

import os
import re
import subprocess

from burr.core import State, action, expr
from burr.core.application import ApplicationBuilder

# Persist the winning shader locally on every keep, so the research output (the
# actual optimized kernel, not just the number) survives the run.
BEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_gemm.comp")

PI_HOST = "root@pi.local"
RESEARCH = "/root/v3d-research"
SHADER = f"{RESEARCH}/gemm.comp"
SPV = f"{RESEARCH}/gemm.spv"

# vkgemm_nmse prints "correct=perf" (regex won't match) for perf-only sizes and
# "correct=yes|no" (NMSE-checked) for correctness sizes, so only checked sizes count.
_VKGEMM_LINE = re.compile(
    r"SZ=(\d+)\s+([\d.]+)\s+GFLOP/s.*?correct=(yes|no)", re.IGNORECASE
)
_METRIC_SZ = "512"  # largest NMSE-checked size; keep/revert metric


# ---------------------------------------------------------------------------
# Pi interaction (SSH). These are the live-hardware side-effects the FSM drives.
# ---------------------------------------------------------------------------
def _ssh(
    cmd: str, timeout: int = 120, stdin: str | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", PI_HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin,
    )


def read_shader() -> str:
    return _ssh(f"cat {SHADER}").stdout


def write_shader(src: str) -> None:
    # Write via stdin so arbitrary shader text is safe (no shell interpolation).
    _ssh(f"cat > {SHADER}", stdin=src)


def compile_shader() -> tuple[bool, str]:
    p = _ssh(f"glslangValidator -V {SHADER} -o {SPV}")
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def run_vkgemm() -> tuple[bool, float, str]:
    """Return (all_correct, gflops_at_metric_size, raw)."""
    p = _ssh(f"cd {RESEARCH} && timeout 90 ./vkgemm_nmse", timeout=120)
    raw = p.stdout + p.stderr
    all_correct, metric = True, 0.0
    saw = False
    for m in _VKGEMM_LINE.finditer(raw):
        saw = True
        sz, gflops, correct = m.group(1), float(m.group(2)), m.group(3)
        if correct.lower() != "yes":
            all_correct = False
        if sz == _METRIC_SZ:
            metric = gflops
    if not saw:
        return False, 0.0, raw  # crashed / no output = not correct
    return all_correct, metric, raw


# ---------------------------------------------------------------------------
# FSM actions. The AGENT owns hypothesize + implement (it supplies the shader);
# the machine owns compile/verify/benchmark/evaluate and cannot be bypassed.
# ---------------------------------------------------------------------------
@action(reads=[], writes=["hardware_constraints"])
def characterize(state: State) -> tuple[dict, State]:
    caps = {
        "max_wg_invocations": 256,
        "shared_mem_bytes": 16384,
        "roofline_gflops": 43.7,
    }
    return {"caps": caps}, state.update(hardware_constraints=caps)


@action(
    reads=[], writes=["current_shader", "best_gflops", "baseline_gflops", "best_shader"]
)
def baseline(state: State) -> tuple[dict, State]:
    src = read_shader()
    write_shader(src)
    ok, errs = compile_shader()
    correct, gflops, _ = run_vkgemm() if ok else (False, 0.0, errs)
    return {
        "baseline_gflops": gflops,
        "correct": correct,
        "compile_ok": ok,
    }, state.update(
        current_shader=src, best_shader=src, best_gflops=gflops, baseline_gflops=gflops
    )


@action(reads=["best_shader", "best_gflops", "variant_log"], writes=["exp_idx"])
def hypothesize(state: State) -> tuple[dict, State]:
    # Hand the agent the current BEST shader + history so it can propose an edit.
    idx = state["exp_idx"] + 1
    return {
        "experiment": idx,
        "best_gflops": state["best_gflops"],
        "roofline_gflops": 43.7,
        "current_best_shader": state["best_shader"],
        "history": state["variant_log"][-6:],
        "instruction": "Rewrite the shader (pass full source in implement.inputs.shader) "
        "to raise GFLOP/s while keeping correct=yes and shared memory <= 16 KB.",
    }, state.update(exp_idx=idx)


@action(reads=["exp_idx"], writes=["proposed_shader", "implement_ok"])
def implement(state: State, shader: str = "") -> tuple[dict, State]:
    # The agent's edit arrives here as `shader` (full source).
    if not shader.strip():
        return {"implement_ok": False, "error": "no shader provided"}, state.update(
            proposed_shader="", implement_ok=False
        )
    write_shader(shader)
    return {"implement_ok": True, "bytes": len(shader)}, state.update(
        proposed_shader=shader, implement_ok=True
    )


@action(reads=[], writes=["compile_ok", "compile_err"])
def compile_action(state: State) -> tuple[dict, State]:
    ok, errs = compile_shader()
    return {"compile_ok": ok, "compile_err": errs[:500]}, state.update(
        compile_ok=ok, compile_err=errs[:500]
    )


@action(reads=[], writes=["verify_ok", "pending_gflops", "vkgemm_raw"])
def verify(state: State) -> tuple[dict, State]:
    correct, gflops, raw = run_vkgemm()
    return {"correct": correct, "gflops_if_correct": gflops}, state.update(
        verify_ok=correct, pending_gflops=gflops, vkgemm_raw=raw[-400:]
    )


@action(reads=["pending_gflops"], writes=["measured_gflops", "benchmark_done"])
def benchmark_action(state: State) -> tuple[dict, State]:
    # Reachable only if verify passed. The GFLOP/s is trusted only here.
    return {"measured_gflops": state["pending_gflops"]}, state.update(
        measured_gflops=state["pending_gflops"], benchmark_done=True
    )


@action(
    reads=["measured_gflops", "best_gflops", "proposed_shader", "consecutive_no_gain"],
    writes=[
        "verdict",
        "best_gflops",
        "best_shader",
        "current_shader",
        "consecutive_no_gain",
    ],
)
def evaluate(state: State) -> tuple[dict, State]:
    g, best = state["measured_gflops"], state["best_gflops"]
    if g > best * 1.01:  # >1% real improvement
        with open(BEST_PATH, "w") as f:
            f.write(
                f"// best: {g:.2f} GFLOP/s (SZ={_METRIC_SZ})\n{state['proposed_shader']}"
            )
        return {"verdict": "keep", "gflops": g}, state.update(
            verdict="keep",
            best_gflops=g,
            best_shader=state["proposed_shader"],
            current_shader=state["proposed_shader"],
            consecutive_no_gain=0,
        )
    return {"verdict": "revert", "gflops": g, "best": best}, state.update(
        verdict="revert", consecutive_no_gain=state["consecutive_no_gain"] + 1
    )


@action(
    reads=[
        "exp_idx",
        "verify_ok",
        "benchmark_done",
        "verdict",
        "measured_gflops",
        "best_gflops",
        "compile_ok",
        "consecutive_no_gain",
        "variant_log",
        "max_experiments",
    ],
    writes=["variant_log", "verdict", "consecutive_no_gain", "should_stop"],
)
def log_variant(state: State) -> tuple[dict, State]:
    verdict = state["verdict"]
    consecutive = state["consecutive_no_gain"]
    if verdict == "init":  # compile/verify failed -> bypassed evaluate
        verdict = "revert"
        consecutive += 1
    entry = {
        "exp": state["exp_idx"],
        "compile_ok": state["compile_ok"],
        "verify_ok": state["verify_ok"],
        "benchmarked": state["benchmark_done"],
        "gflops": state["measured_gflops"] if state["benchmark_done"] else None,
        "best_gflops": state["best_gflops"],
        "verdict": verdict,
    }
    log = [*state["variant_log"], entry]
    should_stop = consecutive >= 4 or len(log) >= state["max_experiments"]
    return {"logged": entry, "consecutive_no_gain": consecutive}, state.update(
        variant_log=log,
        verdict=verdict,
        consecutive_no_gain=consecutive,
        should_stop=should_stop,
    )


@action(
    reads=["variant_log", "best_gflops", "baseline_gflops"], writes=["final_summary"]
)
def stop(state: State) -> tuple[dict, State]:
    summary = {
        "experiments": len(state["variant_log"]),
        "kept": sum(1 for e in state["variant_log"] if e["verdict"] == "keep"),
        "baseline_gflops": state["baseline_gflops"],
        "best_gflops": state["best_gflops"],
        "roofline_gflops": 43.7,
        "speedup": state["best_gflops"] / state["baseline_gflops"]
        if state["baseline_gflops"]
        else 0.0,
    }
    return {"summary": summary}, state.update(final_summary=summary)


def build_explore_app(max_experiments: int = 12):
    init = {
        "current_shader": "",
        "best_shader": "",
        "proposed_shader": "",
        "baseline_gflops": 0.0,
        "best_gflops": 0.0,
        "measured_gflops": 0.0,
        "pending_gflops": 0.0,
        "exp_idx": 0,
        "implement_ok": False,
        "compile_ok": False,
        "compile_err": "",
        "verify_ok": False,
        "benchmark_done": False,
        "verdict": "init",
        "consecutive_no_gain": 0,
        "variant_log": [],
        "should_stop": False,
        "hardware_constraints": {},
        "vkgemm_raw": "",
        "final_summary": {},
        "max_experiments": max_experiments,
    }
    return (
        ApplicationBuilder()
        .with_actions(
            characterize=characterize,
            baseline=baseline,
            hypothesize=hypothesize,
            implement=implement,
            compile_=compile_action,
            verify=verify,
            benchmark=benchmark_action,
            evaluate=evaluate,
            log_variant=log_variant,
            stop=stop,
        )
        # Mutually exclusive branches so the agent can't bypass the gate.
        .with_transitions(
            ("characterize", "baseline"),
            ("baseline", "hypothesize"),
            ("hypothesize", "implement"),
            ("implement", "compile_", expr("implement_ok")),
            ("implement", "hypothesize", expr("not implement_ok")),
            ("compile_", "verify", expr("compile_ok")),
            (
                "compile_",
                "log_variant",
                expr("not compile_ok"),
            ),  # bad shader -> logged revert
            ("verify", "benchmark", expr("verify_ok")),
            ("verify", "log_variant", expr("not verify_ok")),  # THE GUARD
            ("benchmark", "evaluate"),
            ("evaluate", "log_variant"),
            ("log_variant", "stop", expr("should_stop")),
            ("log_variant", "hypothesize", expr("not should_stop")),
        )
        .with_entrypoint("characterize")
        .with_state(**init)
        .build()
    )
