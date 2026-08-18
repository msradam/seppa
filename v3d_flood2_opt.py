"""Flood-stencil exploration FSM over the parameterized vkflood2 harness.

Unlike v3d_flood_opt (which can only rewrite shader bodies under a fixed
host contract), implement here takes the host contract as inputs too:
`shader` (the flux or fused kernel), optional `height_shader` (empty means
a fused single-dispatch step), and `strip` (cells per invocation in y,
sets the dispatch geometry). That widened action space is what contains
the fused strip-mined kernel; the shader-only space does not.

verify runs vkflood2's physics gate: NMSE vs a double CPU reference, mass
conservation vs rain injected, and basin pooling — all three must pass
before benchmark is reachable. The keep metric is simulation steps/s at
256^2 x 400.

Runs on the Pi (the Theodosia server host): local subprocess, no SSH.
"""

from __future__ import annotations

import os
import re
import subprocess

from burr.core import State, action, expr
from burr.core.application import ApplicationBuilder

BEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_flood.comp")

RESEARCH = "/root/v3d-research"
FLUX_SRC = f"{RESEARCH}/fsm_flood.comp"
FLUX_SPV = f"{RESEARCH}/fsm_flood.spv"
HEIGHT_SRC = f"{RESEARCH}/fsm_height.comp"
HEIGHT_SPV = f"{RESEARCH}/fsm_height.spv"

_TIME = re.compile(r"time=([0-9.]+)s")
GATES = ("correct(NMSE vs CPU)=yes", "conserved vs rain=yes", "pools in basin=yes")


def _run(cmd: list[str], timeout: int = 120, env: dict | None = None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=RESEARCH, env=e)


def compile_shaders(fused: bool) -> tuple[bool, str]:
    p = _run(["glslangValidator", "-V", FLUX_SRC, "-o", FLUX_SPV])
    if p.returncode != 0:
        return False, (p.stdout + p.stderr)[-500:]
    if not fused:
        p = _run(["glslangValidator", "-V", HEIGHT_SRC, "-o", HEIGHT_SPV])
        if p.returncode != 0:
            return False, (p.stdout + p.stderr)[-500:]
    return True, ""


def run_gate(fused: bool, strip: int) -> tuple[bool, float, str]:
    """Physics-gated run; returns (all_gates_pass, steps_per_sec, raw)."""
    env = {"FLUX_SPV": FLUX_SPV, "STRIP": str(strip)}
    if fused:
        env["FUSED"] = "1"
    else:
        env["HEIGHT_SPV"] = HEIGHT_SPV
    try:
        p = _run(["./vkflood2", "256", "400"], timeout=180, env=env)
    except subprocess.TimeoutExpired:
        return False, 0.0, "gate timeout"
    raw = p.stdout + p.stderr
    ok = p.returncode == 0 and all(g in raw for g in GATES)
    m = _TIME.search(raw)
    sps = 400.0 / float(m.group(1)) if (m and float(m.group(1)) > 0) else 0.0
    return ok, sps, raw[-500:]


@action(reads=[], writes=["hardware_constraints"])
def characterize(state: State) -> tuple[dict, State]:
    caps = {
        "max_wg_invocations": 256,
        "shared_mem_bytes": 16384,
        "subgroup_size": 16,
        "note": "stencil is bound by fixed per-invocation cost; host contract "
        "(fused, strip) is part of the search space, not just shader bodies",
    }
    return {"caps": caps}, state.update(hardware_constraints=caps)


@action(reads=[], writes=["best_sps", "baseline_sps", "best_variant"])
def baseline(state: State) -> tuple[dict, State]:
    # Baseline = the packed two-pass, one cell per invocation (matches the
    # original flux/height performance).
    with open(f"{RESEARCH}/flux2.comp") as f:
        flux = f.read()
    with open(f"{RESEARCH}/height2.comp") as f:
        height = f.read()
    with open(FLUX_SRC, "w") as f:
        f.write(flux)
    with open(HEIGHT_SRC, "w") as f:
        f.write(height)
    ok, _ = compile_shaders(fused=False)
    gate_ok, sps, _raw = run_gate(fused=False, strip=1) if ok else (False, 0.0, "")
    variant = {"shader": flux, "height_shader": height, "strip": 1}
    return {"baseline_steps_per_sec": sps, "gate_ok": gate_ok}, state.update(
        best_sps=sps, baseline_sps=sps, best_variant=variant
    )


@action(reads=["best_sps", "best_variant", "variant_log"], writes=["exp_idx"])
def hypothesize(state: State) -> tuple[dict, State]:
    idx = state["exp_idx"] + 1
    return {
        "experiment": idx,
        "best_steps_per_sec": state["best_sps"],
        "best_variant_strip": state["best_variant"].get("strip"),
        "best_variant_fused": state["best_variant"].get("height_shader", "") == "",
        "history": state["variant_log"][-6:],
        "instruction": "Propose a variant via implement inputs: `shader` (flux "
        "or fused kernel source), `height_shader` (full source, or empty "
        "string for a fused single-dispatch step), `strip` (cells per "
        "invocation in y; sets dispatch geometry). All three physics gates "
        "must pass. 256 max invocations, 16 KB shared, subgroup 16.",
    }, state.update(exp_idx=idx)


@action(
    reads=["exp_idx"],
    writes=["proposed_variant", "implement_ok"],
)
def implement(
    state: State, shader: str = "", height_shader: str = "", strip: int = 1
) -> tuple[dict, State]:
    if not shader.strip():
        return {"implement_ok": False, "error": "no shader provided"}, state.update(
            proposed_variant={}, implement_ok=False
        )
    with open(FLUX_SRC, "w") as f:
        f.write(shader)
    if height_shader.strip():
        with open(HEIGHT_SRC, "w") as f:
            f.write(height_shader)
    variant = {"shader": shader, "height_shader": height_shader, "strip": int(strip)}
    return {
        "implement_ok": True,
        "fused": not height_shader.strip(),
        "strip": int(strip),
    }, state.update(proposed_variant=variant, implement_ok=True)


@action(
    reads=["proposed_variant"],
    writes=["compile_ok", "compile_err", "verify_ok", "benchmark_done"],
)
def compile_action(state: State) -> tuple[dict, State]:
    fused = not state["proposed_variant"].get("height_shader", "").strip()
    ok, errs = compile_shaders(fused)
    return {"compile_ok": ok, "compile_err": errs}, state.update(
        compile_ok=ok, compile_err=errs, verify_ok=False, benchmark_done=False
    )


@action(
    reads=["proposed_variant"],
    writes=["verify_ok", "pending_sps", "gate_raw", "benchmark_done"],
)
def verify(state: State) -> tuple[dict, State]:
    v = state["proposed_variant"]
    fused = not v.get("height_shader", "").strip()
    ok, sps, raw = run_gate(fused, v.get("strip", 1))
    return {"verify_ok": ok, "steps_per_sec_if_kept": sps}, state.update(
        verify_ok=ok, pending_sps=sps, gate_raw=raw, benchmark_done=False
    )


@action(reads=["pending_sps"], writes=["measured_sps", "benchmark_done"])
def benchmark_action(state: State) -> tuple[dict, State]:
    return {"measured_steps_per_sec": state["pending_sps"]}, state.update(
        measured_sps=state["pending_sps"], benchmark_done=True
    )


@action(
    reads=["measured_sps", "best_sps", "proposed_variant", "consecutive_no_gain"],
    writes=["verdict", "best_sps", "best_variant", "consecutive_no_gain"],
)
def evaluate(state: State) -> tuple[dict, State]:
    sps, best = state["measured_sps"], state["best_sps"]
    if sps > best * 1.01:
        v = state["proposed_variant"]
        with open(BEST_PATH, "w") as f:
            f.write(
                f"// best: {sps:.0f} steps/s (256^2 x 400, vkflood2 physics gate)"
                f" fused={not v.get('height_shader', '').strip()}"
                f" strip={v.get('strip', 1)}\n{v['shader']}"
            )
        return {"verdict": "keep", "steps_per_sec": sps}, state.update(
            verdict="keep", best_sps=sps, best_variant=v, consecutive_no_gain=0
        )
    return {"verdict": "revert", "steps_per_sec": sps, "best": best}, state.update(
        verdict="revert", consecutive_no_gain=state["consecutive_no_gain"] + 1
    )


@action(
    reads=[
        "exp_idx",
        "verify_ok",
        "benchmark_done",
        "verdict",
        "measured_sps",
        "best_sps",
        "compile_ok",
        "consecutive_no_gain",
        "variant_log",
        "max_experiments",
        "proposed_variant",
    ],
    writes=["variant_log", "verdict", "consecutive_no_gain", "should_stop"],
)
def log_variant(state: State) -> tuple[dict, State]:
    verdict = state["verdict"]
    consecutive = state["consecutive_no_gain"]
    if not state["benchmark_done"]:  # compile/verify failed -> bypassed evaluate
        # correctness failures don't count toward the 3-strike stop
        # (phase1_harness_fsm.md: the agent pivots on failure); the budget still bounds
        verdict = "revert"
    v = state["proposed_variant"]
    entry = {
        "exp": state["exp_idx"],
        "fused": not v.get("height_shader", "").strip() if v else None,
        "strip": v.get("strip") if v else None,
        "compile_ok": state["compile_ok"],
        "verify_ok": state["verify_ok"],
        "steps_per_sec": state["measured_sps"] if state["benchmark_done"] else None,
        "best_sps": state["best_sps"],
        "verdict": verdict,
    }
    log = [*state["variant_log"], entry]
    should_stop = consecutive >= 3 or len(log) >= state["max_experiments"]
    return {"logged": entry, "consecutive_no_gain": consecutive}, state.update(
        variant_log=log,
        verdict=verdict,
        consecutive_no_gain=consecutive,
        should_stop=should_stop,
    )


@action(
    reads=["variant_log", "best_sps", "baseline_sps", "best_variant"],
    writes=["final_summary"],
)
def stop(state: State) -> tuple[dict, State]:
    summary = {
        "experiments": len(state["variant_log"]),
        "kept": sum(1 for e in state["variant_log"] if e["verdict"] == "keep"),
        "baseline_steps_per_sec": state["baseline_sps"],
        "best_steps_per_sec": state["best_sps"],
        "speedup": state["best_sps"] / state["baseline_sps"] if state["baseline_sps"] else 0.0,
        "best_fused": state["best_variant"].get("height_shader", "") == "",
        "best_strip": state["best_variant"].get("strip"),
    }
    return {"summary": summary}, state.update(final_summary=summary)


def build_flood2_app(max_experiments: int = 12):
    init = {
        "best_variant": {},
        "proposed_variant": {},
        "baseline_sps": 0.0,
        "best_sps": 0.0,
        "measured_sps": 0.0,
        "pending_sps": 0.0,
        "exp_idx": 0,
        "implement_ok": False,
        "compile_ok": False,
        "compile_err": "",
        "verify_ok": False,
        "gate_raw": "",
        "benchmark_done": False,
        "verdict": "init",
        "consecutive_no_gain": 0,
        "variant_log": [],
        "should_stop": False,
        "hardware_constraints": {},
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
        .with_transitions(
            ("characterize", "baseline"),
            ("baseline", "hypothesize"),
            ("hypothesize", "implement"),
            ("implement", "compile_", expr("implement_ok")),
            ("implement", "hypothesize", expr("not implement_ok")),
            ("compile_", "verify", expr("compile_ok")),
            ("compile_", "log_variant", expr("not compile_ok")),
            ("verify", "benchmark", expr("verify_ok")),
            ("verify", "log_variant", expr("not verify_ok")),  # guard
            ("benchmark", "evaluate"),
            ("evaluate", "log_variant"),
            ("log_variant", "stop", expr("should_stop")),
            ("log_variant", "hypothesize", expr("not should_stop")),
        )
        .with_entrypoint("characterize")
        .with_state(**init)
        .build()
    )
