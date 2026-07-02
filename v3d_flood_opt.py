"""Agentic optimization of the V3D flood kernel — the closing loop.

Same correctness-gated exploration as v3d_explore.py, but the artifact is the
TWO-shader flux-limited overland-flow simulation (flux.comp + height.comp), and
the gate is richer: a variant is only VALID if it still matches the CPU
reference (NMSE < 1e-3) AND conserves mass vs rainfall AND pools water in the
basin. So the agent may retile/vectorize the implementation for speed, but it
physically cannot cheat the hydrology — a variant that breaks conservation or
pooling is rejected even if it's faster. Baseline ~1.44 GFLOP/s (memory-bound,
2-pass); headroom is shared-memory tiling to raise arithmetic intensity so the
flood sim becomes a good compute-bound co-processing citizen.

Run via: v3d_drive.py --flood   (agent rewrites both shaders over MCP)
"""

from __future__ import annotations

import os
import re
import subprocess

from burr.core import State, action, expr
from burr.core.application import ApplicationBuilder

PI_HOST = "root@pi.local"
RES = "/root/v3d-research"
FLUX = f"{RES}/flux.comp"
HEIGHT = f"{RES}/height.comp"
BEST_DIR = os.path.dirname(os.path.abspath(__file__))

_GFLOPS = re.compile(r"([\d.]+)\s+GFLOP/s")
_NMSE = re.compile(r"correct\(NMSE vs CPU\)=(yes|no)\s+NMSE=([0-9.eE+\-]+)")
_MASS = re.compile(r"conserved vs rain=(yes|NO)")
_POOL = re.compile(r"pools in basin=(yes|no)")


def _ssh(
    cmd: str, timeout: int = 180, stdin: str | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", PI_HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin,
    )


def read_shaders() -> dict:
    return {"flux": _ssh(f"cat {FLUX}").stdout, "height": _ssh(f"cat {HEIGHT}").stdout}


def write_shaders(flux: str, height: str) -> None:
    _ssh(f"cat > {FLUX}", stdin=flux)
    _ssh(f"cat > {HEIGHT}", stdin=height)


def compile_shaders() -> tuple[bool, str]:
    p = _ssh(
        f"cd {RES} && glslangValidator -V flux.comp -o flux.spv && "
        f"glslangValidator -V height.comp -o height.spv"
    )
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def run_vkflood() -> dict:
    """Return {nmse_ok, mass_ok, pools_ok, gflops, raw}."""
    p = _ssh(f"cd {RES} && timeout 90 ./vkflood 256 400", timeout=120)
    raw = p.stdout + p.stderr
    nm = _NMSE.search(raw)
    g = _GFLOPS.search(raw)
    return {
        "nmse_ok": bool(nm and nm.group(1) == "yes"),
        "nmse": float(nm.group(2)) if nm else 1.0,
        "mass_ok": bool(_MASS.search(raw) and _MASS.search(raw).group(1) == "yes"),
        "pools_ok": bool(_POOL.search(raw) and _POOL.search(raw).group(1) == "yes"),
        "gflops": float(g.group(1)) if g else 0.0,
        "raw": raw[-400:],
    }


@action(reads=[], writes=["hw"])
def characterize(state: State) -> tuple[dict, State]:
    return {"hw": "V3D 256/16KB/fp32"}, state.update(hw="V3D")


@action(reads=[], writes=["cur_flux", "cur_height", "best_gflops", "baseline_gflops"])
def baseline(state: State) -> tuple[dict, State]:
    s = read_shaders()
    write_shaders(s["flux"], s["height"])
    ok, _ = compile_shaders()
    r = run_vkflood() if ok else {"gflops": 0.0}
    g = r.get("gflops", 0.0)
    return {"baseline_gflops": g, "correct": r.get("nmse_ok")}, state.update(
        cur_flux=s["flux"], cur_height=s["height"], best_gflops=g, baseline_gflops=g
    )


@action(
    reads=["cur_flux", "cur_height", "best_gflops", "variant_log"], writes=["exp_idx"]
)
def hypothesize(state: State) -> tuple[dict, State]:
    idx = state["exp_idx"] + 1
    return {
        "experiment": idx,
        "best_gflops": state["best_gflops"],
        "current_flux": state["cur_flux"],
        "current_height": state["cur_height"],
        "history": state["variant_log"][-5:],
        "instruction": "Rewrite flux.comp and/or height.comp to raise GFLOP/s while keeping "
        "the numerical result IDENTICAL (NMSE vs the fixed CPU reference < 1e-3), mass "
        "conserved, and water pooling in the basin. Pass full source for both in "
        "implement.inputs.flux and implement.inputs.height. Main lever: shared-memory tiling "
        "(load an 18x18 halo tile of H+W into shared memory, <16KB, reuse across the 16x16 "
        "workgroup) to cut global neighbour reads and raise arithmetic intensity.",
    }, state.update(exp_idx=idx)


@action(
    reads=["cur_flux", "cur_height"],
    writes=["prop_flux", "prop_height", "implement_ok"],
)
def implement(state: State, flux: str = "", height: str = "") -> tuple[dict, State]:
    f = flux.strip() or state["cur_flux"]
    h = height.strip() or state["cur_height"]
    if not flux.strip() and not height.strip():
        return {"implement_ok": False, "error": "no shader provided"}, state.update(
            prop_flux="", prop_height="", implement_ok=False
        )
    write_shaders(f, h)
    return {
        "implement_ok": True,
        "flux_bytes": len(f),
        "height_bytes": len(h),
    }, state.update(prop_flux=f, prop_height=h, implement_ok=True)


@action(reads=[], writes=["compile_ok", "compile_err"])
def compile_action(state: State) -> tuple[dict, State]:
    ok, errs = compile_shaders()
    return {"compile_ok": ok, "compile_err": errs[:500]}, state.update(
        compile_ok=ok, compile_err=errs[:500]
    )


@action(reads=[], writes=["verify_ok", "pending_gflops", "vk_raw"])
def verify(state: State) -> tuple[dict, State]:
    r = run_vkflood()
    # PHYSICS GATE: correct numerics AND mass conservation AND pooling.
    ok = r["nmse_ok"] and r["mass_ok"] and r["pools_ok"]
    return {
        "correct": r["nmse_ok"],
        "mass_ok": r["mass_ok"],
        "pools_ok": r["pools_ok"],
        "gflops_if_ok": r["gflops"],
        "nmse": r["nmse"],
    }, state.update(verify_ok=ok, pending_gflops=r["gflops"], vk_raw=r["raw"])


@action(reads=["pending_gflops"], writes=["measured_gflops", "benchmark_done"])
def benchmark_action(state: State) -> tuple[dict, State]:
    return {"measured_gflops": state["pending_gflops"]}, state.update(
        measured_gflops=state["pending_gflops"], benchmark_done=True
    )


@action(
    reads=[
        "measured_gflops",
        "best_gflops",
        "prop_flux",
        "prop_height",
        "consecutive_no_gain",
    ],
    writes=["verdict", "best_gflops", "cur_flux", "cur_height", "consecutive_no_gain"],
)
def evaluate(state: State) -> tuple[dict, State]:
    g, best = state["measured_gflops"], state["best_gflops"]
    if g > best * 1.01:
        with open(os.path.join(BEST_DIR, "best_flux.comp"), "w") as fp:
            fp.write(f"// best flood flux: {g:.2f} GFLOP/s\n{state['prop_flux']}")
        with open(os.path.join(BEST_DIR, "best_height.comp"), "w") as fp:
            fp.write(f"// best flood height: {g:.2f} GFLOP/s\n{state['prop_height']}")
        return {"verdict": "keep", "gflops": g}, state.update(
            verdict="keep",
            best_gflops=g,
            cur_flux=state["prop_flux"],
            cur_height=state["prop_height"],
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
    if verdict == "init":
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
    return {"logged": entry}, state.update(
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
        "speedup": state["best_gflops"] / state["baseline_gflops"]
        if state["baseline_gflops"]
        else 0.0,
    }
    return {"summary": summary}, state.update(final_summary=summary)


def build_flood_app(max_experiments: int = 10):
    init = {
        "hw": "",
        "cur_flux": "",
        "cur_height": "",
        "prop_flux": "",
        "prop_height": "",
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
        "vk_raw": "",
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
            ("verify", "benchmark", expr("verify_ok")),  # PHYSICS GATE
            ("verify", "log_variant", expr("not verify_ok")),
            ("benchmark", "evaluate"),
            ("evaluate", "log_variant"),
            ("log_variant", "stop", expr("should_stop")),
            ("log_variant", "hypothesize", expr("not should_stop")),
        )
        .with_entrypoint("characterize")
        .with_state(**init)
        .build()
    )
