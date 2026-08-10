"""AutoKernel-style exploration loop for llama.cpp's mul_mat_vec shader on V3D.

Same FSM shape as v3d_explore.py, retargeted at real LLM inference: the agent
rewrites ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp in the Pi's
llama.cpp tree (bb28c1f + the V3D 256-invocation class fix and the
GGML_VK_MMV_MAX_COLS / GGML_VK_DISABLE_FLASH_ATTN gates). compile_ rebuilds
llama.cpp; verify runs test-backend-ops against the exported Granite model
graph (NMSE vs the CPU oracle at real model shapes); benchmark measures
llama-bench GPU decode t/s. A variant that fails the graph gate never reaches
benchmark.

The benchmark runs at -ngl 6: full offload (-ngl 99) decodes incoherently on
this llama.cpp commit on BOTH v3dv and llvmpipe, stock and patched (upstream
bug, not V3D's), while ngl<=6 output is temp-0 coherent. At ngl 6 the GPU
mmv work is the majority of token time, so shader speedups still move t/s.

Runs on the Pi itself (the Theodosia server host), so actions use local
subprocess, not SSH.
"""

from __future__ import annotations

import os
import re
import subprocess

from burr.core import State, action, expr
from burr.core.application import ApplicationBuilder

BEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_mul_mat_vec.comp")

LLAMA = "/root/v3d-research/llama.cpp"
SHADER = f"{LLAMA}/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp"
BIN = f"{LLAMA}/build-vulkan/bin"
MODEL = "/root/v3d-research/models/granite-4.0-1b-dense/granite-4.0-1b-Q4_0.gguf"
GRAPH_OPS = "/root/v3d-research/granite_ops.txt"

ENV = {
    **os.environ,
    "GGML_VK_MMV_MAX_COLS": "1",
    "GGML_VK_DISABLE_FLASH_ATTN": "1",
}

_TG_LINE = re.compile(r"tg\d+\s*\|\s*([\d.]+)")


def _run(cmd: list[str], timeout: int, cwd: str | None = None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=ENV)


def read_shader() -> str:
    with open(SHADER) as f:
        return f.read()


def write_shader(src: str) -> None:
    with open(SHADER, "w") as f:
        f.write(src)


def compile_llama() -> tuple[bool, str]:
    p = _run(
        [
            "cmake",
            "--build",
            "build-vulkan",
            "--target",
            "test-backend-ops",
            "llama-bench",
            "-j4",
        ],
        timeout=1800,
        cwd=LLAMA,
    )
    return p.returncode == 0, (p.stdout + p.stderr)[-800:]


def run_graph_gate() -> tuple[bool, int, int, str]:
    """NMSE-verify every op of the Granite graph on V3D vs the CPU oracle."""
    try:
        p = _run(
            [
                "./test-backend-ops",
                "test",
                "-b",
                "Vulkan0",
                "--test-file",
                GRAPH_OPS,
            ],
            timeout=1200,
            cwd=BIN,
        )
    except subprocess.TimeoutExpired:
        return False, 0, 0, "verify timeout (pathological pipeline compile?)"
    raw = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + p.stderr)
    m = re.search(r"(\d+)/(\d+) tests passed", raw)
    n_ok, n_total = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    n_fail = n_total - n_ok
    ok = p.returncode == 0 and m is not None and n_ok == n_total and n_ok >= 20
    return ok, n_ok, n_fail, raw[-600:]


def run_decode_bench() -> tuple[float, str]:
    try:
        p = _run(
            [
                "./llama-bench",
                "-m",
                MODEL,
                "-ngl",
                "6",
                "-fa",
                "0",
                "-p",
                "0",
                "-n",
                "32",
                "-r",
                "1",
            ],
            timeout=900,
            cwd=BIN,
        )
    except subprocess.TimeoutExpired:
        return 0.0, "benchmark timeout"
    raw = p.stdout + p.stderr
    m = _TG_LINE.search(raw)
    return (float(m.group(1)) if m else 0.0), raw[-400:]


@action(reads=[], writes=["hardware_constraints"])
def characterize(state: State) -> tuple[dict, State]:
    caps = {
        "max_wg_invocations": 256,
        "shared_mem_bytes": 16384,
        "subgroup_size": 16,
        "fp16": False,
        "int_dot": False,
        "note": "v3dv register allocator falls back through strategies; "
        "shaders that compile at 4 QPU threads without spills run fastest",
    }
    return {"caps": caps}, state.update(hardware_constraints=caps)


@action(reads=[], writes=["current_shader", "best_tps", "baseline_tps", "best_shader"])
def baseline(state: State) -> tuple[dict, State]:
    src = read_shader()
    ok, errs = compile_llama()
    gate_ok, n_ok, n_fail, _ = run_graph_gate() if ok else (False, 0, 0, errs)
    tps, _ = run_decode_bench() if gate_ok else (0.0, "")
    return {
        "baseline_tps": tps,
        "gate_ok": gate_ok,
        "gate_cases_ok": n_ok,
        "compile_ok": ok,
    }, state.update(current_shader=src, best_shader=src, best_tps=tps, baseline_tps=tps)


@action(reads=["best_shader", "best_tps", "variant_log"], writes=["exp_idx"])
def hypothesize(state: State) -> tuple[dict, State]:
    idx = state["exp_idx"] + 1
    return {
        "experiment": idx,
        "best_tps": state["best_tps"],
        "current_best_shader": state["best_shader"],
        "history": state["variant_log"][-6:],
        "instruction": "Rewrite mul_mat_vec.comp (pass full source in "
        "implement.inputs.shader) to raise GPU decode t/s while keeping the "
        "model-graph NMSE gate green. 256 max invocations, 16 KB shared, "
        "subgroup 16, no fp16 arithmetic.",
    }, state.update(exp_idx=idx)


@action(reads=["exp_idx"], writes=["proposed_shader", "implement_ok"])
def implement(state: State, shader: str = "") -> tuple[dict, State]:
    if not shader.strip():
        return {"implement_ok": False, "error": "no shader provided"}, state.update(
            proposed_shader="", implement_ok=False
        )
    write_shader(shader)
    return {"implement_ok": True, "bytes": len(shader)}, state.update(
        proposed_shader=shader, implement_ok=True
    )


@action(reads=[], writes=["compile_ok", "compile_err", "verify_ok", "benchmark_done"])
def compile_action(state: State) -> tuple[dict, State]:
    ok, errs = compile_llama()
    return {"compile_ok": ok, "compile_err": errs[-500:]}, state.update(
        compile_ok=ok, compile_err=errs[-500:], verify_ok=False, benchmark_done=False
    )


@action(
    reads=[],
    writes=[
        "verify_ok",
        "gate_ok_count",
        "gate_fail_count",
        "gate_raw",
        "benchmark_done",
    ],
)
def verify(state: State) -> tuple[dict, State]:
    ok, n_ok, n_fail, raw = run_graph_gate()
    return {"verify_ok": ok, "cases_ok": n_ok, "cases_fail": n_fail}, state.update(
        verify_ok=ok,
        gate_ok_count=n_ok,
        gate_fail_count=n_fail,
        gate_raw=raw,
        benchmark_done=False,
    )


@action(reads=[], writes=["measured_tps", "benchmark_done"])
def benchmark_action(state: State) -> tuple[dict, State]:
    tps, _ = run_decode_bench()
    return {"measured_tps": tps}, state.update(measured_tps=tps, benchmark_done=True)


@action(
    reads=["measured_tps", "best_tps", "proposed_shader", "consecutive_no_gain"],
    writes=[
        "verdict",
        "best_tps",
        "best_shader",
        "current_shader",
        "consecutive_no_gain",
    ],
)
def evaluate(state: State) -> tuple[dict, State]:
    t, best = state["measured_tps"], state["best_tps"]
    if t > best * 1.01:
        with open(BEST_PATH, "w") as f:
            f.write(
                f"// best: {t:.3f} t/s decode (granite-4.0-1b Q4_0, -ngl 6 "
                f"-fa 0)\n{state['proposed_shader']}"
            )
        return {"verdict": "keep", "tps": t}, state.update(
            verdict="keep",
            best_tps=t,
            best_shader=state["proposed_shader"],
            current_shader=state["proposed_shader"],
            consecutive_no_gain=0,
        )
    return {"verdict": "revert", "tps": t, "best": best}, state.update(
        verdict="revert", consecutive_no_gain=state["consecutive_no_gain"] + 1
    )


@action(
    reads=[
        "exp_idx",
        "verify_ok",
        "benchmark_done",
        "verdict",
        "measured_tps",
        "best_tps",
        "compile_ok",
        "consecutive_no_gain",
        "variant_log",
        "max_experiments",
        "best_shader",
    ],
    writes=["variant_log", "verdict", "consecutive_no_gain", "should_stop"],
)
def log_variant(state: State) -> tuple[dict, State]:
    verdict = state["verdict"]
    consecutive = state["consecutive_no_gain"]
    if not state["benchmark_done"]:  # compile/verify failed -> bypassed evaluate
        # ponytail: correctness failures don't count toward the 3-strike stop
        # (phase1_harness_fsm.md: the agent pivots on failure); budget still bounds
        verdict = "revert"
    if verdict == "revert" and state["best_shader"]:
        # never leave the tree sitting on a rejected edit
        write_shader(state["best_shader"])
    entry = {
        "exp": state["exp_idx"],
        "compile_ok": state["compile_ok"],
        "verify_ok": state["verify_ok"],
        "benchmarked": state["benchmark_done"],
        "tps": state["measured_tps"] if state["benchmark_done"] else None,
        "best_tps": state["best_tps"],
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
    reads=["variant_log", "best_tps", "baseline_tps", "best_shader"],
    writes=["final_summary"],
)
def stop(state: State) -> tuple[dict, State]:
    write_shader(state["best_shader"])
    summary = {
        "experiments": len(state["variant_log"]),
        "kept": sum(1 for e in state["variant_log"] if e["verdict"] == "keep"),
        "baseline_tps": state["baseline_tps"],
        "best_tps": state["best_tps"],
        "speedup": state["best_tps"] / state["baseline_tps"] if state["baseline_tps"] else 0.0,
    }
    return {"summary": summary}, state.update(final_summary=summary)


def build_llama_mmv_app(max_experiments: int = 8):
    init = {
        "current_shader": "",
        "best_shader": "",
        "proposed_shader": "",
        "baseline_tps": 0.0,
        "best_tps": 0.0,
        "measured_tps": 0.0,
        "exp_idx": 0,
        "implement_ok": False,
        "compile_ok": False,
        "compile_err": "",
        "verify_ok": False,
        "gate_ok_count": 0,
        "gate_fail_count": 0,
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
