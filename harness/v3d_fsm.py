"""Phase-1 single-kernel V3D optimization FSM (Burr).

    CHARACTERIZE -> BASELINE -> HYPOTHESIZE -> IMPLEMENT -> COMPILE
                 -> VERIFY -> BENCHMARK -> EVALUATE -> LOG -> (loop | STOP)

Adapted from ~/v3d-pi-ai/burr_fsm/fsm.py, trimmed to ONE kernel (no
multi-kernel orchestration, no Amdahl), and wired to the v3d_verify gate.

The load-bearing property this file demonstrates: the VERIFY -> BENCHMARK
edge is guarded by `not (verify_ok and verify_complete)`, so a kernel that
fails correctness (or writes only part of its output)
is routed straight to LOG with a revert verdict and is NEVER benchmarked.
The agent (in the real system) owns HYPOTHESIZE/IMPLEMENT; the machine owns
COMPILE/VERIFY/BENCHMARK/EVALUATE and the agent cannot skip the gate.

Live vs replay share the graph; only COMPILE consults an injected `run_fn`
(replay returns canned per-variant results; live calls the Pi's
test-backend-ops / MNN and hands back the op stdout + output tensor).

Run `python v3d_fsm.py` for the end-to-end guard proof.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from burr.core import State, action, expr
from burr.core.application import ApplicationBuilder

import v3d_verify

# A run_fn maps a variant dict -> a canned/live result:
#   {"compile_oom": bool, "op_stdout": str, "op_returncode": int,
#    "output": np.ndarray | None, "ts": float}
# `ts` (throughput) must only ever be read in BENCHMARK, never before.
RunFn = Callable[[dict], dict]

# Theodosia serializes Burr State to JSON on every step (ledger + step response),
# so State must stay JSON-clean: no callables, no numpy arrays. The run_fn and the
# raw output tensor live here as module side-channels instead of in State.
# Single-session globals. Per-session isolation (factory/http multi-
# client) would key these by app_id; Phase 1 drives one session at a time.
_RUN_FN: RunFn | None = None
_LAST_OUTPUT: np.ndarray | None = None


@action(reads=[], writes=["hardware_constraints", "started_at"])
def characterize(state: State) -> tuple[dict, State]:
    # Live: run v3d_caps. Replay: the V3D envelope is fixed and known.
    caps = {
        "max_wg_invocations": 256,
        "shared_mem_bytes": 16384,
        "subgroup": 16,
        "fp16_arith": False,
        "coop_matrix": False,
    }
    return {"caps": caps}, state.update(hardware_constraints=caps, started_at=time.time())


@action(reads=["baseline_ts"], writes=["best_ts"])
def baseline(state: State) -> tuple[dict, State]:
    return {"baseline_ts": state["baseline_ts"]}, state.update(best_ts=state["baseline_ts"])


@action(
    reads=["variants", "variant_idx"],
    writes=[
        "variant_idx",
        "selected_variant",
        "search_exhausted",
        "verify_ok",
        "verify_complete",
        "benchmark_done",
        "verdict",
    ],
)
def hypothesize(state: State) -> tuple[dict, State]:
    idx = state["variant_idx"] + 1
    variants = state["variants"]
    if idx >= len(variants):
        return {"exhausted": True}, state.update(search_exhausted=True)
    # Reset per-iteration flags so nothing leaks from the previous variant.
    return {"selected": variants[idx]["id"]}, state.update(
        variant_idx=idx,
        selected_variant=variants[idx],
        search_exhausted=False,
        verify_ok=False,
        verify_complete=False,
        benchmark_done=False,
        verdict="init",
    )


@action(reads=["selected_variant"], writes=["implement_ok"])
def implement(state: State) -> tuple[dict, State]:
    # Live: write kernel.comp + host patch, git commit. Replay: no-op.
    return {"applied": state["selected_variant"]["id"]}, state.update(implement_ok=True)


@action(
    reads=["selected_variant"],
    writes=["compile_ok", "compile_oom", "current_result"],
)
def compile_action(state: State) -> tuple[dict, State]:
    # Live: glslangValidator + offline SPIR-V repack + relink. This is where
    # the variant is actually built and its raw result obtained.
    global _LAST_OUTPUT
    res = _RUN_FN(state["selected_variant"])
    _LAST_OUTPUT = res.get("output")  # numpy array stays out of State
    oom = bool(res.get("compile_oom"))
    # current_result keeps only JSON-serializable fields (Theodosia serializes State).
    summary = {
        "op_stdout": res.get("op_stdout", ""),
        "op_returncode": int(res.get("op_returncode", 0)),
        "ts": float(res.get("ts", 0.0)),
        "has_output": res.get("output") is not None,
    }
    return {"compile_oom": oom}, state.update(
        compile_ok=not oom, compile_oom=oom, current_result=summary
    )


@action(reads=["current_result"], writes=["verify_ok", "verify_complete"])
def verify(state: State) -> tuple[dict, State]:
    res = state["current_result"]
    op = v3d_verify.parse_test_backend_ops(res["op_stdout"], res["op_returncode"])
    vr = v3d_verify.verify(op, _LAST_OUTPUT)
    return {
        "verify_ok": vr.verify_ok,
        "verify_complete": vr.verify_complete,
        "reasons": (vr.completeness.reasons if vr.completeness else []),
    }, state.update(verify_ok=vr.verify_ok, verify_complete=vr.verify_complete)


@action(reads=["current_result"], writes=["benchmark_done", "measured_ts"])
def benchmark_action(state: State) -> tuple[dict, State]:
    # Reachable ONLY when verify_ok and verify_complete. This is the first and
    # only place a timing value is trusted.
    ts = float(state["current_result"]["ts"])
    return {"measured_ts": ts}, state.update(benchmark_done=True, measured_ts=ts)


@action(
    reads=["measured_ts", "best_ts", "consecutive_no_gain"],
    writes=["verdict", "best_ts", "consecutive_no_gain"],
)
def evaluate(state: State) -> tuple[dict, State]:
    # Deterministic keep/revert. The agent does not decide this.
    ts, best = state["measured_ts"], state["best_ts"]
    if ts > best * 1.02:  # >2% improvement
        return {"verdict": "keep"}, state.update(verdict="keep", best_ts=ts, consecutive_no_gain=0)
    return {"verdict": "noise"}, state.update(
        verdict="noise", consecutive_no_gain=state["consecutive_no_gain"] + 1
    )


@action(
    reads=[
        "selected_variant",
        "variant_idx",
        "verify_ok",
        "verify_complete",
        "benchmark_done",
        "verdict",
        "measured_ts",
        "best_ts",
        "consecutive_no_gain",
        "variant_log",
    ],
    writes=["variant_log", "verdict", "consecutive_no_gain", "should_stop"],
)
def log_variant(state: State) -> tuple[dict, State]:
    verdict = state["verdict"]
    consecutive = state["consecutive_no_gain"]
    # verify-fail bypassed evaluate, so verdict is still the "init" sentinel.
    if verdict == "init":
        verdict = "revert"  # correctness failure -> always revert
        # correctness failures do NOT count toward the no-gain stop counter.

    entry = {
        "variant": state["selected_variant"]["id"],
        "idx": state["variant_idx"],
        "verify_ok": state["verify_ok"],
        "verify_complete": state["verify_complete"],
        "benchmarked": state["benchmark_done"],
        "measured_ts": state["measured_ts"] if state["benchmark_done"] else None,
        "best_ts": state["best_ts"],
        "verdict": verdict,
    }
    log = [*state["variant_log"], entry]
    should_stop = consecutive >= 3
    return {"logged": entry}, state.update(
        variant_log=log,
        verdict=verdict,
        consecutive_no_gain=consecutive,
        should_stop=should_stop,
    )


@action(reads=["variant_log", "best_ts", "baseline_ts"], writes=["final_summary"])
def stop(state: State) -> tuple[dict, State]:
    kept = [e for e in state["variant_log"] if e["verdict"] == "keep"]
    summary = {
        "experiments": len(state["variant_log"]),
        "kept": len(kept),
        "baseline_ts": state["baseline_ts"],
        "best_ts": state["best_ts"],
        "speedup": state["best_ts"] / state["baseline_ts"] if state["baseline_ts"] else 0.0,
    }
    return {"summary": summary}, state.update(final_summary=summary)


def build_app(
    run_fn: RunFn | None = None,
    variants: list[dict] | None = None,
    baseline_ts: float = 4.0,
    target_op: str = "MUL_MAT",
):
    """Factory. Zero-arg callable for Theodosia mount; args for replay/live."""
    global _RUN_FN
    if run_fn is None:
        run_fn, variants = _replay_demo()
    _RUN_FN = run_fn  # module side-channel; kept out of State for serializability
    init = {
        "library": "llama.cpp",
        "target_op": target_op,
        "variants": variants or [],
        "variant_idx": -1,
        "selected_variant": {},
        "search_exhausted": False,
        "implement_ok": False,
        "compile_ok": False,
        "compile_oom": False,
        "verify_ok": False,
        "verify_complete": False,
        "current_result": {},
        "benchmark_done": False,
        "measured_ts": 0.0,
        "verdict": "init",
        "baseline_ts": baseline_ts,
        "best_ts": baseline_ts,
        "consecutive_no_gain": 0,
        "variant_log": [],
        "should_stop": False,
        "hardware_constraints": {},
        "started_at": 0.0,
        "final_summary": {},
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
        # Every branch pair is MUTUALLY EXCLUSIVE (guarded + negated-guard), not
        # guarded + default. Theodosia lets the agent choose among all reachable
        # transitions, and an unconditional "default" branch is always reachable
        # -- so a default `benchmark` edge would let the agent bypass the gate on
        # a broken kernel. Explicit negation makes exactly one branch reachable.
        .with_transitions(
            ("characterize", "baseline"),
            ("baseline", "hypothesize"),
            ("hypothesize", "stop", expr("search_exhausted")),
            ("hypothesize", "implement", expr("not search_exhausted")),
            ("implement", "compile_"),
            ("compile_", "hypothesize", expr("compile_oom")),
            ("compile_", "verify", expr("not compile_oom")),
            # THE GUARD: benchmark is reachable ONLY if correctness passed.
            ("verify", "log_variant", expr("not (verify_ok and verify_complete)")),
            ("verify", "benchmark", expr("verify_ok and verify_complete")),
            ("benchmark", "evaluate"),
            ("evaluate", "log_variant"),
            ("log_variant", "stop", expr("should_stop")),
            ("log_variant", "hypothesize", expr("not should_stop")),
        )
        .with_entrypoint("characterize")
        .with_state(**init)
        .build()
    )


def _replay_demo() -> tuple[RunFn, list[dict]]:
    """Canned variants exercising every path, including a partial-output kernel."""
    rng = np.random.default_rng(0)
    good = rng.standard_normal((8, 16)).astype(np.float32)
    ok_stdout = "  MUL_MAT(m=16,n=1,k=256): OK\n  1/1 tests passed\n  Backend Vulkan0: OK\n"
    fail_stdout = "  MUL_MAT(m=64,n=64,k=64): NMSE = 0.5 > 0.0005 FAIL\n  0/1 tests passed\n"

    partial = good.copy()
    partial[4:, :] = np.nan  # wrote half the output channels; a partial-output kernel

    table = {
        # correct + faster -> KEEP, benchmarked
        "lws240": {
            "compile_oom": False,
            "op_stdout": ok_stdout,
            "op_returncode": 0,
            "output": good,
            "ts": 5.5,
        },
        # compile OOM -> loops back, never verified/benchmarked
        "lws512": {
            "compile_oom": True,
            "op_stdout": "",
            "op_returncode": 1,
            "output": None,
            "ts": 0.0,
        },
        # THE MIRAGE: op oracle says OK (autotuner has no correctness check) but
        # only half the output was written. Completeness must catch it and block
        # benchmark, even though ts=9.9 would look like a huge win.
        "pack_partial": {
            "compile_oom": False,
            "op_stdout": ok_stdout,
            "op_returncode": 0,
            "output": partial,
            "ts": 9.9,
        },
        # op oracle FAIL (real correctness break) -> revert, no benchmark
        "bad_stride": {
            "compile_oom": False,
            "op_stdout": fail_stdout,
            "op_returncode": 1,
            "output": None,
            "ts": 8.0,
        },
        # correct but not faster -> noise/revert, benchmarked
        "lws208": {
            "compile_oom": False,
            "op_stdout": ok_stdout,
            "op_returncode": 0,
            "output": good,
            "ts": 3.9,
        },
    }
    variants = [{"id": k} for k in table]
    return (lambda var: table[var["id"]]), variants


def _demo() -> None:
    app = build_app()
    _, _, state = app.run(halt_after=["stop"])
    log = {e["variant"]: e for e in state["variant_log"]}

    # The mirage: op oracle passed, but it was NEVER benchmarked (guard held),
    # and it was reverted. ts=9.9 was never trusted.
    m = log["pack_partial"]
    assert m["verify_ok"] and not m["verify_complete"], m
    assert m["benchmarked"] is False, m
    assert m["verdict"] == "revert", m

    # Real correctness failure: also never benchmarked.
    assert log["bad_stride"]["benchmarked"] is False
    assert log["bad_stride"]["verdict"] == "revert"

    # The genuine win: verified, benchmarked, kept.
    g = log["lws240"]
    assert g["verify_ok"] and g["verify_complete"] and g["benchmarked"] is True, g
    assert g["verdict"] == "keep", g

    # OOM variant looped back without ever reaching verify/benchmark.
    assert "lws512" not in log, "compile_oom variant should not be logged"

    # Correct-but-slow: benchmarked (correct) but not kept.
    s = log["lws208"]
    assert s["benchmarked"] is True and s["verdict"] == "noise", s

    print("v3d_fsm: guard proof passed")
    print(f"  experiments logged: {len(state['variant_log'])}")
    print(
        f"  kept: {state['final_summary']['kept']}  "
        f"best_ts: {state['final_summary']['best_ts']:.2f}  "
        f"speedup: {state['final_summary']['speedup']:.3f}x"
    )
    print(
        "  mirage (pack_partial): verify_ok=%s verify_complete=%s benchmarked=%s verdict=%s"
        % (m["verify_ok"], m["verify_complete"], m["benchmarked"], m["verdict"])
    )


if __name__ == "__main__":
    _demo()
