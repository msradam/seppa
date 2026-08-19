"""Offline check of the graph property the paper stakes its claims on:
`benchmark` is reachable only through a green `verify`. No GPU or Pi is
needed; compile and gate calls are stubbed, while the graph, guards, and
ledger logic are the real ones from v3d_flood2_opt.

    uv run python harness/test_gate.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3d_flood2_opt as m


def make_app(gate_ok: bool):
    td = tempfile.mkdtemp()
    for f in ("flux2.comp", "height2.comp"):
        with open(os.path.join(td, f), "w") as fh:
            fh.write("// stub\n")
    m.RESEARCH = td
    m.BEST_PATH = os.path.join(td, "best_flood.comp")
    m.FLUX_SRC = os.path.join(td, "fsm_flood.comp")
    m.HEIGHT_SRC = os.path.join(td, "fsm_height.comp")
    m.compile_shaders = lambda fused: (True, "")
    calls = {"n": 0}

    def fake_gate(fused, strip):
        calls["n"] += 1
        if calls["n"] == 1:
            return True, 1000.0, "baseline gate"
        return gate_ok, 1500.0, "candidate gate"

    m.run_gate = fake_gate
    return m.build_flood2_app()


def reachable(app):
    """Actions reachable from the current position, as the MCP server
    computes them: transitions from the last-run node whose condition
    holds in the current state."""
    last = app.state.get("__PRIOR_STEP")
    out = []
    for t in app.graph.transitions:
        if t.from_.name == last:
            res = t.condition.run(app.state)
            if res.get(t.condition.KEY, res.get("proceed", False)):
                out.append(t.to.name)
    return out


def drive(app, until, inputs_for=None):
    inputs_for = inputs_for or {}
    for _ in range(20):
        nxt = app.get_next_action()
        if nxt is None:
            break
        act, result, state = app.step(inputs=inputs_for.get(nxt.name, {}))
        if act.name == until:
            return app
    raise AssertionError(f"never reached {until}")


IMPL = {"implement": {"shader": "// candidate\n", "height_shader": "", "strip": 2}}


def test_refusal():
    app = make_app(gate_ok=False)
    drive(app, "verify", IMPL)
    nxt = reachable(app)
    assert "benchmark" not in nxt, nxt
    assert nxt == ["log_variant"], nxt
    assert app.state["verify_ok"] is False
    drive(app, "log_variant", IMPL)
    entry = app.state["variant_log"][-1]
    assert entry["verdict"] == "revert" and entry["steps_per_sec"] is None, entry


def test_pass():
    app = make_app(gate_ok=True)
    drive(app, "verify", IMPL)
    nxt = reachable(app)
    assert nxt == ["benchmark"], nxt
    drive(app, "log_variant", IMPL)
    entry = app.state["variant_log"][-1]
    assert entry["verdict"] == "keep" and entry["steps_per_sec"] == 1500.0, entry


if __name__ == "__main__":
    test_refusal()
    test_pass()
    print("gate property holds: benchmark unreachable after a failed verify,")
    print("failed variants ledgered as revert with null steps_per_sec.")
