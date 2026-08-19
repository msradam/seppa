"""Recreate the flood optimization through the FSM: baseline, then submit
the fused strip-2 kernel as experiment 1 and let the machine judge it."""

import json
import os

import v3d_flood2_opt

app = v3d_flood2_opt.build_flood2_app()
_HERE = os.path.dirname(os.path.abspath(__file__))
fused_src = open(os.path.join(_HERE, "..", "pi", "flood", "fused2s.comp")).read()

steps = [
    ("characterize", {}),
    ("baseline", {}),
    ("hypothesize", {}),
    ("implement", {"shader": fused_src, "height_shader": "", "strip": 2}),
    ("compile_", {}),
    ("verify", {}),
    ("benchmark", {}),
    ("evaluate", {}),
    ("log_variant", {}),
]
for name, inputs in steps:
    action, result, state = app.step(inputs=inputs)
    assert action.name == name, f"expected {name}, FSM chose {action.name}"
    slim = {k: v for k, v in result.items() if not isinstance(v, str) or len(v) < 120}
    print(f"{name}: {json.dumps(slim)}")
print("LEDGER:", json.dumps(state["variant_log"]))
