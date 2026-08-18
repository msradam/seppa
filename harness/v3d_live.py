"""Live run_fn: drive the FSM against the real Pi V3D GPU over SSH.

Each variant names an op and a backend; the run_fn SSH-runs llama.cpp's
`test-backend-ops test -o <OP> -b <BACKEND>` on the Pi, captures the real
stdout + returncode, and hands them to the v3d_verify gate. No shader rebuild
yet (Phase 2): this exercises the gate against real hardware, where V3D matmul
aborts on the 16 KB shared-memory wall and CPU passes.

`ts` here is a placeholder (inverse wall-clock of the correctness run) so a
passing op has *a* monotonic number to benchmark; the real perf metric is
llama-bench decode t/s, wired in Phase 2. V3D ops abort before benchmark, so
their ts is never used.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

PI_HOST = "root@pi.local"
BIN_DIR = "/root/v3d-research/llama.cpp/build-vulkan/bin"

# The live experiment set: a CPU positive control that really passes, and the
# two V3D ops that really hit the 16 KB matmul wall.
LIVE_VARIANTS = [
    {"id": "rmsnorm_cpu", "op": "RMS_NORM", "backend": "CPU"},
    {"id": "rmsnorm_v3d", "op": "RMS_NORM", "backend": "Vulkan0"},
    {"id": "mulmat_v3d", "op": "MUL_MAT", "backend": "Vulkan0"},
]


def make_live_run_fn(
    host: str = PI_HOST, bin_dir: str = BIN_DIR, timeout: int = 180
) -> Callable[[dict], dict]:
    """Return a run_fn that runs test-backend-ops on the Pi for one variant."""

    def run_fn(variant: dict) -> dict:
        op = variant["op"]
        backend = variant["backend"]
        remote = f"cd {bin_dir} && ./test-backend-ops test -o {op} -b {backend}"
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout, rc = proc.stdout + proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            stdout, rc = "TIMEOUT: test-backend-ops exceeded budget", 124
        elapsed = max(time.time() - t0, 1e-3)
        return {
            "compile_oom": False,  # abort is a runtime/verify failure, not a build OOM
            "op_stdout": stdout,
            "op_returncode": rc,
            "output": None,  # test-backend-ops compares internally; no tensor to us
            "ts": round(1.0 / elapsed, 4),  # placeholder inverse-wall-clock metric
        }

    return run_fn


if __name__ == "__main__":
    # Smoke: run each live variant against the Pi and print the real verdict.
    import v3d_verify as v

    fn = make_live_run_fn()
    for var in LIVE_VARIANTS:
        res = fn(var)
        op = v.parse_test_backend_ops(res["op_stdout"], res["op_returncode"])
        print(
            f"{var['id']:14s} op={var['op']:9s} backend={var['backend']:8s} "
            f"-> passed={op.passed}  {op.raw_summary or (op.failures[:1] or [''])[0]}"
        )
