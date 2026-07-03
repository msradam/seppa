"""V3D correctness gate, the load-bearing component of the FSM.

Two independent halves, matching the two ways a kernel gets checked on V3D:

1. parse_test_backend_ops reads llama.cpp `test-backend-ops -b Vulkan0`'s
   verdict. That tool runs each op on Vulkan vs the CPU backend and compares
   with NMSE internally; it only hands back OK/FAIL text, so this parses it.
   Full-output coverage is already enforced by its energy-normalized NMSE.

2. check_completeness holds the anti-gaming filters for the path where this
   code controls the output buffer (MNN path, custom dispatch). Poison-fill the
   buffer, run the kernel, then assert every element was written and the output
   is non-degenerate. This catches a partial-compute kernel (for example a
   dispatch that writes only some output channels) that an unguarded autotuner
   would crown as a large speedup.

`verify()` combines them into the predicate on the FSM's `VERIFY -> BENCHMARK`
edge: benchmarking is unreachable unless this returns verify_ok and (when an
output tensor is available) verify_complete.

Thresholds and mechanisms sourced in ../v3d-investigation/docs/verification_design.md.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

import numpy as np

# test-backend-ops per-op line. The tail after "): " varies by version/outcome:
#   "  MUL_MAT(type_a=f16,...): OK"
#   "  MUL_MAT(type_a=q5_1,...): NMSE = 0.000508874 > 0.000500000 FAIL"
#   "  ...: NOT SUPPORTED"
# so we capture the whole tail and classify it (only a bare "OK" tail passes).
_OP_LINE = re.compile(r"^\s*([A-Z0-9_]+)\((.*?)\):\s*(.+?)\s*$", re.MULTILINE)
_NMSE = re.compile(r"NMSE\s*=\s*([0-9.eE+\-]+)")
_SUMMARY = re.compile(r"(\d+)\s*/\s*(\d+)\s+tests passed")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")  # test-backend-ops colors OK/FAIL green/red
# Markers that mean the backend aborted before finishing (real V3D behaviour:
# "Shared memory size too small for matrix multiplication" -> SIGABRT).
_ABORT_MARKERS = (
    "Shared memory size too small",
    "terminate called",
    "Aborted",
    "GGML_ASSERT",
    "error:",
)

# robust-kbench degenerate-output thresholds (arXiv:2509.14279)
_TRIVIAL_BAND = 0.01  # output clamped inside [-band, band] is degenerate
_MIN_STD = 0.01  # overall and per-axis variation floor


@dataclass
class OpResult:
    passed: bool
    op: str | None
    total: int
    failed: int
    nmse_max: float | None
    failures: list[str] = field(default_factory=list)
    raw_summary: str = ""


def parse_test_backend_ops(
    stdout: str, returncode: int, op: str | None = None
) -> OpResult:
    """Parse `test-backend-ops test -b VulkanN [-o OP]` output into a verdict.

    Authoritative pass signal is: returncode == 0 AND the "N/M tests passed"
    summary shows N == M AND no per-op FAIL/NaN/NOT SUPPORTED line. We collect
    per-line failures (with NMSE when present) for the audit ledger.
    """
    stdout = _ANSI.sub("", stdout)  # strip color so "OK"/"FAIL" classify cleanly
    failures: list[str] = []
    nmse_values: list[float] = []
    total = failed = 0

    for m in _OP_LINE.finditer(stdout):
        op_name, params, tail = m.group(1), m.group(2), m.group(3)
        total += 1
        nm = _NMSE.search(tail)
        if nm:
            nmse_values.append(float(nm.group(1)))
        # Only a bare "OK" tail is a pass. FAIL / NaN / NOT SUPPORTED / an NMSE
        # exceedance ("... > ...") all count as failures for the gate.
        if tail.strip() != "OK":
            failed += 1
            failures.append(f"{op_name}({params}): {tail.strip()}")

    summary = _SUMMARY.search(stdout)
    raw_summary = summary.group(0) if summary else ""
    summary_ok = bool(summary) and summary.group(1) == summary.group(2)

    # A clean run: process exited 0, summary reports all-passed, no FAIL lines.
    # If we never matched a summary line, fall back to returncode + no failures.
    if summary:
        passed = returncode == 0 and summary_ok and failed == 0
    else:
        passed = returncode == 0 and failed == 0 and total > 0

    # No summary + nonzero exit means an abort (e.g. the 16 KB matmul limit on
    # V3D). Surface the reason for the ledger rather than an empty failure list.
    if not passed and not summary:
        for marker in _ABORT_MARKERS:
            if marker in stdout:
                failures.append(f"aborted: {marker}")
                break

    return OpResult(
        passed=passed,
        op=op,
        total=total,
        failed=failed,
        nmse_max=max(nmse_values) if nmse_values else None,
        failures=failures,
        raw_summary=raw_summary,
    )


def run_test_backend_ops(
    op: str,
    backend: str = "Vulkan0",
    binary: str = "test-backend-ops",
    timeout: int = 600,
) -> OpResult:
    """Run the op oracle on the Pi and parse it. Fresh inputs each run (the tool
    seeds with std::random_device); log the invocation for reproducibility."""
    proc = subprocess.run(
        [binary, "test", "-b", backend, "-o", op],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return parse_test_backend_ops(proc.stdout + proc.stderr, proc.returncode, op=op)


@dataclass
class CompletenessResult:
    complete: bool
    reasons: list[str] = field(default_factory=list)


def check_completeness(
    output: np.ndarray, poison_was_nan: bool = True
) -> CompletenessResult:
    """Anti-gaming filters for a controlled output buffer.

    Pre-condition: the buffer was poison-filled (NaN) before the kernel ran, so
    any un-written element is still NaN. Catches partial-compute, constant/
    broadcast fills, and clamped-to-trivial-band degenerate outputs.
    """
    reasons: list[str] = []
    out = np.asarray(output)

    # 1. Full-output coverage: no poison survivors (a partially-written output).
    coverage_ok = not (poison_was_nan and bool(np.isnan(out).any()))
    if not coverage_ok:
        reasons.append(f"coverage: {int(np.isnan(out).sum())} unwritten (NaN) elements")
    clean = out[~np.isnan(out)] if poison_was_nan else out
    if clean.size == 0:
        reasons.append("coverage: no written elements")
        return CompletenessResult(False, reasons)

    # 2. Range: not clamped inside a trivial band (robust-kbench Output Range).
    if np.all(np.abs(clean) <= _TRIVIAL_BAND):
        reasons.append(f"range: all |values| <= {_TRIVIAL_BAND} (degenerate)")

    # 3. Overall variation (robust-kbench Std Check).
    if float(clean.std()) <= _MIN_STD:
        reasons.append(f"std: {clean.std():.4g} <= {_MIN_STD} (near-constant output)")

    # 4. Per-axis variation (robust-kbench Axes Check): a constant/broadcast axis
    #    (e.g. unwritten output channels) has ~zero variation along it. Only
    #    meaningful on a fully-written grid, so skip if coverage already failed.
    if coverage_ok and out.ndim >= 1 and out.size > 1:
        for ax in range(out.ndim):
            if out.shape[ax] < 2:
                continue
            axis_std = out.std(axis=ax)
            if float(axis_std.mean()) <= _MIN_STD:
                reasons.append(
                    f"axes: axis {ax} variation {axis_std.mean():.4g} <= {_MIN_STD}"
                )

    return CompletenessResult(not reasons, reasons)


@dataclass
class VerifyResult:
    verify_ok: bool  # op oracle passed (NMSE vs CPU within tolerance)
    verify_complete: bool  # completeness filters passed (or n/a)
    verdict: str  # "pass" -> benchmark reachable; else "revert"
    op: OpResult | None = None
    completeness: CompletenessResult | None = None


def verify(op_result: OpResult, output: np.ndarray | None = None) -> VerifyResult:
    """The predicate on VERIFY -> BENCHMARK. `output` is optional: supply it only
    when we control the buffer (MNN / custom dispatch). For the pure
    test-backend-ops path, NMSE already enforces full-output coverage."""
    verify_ok = op_result.passed
    comp = check_completeness(output) if output is not None else None
    verify_complete = comp.complete if comp is not None else True
    verdict = "pass" if (verify_ok and verify_complete) else "revert"
    return VerifyResult(
        verify_ok, verify_complete, verdict, op=op_result, completeness=comp
    )


def _demo() -> None:
    """Runnable self-check: asserts the gate catches each documented exploit class."""
    rng = np.random.default_rng(42)

    # --- op-oracle parsing ---
    ok_out = (
        "  MUL_MAT(type_a=f16,type_b=f32,m=16,n=1,k=256): OK\n"
        "  MUL_MAT(type_a=f32,type_b=f32,m=32,n=2,k=64): OK\n"
        "  2/2 tests passed\n  Backend Vulkan0: OK\n"
    )
    r = parse_test_backend_ops(ok_out, 0, op="MUL_MAT")
    assert r.passed and r.total == 2 and r.failed == 0, r

    # real test-backend-ops colors OK/FAIL with ANSI codes; must still classify.
    ansi_ok = "  RMS_NORM(type=f32,ne=[64,5,4,3],v=0): \x1b[1;32mOK\x1b[0m\n  1/1 tests passed\n"
    assert parse_test_backend_ops(ansi_ok, 0).passed, "ANSI-wrapped OK misparsed"

    # real V3D abort: 16 KB matmul limit -> SIGABRT (exit 134), no summary line.
    abort_out = (
        "ggml_vulkan: Found 1 Vulkan devices:\nTesting 2 devices\n\n"
        "ggml_vulkan: Error: Shared memory size too small for matrix multiplication.\n"
        "terminate called after throwing an instance of 'std::runtime_error'\n"
    )
    ra = parse_test_backend_ops(abort_out, 134, op="RMS_NORM")
    assert not ra.passed and any("Shared memory" in f for f in ra.failures), ra

    # the real pre-patch V3D catch: wrong finite number vs CPU
    fail_out = (
        "  MUL_MAT(type_a=q5_1,m=64,n=64,k=64): NMSE = 0.000508874 > 0.000500000 FAIL\n"
        "  1/2 tests passed\n  Backend Vulkan0: FAIL\n"
    )
    r = parse_test_backend_ops(fail_out, 1, op="MUL_MAT")
    assert not r.passed and r.failed == 1 and r.nmse_max == 0.000508874, r

    # no summary line + nonzero exit -> not passed
    assert not parse_test_backend_ops("garbage traceback", 1).passed

    # --- completeness filters ---
    good = rng.standard_normal((8, 16)).astype(np.float32)
    assert check_completeness(good).complete

    # partial compute: bottom half of channels never written (still NaN)
    partial = good.copy()
    partial[4:, :] = np.nan
    c = check_completeness(partial)
    assert not c.complete and any("coverage" in x for x in c.reasons), c

    # constant output (matches loose tolerance without computing)
    assert not check_completeness(np.full((8, 16), 0.5, np.float32)).complete

    # clamped into trivial band
    assert not check_completeness(
        rng.standard_normal((8, 16)).astype(np.float32) * 1e-4
    ).complete

    # a constant axis (unwritten-channel / broadcast signature): axis 0 identical
    # across rows, columns still vary -> axes filter must still reject it.
    flat_axis = np.tile(rng.standard_normal((1, 16)).astype(np.float32), (8, 1))
    c = check_completeness(flat_axis)
    assert not c.complete and any("axes" in x for x in c.reasons), c

    # --- combined verdict ---
    assert verify(parse_test_backend_ops(ok_out, 0), good).verdict == "pass"
    assert verify(parse_test_backend_ops(ok_out, 0), partial).verdict == "revert"
    assert verify(parse_test_backend_ops(fail_out, 1)).verdict == "revert"

    print("v3d_verify: all self-checks passed")


if __name__ == "__main__":
    _demo()
