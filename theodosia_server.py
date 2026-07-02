"""Mount the Phase-1 V3D FSM as an MCP server (Theodosia).

Run ON the Pi:
    python theodosia_server.py                 # stdio (for a local MCP client)
    python theodosia_server.py --http          # streamable-http on 0.0.0.0:8000

The laptop agent connects an MCP client to http://<pi>:8000/mcp and drives the
loop via the single generic step(action, inputs) tool. Theodosia constrains the
action to the graph's enum and refuses out-of-order calls, so the agent cannot
skip VERIFY or reach BENCHMARK on a kernel that failed correctness.

Default is replay mode (canned variants) so the server is pokeable without the
Pi toolchain. Live mode is wired by passing a run_fn into build_app that shells
out to test-backend-ops / MNN; see v3d_live.py (to come).
"""

from __future__ import annotations

import argparse
import os
import sys

# When the Agent SDK launches this as a stdio MCP server, cwd may differ; make
# the sibling modules (v3d_fsm, v3d_verify) importable regardless.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from theodosia import mount

from v3d_fsm import build_app


def make_server(
    name: str = "v3d", live: bool = False, explore: bool = False, flood: bool = False
):
    # Factory (not a built app) so each MCP session gets isolated state and
    # fork/reset stay enabled.
    if flood:
        # Agent optimizes the two-shader flood sim under the physics gate.
        import v3d_flood_opt

        return mount(v3d_flood_opt.build_flood_app, name=name)
    if explore:
        # Real AutoKernel-style loop: the agent rewrites gemm.comp on the Pi.
        import v3d_explore

        return mount(v3d_explore.build_explore_app, name=name)
    if live:
        import v3d_live

        def factory():
            return build_app(
                run_fn=v3d_live.make_live_run_fn(),
                variants=v3d_live.LIVE_VARIANTS,
                baseline_ts=0.01,
            )

        return mount(factory, name=name)
    return mount(build_app, name=name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Theodosia MCP server for the V3D FSM")
    ap.add_argument(
        "--http", action="store_true", help="serve streamable-http instead of stdio"
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--live", action="store_true", help="drive the real Pi over SSH")
    ap.add_argument(
        "--explore", action="store_true", help="agent rewrites gemm.comp on the Pi"
    )
    ap.add_argument(
        "--flood", action="store_true", help="agent optimizes the flood sim (2 shaders)"
    )
    args = ap.parse_args()

    server = make_server(live=args.live, explore=args.explore, flood=args.flood)
    if args.http:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
