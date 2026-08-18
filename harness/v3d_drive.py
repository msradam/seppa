"""Drive the V3D optimization FSM with a real Claude agent over MCP.

Uses the Claude Agent SDK authenticated with the local Claude Code SSO session
(no ANTHROPIC_API_KEY). The SDK launches the Theodosia MCP server (theodosia_server.py)
as a stdio subprocess; the agent acts only through the generic `mcp__v3d__step`
tool. Theodosia's guard means a kernel that fails verification is never offered
the `benchmark` action, so the agent cannot bypass correctness.

    .venv/bin/python v3d_drive.py

Replay mode (canned variants) by default, so this runs without the Pi.
"""

from __future__ import annotations

import os
import sys

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, ".venv", "bin", "python")

SYSTEM = """You are an autonomous GPU-kernel optimization agent. You optimize a
Vulkan compute kernel for the Raspberry Pi 5 V3D GPU by driving a finite-state
machine through ONE tool: mcp__v3d__step(action, inputs).

Rules:
- Start by calling step with action="characterize" and inputs={}.
- Every result includes "valid_next_actions". Call step again with ONE of those.
- The optimization loop is: hypothesize -> implement -> compile -> verify ->
  (benchmark) -> evaluate -> log_variant -> (repeat or stop).
- The machine enforces correctness. If a variant fails verification you will NOT
  be offered "benchmark" in valid_next_actions -- that is the safety gate, do not
  fight it; take the offered action (log_variant).
- Keep inputs={} unless you have a specific hypothesis to pass.
- Continue until valid_next_actions is just ["stop"]; call stop, then in ONE short
  paragraph summarize: how many experiments ran, which were kept, and call out any
  variant that verified as INCOMPLETE and was therefore never benchmarked.
Narrate each step in one short line before you call it."""

EXPLORE_SYSTEM = """You are a GPU kernel optimization researcher. You optimize a tiled
SGEMM (single-precision C=A*B) Vulkan/GLSL compute shader for the Raspberry Pi 5 V3D GPU,
exploring the space of shader rewrites to raise GFLOP/s. You act ONLY through
mcp__v3d__step(action, inputs).

Loop: characterize -> baseline -> hypothesize -> implement -> compile_ -> verify ->
benchmark -> evaluate -> log_variant -> (repeat or stop).

Procedure:
1. step("characterize"), then step("baseline").
2. step("hypothesize") returns best_gflops, roofline_gflops (43.7), current_best_shader
   (the FULL GLSL source) and history. Read current_best_shader carefully.
3. Choose ONE focused change likely to raise GFLOP/s. Call
   step("implement", inputs={"shader": "<the ENTIRE modified GLSL source>"}). Pass the
   whole shader text, not a diff.
4. step("compile_"). If compile_ok is false, read compile_err and go back to hypothesize.
5. step("verify") runs it on REAL V3D hardware and checks correctness. If verify_ok is
   false you will NOT be offered benchmark (safety gate) -- take log_variant and try again.
6. If correct: step("benchmark"), step("evaluate") (keeps only if >1% faster),
   step("log_variant"). Then continue from hypothesize, building on what worked.
7. Continue until only "stop" is offered; call it and summarize the best GFLOP/s reached.

HARD CONSTRAINTS (violate => won't compile or won't run):
- local_size_x * local_size_y * local_size_z <= 256 invocations.
- total shared memory <= 16384 bytes (two 32x32 float tiles = 8 KB).
- V3D has NO fp16 arithmetic, NO cooperative matrix, subgroup size 16.
- Output must stay correct: C = A*B, row-major, square, size SZ from the push constant.

Ideas to try (one per experiment): larger register micro-tiles (TM x TN per thread) for
higher arithmetic intensity; vec4 loads for coalesced TMU access; different tile shapes;
avoiding shared-memory bank conflicts; unrolling the k-loop. Baseline ~7 GFLOP/s; ceiling
43.7. Narrate each step in one short line before calling it."""

PROMPT = "Drive the V3D kernel-optimization FSM from characterize through to stop."
EXPLORE_PROMPT = (
    "Explore shader rewrites to maximize GFLOP/s of the V3D SGEMM kernel, "
    "staying correct and within the 16 KB / 256-invocation envelope. "
    "Run experiments until the FSM stops."
)

FLOOD_SYSTEM = """You are a GPU kernel optimization researcher. You optimize a two-shader
flux-limited overland-flow FLOOD simulation (flux.comp + height.comp, GLSL compute) for the
Raspberry Pi 5 V3D GPU, raising GFLOP/s WITHOUT changing the numerical result. You act ONLY
through mcp__v3d__step(action, inputs).

Loop: characterize -> baseline -> hypothesize -> implement -> compile_ -> verify -> benchmark
-> evaluate -> log_variant -> (repeat or stop).

Procedure:
1. step("characterize"), then step("baseline").
2. step("hypothesize") returns best_gflops, current_flux and current_height (the FULL GLSL
   source of both shaders), and history. Read both shaders carefully.
3. Choose ONE focused optimization. Call step("implement", inputs={"flux": "<full flux.comp>",
   "height": "<full height.comp>"}). Pass BOTH shaders' entire source (send the unchanged one
   as-is). Do NOT change the algorithm or the numbers -- only how they are computed.
4. step("compile_"). If compile_ok is false, read compile_err and go back to hypothesize.
5. step("verify") runs it on real V3D and checks THREE things: NMSE vs a fixed CPU reference
   (< 1e-3), mass conservation vs rainfall, and that water pools in the basin. If any fails you
   will NOT be offered benchmark -- your optimization changed the physics; revert and try again.
6. If valid: step("benchmark"), step("evaluate") (keeps only if >1% faster), step("log_variant").
   Continue from hypothesize, building on what worked, until only "stop" is offered.

HARD CONSTRAINTS:
- local_size_x * local_size_y <= 256; total shared memory <= 16384 bytes; V3D has NO fp16.
- The output raster MUST stay numerically identical (the CPU reference is fixed) -- you are
  optimizing the implementation, not the model.

The kernel is currently MEMORY-BOUND (~1.4 GFLOP/s): each cell reads its 4 neighbours from
global memory. The main lever is SHARED-MEMORY TILING: have the 16x16 workgroup cooperatively
load an 18x18 halo tile of (H + water) into shared memory (~2.6 KB, well under 16 KB), then read
neighbours from shared memory instead of global -- cutting global traffic ~5x and raising
arithmetic intensity. Also consider vec4 loads and fusing work. Narrate each step in one line."""

FLOOD_PROMPT = (
    "Optimize the two-shader V3D flood simulation to maximize GFLOP/s while keeping the "
    "result numerically identical, mass-conserving, and physically pooling. Run experiments "
    "until the FSM stops."
)


async def main() -> None:
    live = "--live" in sys.argv
    explore = "--explore" in sys.argv
    flood = "--flood" in sys.argv
    server_args = [os.path.join(HERE, "theodosia_server.py")]
    if flood:
        server_args.append("--flood")
    elif explore:
        server_args.append("--explore")
    elif live:
        server_args.append("--live")
    mode = (
        "FLOOD (agent optimizes flood sim on real Pi)"
        if flood
        else "EXPLORE (agent rewrites gemm.comp on real Pi)"
        if explore
        else ("LIVE (real Pi)" if live else "replay")
    )
    print(f"[driving FSM over MCP, {mode} mode]\n")
    system = FLOOD_SYSTEM if flood else EXPLORE_SYSTEM if explore else SYSTEM
    prompt = FLOOD_PROMPT if flood else EXPLORE_PROMPT if explore else PROMPT
    options = ClaudeAgentOptions(
        system_prompt=system,
        mcp_servers={
            "v3d": {
                "type": "stdio",
                "command": VENV_PY,
                "args": server_args,
            }
        },
        allowed_tools=[
            "mcp__v3d__step",
            "mcp__v3d__list_resources",
            "mcp__v3d__read_resource",
        ],
        permission_mode="bypassPermissions",
        max_turns=200 if (explore or flood) else 50,
        cwd=HERE,
    )

    steps = 0
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    print(f"  {block.text.strip()}")
                elif isinstance(block, ToolUseBlock) and block.name.endswith("step"):
                    steps += 1
                    inp = block.input or {}
                    action = inp.get("action", "?")
                    extra = ""
                    if action == "implement":
                        ins = inp.get("inputs") or {}
                        nbytes = sum(len(v) for v in ins.values() if isinstance(v, str))
                        extra = f"  ({nbytes} bytes of shader)"
                    print(f"→ step[{steps}]: {action}{extra}")
        elif isinstance(msg, ToolResultBlock):
            pass
        elif isinstance(msg, ResultMessage):
            print(f"\n=== done: {steps} step() calls, {getattr(msg, 'num_turns', '?')} turns ===")


if __name__ == "__main__":
    anyio.run(main)
