# AgentVoodoo

AgentVoodoo is [AutoKernel](https://github.com/RightNow-AI/autokernel) (MIT), an
existing GPU kernel-optimization harness, ported to a state-machine framework for
agentic AI kernel optimization. Its edit -> benchmark -> keep/revert loop is
reimplemented as an explicit finite state machine (Burr), served to an LLM agent
over MCP (Theodosia), and driven by the Claude Agent SDK, with a correctness gate
the agent cannot bypass. It targets the Raspberry Pi 5 integrated GPU (Broadcom
VideoCore VII / V3D, Vulkan compute); the loop itself is backend-agnostic.

The purpose is enablement: kernels optimized this way are fast enough that the
otherwise-idle GPU becomes a useful co-processor alongside the CPU. The downstream
application is [Bonbibi](https://github.com/msradam/bonbibi), a concurrent GPU +
CPU workload where the GPU runs a flood simulation while the CPU runs routing and
a language model.

The state machine is the mechanism for the correctness gate: a variant cannot be
benchmarked or kept unless it first passes an NMSE check against a CPU
reference. The benchmark state is unreachable from an incorrect kernel, so the
agent cannot win by producing a fast wrong kernel.

## How it works

The loop is a Burr application. Each state is an `@action`; transitions carry
mutually exclusive guard conditions so the agent, which selects among reachable
transitions over MCP, cannot bypass the gate:

```
characterize -> baseline -> hypothesize -> implement -> compile -> verify
                                 ^                                    |
                                 |                          verify_ok / not verify_ok
                                 |                                    |
                 log_variant <- evaluate <- benchmark <--------------+
```

- `verify` runs the candidate on the real GPU and parses an NMSE-vs-CPU result.
- The only transition into `benchmark` is guarded by `verify_ok`; a failing
  variant is routed to `log_variant` and reverted.
- `evaluate` keeps a variant only if it is both correct and faster than the
  current best. The best kernel is persisted.

`v3d_verify.py` is the oracle: it strips ANSI, classifies the eval tail (only a
bare `OK` passes), and rejects incomplete outputs (NaN poisoning, zero
variance, degenerate ranges) so a silent failure cannot read as a pass.

## Layout

- `v3d_verify.py` — correctness oracle (NMSE parse + completeness checks).
- `v3d_fsm.py` — single-kernel optimization FSM.
- `v3d_explore.py` — agentic exploration FSM (agent rewrites the shader each round).
- `v3d_flood_opt.py` — the same loop applied to a second kernel (a flood stencil),
  with a physics gate (NMSE + mass conservation + pooling).
- `v3d_coproc.py` — thermal-guarded CPU||GPU co-processing measurement FSM.
- `v3d_live.py` — live-Pi run helpers.
- `theodosia_server.py` — mounts an FSM as an MCP server (`--explore`, `--flood`).
- `v3d_drive.py` — drives the FSM over MCP with the Claude Agent SDK.
- `best_gemm.comp` — best SGEMM shader found by the loop.
- `pi/` — the on-device evaluator: `vkgemm_nmse.cpp` (random inputs, double CPU
  reference, prints `correct=yes NMSE=...`) and `gemm.comp` (baseline shader).
- `docs/` — design notes.

## Install

```
uv venv && uv pip install -e .
```

The agent authenticates with the local Claude session; no API key is required.
The evaluator builds on the Pi with `g++ -O3 vkgemm_nmse.cpp -lvulkan`.

## Usage

Point the helpers at your Pi (`PI_HOST` in the `v3d_*` modules), then:

```
python v3d_drive.py --explore   # agent optimizes gemm.comp on the Pi
python v3d_drive.py --flood     # agent optimizes the flood stencil under the physics gate
```

## Attribution

The optimization-loop concept (edit a kernel, benchmark against a reference,
keep or revert, log the variant) is from AutoKernel by RightNow AI, MIT
licensed. This repository reimplements that loop as a state machine with an
enforced correctness gate and a Vulkan/V3D backend. See `LICENSE`.

## License

MIT. See `LICENSE`.
