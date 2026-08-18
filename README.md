# Seppa

Correctness-gated, LLM-driven GPU kernel optimization for the Raspberry
Pi 5's integrated GPU (Broadcom VideoCore VII / V3D, Vulkan compute).

Seppa is [AutoKernel](https://github.com/RightNow-AI/autokernel) (MIT), an
existing prompt-driven kernel-optimization loop, ported onto an explicit
finite-state machine ([Burr](https://github.com/apache/burr)) and served
to an LLM agent over MCP (Theodosia). The model proposes kernel and
host-contract changes; the machine owns compilation, verification against
physics oracles, benchmarking, and the keep-or-revert verdict, and it
refuses out-of-order requests, so an unverified kernel can never reach a
benchmark. The name is the Finnish word for smith (seppä, the epithet of
the Kalevala smith Ilmarinen) and a backronym: State-machine-Enforced
Parallel-Program Accelerator.

The point of the optimization is co-processing: the tuned kernels run
fast enough that the otherwise-idle GPU becomes a useful second engine
beside the CPU. The sample application is
[Bonbibi](https://github.com/msradam/bonbibi), which runs a flood
simulation on the GPU while the CPU runs routing and a language model.

## Results

All measured on a Pi 5 (V3D 7.1.10.2, Mesa v3dv 25.0.7), every number
gated on correctness against a double-precision CPU reference. Raw logs
and transcripts for each claim are in `docs/paper/artifacts/`; notebook
02 recomputes the Section VII tables from them, and notebook 01 replays
the Section VI transcripts.

- **Flood stencil (the paper's case study):** 1.58x at 256x256
  (2,127.7 vs 1,348.6 steps/s) via a fused, strip-mined kernel, verified by
  three physics gates (NMSE, mass conservation, basin pooling). The
  machine re-derived the result over MCP five times out of five
  (`passk_flood2.py`), refusing each time to benchmark a deliberately
  mass-violating kernel.
- **Concurrency (steady-state, thermally soaked):** with CPU LLM decode
  saturating the cores, the optimized kernel sustains 883 steps/s while
  decode keeps 84% of its solo rate; the kernel's advantage grows from
  1.58x alone to 2.20x under contention, and the split beats the best
  CPU-only scheme on both axes. Against generic CPU loads the GPU keeps
  99% (compute-bound) to 92% (memory-bound); a weight-streaming LLM is
  the harshest measured case.
- **Other targets, same harness:** a GEMM kernel from 7.02 to 13.42
  GFLOP/s over two FSM rounds; llama.cpp's matrix-vector kernel
  de-unrolled for +28% end-to-end decode.

The paper (`docs/paper/paper.pdf`, built from `paper.md` by `build.py`)
documents the method, the falsification sweep, and the measurement
campaigns. `docs/autokernel_fidelity.md` audits the port against
upstream AutoKernel rule by rule.

## Quick start

Explore the results without any hardware:

```
uv venv && uv sync
jupyter lab notebooks/
```

- `notebooks/01_reproduction_and_repeatability.ipynb` replays the
  machine-checked reproduction, the gate refusal, the pass^5 campaign,
  and the agent-driven session from the committed transcripts.
- `notebooks/02_concurrency_envelope.ipynb` recomputes the concurrency
  tables and the generic-load results from the raw benchmark logs.

## Running against a Pi

On the Pi (needs `glslangValidator`, a Vulkan-enabled Mesa, and the
harness binaries; build commands in `pi/flood/README.md`):

```
.venv/bin/python theodosia_server.py --http --flood2
```

From any machine on the network:

```
.venv/bin/python drive_flood2_mcp.py http://<pi>:8000/mcp  # scripted reproduction
python3 passk_flood2.py 5 http://<pi>:8000/mcp             # scored repeatability campaign
./claude_driver.sh http://<pi>:8000/mcp                    # an LLM drives the loop
```

`claude_driver.sh` uses the signed-in Claude Code session (model,
reasoning effort, and budget are recorded per run under
`docs/paper/artifacts/claude_sessions/`). Server modes `--explore`
(GEMM) and `--llama-mmv` (llama.cpp matrix-vector) target the other
kernels.

## How it works

The loop is a Burr application; each state is an `@action`, and
transitions carry mutually exclusive guards, so the agent, which selects
among reachable transitions over MCP, cannot bypass the gate:

```
characterize -> baseline -> hypothesize -> implement -> compile -> verify
                                 ^                                    |
                                 |                          verify_ok / not verify_ok
                                 |                                    |
                 log_variant <- evaluate <- benchmark <--------------+
```

- `verify` runs the candidate on the real GPU under the target's oracle
  (for the flood target: NMSE vs a double CPU reference, mass
  conservation, basin pooling). The only transition into `benchmark` is
  guarded by `verify_ok`.
- `evaluate` keeps a variant only if it beats the best by more than 1%;
  the winner is persisted, and the ledger (`variant_log`) records every
  experiment either way.
- A run ends after three consecutive non-improvements on gate-passing
  variants, or at the experiment budget.

## Layout

- `v3d_flood2_opt.py` — the paper's FSM target: flood stencil with the
  host contract (fusion, strip) in the action space
- `theodosia_server.py` — mounts an FSM target as an MCP server
- `drive_flood2_mcp.py`, `passk_flood2.py`, `replay_flood2.py` —
  scripted reproduction, scored repeatability, in-process replay
- `claude_driver.sh` + `claude_driver_prompt.md` — an LLM as the proposer
- `v3d_explore.py`, `v3d_llama_mmv.py`, `v3d_flood_opt.py`, `v3d_fsm.py`
  — the other FSM targets and the original single-kernel loop
- `v3d_verify.py` — the GEMM-path oracle (NMSE parse + completeness checks)
- `pi/` — on-device evaluators, kernels, benchmark scripts
  (`pi/flood/` is the flood kit)
- `notebooks/` — executed notebooks deriving the paper's numbers
- `docs/paper/` — paper source, build pipeline, artifacts, archived
  references
- `docs/notes/NOTES.md` — running log, one entry per confirmed win and
  dead end

## Attribution

The optimization-loop concept (edit a kernel, benchmark against a
reference, keep or revert, log the variant) is from AutoKernel by
RightNow AI, MIT licensed. This repository reimplements that loop as a
state machine with an enforced correctness gate and a Vulkan/V3D
backend.

## License

MIT. See `LICENSE`.
