# V3D GPU compute survey: what else could run here, beyond GEMM

Seppa currently optimizes one kernel (a standalone GEMM shader) and measures
one co-processing scenario (GPU flood sim alongside CPU LLM decode, in
Bonbibi). This is a survey of the actual V3D device limits, verified
directly on the Pi 5, and a ranked list of other GPU-compute work this
project's stack could plausibly use.

## Verified device limits (Pi 5, `vulkaninfo`, V3D 7.1.10.2 device only —
`vulkaninfo` also reports a `llvmpipe` software Vulkan device on this system;
that's a CPU fallback, not real GPU capability, and is excluded below)

| Limit | Value |
|---|---|
| `maxComputeWorkGroupInvocations` | 256 |
| `maxComputeSharedMemorySize` | 16384 bytes (16 KB) |
| subgroup size | fixed at 16 (min = max = 16) |
| `shaderFloat16` | false — no fp16 arithmetic |
| `shaderInt16` / `shaderInt64` | false |
| `VK_KHR_shader_integer_dot_product` | extension present, but llama.cpp's own Vulkan backend probe reports `int dot: 0` on this device — advertised, not usably accelerated |
| cooperative matrix / matrix cores | none |
| GPU clock | ~960 MHz |

This is the same story as the CPU side (A76: no i8mm, no SVE/SME2) — no fast
low-precision integer path on either compute unit. Any optimization idea
that assumes int8 dot-product acceleration or fp16 ALU speed is a dead end
on this specific board, GPU or CPU.

## What's already GPU-accelerated

- **The flood-sim stencil** (`vkflood.cpp` in Bonbibi, `flux.comp`/`height.comp`)
  — real, verified, NMSE-checked, mass-conserving. This is genuine production
  GPU work, not a benchmark.
- **The GEMM demo shader** (`best_gemm.comp`) — a standalone benchmark used to
  prove and measure the FSM optimization loop (7.02 → 12.56 GFLOPS). It is
  not plugged into a real workload; it exists to validate the methodology.

## Ranked candidates for further GPU-compute work

**1. Real `llama.cpp` GPU inference (`MUL_MAT` inside the actual forward
pass) — highest value, currently broken, not done.** This was the original
target this project's planning docs (`integration_plan.md`,
`phase1_harness_fsm.md`) described before the simpler standalone-GEMM MVE was
built instead. Verified directly on the Pi this session: `llama-bench -ngl
99` on the current llama.cpp build **hangs** (killed by a 60s timeout, no
crash, no output past device detection) rather than completing a forward
pass — V3D's 16 KB shared-memory ceiling is smaller than `mul_mat` assumes.
Bonbibi's README previously claimed a working "small-shared-memory matmul
fix" for this; that claim did not hold up under direct testing and has been
corrected there. Getting one real op (`MUL_MAT`, then `RMS_NORM`/`SOFT_MAX`/
`ROPE` for a fully GPU-resident forward pass) working through Seppa's FSM
loop — with the correctness gate this project already has — would mean
actual GPU-accelerated LLM decode, not a side benchmark. That changes the
co-processing story from "GPU does flood physics, CPU does everything else"
to genuine concurrent, mixed GPU/CPU inference. This is the single biggest
unclaimed opportunity found in this survey.

**2. `whisper.cpp` Vulkan encoder for speech-to-text** — not yet deployed to
the Pi at all (checked; no `whisper.cpp` present on-device). A voice
interface is the natural accessibility front-end for Bonbibi (a wheelchair
user or a blind person in a flood can't fill a form), and Whisper's encoder
is convolution + attention layers of a size very different from an
autoregressive LLM decode — worth its own kernel-search pass rather than
assuming the GEMM tuning transfers directly. Real, concrete, unclaimed work.

**3. The remaining `ggml` ops needed for a fully GPU-resident forward pass**
(`RMS_NORM`, `SOFT_MAX`, `ROPE`, `FLASH_ATTN_EXT`) — follows naturally once
(1) is solved; each is small enough to be its own FSM target the way the
GEMM shader already is.

**4. Lower priority / speculative:** additional physics stencils beyond
flood (wildfire spread, structural stress — same flux-limited-stencil shape,
no new methodology needed, but no concrete use case yet); DEM
upsampling/smoothing on GPU (one-time preprocessing cost, not a hot loop, low
value); vector/embedding search acceleration for a future RAG-style
component (nothing in the current stack needs it).

## Why this matters for the paper/hackathon story

Item 1 is the more ambitious, more defensible claim than the standalone-GEMM
MVE this project actually shipped: real GPU-accelerated LLM inference on a
GPU the community has documented as unable to run `mul_mat` at all
([ramalama #2592](https://github.com/containers/ramalama/issues/2592),
[llama.cpp #9801](https://github.com/ggml-org/llama.cpp/issues/9801)). The
correctness-gated FSM this project already built is exactly the tool suited
to closing that gap safely — an ungated attempt at this exact op is what
produced MNN's documented 72% spurious-speedup bug from a partial-output
kernel (see `verification_design.md`). Whoever picks this up next should
read `verification_design.md` and `phase1_harness_fsm.md` first — the
guard design for exactly this op is already worked out there.
