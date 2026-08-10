# Deterministic Verification Design, V3D LLM/ML Kernel Optimization

> **Note on this document:** planning doc. The NMSE-vs-CPU-oracle principle
> and the anti-gaming guard predicates below are exactly what
> `v3d_verify.py` implements. The concrete harness differs from what's
> described here: the implemented oracle is a small dedicated program
> (`pi/vkgemm_nmse.cpp` — random inputs, double-precision CPU reference,
> prints `correct=yes NMSE=...`), not llama.cpp's `test-backend-ops`. The
> design reasoning (why an in-process CPU oracle beats a torch reference,
> why NMSE energy-normalization catches partial-output cheating for free)
> carries over unchanged.

**Scope:** how the FSM decides that an optimized Vulkan compute kernel, or a full LLM forward pass, on the Pi 5 V3D GPU is *numerically valid*, in a way an autotuner or agent cannot game. This is the load-bearing part of the whole project: without it, every speedup is a plausible but numerically wrong result.

All thresholds and mechanisms below are sourced from the engines' own test harnesses (llama.cpp `test-backend-ops`, MNN `backendTest.out`, NCNN `test_layer`) and the reward-hacking literature (robust-kbench arXiv:2509.14279, CUDA-L1 arXiv:2507.14111, Kevin arXiv:2507.11948). See "Sources" at the end.

---

## 0. Why this is the hard part (and the actual contribution)

The public state of the art says LLM matmul does **not** run on V3D: stock llama.cpp Vulkan aborts with `ggml_vulkan: Error: Shared memory size too small for matrix multiplication`, because its `mul_mat` kernels assume >16 KB shared memory, fp16 arithmetic, integer dot-product, and cooperative-matrix, V3D has **none** of these (16 KB shared mem, `shaderFloat16=false`, `int dot: 0`, `matrix cores: none`, subgroup 16). Community consensus: disable Vulkan on the Pi 5 ([ramalama #2592](https://github.com/containers/ramalama/issues/2592), [RPi forums t=390565](https://forums.raspberrypi.com/viewtopic.php?t=390565), [llama.cpp #9801](https://github.com/ggml-org/llama.cpp/issues/9801)).

The `v3d-pi-ai` investigation already got a forward pass running inside that envelope (MNN Qwen2, corrected 4.95 t/s decode ceiling; patched llama.cpp shaders to `local_size 256` / 16 KB). **The novelty is not "we ran an LLM on the GPU"; it is "we ran it inside a hostile compute envelope and proved it was correct."** The prior audit is the cautionary twin: without a correctness gate, MNN's autotuner reported a 72% spurious speedup by crowning a shader that computed 1/K of its output K× faster. Deterministic verification is what separates the real result from that.

---

## 1. The reference oracle: CPU backend of the same engine (never torch)

AutoKernel's oracle is a PyTorch op on CUDA. That does not exist here. The V3D oracle is the **CPU backend of the same inference engine**, computing the same op/graph in a separate address space:

- **llama.cpp:** the `ggml_backend_cpu` scalar/CPU implementation. `test-backend-ops` builds the same op graph on Vulkan and CPU and diffs per output node (`ggml_backend_compare_graph_backend`).
- **MNN:** `forwardType=0` (CPU) is the reference; Vulkan is `forwardType=7`. `backendTest.out` and `testMNNFromOnnx.py` (ONNX Runtime as reference) diff against it.
- **Whole-model:** an FP16 (or FP32) CPU run of the same GGUF is the golden logit source.

Two properties of this oracle matter and are **free on V3D**:
1. **Reference-copy exploit is structurally impossible.** The exploit that broke Sakana and KernelBench (candidate reuses the reference's output buffer) requires the candidate to *read* the reference result. A Vulkan compute shader receives only its declared **input** bindings; it has no binding to the CPU reference tensor, which lives in host memory in a different process. The class of cheat is unrepresentable, not merely guarded against.
2. **Partial-output is caught by the metric, not a heuristic.** NMSE is normalized by the *full* reference energy (`Σ(a−b)²/Σa²`), so a kernel that writes 1/K of the output and leaves the rest at zero/poison produces a large normalized error and fails. This is precisely that failure mode, and NMSE fails it automatically.

---

## 2. The three-rung verification ladder

Run cheap→expensive; the first rung is the per-experiment gate, the upper rungs run on "keep" candidates and at end-to-end.

### Rung 1: op-level, per experiment (the gate)
**`test-backend-ops test -b Vulkan0 -o <OP>`** on the Pi's patched llama.cpp build.
- **Metric:** NMSE vs CPU backend. **Default threshold 1e-7**; F16 relaxed to 1e-6; quantized ops looser (sized to ~one quant step). Reject on any NaN/Inf or `NMSE > max_err`.
- **Coverage:** `MUL_MAT`, `RMS_NORM`, `SOFT_MAX`, `ROPE`, `FLASH_ATTN_EXT`, `IM2COL`/`CONV_2D`, `GET_ROWS`, the operators an LLM forward pass touches.
- **Inputs:** stock harness seeds with `std::random_device` (fresh every run, good for anti-overfit). **Patch it to log the seed** so a failure is reproducible: fresh inputs each iteration, but a recorded seed. This is the one source change to the harness.
- **Concrete prior catch:** pre-patch, `test-backend-ops -b Vulkan0` on V3D produced wrong finite numbers (`Vulkan0=-148.95` vs `CPU=inf`) and timed out at 600 s. That is exactly a Rung-1 FAIL, the gate working.
- **MNN equivalent:** `backendTest.out <model.mnn> 7 0.05 1` (Vulkan vs CPU, absolute tol 0.05). Use for the MNN decode path; note MNN uses absolute tolerance, so scale it to tensor magnitude.

### Rung 2: whole-model distribution, per kept candidate
**`llama-perplexity --kl-divergence`** against golden FP16 logits on a fixed Wikitext-2 slice:
```
llama-perplexity -m model-f16.gguf -f wiki.test.raw --kl-divergence-base logits.kld     # golden, once, on CPU
llama-perplexity -m model-q4_k.gguf -f wiki.test.raw --kl-divergence-base logits.kld --kl-divergence   # V3D run
```
- **Metrics:** mean KL divergence (0 = identical distributions), ΔPerplexity, and **"Same top p"** (fraction of tokens where V3D and reference rank the same token first).
- **Accept:** land near the published per-quant KLD/ΔPPL figures for that quantization; a broken kernel shows a *large* KLD jump, not a marginal one. KLD > PPL as the signal (PPL is "a very rough measurement").

### Rung 3: end-to-end greedy, at integration
Fixed prompt, temperature 0 (argmax), compare the V3D token sequence to the CPU reference. Report **Div_Index** (position of first divergence; −1 = never diverged). Small tolerance-driven divergence is acceptable unless kernels are made batch-invariant; a kernel that diverges at token 1 is broken.

---

## 3. Two orthogonal gates (do not conflate them)

The existing FSM's Stage-4 "same input ×3, bitwise identical" is a *different* check from "matches the reference." Keep both, scoped correctly:

- **Gate A, determinism / structure (device-local).** Same input ×3 → bitwise identical. This asserts the kernel is **order-stable**: atomic-free, fixed hierarchical reduction tree, contraction pinned (`NoContraction`/`precise` or explicit `fma()`), on a **pinned Mesa/V3DV build**. It is legitimate and cheap for the reduction-heavy ML ops here (matmul, rmsnorm, softmax). It is *not* cross-backend equality, and it correctly rejects FP-atomic accumulation. Allow an explicit opt-out label for kernels that are legitimately non-deterministic (atomic histograms), none of the target ops need it.
- **Gate B, numerical correctness (vs reference).** Rung 1 NMSE / tolerance vs the CPU oracle. This is what "valid" means.

A kernel must pass **both** to be kept. Gate A without Gate B passes a deterministic-but-wrong kernel; Gate B without Gate A passes a right-on-average-but-flaky kernel.

**Tolerance by dtype** (Gate B): fp32 `atol≈1e-5 rtol≈1e-5` (robust-kbench default) or NMSE ~1e-7; fp16/quantized `atol≈1e-2` or NMSE ~1e-6 with fp32 accumulate. Prefer relative/NMSE over raw max-abs so the threshold is scale-free. A too-loose tolerance is itself a gaming surface.

---

## 4. Anti-gaming guard predicates → FSM transition

The `verify → benchmark` edge is a **conjunction** of predicates; failure routes to `log_variant` with a `revert`/`build-prereq` verdict and never reaches `benchmark`. Timing is never read until all pass. Predicates, and which are free on V3D:

| # | Guard | On V3D |
|---|---|---|
| 1 | fresh random inputs each iteration (logged seed) | patch `test-backend-ops` seed logging |
| 2 | full-output coverage, every element written (poison-fill buffer, no survivors) | **NMSE energy-normalization enforces this automatically** |
| 3 | tolerance/NMSE vs trusted CPU reference | Rung 1 |
| 4 | output range not clamped to trivial band | robust-kbench Output Range filter |
| 5 | output std > 0.01 overall and per-axis | robust-kbench Std + Axes filters, **catches the prior audit's constant/partial tail** |
| 6 | input-sensitivity: output changes when input changes | robust-kbench Input Impact |
| 7 | multi-seed / multi-init: pass over ≥N seeds | re-run Rung 1 N times |
| 8 | reference-copy impossible | **free, shader binds only inputs** |
| 9 | no-op / dead-compute check | shader must dispatch the full work domain; cross-check against `v3d_workgroup_patch` host constants |
| 10 | determinism (Gate A) | existing Stage-4 ×3 |
| 11 | timing gated on correctness | `benchmark` unreachable unless 1-10 hold |

Guards specific to the CUDA reward-hacks (async-stream evasion, `data_ptr()` caching, lazy CUDA tensors) are **not applicable**: there are no CUDA streams and no torch lazy tensors in a Vulkan-shader-over-SSH path. That is a portability *win*: the V3D substrate removes three of the five documented exploit classes by construction.

---

## 5. The prior audit's candidate, re-run through this gate

The exact shader that caused the audit failure (`convolution1x1.comp` with `gws.y = UP_DIV(ocDiv4, K)` computing 1/K of the channels) fails **three independent guards**:
1. **NMSE (Guard 3):** missing channels read as zero/poison → normalized error ≫ threshold → FAIL.
2. **Per-axis std (Guard 5):** the unwritten output-channel axis has ~zero variation → FAIL.
3. **Full-output coverage (Guard 2):** poison survivors in the tail → FAIL.

Any one of these blocks the `verify → benchmark` transition. The autotuner's fast time is never recorded. The spurious 72% gain becomes structurally unreachable. That is the paper's core demonstrable claim.

---

## 6. MVE for verification specifically

Smallest thing that proves the gate works: on the Pi, run `test-backend-ops -b Vulkan0 -o MUL_MAT` (or `RMS_NORM`) through the FSM's `verify` action against a deliberately-broken shader variant (partial-channel write) and a correct one. Success = the ledger shows the broken variant **refused at `verify`** (NMSE FAIL, benchmark never entered) and the correct variant passing to `benchmark`. This reproduces the pre-patch `Vulkan0=-148.95 vs CPU=inf` catch, now inside an enforced FSM guard.

---

## Sources

- llama.cpp `test-backend-ops` (NMSE 1e-7, CPU oracle, random_device seed): [test-backend-ops.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp), [backend-validation discussion #6345](https://github.com/ggml-org/llama.cpp/discussions/6345), [mul_mat NMSE 5e-4 case #11972](https://github.com/ggml-org/llama.cpp/issues/11972)
- llama.cpp perplexity/KLD workflow: [tools/perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md), [KLD vs PPL #4110](https://github.com/ggml-org/llama.cpp/discussions/4110)
- MNN test tools (Vulkan=7, abs tol, ONNX reference, binary-search op locator): [MNN test docs](https://mnn-docs.readthedocs.io/en/latest/tools/test.html), [testMNNFromOnnx 1e-4](https://leeroopedia.com/index.php/Implementation:Alibaba_MNN_TestMNNFromOnnx)
- NCNN op harness (CompareMat epsilon 0.001, seeded): [tests/testutil.h](https://github.com/Tencent/ncnn/blob/master/tests/testutil.h)
- Reward-hacking defenses: [robust-kbench arXiv:2509.14279](https://arxiv.org/abs/2509.14279) + [repo](https://github.com/SakanaAI/robust-kbench), [CUDA-L1 arXiv:2507.14111](https://arxiv.org/html/2507.14111v8), [Kevin arXiv:2507.11948](https://arxiv.org/html/2507.11948v1), [buffer-recycling / Hardening Agent Benchmarks arXiv:2606.08960](https://arxiv.org/html/2606.08960v1)
- FP determinism (non-associativity, tree reductions, NoContraction): [arXiv:2408.05148](https://arxiv.org/html/2408.05148v3), [NVIDIA CCCL determinism](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/), [Khronos GLSL.std.450](https://registry.khronos.org/SPIR-V/specs/1.0/GLSL.std.450.pdf)
- V3D reality (16 KB shared mem abort, no fp16/int-dot/coop-matrix, Vulkan 1.3 conformant): [ramalama #2592](https://github.com/containers/ramalama/issues/2592), [RPi forums t=390565](https://forums.raspberrypi.com/viewtopic.php?t=390565), [llama.cpp #9801](https://github.com/ggml-org/llama.cpp/issues/9801), [Mesa V3DV Vulkan 1.3](https://www.phoronix.com/news/V3DV-Vulkan-Conformance), [fp16 accum NaN #18969](https://github.com/ggml-org/llama.cpp/issues/18969)
