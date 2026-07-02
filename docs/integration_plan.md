# AutoKernel → V3D Integration Plan

**Fork:** `RightNow-AI/autokernel` → `msradam/autokernel` (proposed rename: `not-quite-voodoo`)
**Target:** Raspberry Pi 5, Broadcom VideoCore VII (V3D 7.1.10.2), Vulkan compute
**Driver:** Burr FSM exposed as an MCP server by Theodosia
**Status:** planning only. No code changed, nothing pushed.

---

## Executive summary

AutoKernel is not an agent framework: it is a set of stateless Python scripts plus a 909-line `program.md` prompt, and the "agent" is a human-launched Claude/Codex that edits `kernel.py`, runs `bench.py`, and does keep/revert with plain `git`. The V3D port keeps AutoKernel's *shape* (profile → extract → edit → compile → verify → benchmark → keep/revert → log, one artifact at a time) but replaces its entire torch/CUDA execution substrate, because every correctness check in `bench.py` is `torch.allclose(kernel_fn(**inputs), pytorch_reference)` on a CUDA tensor and none of that exists on V3D. The good news: the 10-state Burr FSM this task imagines **already exists** at `~/v3d-pi-ai/burr_fsm/fsm.py`, already runs on the Pi via `subprocess`, and already encodes the Session 10 correctness lesson in its `verify`/`evaluate` gates — so the real work is (a) porting AutoKernel's file-loop discipline and playbook into that FSM's `hypothesize`/`implement` actions, and (b) wrapping the FSM with Theodosia so a laptop agent drives it over MCP.

---

## 1. Fork changes — what changes, what stays

AutoKernel splits cleanly into three buckets. The dividing line is **"does it touch torch/CUDA?"**

**Replace entirely (torch/CUDA-coupled, no V3D analog):**
| File | Why it dies |
|---|---|
| `bench.py` | Whole harness is `kernel_fn(**inputs) -> torch.Tensor` compared to a PyTorch reference via `torch.allclose`; timing is `torch.cuda.Event` / `triton.testing.do_bench`; GPU DB (`_KNOWN_GPUS`) is NVIDIA/AMD only. Nothing survives. Becomes `bench.sh` (SSH → Vulkan tools). |
| `reference.py` | PyTorch op oracles. On V3D the oracle is the CPU backend of the same engine (llama.cpp CPU, MNN CPU), not torch. |
| `kernels/` (9 Triton) and `kernels/cuda/` (9 CUDA C++) | Starter kernels in the wrong languages. Replaced by GLSL `.comp` shaders extracted from MNN/llama.cpp. |
| `prepare.py` | Requires `torch.cuda.is_available()`, `nvidia-smi`, Triton; caches torch tensors. Rewritten as Pi provisioning (swap, `vc4-kms-v3d-pi5`, MNN/llama.cpp build check, `v3d_caps` probe). |
| `export_hf.py`, `analysis.py`, `models/`, `kernelbench/` | HF-Kernels export, torch-profiler plots, torch model defs, KernelBench's `ModelNew`-vs-torch harness. Out of scope for the MVE; drop or park. |

**Adapt (concept survives, implementation rewritten):**
| File | Adaptation |
|---|---|
| `profile.py` | Same job (rank bottlenecks, classify compute/memory-bound, write `profile_report.json`) but source is `llm_demo`/`llama-bench`/`benchncnn`/`test-backend-ops` output over SSH, not `torch.profiler`. Roofline classification stays static-lookup style (AutoKernel's is already just an op-type table). |
| `extract.py` | Same job (emit a standalone, self-describing target file for the loop) but pulls a `.comp` shader out of MNN's `AllShader.cpp`/llama.cpp shader tree and writes `kernel.comp` + a metadata sidecar instead of a Triton/CUDA `kernel.py`. |
| `orchestrate.py` | Reusable **almost verbatim** — it is pure Python bookkeeping (JSON state, Amdahl `1/(1-Σfᵢ(1-1/sᵢ))`, move-on criteria, TSV append) with **zero torch and zero git calls**. Retarget its metric from `tflops` to `tok/s` (or `bw_util`) and its move-on thresholds to V3D numbers. |
| `verify.py` | Same job (plug optimized artifact back into the full model, check end-to-end correctness + speedup) but "plug back" means rebuild MNN/llama.cpp with the patched shader and run `test-backend-ops -b Vulkan0` + a full-model decode, not torch monkeypatching. |
| `program.md` | Rewritten as the V3D playbook (Q5). Structure (phases, decision framework, anti-patterns, hard constraints) is reused; every tier is replaced. |

**Keep as-is (generic, infra-free):**
- `orchestrate.py`'s scheduling/Amdahl/TSV core (retarget constants only).
- The **git-per-experiment keep/revert protocol** from `program.md` (see Q8). Portable verbatim.
- The `results.tsv` format and the profile→extract→loop→verify pipeline skeleton.
- `.gitignore` discipline (results.tsv / run.log untracked).

---

## 2. Kernel representation

**`kernel.comp` — a Vulkan GLSL compute shader.** This is the direct analog of `kernel.py`: one file, one artifact, the only thing the agent edits per experiment.

- **Edit pattern:** identical to AutoKernel. The agent makes one focused change (workgroup size, tile, load width), commits, compiles, verifies, benchmarks, keeps or reverts. Same discipline, same `git` mechanics.
- **Compile:** `glslangValidator -V -Os kernel.comp -o kernel.spv` (V3D toolchain uses `glslangValidator`, confirmed in Session 10 logs; `glslc` is the equivalent and also fine). The critical trap is downstream: MNN and llama.cpp link **pre-compiled SPIR-V byte arrays** (`AllShader.cpp`, `mul_mm.comp.cpp`) and `cmake --build` has **no dependency rule from `.comp` to those arrays**. This is the entire root cause of the Session 10 audit failure. Therefore the COMPILE state must run the *full* offline chain — `glslangValidator` → `makeshader.py`/`spirv-as` repack → relink — and must fail loudly if the repack is skipped. A stale-bytecode guard (hash the `.spv`, assert it changed) belongs here.
- **Contract analog:** AutoKernel's `kernel.py` exports `KERNEL_TYPE` + `kernel_fn`. `kernel.comp` carries an equivalent metadata sidecar (`kernel.meta.json`: `shader_type`, `entry_shader_path_in_engine`, `local_size`, `workgroup_denoms`, host-side patch coordinates) so `extract.py`/`bench.sh` know where the shader plugs into MNN/llama.cpp and which host constants co-vary (the workgroup/stride pair from `v3d_workgroup_patch.py`).

The host-side coupling is the one place V3D is *harder* than CUDA: a shader change (`local_size_x`) usually requires a matching host change (`wg_denoms`, strides). `kernel.comp` alone is insufficient; the unit of change is `(kernel.comp, host-patch)`. `v3d_workgroup_patch.py` already does exactly this coordinated rewrite and is the reference implementation.

---

## 3. Benchmark harness replacement

`bench.py` (torch profiler + CUDA events) → **`bench.sh`**, an SSH-driven wrapper around the Vulkan toolchain already built on the Pi. Mapping AutoKernel's single `bench.py` onto the four V3D tools:

| Purpose | AutoKernel (`bench.py`) | V3D replacement | Command (on Pi) |
|---|---|---|---|
| **Correctness** | 5-stage `torch.allclose` vs reference | `test-backend-ops -b Vulkan0` (op reference-checked against CPU backend) | `test-backend-ops -o MUL_MAT -b Vulkan0` |
| **LLM throughput** | `throughput_tflops` | MNN decode t/s | `LD_PRELOAD=…/libMNN_Vulkan.so llm_demo config.json prompt.txt` → regex `decode speed = ([\d.]+) tok/s` |
| **CNN throughput** | (n/a) | NCNN | `benchncnn 8 <threads> 0 <gpu> 0` |
| **Bandwidth** | roofline `pct_peak_bandwidth` | `bw_stress` triad | `./bw_stress <pad> 6 1` → `ACHIEVED_GBPS` |

- **Primary loop metric:** MNN decode **tok/s** (the investigation's north-star; CPU baseline 37.6, true V3D ceiling 4.95). `bw_util` = achieved GB/s ÷ 17 GB/s LPDDR4X peak is the roofline analog of `pct_peak_bandwidth` (V3D LLM decode is memory-bound, so bandwidth utilization is the honest ceiling signal).
- **Timing without `torch.profiler`:** the engines self-report. MNN prints `decode speed`; `llama-bench` reports t/s with `-r 2` repeats; take warm median over ≥3 reps (the audit's corrected 4.95 t/s is a "warm median, 3 reps" number — match that protocol). No profiler needed; the harness parses stdout, exactly as AutoKernel's agent greps `run.log`.
- **`bench.sh` returns** the same greppable key/value lines AutoKernel emits (`correctness: PASS`, `throughput: <t/s>`, `bw_util: <f>`, `bottleneck: memory_bound`) so the FSM's `benchmark`/`evaluate` actions parse identically to the AutoKernel agent loop.

---

## 4. Correctness harness — the five stages, and the Session 10 lesson

AutoKernel's 5 stages, mapped to V3D. The mapping is favorable because `test-backend-ops` already does reference-vs-Vulkan comparison internally.

| AutoKernel stage | What it did | V3D equivalent |
|---|---|---|
| 1. Smoke | tiny input, tight tol vs torch | `test-backend-ops -o <OP> -b Vulkan0` single op (e.g. `RMS_NORM`, 21-test sanity) |
| 2. Shape sweep | 10+ sizes × 3 dtypes | `test-backend-ops` runs its own multi-shape matrix per op; add a couple of model-native shapes |
| 3. Numerical stability | adversarial inputs (near-max, near-zero, mixed) | `test-backend-ops` tolerance checks catch the finite-but-wrong case (pre-patch it caught `Vulkan0=-148.95 CPU=inf`) |
| 4. Determinism | same input ×3, bitwise identical | run the op twice via `test-backend-ops`/decode twice, assert identical — cheap, keep it |
| 5. Edge cases | non-power-of-2 sizes | V3D's real edge is `local_size` not dividing the work domain (the exact Session 10 failure mode: partial output, undefined tail). Add an assertion that **all** output elements are written (compare full output image, not a sampled tile) |

**The Session 10 finding IS the harness design constraint.** MNN's autotuner crowned a shader candidate that did `1/K` of the work and ran ~K× faster, because *the autotuner times candidates with no reference-output check*. AutoKernel's whole architecture — "correctness gate before you ever look at the number, revert instantly on FAIL" — is the antidote, and it is exactly what the investigation lacked. So the V3D VERIFY state must:

1. Run `test-backend-ops -b Vulkan0` (reference-checked) **before** any timing is trusted.
2. Assert the output is **complete** (every channel/element written), not just close on a sampled region — this is what would have caught the `gws.y = UP_DIV(ocDiv4, K)` partial-compute.
3. Treat the autotuner's chosen time as *untrusted* until (1) and (2) pass. A fast number from an unverified candidate is a FAIL, not a win.

This is the single most important thing to get right (see final section).

---

## 5. Optimization playbook (`program.md` → V3D)

Reuse `program.md`'s structure (tiered, gains-first, with anti-patterns and hard constraints) and replace all six tiers with V3D-grounded ones. Every number below is sourced from the investigation's `v3d_caps.c` / `replay_data.py` / `shader_optimization_log.md`.

| Tier | V3D lever | Hard limit / candidates |
|---|---|---|
| **1. Workgroup size** | `local_size_x` sweep | ≤ **256** `maxComputeWorkGroupInvocations`; must be multiple of **16** `subgroupSize`; non-pow2 candidates **48,80,112,144,176,208,240** |
| **2. Shared-memory tiling** | `shared` tile sizing | ≤ **16 KB** `maxComputeSharedMemorySize`; IQ1/IQ2 quants overflow it, Q4_0 fits |
| **3. Load width** | `uvec4` loads for TMU coalescing | 16-byte (`uvec4` = 4×i32) is V3D TMU's native coalesced width |
| **4. Row/channel packing** | output channels per invocation | register-bound: threads=4 needs ≤32 regs, ~64-reg envelope; **this is the tier the Session 10 bug lived in** — packing must write *all* channels |
| **5. Autotune params** | LWS candidate set + mode | WIDE = 4×subgroup = 64 budget; **HEAVY = 256** budget; lower the WIDE LWS-skip floor (≤16 → ≤4) |
| **6. fp16 storage** | `R16G16B16A16_SFLOAT` image2D | storage only — **no fp16 arithmetic** (`shaderFloat16=false`); driver widens fp16→fp32 vec4 in ALU. Also: no `cooperative_matrix`, no dp4a |

Anti-patterns section (analog to AutoKernel's): channel packing without full-output verification (Session 10), assuming fp16 ALU speedup (there is none), random-gather access patterns (36× bandwidth collapse: 4.0 → 0.11 GB/s), workgroups > 256, shared mem > 16 KB, forgetting the host-side `wg_denoms` co-change.

---

## 6. Burr FSM design — mostly already built

The task's hypothesized states **are the existing FSM** at `~/v3d-pi-ai/burr_fsm/fsm.py`. Do not rebuild it; adapt it.

Existing states (10): `CHARACTERIZE → BASELINE → HYPOTHESIZE → IMPLEMENT → COMPILE → VERIFY → BENCHMARK → EVALUATE → LOG → STOP`.

**AutoKernel concept → Burr state:**
| AutoKernel | Burr state |
|---|---|
| `profile.py` (rank bottlenecks) | `CHARACTERIZE` (runs `v3d_caps` probe) |
| baseline `bench.py` run | `BASELINE` |
| "Hypothesize" step of the loop | `HYPOTHESIZE` (picks next `search_space` point) |
| edit `kernel.py` | `IMPLEMENT` (edit `kernel.comp` + host patch) |
| *(implicit in torch/Triton JIT)* | `COMPILE` — **new state, V3D-specific**: the offline SPIR-V repack that Session 10 proved you cannot skip |
| 5-stage correctness | `VERIFY` (`test-backend-ops`) |
| perf section of `bench.py` | `BENCHMARK` (decode t/s) |
| keep/revert decision | `EVALUATE` |
| `results.tsv` append + git commit | `LOG` (`log_variant`) |
| orchestrator `DONE` | `STOP` |

**New states beyond AutoKernel's implicit loop:** `COMPILE` (the offline-shader-repack gate) and an explicit `CHARACTERIZE` (hardware probe — CUDA code assumes it knows the GPU; V3D must probe caps first).

**Existing guards (keep) and the audit-derived guard (add):**
- `compile_ → hypothesize` on `compile_oom` (re-loop, don't crash) — keep.
- `verify → log_variant` on `not verify_ok` (correctness fail **bypasses** benchmark/evaluate — a wrong kernel never gets timed) — keep; this is the Session 10 guard.
- **Add:** a `verify_complete` guard (all outputs written, not just close) so a partial-compute candidate cannot reach `BENCHMARK`. This is the one guard the original investigation lacked and the direct fix for the spurious-gain bug.
- Stopping conditions (keep, retarget): `best_bw_util ≥ 0.40`, `consecutive_no_gain ≥ 3`, `search_exhausted`, `best_ts ≥ cpu_baseline_ts`, `elapsed ≥ budget` (240 min). Correctness-fails do **not** count toward the no-gain counter (they are patch-chain prerequisites).

Burr `State` already carries the right keys (`variant_log`, `best_ts`, `best_bw_util`, `consecutive_no_gain`, `verify_ok`, `compile_oom`, `hardware_constraints`, …). Add `verify_complete` and the `kernel.comp` git SHA per variant.

---

## 7. Theodosia integration

Theodosia mounts a Burr `Application` as a FastMCP server via `theodosia.mount(factory, name=...)`, exposing **one generic `step(action, inputs)` tool** (not one tool per transition) plus `reset_session`/`fork_at`/`fork_from_past`. The agent's `action` argument is constrained to a JSON-Schema `enum` of the graph's action names, so it cannot invoke an out-of-order or hallucinated state.

- **What the agent sees:** every `step` response returns `valid_next_actions` and, on refusal, `next_action_schemas`. The agent also reads `theodosia://graph` (topology + conditions), `theodosia://state`, `theodosia://next`, `theodosia://history`, `theodosia://trace`. A guard failure is a **structured refusal payload** (`{"error":"invalid_transition","valid_next_actions":[…]}`), not a crash — the agent recovers by picking a legal action.
- **Where it runs — decisive design choice:** Theodosia is confirmed to run on aarch64 / Python 3.13 (all runtime deps pure-Python or have prebuilt aarch64 wheels; `psutil` is dev-only). The existing FSM *already runs on the Pi via local `subprocess`*. So **run the Theodosia MCP server on the Pi** (`theodosia serve fsm:build_app --transport streamable-http --host 0.0.0.0 --port 8000`) and point the laptop agent's MCP client at `http://<pi>:8000/mcp`. The FSM keeps calling local binaries with zero SSH inside the FSM code — this is the laziest correct topology: no SSH marshaling in the hot loop, the Pi owns its own tools.
- **SSH's remaining role:** provisioning and `bench.sh`-from-laptop fallback only. In the Theodosia-on-Pi topology, the FSM's `benchmark`/`compile` actions run `subprocess.run([...])` locally (already true in `benchmark.py`, `probe.py`); SSH is used by humans for setup (`scp`/`ssh root@pi.local`, the Pi from the investigation notes).
- **Use factory mode** (`mount(build_app, ...)`), not a pre-built app, so each MCP session gets isolated state and fork/reset stay enabled (a shared built-app disables them and warns).

Net: the laptop agent replaces AutoKernel's human-reading-`program.md`. Instead of "edit file, run bash, git revert" typed by hand, it calls `step("implement", {...})`, `step("compile", {})`, `step("verify", {})` — and Theodosia's enum + guards enforce the loop discipline that AutoKernel could only *ask* the agent to follow in prose.

---

## 8. Git audit trail — keep both, they cover different layers

AutoKernel's keep/revert is git on `kernel.py` (`git commit` before each run, `git reset --hard HEAD~1` on FAIL/regression), done by the agent, not by any script. It is elegant and portable. Theodosia adds a hash-chained ledger (`ledger.jsonl`, `prev`+`sha256`, tamper-evident) plus Burr's SQLite/JSONL tracker.

**Keep both, at different granularities:**
- **git** = the artifact layer. Commit `(kernel.comp, host-patch)` per experiment; `git reset --hard HEAD~1` is the revert. This is the *source of truth for what code produced a number* and survives independent of any framework. Keep it verbatim from AutoKernel.
- **Theodosia ledger + Burr tracker** = the decision layer. Every `step` (including refusals — a rejected `benchmark` on an unverified candidate) is recorded immutably. This is what a paper needs: proof that no timed result ever bypassed the correctness gate. The ledger records *transitions*; git records *diffs*.

They are complementary, not redundant: git can't prove "we never benchmarked an unverified kernel" (that's a control-flow claim); the ledger can't reproduce the shader. `LOG`/`log_variant` writes the git SHA into the Burr `variant_log` entry, stitching the two.

---

## 9. What stays from AutoKernel — explicit ledger

**Kept verbatim:**
- `orchestrate.py` scheduling core: JSON state machine, Amdahl `1/(1-Σfᵢ(1-1/sᵢ))`, move-on criteria pattern, `_append_result_row` TSV writer. (Zero torch, zero git — retarget constants only.)
- `results.tsv` schema and the "don't commit results.tsv/run.log" rule.
- git-per-experiment keep/revert protocol.
- `program.md`'s document architecture (phases A/B/C, decision framework, anti-patterns, hard-constraints list).
- The one-artifact-per-experiment, single-file-edit discipline.

**Adapted:**
- `profile.py`, `extract.py`, `verify.py`, `prepare.py` — same responsibilities, Vulkan/SSH implementations.
- `program.md` tiers → V3D playbook.
- orchestrator metric `tflops` → `tok/s`/`bw_util`, thresholds → V3D.

**Replaced/dropped:**
- `bench.py`, `reference.py`, `kernels/`, `kernels/cuda/`, `models/`, `kernelbench/`, `export_hf.py`, `analysis.py`.

---

## 10. Minimum viable experiment

**Optimize `convolution1x1.comp` (MNN's shader, the exact Session 10 culprit) for V3D's 256-invocation limit, through the Theodosia-driven FSM, and prove the loop rejects the partial-compute candidate.**

Smallest thing that proves "Theodosia-driven agentic Vulkan kernel optimization on Pi 5":
1. Agent connects to Theodosia-on-Pi over streamable-http.
2. `step("characterize")` → probes `v3d_caps` (256 / 16 KB / subgroup 16).
3. `step("baseline")` → MNN decode t/s on Qwen2-0.5B (the corrected 4.95 t/s ceiling shader).
4. One HYPOTHESIZE→IMPLEMENT→COMPILE→VERIFY cycle where the agent tries a channel-packing variant (`gws.y = UP_DIV(ocDiv4, K)`), and **VERIFY rejects it** because `verify_complete` fails (not all channels written) — reproducing the Session 10 catch, this time *inside the gate*.
5. A second cycle with a legitimate `local_size` change (e.g. `(1,240,1)`) that VERIFIES and is kept or reverted on honest t/s.

Success = the ledger shows a benchmarked kept variant **and** a refused partial-compute variant. That single trace is the paper's core claim: the loop makes the Session 10 failure structurally impossible.

---

## 11. Proposed repo structure

```
not-quite-voodoo/
├── README.md                 # what it is, Pi requirements, run
├── program.md                # V3D optimization playbook (6 V3D tiers, Q5)
├── kernel.comp               # the Vulkan GLSL shader under optimization
├── kernel.meta.json          # shader→engine plug-in coords + host-patch coords
├── bench.sh                  # V3D correctness + timing (test-backend-ops / llm_demo / bw_stress)
├── fsm.py                    # Burr FSM (adapted from v3d-pi-ai/burr_fsm/fsm.py)
├── theodosia_server.py       # mount(build_app, name="v3d").run(transport="streamable-http")
├── prepare.py                # Pi setup: swap, vc4-kms-v3d-pi5, MNN/llama.cpp build check, probe
├── profile.py                # rank bottleneck shaders (adapted)
├── extract.py                # pull target shader out of MNN/llama.cpp (adapted)
├── orchestrate.py            # scheduler — reused near-verbatim, metric = tok/s
├── results.tsv               # per-experiment log (AutoKernel format, untracked)
└── src/
    ├── v3d_caps.c            # hardware probe (from investigation)
    ├── bw_stress.cpp         # bandwidth stress (from investigation)
    └── v3d_workgroup_patch.py # known-good coordinated shader+host patch reference
```

`fsm.py`, `src/*` are lifts from `~/v3d-pi-ai/burr_fsm/` and `~/v3d-investigation/src/`. `orchestrate.py` is a lift from the fork. Everything else is new-but-small.

---

## 12. Paper contribution statement

AutoKernel and KernelBench both assume a CUDA/torch substrate: kernels are Triton or CUDA C++, correctness is `torch.allclose` against a PyTorch reference, and timing is CUDA events — an abundant, well-instrumented, tensor-core-rich environment. This fork ports the "edit one artifact, gate on correctness, keep-or-revert, log every experiment" methodology to the opposite regime: an $80 Raspberry Pi 5's VideoCore VII, where the artifact is a Vulkan GLSL compute shader, correctness comes from `test-backend-ops -b Vulkan0` against a CPU reference backend, and the hardware ceiling is a 17 GB/s LPDDR4X memory bus with a 256-invocation, 16 KB-shared-memory, fp16-storage-only compute unit and no tensor cores. Its distinct contribution is twofold: it replaces the ad-hoc "human reads `program.md`" agent loop with a **Burr finite-state machine served over MCP by Theodosia**, making the optimization protocol an enforced, tamper-evidently audited state machine rather than a prompt the agent may ignore; and it hard-codes into that machine's guards the lesson of a real prior failure (the Session 10 audit, where MNN's correctness-free autotuner reported 72% spurious speedups by crowning a shader that computed 1/K of its output K× faster) — demonstrating that agentic kernel optimization on commodity edge GPUs is only trustworthy when a completeness-checked correctness gate structurally precedes every performance measurement.

---

## Dependency graph — what must exist before what

```
v3d_caps probe (exists) ─┐
MNN/llama.cpp built on Pi (exists) ─┼─► prepare.py (Pi provisioning verified)
8 GiB swap on Pi (exists) ─┘
                                      │
                                      ▼
                       bench.sh  ◄──── the correctness+timing primitive
                        │  (test-backend-ops, llm_demo, bw_stress wrappers)
                        │
         ┌──────────────┼───────────────┐
         ▼              ▼               ▼
   profile.py      extract.py       kernel.comp + kernel.meta.json
   (rank)          (pull shader)    (the artifact)
         └──────────────┼───────────────┘
                        ▼
                     fsm.py  ◄──── adapt existing burr_fsm/fsm.py
                        │           (add COMPILE gate + verify_complete guard)
                        ▼
                orchestrate.py (retarget metric/thresholds)  ── reused
                        ▼
                theodosia_server.py  ◄── mount(fsm) on Pi, streamable-http
                        ▼
                laptop agent (MCP client)  ──► MVE run (Q10)
```

**Critical path:** `bench.sh` gates everything — it is the correctness primitive every state depends on. Build and trust it first. `fsm.py` and `orchestrate.py` are low-risk (mostly exist). `theodosia_server.py` is a thin wrapper. `extract.py` (host-side patch-coordinate discovery) is the highest-uncertainty new component.

## Estimated complexity per component (hours)

| Component | Hours | Notes |
|---|---|---|
| `bench.sh` (correctness + timing wrapper) | 6–10 | Wrapping existing tools; the completeness-check is the hard part |
| `verify_complete` guard + COMPILE offline-repack gate | 4–6 | The audit-critical logic; must be exactly right |
| `prepare.py` (Pi provisioning) | 3–5 | Mostly checks; Pi already provisioned |
| `profile.py` (adapt) | 4–6 | Parse engine stdout instead of torch.profiler |
| `extract.py` (adapt) | 8–12 | Highest uncertainty: locate shader + co-varying host constants |
| `fsm.py` (adapt existing) | 4–6 | Add COMPILE state + verify_complete; retarget guards |
| `orchestrate.py` (retarget) | 1–2 | Constants only |
| `theodosia_server.py` | 1–2 | `mount(build_app).run(...)` |
| `program.md` (V3D playbook) | 4–6 | Rewrite six tiers |
| MVE run + debugging | 6–10 | First real Theodosia-on-Pi loop |
| **Total** | **~41–65 h** | |

## The single most important design decision to get right first

**The VERIFY gate must compare *complete reference output*, not a fast timing, and it must run before any performance number is trusted — enforced by a Burr guard, not by agent goodwill.** Everything else is plumbing. This one decision is the reason to do the project at all: the Session 10 audit proved that a correctness-free timing loop (MNN's autotuner) will confidently report large fake speedups by rewarding a kernel that skips most of its work. AutoKernel's architecture already embodies the fix (correctness precedes measurement, revert on fail), but AutoKernel enforces it only in `program.md` prose. Moving that gate into a Theodosia-served FSM guard (`verify → benchmark` is *unreachable* unless `verify_ok AND verify_complete`) is the actual novel, defensible contribution. Get the completeness check wrong — sample a tile instead of the full output — and the fork reproduces the exact bug it exists to prevent.
