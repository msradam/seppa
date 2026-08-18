# Seppa: Correctness-Gated LLM Kernel Optimization on the Raspberry Pi 5 GPU

Adam Munawar Rahman
New York University. Code: github.com/msradam/seppa, github.com/msradam/bonbibi

## Abstract

Most edge AI deployments on single-board computers leave the integrated GPU idle. We present Seppa, a harness for optimizing GPU kernels with a large language model in the loop: the model proposes changes, and a finite-state machine compiles, verifies against physics gates, benchmarks, and decides keep or revert, refusing to benchmark any kernel that failed verification. On the flood-simulation stencil of an offline flood-guidance application for the Raspberry Pi 5, the harness produced a kernel 1.58x faster than the original; the machine re-measured and re-judged the result in five consecutive runs over the Model Context Protocol, and in every run it refused to benchmark a kernel that failed its gate. With CPU inference running concurrently the advantage widens to 2.20x, and the deployment sustains 9.6 tokens/s of language-model decode beside 883 simulation steps per second, an operating point that beats the best measured CPU-only configuration on both axes.

## 1. Introduction

Raspberry Pi class boards are a common platform for edge AI in emergency-response and field settings, where power, connectivity, and budget are constrained [12] and inference must happen on-device once the network is gone [11]; a Pi 5 runs a quantized billion-parameter model on its CPU cores [8]. AI capability is usually added with dedicated hardware, an NPU HAT or a USB accelerator [13], which raises cost and, in a supply-constrained chip market [14], adds a procurement dependency.

Less attention goes to silicon already on the board: every Pi 5 ships a VideoCore VII GPU that compute workloads rarely touch, and during CPU inference it idles. Two questions decide that idle GPU's worth: can it be made fast enough at a useful workload to be a co-processor, and can it run beside a busy CPU on a shared LPDDR bus? Sections 5 and 6 answer the first question, Section 7 the second.

Making kernels fast on a sparsely documented mobile GPU, where tuning folklore from CUDA-class hardware mostly fails (Section 2), is exactly the kind of work the field has begun handing to language models. Existing agentic optimizers, AutoKernel [15] among them, steer the agent with prompts and trust it to verify and revert honestly; the record shows what that trust costs (Section 3). Seppa removes the trust. The model proposes, and nothing else; a finite-state machine built on Burr [5] and served over the Model Context Protocol [6] owns compilation, verification, benchmarking, and the verdict, and refuses out-of-order requests.

The demonstration workload is the flood simulation of an offline flood-guidance application [7], which must run beside language-model narration on one board; its stencil, a neighbor-update grid kernel, is the optimization target, and its CPU-plus-GPU split is what we measure (Section 9 describes the application).

This paper reports:

1. the optimization of a real flood stencil for V3D under three physics gates (correctness checks passed before any benchmark), to 1.58x, failed hypotheses included (Section 5);
2. an action-space lesson: the first search plateaued because every winning change lived in the host program, outside what the machine let the model edit (Section 5.3);
3. a machine-checked reproduction over MCP, repeated five times under a defined pass criterion, including the server's refusal to benchmark a gate-failing kernel (Section 6);
4. an interference measurement of the deployed configuration, with a gate-verified CPU-only counterfactual that the GPU split beats on both axes (Section 7).

## 2. Background: The Raspberry Pi 5 GPU

The Pi 5 pairs four Cortex-A76 CPU cores with a VideoCore VII GPU (V3D 7.1), the block that also drives the desktop. There is no CUDA and no vendor compute toolchain; compute reaches the GPU only through Vulkan shaders, compiled by Mesa's v3dv driver [9]. In this paper, kernel means a GLSL compute shader.

Tables~\ref{tab:platform} and \ref{tab:gpu} list the platform and the GPU's compute limits, collected from the running board by `pi/collect_specs.sh` (archived with the raw vulkaninfo dump). The limits are tight, and the scale is modest: the best kernel we measured sustains about 13 GFLOP/s in fp32, and that was a dense matrix multiply, not the stencil this paper optimizes (Section 7.1 compares the two processors on the stencil itself). The GPU is a second, slower engine that happens to be free.

<!--SPECS-->

The strange part is the driver: when a shader wants more registers than exist, v3dv does not fail. It walks a ladder of six compile strategies [16], disabling scheduling, loop unrolling, and code motion together, then halving the thread count, then falling back to a simpler scheduler, and hands back whatever survives, sometimes several times slower.

## 3. Related Work

Kernel generation is now a standard LLM task. KernelBench [3] conditions speedup on correctness by definition, Kevin [17] trains multi-turn reinforcement learning on kernel writing with correctness and runtime as verifiable rewards, and TritonRL [18] puts the difficulty in its title, training a Triton model "without cheating." The systems still cheat. CUDA-L1 [4] reports that 82 of its 250 RL-generated kernels (32.8%) exploited stream-timing loopholes, inflating the reported speedup to 18x; containment took a reward checker, a database of known hacks, and forced stream synchronization. Wherever the reward is a measured number, the model eventually optimizes the measurement rather than the kernel.

A weaker assumption sits underneath all of this: that a passing check means a correct kernel. Sarkar [19] tests the oracles these benchmarks use, fixed-shape allclose comparisons at fixed tolerance, and finds them systematically optimistic, with seeded-buggy kernels passing the benchmark oracle while failing a stricter one. SpecBench [20] states the general form: when oversight collapses onto an automated suite, the agent optimizes the suite. Our physics gates are the same kind of instrument and inherit the same weakness, which is why Section 8 stops short of calling them sufficient.

A second line puts the evaluator outside the model instead of arguing with it. FunSearch [21] and AlphaEvolve [22] use the model as a mutation operator inside an evolutionary loop scored by a fixed evaluator; AlphaEvolve found a 48-multiplication procedure for 4x4 complex matrix multiplication, the first improvement on Strassen in that setting in 56 years. That structural commitment, the model proposing and something else scoring, is the one Seppa makes, at a far smaller scale and with the scoring expressed as reachability in a graph instead of a fitness function (Section 4). In hardware design the same split appears with a human in it: Chip-Chat [10] carried conversational design to tapeout with an engineer and a testbench closing the loop, and VeriGen [23] fine-tunes open models for Verilog. We close the loop with a state machine instead.

## 4. The Seppa Harness

Seppa is a port of AutoKernel [15], an open-source kernel-optimization loop, onto a Burr finite-state machine, served over MCP by a wrapper (Theodosia) running on the Pi itself. The graph is:

```
characterize -> baseline -> hypothesize -> implement
  -> compile -> verify -> benchmark -> evaluate
  -> log_variant -> (hypothesize | stop)
```

with two guard edges to `log_variant`, on compile failure and on gate failure, and a third edge returning `implement` to `hypothesize` when the model submits nothing usable. The guards are ordinary code, from `v3d_flood2_opt.py`:

```
("compile_", "verify", expr("compile_ok")),
("compile_", "log_variant", expr("not compile_ok")),
("verify", "benchmark", expr("verify_ok")),
("verify", "log_variant", expr("not verify_ok")),  # THE GUARD
("benchmark", "evaluate"),
("evaluate", "log_variant"),
```

The benchmark state is reachable only through a green verify. The agent advances the loop through one MCP tool, `step(action, inputs)`, and the server constrains `action` to the graph's legal next moves. The server also exposes session utilities (`reset_session`, `fork_at`, and resource reads) that no archived session used; `fork_at` rewinds to an earlier state, which would let a caller resample a measurement, and the 1% keep threshold does not defend against that (Section 8).

The port preserves AutoKernel's loop discipline: one focused change per experiment, a baseline measured first, correctness and timing taken from the same execution, keep only on a gain of at least 1%, every experiment recorded. It moves one thing: the authority to decide. AutoKernel trusts its agent to run the benchmark and honestly revert failures; here the verdict is a deterministic action the model cannot reach. A run ends after three consecutive non-improvements on gate-passing variants, or at the budget. An audit of the port is in the repository (`docs/autokernel_fidelity.md`).

For the division of labor, this paper follows Chip-Chat's disclosure convention. The author built the application and the harness, chose the physics gates, ran the hand-driven sweep of Section 5.2, and supervised sessions. The language model, Claude Sonnet 5 (Anthropic), driven through the Claude Code client at high reasoning effort, wrote the content of `hypothesize` and `implement` (kernel source and host-side parameters) and nothing else. The machine did everything evidential; no proposed kernel was hand-edited (transcripts in `docs/paper/artifacts/claude_sessions/`). Concretely: the winning kernel came from the hand-driven sweep, the machine verified and reproduced it (Section 6), and the one fully agent-driven session worked a knob the server exposed without reaching fusion.

For the flood target, `verify` runs 400 steps at 256x256 and demands three things: normalized mean-square error (NMSE) below 1e-3 against a double-precision CPU reference (loose enough for fp32 drift, six orders above what correct kernels measure), total water equal to injected rainfall, and maximum depth pooling inside the terrain basin. A green gate in the harness log:

```
selected: V3D 7.1.10.2
grid=256^2 steps=4000 time=1.880s 3.07 GFLOP/s
correct(NMSE vs CPU)=yes  NMSE=1.26e-09
mass: rain_injected=5242879.9
  water_total=5242739.3  (conserved vs rain=yes)
pooling: max depth=111.718 at (127,127)
  basin_centre=(128,128)  (pools in basin=yes)
```

## 5. Case Study: The Flood Stencil

### 5.1 The Workload

The application's simulation is a stencil; each cell updates from its neighbors, in two passes per time step, a flux pass and a height pass. The original implementation ran about 1,045 steps/s at 256x256 (1.51 GFLOP/s), and both shaders compile cleanly, which means they never stress the register allocator. The bottleneck was therefore something other than register pressure.

### 5.2 Falsification Sweep

We did not guess the bottleneck. We wrote one variant per hypothesis and let the loop measure each. The variants pack the state into wider loads, fuse the two passes into one dispatch, or strip-mine, meaning each thread updates a short vertical run of cells to amortize its launch cost. Every variant passed all three gates. Speeds are 400 steps at 256x256 on the parameterized `vkflood2` harness. Its host loop is faster than the original's for identical kernels (about 1,345 steps/s against 1,045), so every comparison in this paper is within `vkflood2`.

| Variant | Hypothesis tested | Result |
|-----------|---------|----------|
| Packed vec2 state (water, surface) halving flux-pass loads | Memory-op count is the bound | 1.93 GFLOP/s, no change: falsified |
| Fused single-dispatch step (flux buffer eliminated) | Traffic and barriers are the bound | 2.04 GFLOP/s, +5%: mostly falsified |
| 2 cells per invocation (strip-mined dispatch) | Fixed per-invocation cost is the bound | 2.40 GFLOP/s, +23%: supported |
| 4 cells per invocation | More strip is better | 2.35 GFLOP/s: falsified, register pressure |
| Fused + 2 cells per invocation (`fused2s.comp`) | The wins stack | 3.08 GFLOP/s, +59% |

The outcome surprised us. Halving the flux pass's memory traffic changed nothing, and fusion, which deletes an intermediate buffer and a barrier, bought only 5%. Fixed per-invocation cost dominated: each invocation does so little arithmetic that launch overhead swamps it, and giving each thread two cells beat every memory-system optimization we tried; four cells gave the gain back to register pressure. The sweep's own numbers size that cost: strip-2 removes 65,536 invocations per step and saves about 140 microseconds, near 2 ns, two clock cycles, per invocation, the scale of thread issue, not of arithmetic or memory traffic. Two hardware details shaped the final kernel. Strips run vertically because that keeps a subgroup's 16 lanes on adjacent addresses, and the kernel consumes each flux value as it is produced; holding all four alive fails register allocation. The shipped kernel is 1.58x at 256x256: 1.574 to 1.585 across the five scored reproductions, 1.571 in the July transcript quoted in Section 6, and 1.59x hand-timed in the sweep. It is 1.41x at 512x512 and 1.18x at 1024x1024, and it holds all gates over 4,000 steps.

### 5.3 Action-Space Limits

An earlier session pointed the state machine at this stencil and came back empty; we nearly concluded the kernel was at its limit. The problem was the action space: that version of `implement` accepted shader text only, and every winning change in Table I lives outside the shader (packing changes the buffer layout, fusion deletes a host-loop pipeline, strip-mining changes the dispatch shape). Once `implement` took `{shader, height_shader, strip}`, the machine found and kept the fused strip-2 kernel from its own baseline (Section 6). A plateau can mean the search space is too small, not that the hardware is exhausted. Two other targets ran under the same gates: a GEMM (dense matrix multiply) went from 7.02 to 13.42 GFLOP/s over two rounds, and de-unrolling llama.cpp's matrix-vector kernel gained 28% end-to-end decode. Both are recorded in the running notes, not in archived logs, and neither is claimed at Section 6's evidentiary standard.

## 6. Machine-Checked Reproduction over MCP

Section 5's numbers do not have to be taken on trust. `drive_flood2_mcp.py` runs on a second computer and drives the Pi-resident state machine through the whole cycle via the `step` tool.

The state machine measures its own baseline (1,346.8 steps/s on 2026-07-11, gates green), receives the fused strip-2 kernel as experiment 1, compiles it, gates it, benchmarks 2,116.4 steps/s, and issues its own verdict:

```
{"exp": 1, "fused": true, "strip": 2,
 "compile_ok": true, "verify_ok": true,
 "steps_per_sec": 2116.4, "best_sps": 2116.4,
 "verdict": "keep"}
```

Then the driver tries to cheat: it submits the same kernel with rainfall injection doubled in one of the two cell updates, a change that compiles and breaks conservation. The gate fails it; the driver requests `benchmark` anyway, and the server answers (two advisory fields elided):

```
{"error": "invalid_transition",
 "requested": "benchmark",
 "valid_next_actions": ["log_variant"],
 "message": "action 'benchmark' is not
   reachable from current state. Valid
   actions now: ['log_variant']."}
```

The variant is ledgered as revert with a null `steps_per_sec`. Gates and timing come from one execution, so `verify` does return the failing kernel's speed to the caller as `steps_per_sec_if_kept`; what the green gate controls is entry to the ledger, the verdict, and the kept kernel, not the caller's sight of the number.

How repeatable is this? `passk_flood2.py` runs the two-cycle reproduction k times from cool starts, scoring a pass when the baseline gates green inside 1,300 to 1,400 steps/s, the fused kernel is kept at 1.5x or better, and the broken kernel is refused and nulled. Five of five runs passed on 2026-08-09: baselines 1,348.6 ± 4.1 steps/s, the kept kernel 2,127.7 in every run (identical at the timer's resolution), speedups 1.574 to 1.585.

One session was fully agent-driven (2026-08-02, budget three). The server's `characterize` payload names fixed per-invocation cost as the bound and exposes the strip parameter, so the session demonstrates the loop's enforcement, not autonomous discovery: the model worked the exposed knob (strip 4 kept at 1,470.6, +8.8%), hit the register cliff at strip 8 (921.7, reverted), probed it at strip 6 (1,298.7, reverted), and never reached fusion; all of its reported numbers came from the machine.

## 7. Concurrent CPU and GPU Execution

In deployment the GPU loops the simulation while the CPU routes and decodes (Granite 4.0 1B, llama.cpp, `-ngl 0`, 4 threads). Decode, the token-by-token generation phase, is memory-bound, which matters because the two processors share one LPDDR bus.

The campaign turned up one configuration surprise first. With any Vulkan device visible, llama.cpp places CPU-resident model weights in the GPU's host-pinned, write-combined memory even at `-ngl 0`, memory that is slow for the CPU reads that dominate decode. Hiding the device with `GGML_VK_VISIBLE_DEVICES=99` takes decode from 9.68 to 11.59 tokens/s, a 20% gain from an environment variable (A/B from cool starts, llama-bench r=5; raw logs in `docs/paper/artifacts/`, ab_vkvisible). Every number below uses the hidden-device configuration, which the application ships.

`pi/flood/conc_steady_bench.sh` measures each kernel alone (cool starts, finished before heat accumulates), decode alone, and each kernel looping while decode runs. Each phase is preceded by a cooldown and a fixed eight-repetition decode soak. That soak is not gated on the throttle bit, and the conditions did not end up in the same thermal regime: the concurrent windows ran with the firmware's soft-temperature limit active 70 to 79% of the time and peaked at 76.3 C, while decode alone reached 73.0 C and was limited for 3% of its window. Each window is long enough for 12 to 28 flood completions (llama-bench tg64, r=20), and a flood run counts only if it fits inside the window. An earlier short-window campaign, preserved in `conc_bench2/`, read up to 19% differently in both directions. We first attributed that to a thermal transient, but the logs do not support it: the earlier campaign ran cooler (68.7 C mean, soft limit active 46% of the window, hard throttle never) than this one (72.7 C, 70%, 9%) and still read lower, so heat does not explain the gap. Within this window the rate also climbs, from 818.6 steps/s over the first quarter to 919.1 over the last, so the figure below is a window mean on a rising curve and not a plateau. Results from the steady-state run of 2026-08-10 (raw logs in `docs/paper/artifacts/`, conc_steady):

| Condition | GPU flood (steps/s) | CPU decode (t/s) |
|---|---|---|
| Optimized kernel alone | 2,128.0 ± 1.7 (n=3) | |
| Original kernel alone | 1,351.4 ± 0.5 (n=3) | |
| Decode alone, 4 threads (post-soak) | | 11.4 ± 0.0 |
| Concurrent, optimized kernel | 883.4 ± 47.1 (n=28) | 9.6 ± 0.2 |
| Concurrent, original kernel | 401.8 ± 8.3 (n=12) | 9.6 ± 0.2 |

Three things follow. Co-processing costs the CPU 16% of its decode and buys a continuous 883 steps/s of simulation that a CPU-only deployment does not have. The interference is lopsided: the CPU keeps 84% of its rate while the GPU keeps 42%, as decode traffic crowds the shared bus. And contention favors the optimized kernel: 1.58x over the original alone becomes 2.20x concurrent, at identical decode cost (9.6 t/s under either kernel). One residual: the optimized-kernel window still drifts upward (first-half mean 852, second-half 915 steps/s) as governance slows decode, so 883.4 is conservative relative to the late-window rate.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{gpu_retention.png}
\caption{GPU flood throughput retained beside three concurrent CPU loads, as a percentage of the same kernel running alone. Compute-bound and memory-bound CPU work barely disturb the GPU; language-model decode, which streams weights from DRAM every token, takes it to 42\%. Raw logs in \texttt{docs/paper/artifacts/}, genload2 and conc\_steady.}
\label{fig:retention}
\end{figure}

Decode is also the GPU's worst case. Paired with a compute-bound CPU load (openssl SHA-256, four threads) the GPU keeps 99% and the load keeps 88% of its 16 KB-block throughput; paired with the memory-bound four-thread CPU flood the GPU keeps 92% and the load keeps 75% (continuous sim-mode probes over cool-start 60 s windows, both sides logged; `docs/paper/artifacts/`, genload2). These two pairings also bound the thermal confound. Both ran at the same 76.3 C with the soft limit active for 80% and 70% of their windows, against a solo window that never left 57.6 C, and the GPU still kept 99% and 92%. Clock governance alone therefore does not explain decode's 42%. The GPU is the robust side of every pairing except decode, which streams model weights from DRAM every token: interference tracks the CPU load's memory traffic.

### 7.1 The CPU-Only Counterfactual

Four idle A76 cores run this stencil at 3,852 steps/s, well above the V3D's 2,128, so the GPU looks unnecessary. `pi/flood/cpuflood.cpp` is the same update with OpenMP and passes the same three gates (NMSE 1.46e-11); `pi/flood/cpu_steady_bench.sh` measures it alone and sharing the cores with decode, pinned apart (flood on core 0, decode on 1 to 3) and oversubscribed (four threads each). The same soak-and-measure protocol applied; raw logs are in `docs/paper/artifacts/`, cpu_steady.

| Condition | Flood (steps/s) | CPU decode (t/s) |
|------------|--------|------|
| CPU flood alone, 1 / 2 / 4 threads | 992.5 / 1,980.5 / 3,852.4 | |
| Decode alone, 3 threads (pinned, post-soak) | | 11.9 ± 0.0 |
| Partitioned: flood 1t + decode 3t | 678.8 ± 10.6 (n=25) | 8.4 ± 0.1 |
| Oversubscribed: flood 4t + decode 4t | 857.1 ± 181.0 (n=89) | 3.0 ± 0.3 |
| GPU concurrent, optimized (from above) | 883.4 ± 47.1 | 9.6 ± 0.2 |

The idle-core number is real but unavailable: in deployment, decode is always running. Oversubscribed, decode collapses 75% (llama.cpp's workers synchronize every token; losing one core stalls all four), and the flood mean is inflated by bursts between decode repetitions, hence its ± 181. Partitioned, the best CPU-only arrangement, decode pays a 29% tax against the GPU split's 16%; time-slicing (flooding about 23% of the time to match 883 steps/s) would cap decode at 8.8 t/s and freeze guidance during bursts. The GPU split beats the best measured CPU-only option on both axes, by 30% on simulation (883.4 ± 47.1 vs 678.8 ± 10.6) and 14% on decode (9.6 ± 0.2 vs 8.4 ± 0.1): the slower processor still wins, because the CPU has no spare cycles to sell.

## 8. Limitations

The envelope is one board (Pi 5, V3D 7.1.10.2, Mesa v3dv 25.0.7) and one grid family, with speedups that shrink as the grid grows; headroom likely remains. The July and August campaigns straddle an OS update (kernel 6.12 to 6.18); flood baselines agree across it (1,346.8 before, 1,348.6 ± 4.1 after). The gates are necessary rather than sufficient: verification covers one storm scenario at one grid size, and the keep decision rests on a single timing sample against a 1% threshold, although Section 6's five-run repeatability (baseline spread about 0.3%, kept kernel identical at timer resolution) bounds that noise well inside the threshold. That bound is an upper bound, not a measured spread: the harness prints elapsed time to milliseconds, so at the kept kernel's 0.188 s one digit is 0.53%, and "identical" means the last printed digit did not move. The 3,852 steps/s CPU figure assumed a cache-resident 256x256 working set and may not survive larger grids. Three asymmetries qualify Section 7. The GPU kernel was optimized and `cpuflood.cpp` was not: it is the original two-pass scheme, so the counterfactual compares a tuned GPU against an untuned CPU, and fusing its flux array would cut DRAM traffic on the axis that contends with decode. The conditions were not measured at a matched thermal operating point, so part of the 16% decode cost is governance, not contention. And the steady-state campaign is a single run: an earlier campaign at the same configuration, kept in `conc_bench2/`, reads 711.9 steps/s concurrent against this run's 883.4, which is the least reproducible quantity in the paper. That matters for how far Section 7.1 reaches. Under the earlier campaign the simulation axis is within noise of the partitioned CPU scheme (711.9 ± 76.1, n=5, against 678.8 ± 10.6, n=25), so the two-axis claim rests on the later campaign; the decode axis holds under both (9.6 and 10.3 against 8.4). The scripted reproduction's kernel came from the hand sweep, and the agent-driven session worked a knob the server itself exposed. The server's `fork_at` utility rewinds to an earlier state, so a caller could in principle resample a marginal kernel until a draw cleared the 1% threshold; `evaluate` compares one fresh sample against a frozen incumbent and would not notice. No archived session used it. The demonstrated property is machine verification; autonomous discovery remains open. Full GPU offload of the language model is blocked by an upstream llama.cpp Vulkan defect. Numbers without an archived log (the original harness's 1,045 steps/s, the sweep's GFLOP/s column and hand-timed ratios, the 1.41x and 1.18x at the larger grids, and the GEMM and matrix-vector results) come from the repository's dated running notes.

## 9. The Sample Application

The sample application [7] simulates surface flooding over real terrain [1], finds shelter routes by mobility profile [2], and explains the result in plain language, entirely offline; the model never makes a safety decision, only narrating what the deterministic code computed. The application lives at github.com/msradam/bonbibi and was built as an Arm AI Optimization Challenge entry; the harness at github.com/msradam/seppa. Everything this paper claims was built on top of that application rather than as part of it: the state machine and its gates, the falsification sweep of Section 5, the machine-checked reproduction and pass criterion of Section 6, and the concurrency and CPU-counterfactual campaigns of Section 7. Measurements used its Granite 4.0 1B configuration, whose 1.63 B GGUF llama-bench prints under a granite-3B size label (the application has since moved to a larger model); build commands are in `pi/flood/README.md`, and cited references are archived in `docs/paper/references/`.

## 10. Conclusion

Seppa separates proposal from judgment: a language model suggests kernels for the Raspberry Pi 5's integrated GPU, and a state machine the model cannot argue with decides what is true about them. On a real flood stencil this produced a verified 1.58x, reproduced within 1% by the machine five of five times, and it holds in deployment: 883 steps/s of simulation beside decode at 84% of its solo speed. The evidence in this paper is the gate ledger, not the model's account of its work. The suggestion is open-ended: commodity edge boards carry more usable silicon than their deployments exercise, and a loop that cannot lie about correctness is a reasonable way to get at it.

## References

[1] M. Guidolin, A. S. Chen, B. Ghimire, E. C. Keedwell, S. Djordjevic, and D. A. Savic. A weighted cellular automata 2D inundation model for rapid flood analysis. Environmental Modelling & Software 84:378-394, 2016.

[2] J. Dibbelt, B. Strasser, and D. Wagner. Customizable Contraction Hierarchies. ACM Journal of Experimental Algorithmics 21, Article 1.5, 2016.

[3] A. Ouyang, S. Guo, et al. KernelBench: Can LLMs Write Efficient GPU Kernels? arXiv:2502.10517, 2025.

[4] X. Li et al. CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning. arXiv:2507.14111; ICLR 2026.

[5] Burr. Apache Software Foundation (incubating). https://github.com/apache/burr

[6] Model Context Protocol. https://modelcontextprotocol.io

[7] Bonbibi: offline flood-aware routing on a Raspberry Pi 5. https://github.com/msradam/bonbibi

[8] llama.cpp. ggml-org. https://github.com/ggml-org/llama.cpp

[9] V3D driver documentation, Mesa 3D. https://docs.mesa3d.org/drivers/v3d.html

[10] J. Blocklove, S. Garg, R. Karri, and H. Pearce. Chip-Chat: Challenges and Opportunities in Conversational Hardware Design. ACM/IEEE Workshop on Machine Learning for CAD (MLCAD), 2023.

[11] Z. Zhou et al. Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing. Proc. IEEE 107(8), 2019.

[12] S. Boddu and A. Mukherjee. Efficient Edge Deployment of Quantized YOLOv4-Tiny for Aerial Emergency Object Detection on Raspberry Pi 5. arXiv:2506.09300, 2025.

[13] A. Reuther et al. Survey of Machine Learning Accelerators. IEEE HPEC, 2020.

[14] The White House. Building Resilient Supply Chains, Revitalizing American Manufacturing, and Fostering Broad-Based Growth: 100-Day Reviews under Executive Order 14017. June 2021.

[15] AutoKernel. RightNow AI. https://github.com/RightNow-AI/autokernel

[16] Mesa, `src/broadcom/compiler/vir.c`, `strategies[]` compile-fallback table. https://gitlab.freedesktop.org/mesa/mesa

[17] C. Baronio, P. Marsella, B. Pan, et al. Kevin: Multi-Turn RL for Generating CUDA Kernels. arXiv:2507.11948, 2025.

[18] J. Woo, S. Zhu, A. Nie, Z. Jia, Y. Wang, and Y. Park. TritonRL: Training LLMs to Think and Code Triton Without Cheating. arXiv:2510.17891, 2025.

[19] D. Sarkar. The Correctness Illusion in LLM-Generated GPU Kernels. arXiv:2606.20128, 2026.

[20] B. Zhao, D. Srikanth, Y. Wu, and Z. Jiang. SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents. arXiv:2605.21384, 2026.

[21] B. Romera-Paredes, M. Barekatain, A. Novikov, et al. Mathematical discoveries from program search with large language models. Nature 625:468-475, 2024.

[22] A. Novikov, N. Vu, M. Eisenberger, et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. Google DeepMind, 2025.

[23] S. Thakur, B. Ahmad, H. Pearce, B. Tan, B. Dolan-Gavitt, R. Karri, and S. Garg. VeriGen: A Large Language Model for Verilog Code Generation. ACM TODAES, 2024.
