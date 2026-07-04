# Fable session notes: llama.cpp GPU decode on V3D

One lesson per entry. Confirmed wins and dead ends both, with why.
Hardware: Pi 5, V3D 7.1.10.2, Mesa 25.0.7-2+rpt4, llama.cpp bb28c1f.
All work lives on the Pi in `/root/v3d-research/llama.cpp`, branch
`v3d-fixes` (commits 428df1d, bc4d300). The same two commits are exported
as `pi/llama-cpp-v3d-fixes.patch` and `pi/llama-cpp-v3d-mmv-deunroll.patch`
in this repo.

## Repro commands (exact)

Environment for every V3D run:
`GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1`

- Correctness gate (rung 1, real model shapes, NMSE vs CPU oracle;
  thresholds are llama.cpp's own: 1e-7 f32 ops, 5e-4 MUL_MAT/SET_ROWS):
  `cd /root/v3d-research/llama.cpp/build-vulkan/bin && GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1 ./test-backend-ops test -b Vulkan0 --test-file /root/v3d-research/granite_ops.txt`
  Expected: `43/43 tests passed` (plus 11 "not supported" = deliberate CPU
  fallbacks: n>1 matmuls, FLASH_ATTN). The ops file was generated with
  `./export-graph-ops -m <model> -fa off|on -o ...` (both merged, `sort -u`).
- Benchmark:
  `GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1 ./llama-bench -m /root/v3d-research/models/granite-4.0-1b-dense/granite-4.0-1b-Q4_0.gguf -ngl 6 -fa 0 -p 0 -n 32 -r 2`
  Expected: tg32 = 5.56 ± 0.01 t/s (was 4.32 on the stock shader).
- End-to-end sanity:
  `... ./llama-completion -m <same model> -ngl 6 -c 4096 -fa 0 -n 8 --temp 0 -p "The capital of France is"` → mentions Paris.
- The FSM itself: `/root/seppa/v3d_llama_mmv.py` served by
  `.venv/bin/python theodosia_server.py --http --llama-mmv` (port 8000);
  drive with the MCP `step` tool. Baseline action alone takes ~7 min.

## Win: the "llama.cpp Vulkan hangs on Pi 5" is a v3dv compile-time hang, not a GPU hang

`llama-bench -ngl 99` "hangs after device detection" because the v3dv
register allocator never terminates compiling llama.cpp's multi-column
`mul_mat_vec` pipelines (ncols 2..8). Verified: gdb backtrace shows one
thread at 99.9% CPU inside `libvulkan_broadcom.so` under
`ggml_vk_create_pipeline_func`; the n=2 case ran >550 s without finishing.
n=1 compiles in ~30 s after the driver's fallback ladder ('disable general
TMU sched' → 'disable gcm' → 'disable loop unrolling') and passes
correctness. `V3D_DEBUG=opt_compile_time` does NOT rescue it. Fix:
`GGML_VK_MMV_MAX_COLS=1` (new env in supports_op) reports n>1 matmuls
unsupported so prompt processing falls back to CPU; decode is n=1.

## Win: the >256-invocation silent-corruption class

48 shader files hardcode `local_size_x = 512/1024` and 7 more use
`BLOCK_SIZE 512`; V3D's max is 256 and v3dv silently mis-executes instead
of failing pipeline creation. Symptoms: RMS_NORM writes out of bounds
("sentinel mismatch" in test-backend-ops), GET_ROWS (the embedding lookup,
first op of every graph!) returns garbage, SET_ROWS corrupts the KV cache.
Fix is the class sed 512/1024 → 256 in shaders plus the paired host
`wg_denoms` triples, plus one non-obvious host pairing: SET_ROWS's
"elements per workgroup" divisor (`ne = CEIL_DIV(ne, 512)` → 256) — the sed
alone left half the rows unwritten. Elementwise shaders' internal
`y*512 + x` linearization is the host's element-wrap contract, NOT the
local size; do not sed those.

## Win (FSM experiment 1, kept): de-unroll mul_mat_vec → +28% decode

Removing the manual 4x/2x unrolling from `mul_mat_vec.comp` took Granite
Q4_0 decode from 4.32 to 5.55 t/s at -ngl 6, gate green (43/43). Mechanism:
the unrolled bodies blow the QPU register budget, and the driver's rescue
strategy ('disable loop unrolling') discards the unrolling anyway but only
after also losing general TMU scheduling and GCM. Rolled source compiles on
the default strategy. General V3D lesson: optimize for the driver's
register allocator first; source-level unrolling is anti-productive here.

## Dead end (FSM experiment 3, reverted): software-pipelined B prefetch

Double-buffering the B loads (issue i+1's loads before computing i)
verified correct but was performance-neutral (5.55 → 5.55). The default
strategy's TMU scheduler evidently already hides that latency once the
loop is rolled. Not worth register pressure. GLSL gotcha that cost one
compile-fail cycle (experiment 2): arrays sized by specialization
constants cannot be whole-array assigned (`bv0 = nbv0;` fails with
"can't use with types containing arrays sized with a specialization
constant"); copy element-wise.

## Dead end / upstream bug: full offload decodes garbage on ANY Vulkan driver at bb28c1f

With every per-op test green, `-ngl 99` still decodes incoherently
("GGGG...") and quality degrades progressively with ngl (1..6 coherent,
8 marginal, 10+ garbage). NOT V3D's fault: stock bb28c1f on llvmpipe
(reference software driver, `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`
+ `GGML_VK_VISIBLE_DEVICES=0`) reproduces the same garbage, with and
without our patches, for granite Q4_0, granite Q4_K_M, and lfm2-1.2b.
Falsified along the way: fusion (GGML_VK_DISABLE_FUSION), v3dv job sync
(V3D_DEBUG=sync), TMU precision (tmu32), host-visible memory, suballocation
size, graph-optimize, submit granularity (new GGML_VK_NODES_PER_SUBMIT env).
Follow-up: bisect upstream llama.cpp or retest on a newer commit; this is
an upstream ggml-vulkan regression, likely known/fixed by now. Until then
-ngl 6 is the verified-correct ceiling.

## Gotcha: per-op NMSE ~1.5e-5 for quantized MUL_MAT is normal, not a defect

V3D q4_0/q6_K matvec NMSE vs CPU is 8e-6..8e-5. llvmpipe shows the same
magnitudes (7e-6..3e-5). The difference is the CPU reference's own q8_1
activation quantization path, not GPU error. V3D f32 matvec is exact
(1.4e-14). Don't chase this.

## Gotcha: test-backend-ops full MUL_MAT suite cannot finish on V3D

Exotic iq-type (iq1_m etc.) ncols=1 pipelines also hit the unbounded
compile. Filter with `--test-file` (model graph ops) or `-p` regex to
mainstream types. Also: its output is ANSI-colored; `\bOK\b` does not
match `[1;32mOK` (the 'm' kills the word boundary) — parse the
"N/M tests passed" summary instead. `GGML_TEST_PRINT_ERR=1` (added in
tests/test-backend-ops.cpp) prints the measured error for every case.

## Gotcha: Pi ops hygiene

`pkill -f <pattern>` kills your own SSH session when the pattern appears
literally in the remote command line — write it as `patter[n]`. Background
jobs launched via `bash -c "... &"` chains die unpredictably with the
session; use a script file + `setsid`. Redirected test output is
block-buffered: the last line of a log is NOT where a process died.

## State of the MCP server

`theodosia_server.py --http --llama-mmv` on the Pi (port 8000) serves the
`v3d_llama_mmv` FSM. Ledger from this session (app 9466169f):
baseline 4.32 → exp1 keep 5.55 → exp2 revert (compile fail) → exp3 revert
(correct, no gain). `best_mul_mat_vec.comp` in this repo = the kept shader
= what is on disk and committed on the Pi. The old gemm.comp explore mode
is still available with `--explore` (best 12.56 GFLOPS, unchanged).

## Next ideas, in order of expected value

1. Retest full offload on current llama.cpp master (the upstream garbage
   bug may be fixed); if coherent, switch the FSM benchmark to -ngl 99
   where mmv gains are ~5x more leveraged.
2. Host-side rows-per-workgroup (rm_stdq) and DMMV workgroup-size sweep:
   needs the FSM implement step extended to patch ggml-vulkan.cpp
   constants, not just the shader.
3. Attention on GPU: kq/kqv/SOFT_MAX n=1 all verify green already; the
   -nkvo requirement comes from the upstream composition bug, not from
   those ops.
