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

The server on the Pi (port 8000) serves whichever mode `/tmp/serve.sh`
launches; as of the second Fable session it is `--explore` (GEMM). Switch by
editing /tmp/serve.sh (or launching `.venv/bin/python theodosia_server.py
--http --llama-mmv|--explore` from /root/seppa) — kill the old server with
`pkill -f "theodosia_[s]erver"` from a command line that does NOT itself
contain the unescaped server name, then `setsid /tmp/serve.sh ... &`.
Ledgers so far:
- llama-mmv (app 9466169f): baseline 4.32 → exp1 keep 5.55 → exp2 revert
  (compile fail) → exp3 revert (correct, no gain).
- gemm explore round 2 (app 74876a65): baseline 12.56 → exp1 keep 13.42 →
  exp2 revert (8.60, transposed B) → exp3 revert (12.57, BK=64).
`best_mul_mat_vec.comp` and `best_gemm.comp` in this repo are the kept
shaders and match what is on the Pi.

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

## Root cause: the llama-cli abort was a stale binary; beneath it, llama-cli itself is broken upstream

Two layers, both closed:

**Layer 1 — the abort Sonnet reproduced.** "Shared memory size too small for
matrix multiplication" is the exact line commit 428df1d deletes. The string
occurs 0 times in the patched source but 2 times in the llama-cli binary,
which was dated May 21 (the original stock build): ggml-vulkan is a STATIC
library and every `cmake --build --target ...` in the first session listed
only test-backend-ops/llama-bench/llama-completion, so llama-cli was never
relinked and silently carried the unpatched backend. Rebuilding it
(`cmake --build build-vulkan --target llama-cli -j4`) removes the abort;
the binary now loads and decodes on V3D at the expected ~6.2 t/s (ngl 6).
Lesson recorded for every future claim: with a static ggml, name the binary
you tested AND check it postdates the patches
(`strings bin/<tool> | grep -c "Shared memory size too small"` must be 0).

**Layer 2 — what the rebuild exposed.** llama-cli at bb28c1f is a rewritten
frontend over the llama-server slot machinery (verbose log shows
`slot get_availabl ... selected by LRU`, prompt cache, non-unified KV,
graph reservations with n_outputs=16/512), while llama-completion is the
classic decode loop — that is the real "different path". With the patched
backend, llama-cli output is garbage at ANY ngl (even 1) on BOTH v3dv and
llvmpipe (`GGGGGGGG` / `-Identifier-Identifier`), and on pure CPU (-ngl 0)
it generates ZERO tokens (`Generation: 0.0 t/s`, empty reply, two different
prompts, temp 0, `-st < /dev/null`). Control: llama-server — the SAME slot
infrastructure, same build — is coherent on CPU (" Paris. France is known
for its art") and partially degraded on V3D at ngl 6 (" Paris cityscape
amidst clouds/cloud/cloud"). So: the slot infra is fine, the cli frontend
is broken at this commit independent of any GPU, and its Vulkan incoherence
is the same driver-independent upstream composition bug already documented
(server-style graphs just trigger it at lower ngl than llama-completion's).
It cannot be "fixed the same way" because nothing V3D-side is wrong: per-op
NMSE green, slot-infra-on-CPU green, stock llvmpipe reproduces. The fix is
an upstream llama.cpp update/bisect.

Verified entry points on this build, in order of trust: `llama-completion`,
`llama-bench` (coherent/verified at ngl<=6); `llama-server` (works, quality
degrades earlier than llama-completion on GPU); `llama-cli` (do not use at
this commit, broken even on CPU).

Repro:
- `strings build-vulkan/bin/llama-cli | grep -c "Shared memory size too small"` → 0 after rebuild, 2 before.
- cli GPU garbage: `GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1 ./llama-cli -m <granite Q4_0> -ngl 1 -c 4096 -fa 0 -n 8 --temp 0 -st -p "The capital of France is" < /dev/null` → "GGGGGGGG".
- cli CPU empty: same with `-ngl 0` → empty reply, 0.0 t/s.
- server CPU coherent: `./llama-server -m <model> -ngl 0 -c 4096 -fa 0 --port 8081` then `curl -s localhost:8081/completion -d '{"prompt": "The capital of France is", "n_predict": 8, "temperature": 0}'` → " Paris. France is known for its art".

## Win (GEMM FSM round 2, kept): fewer accumulators -> 12.56 -> 13.42 GFLOP/s

The register-allocator insight transfers to the standalone GEMM. Diagnostic
first (MESA_SHADER_CACHE_DISABLE=true V3D_DEBUG=perf ./vkgemm_nmse): the
12.56 shader fails RA at 2 threads and lands on 'disable loop unrolling'.
Collapsing its 16 partial accumulators (4 software-ILP chains per output)
to 4 direct accumulators freed ~12 registers: 13.42 GFLOP/s @ SZ=512,
NMSE 4.5e-14 (also 13.08 @ 256, 13.43 @ 1024). Kept via the --explore FSM
(app 74876a65, exp 1). `best_gemm.comp` in this repo = the kept shader;
`/root/v3d-research/gemm.comp` on the Pi matches and re-measures 13.42.
Repro: `cd /root/v3d-research && glslangValidator -V gemm.comp -o gemm.spv && ./vkgemm_nmse`.

## Dead ends (GEMM FSM round 2, reverted, both verified-correct)

- Transposed k-major vec4 B tile (fewer shared reads per FLOP): 8.60
  GFLOP/s, a 36% LOSS. The old layout had all 16 lanes of a subgroup
  reading adjacent elements; the transpose strides lanes 8 vec4s apart and
  makes the loader's global reads non-coalesced. V3D lesson: per-subgroup
  lane contiguity in shared memory beats per-thread TMU-op count.
- BK 32 -> 64 (half the barriers, full 16 KB shared): 12.57 GFLOP/s, a 6%
  loss. Occupancy (2 resident workgroups at 8 KB each) is worth more than
  halving barrier count. Quirk: v3d_explore's log_variant does NOT restore
  the best shader to disk after a revert (v3d_llama_mmv does) — restore
  gemm.comp from best_gemm.comp manually after a session.

## Audit: which llama.cpp shaders still hit the RA fallback ladder (cold-cache, model shapes)

`MESA_SHADER_CACHE_DISABLE=true V3D_DEBUG=perf test-backend-ops --test-file granite_ops.txt`:
- RMS_NORM: full ladder through 'disable TMU pipelining' (rms_norm.comp has
  six [[unroll]]s) — next de-unroll candidate, same recipe as mul_mat_vec.
- f16 kq matvec: deepest fallback observed, 'lower thread count' — the
  attention pipelines are the most register-starved.
- ROPE and SOFT_MAX: compile clean, no fallback; no hidden win there.
All 43/43 still pass; these are performance candidates, not correctness bugs.

## Breakthrough: the mm (tiled matmul, n>1) path WORKS on V3D

The famous "Shared memory size too small for matrix multiplication" abort
was an over-broad device-init check: it loops over ALL quant types and
throws because iq1's 12 KB LUT overflows 16 KB. The mainstream s-warptile
needs only (32+32)x(32+1)x4 = 8448 bytes with fp32 staging — it fits. With
the per-type disable (commit 428df1d) plus the refined column gate (reject
only ncols 2..8, the pathological mul_mat_vec variants; let n>8 route to
mm), `MUL_MAT n=9` compiles through the ladder (several minutes one-time,
then Mesa-disk-cached) and passes NMSE for f32, q4_0, q6_K, f16 (quantized
types reuse the f32 mm via dequant, so they warm instantly). The l and m
warptiles genuinely do not fit (33792 / 16896 bytes) and stay disabled.
This un-blocks GPU prompt processing and, in principle, whisper.cpp's
encoder (all n~1500 GEMMs). Repro:
`GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1 ./test-backend-ops test -b Vulkan0 -o "MUL_MAT(type_a=f32,type_b=f32,m=16,n=9,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1)"`
(first run ~7 min while the ladder runs; subsequent runs instant).

With the mm path enabled (`GGML_VK_ALLOW_MM=1`, commit 3f2dcc1 makes it
opt-in) the full granite model-graph gate is **49/49 passed** (was 43/43 +
11 unsupported): the n=512 q4_0/q6_K prompt matmuls run and verify ON THE
GPU. Remaining principled CPU fallbacks: 2 FLASH_ATTN (env-gated), the 2
permuted-f16 kq/kqv n=512 layouts (the numerically broken ones — correctly
rejected by contiguity checks), 1 oversized SOFT_MAX. First cold run of
this gate takes ~36 min (new pipeline ladders plus a 210-GFLOP CPU
reference for the q6_K output head). Do not run two GPU jobs at once when
timing anything.

BUT: composed into a real forward pass the mm path corrupts decode even at
-ngl 6 with the short prompt (temp-0 output "/?????" instead of "Paris").
Per-op green + composition garbage = the same upstream defect that breaks
full offload, engaged by the larger GPU graphs. Hence opt-in only; the
default (GGML_VK_MMV_MAX_COLS=1 alone) rejects all n>1 and restores the
verified decode (re-measured tg32 = 5.51 ± 0.02 after the change).

## Envelope correction: the coherent-decode claim only covers SHORT prompts

A ~30-token prompt at -ngl 6 decodes to garbage in EVERY configuration of
this llama.cpp commit (mm on or off): temp-0 reply to the Eiffel-Tower
geography prompt is "GGGG..." while "The capital of France is" gives
"...city named Paris". The session-1 verification used only the short
prompt, so the honest statement is: decode is coherent at ngl<=6 for short
prompts; the upstream composition bug scales with total per-graph GPU work
(tokens x layers), not layer count alone. Any usability claim must name the
prompt length. Repro: same llama-completion command, prompt
"Geography quiz. France is a country in Europe. Its capital city, famous
for the Eiffel Tower and the Louvre, is called", -n 10 --temp 0.

## whisper.cpp on V3D: stood up, builds, runs — output blocked by the same upstream bug

Deployed to /root/v3d-research/whisper.cpp (master, shallow clone), with
the vendored ggml REPLACED by the patched llama.cpp ggml (API-compatible;
`rm -rf ggml && cp -r ../llama.cpp/ggml ggml`, then
`cmake -B build-vulkan -DGGML_VULKAN=1` + build whisper-cli). tiny.en
downloaded via models/download-ggml-model.sh.
- CPU (`-ng`): correct JFK transcript, encode 985 ms. Repro:
  `./build-vulkan/bin/whisper-cli -m models/ggml-tiny.en.bin -f samples/jfk.wav -ng`
- Vulkan (`GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1`, mm path
  active for the encoder GEMMs): runs end-to-end but the transcript is
  repetition garbage ("WITH WITH WITH... anananan") — the encoder is
  all-GPU, i.e. the full-offload composition regime. Attribution to the
  upstream bug is inferred from the identical signature and shared ggml
  code, NOT llvmpipe-proven for whisper specifically (llama's llvmpipe
  proof is in an earlier entry). First-run encode wall time (875 s) is
  meaningless: pipeline-ladder compiles plus GPU contention with a
  concurrently running gate.
Conclusion: whisper GPU is one upstream fix away, not blocked by V3D. The
per-op foundation (256-cap, mm tiles, gates) is already in its tree.

## Deliverable: pi/ is a self-contained deploy kit, proven end-to-end

`pi/setup-llama-v3d.sh` takes a bare Pi 5 from clone to GPU llama.cpp:
shallow-fetches the pinned commit (bb28c1fe246b), `git am`s the three
patches, builds with `-DLLAMA_BUILD_UI=OFF` (the embedded server web UI
needs a prebuilt bundle absent from the tree — the API works without it),
and builds the standalone GEMM demo. Proven by running it in a fresh
directory on the Pi: SETUP-EXIT=0, GEMM 13.42 GFLOP/s correct=yes,
llama-bench tg32 = 5.54 ± 0.02, temp-0 "Paris" with the kit-built
llama-completion. `pi/README.md` states the verified envelope. Caveat
learned during the proof: benchmark numbers taken while another build
saturates the cores read ~3x low (1.85 t/s) — always bench solo.

## Upstream status (2026-07-07 lookup): nobody is fixing our bug; the known-good endpoint was a mirage

- llama.cpp master (bec4772, July) still decodes garbage at -ngl 99 on
  llvmpipe — the composition bug is alive upstream.
- The open upstream reports with our GPU profile (int dot 0, no coopmat):
  #20029 / #20465 bisect to aa6f918c (Scalar Flash Attention Refactor),
  which postdates bb28c1f and is NOT in our tree — theirs is a different,
  later regression in the same neglected path.
- Mesa 26.0/26.1 relnotes contain no v3d compiler work: upgrading Mesa
  will not fix the unbounded register-allocation compile.
- The 871b0b7-was-good assumption does not hold for this bug: only 35
  commits (2 Vulkan: snake fusion, IM2COL) separate it from bb28c1f, none
  plausibly relevant, so the defect predates the range. The old "good"
  evidence was TinyLlama + the 49-file patch set, a different model and
  different shaders. Next discriminator (running): SmolLM2-360M (vanilla
  llama arch) at -ngl 99 on llvmpipe — coherent means the bug is
  arch-specific to granite/lfm2-style graphs; garbage means the
  non-coopmat path is generically broken and old.
- VERDICT: garbage. SmolLM2-360M-Instruct Q8_0 (vanilla llama arch, the
  most CI-exercised path) also decodes to garbage at -ngl 99 on llvmpipe
  ("user user user assistant..."). The composition bug is generic to the
  non-coopmat Vulkan path, model-agnostic, and predates our whole range.
  Likely why it survives: CI runs per-op tests on llvmpipe, not
  end-to-end generation, and real-GPU users are mostly on coopmat paths.
  Decision: stop the archaeology; the evidence is packaged for upstream
  in `upstream-report-draft.md` (llvmpipe two-command repro, three
  architectures, both May and July commits, falsified-knob list). Not
  posted anywhere — review and post it as an issue/comment on #20029.

## Flood stencil (bonbibi): the optimization playbook does not transfer, by measurement

flux.comp/height.comp compile with ZERO fallback-ladder lines
(MESA_SHADER_CACHE_DISABLE=true V3D_DEBUG=perf ./vkflood): they are too
small to stress the register allocator, so the de-unroll family of wins
has nothing to act on — which also explains why the earlier flood
optimization session measured flat. Current: 256^2 grid, 400 steps in
0.383 s (1045 steps/s, 1.51 GFLOP/s, mass conserved). The only real lever
is TMU-op reduction (pack terrain+water into one vec2 buffer: flux drops
10 loads to 5), which needs a vkflood.cpp buffer-layout change — not
worth it, flood speed is not a Bonbibi bottleneck (its gaps are physics
and data plumbing per its own README).

## Sonnet independent verification (2026-07-04): mostly confirmed, one real gap found

Reproduced independently: `llama-bench -ngl 6` gives tg32 ≈ 5.5 t/s (matches
5.55-5.56 claim). `llama-completion` with the documented flags
(`-ngl 6 -c 4096 -fa 0 --temp 0`) loads and generates coherently ("the
capital of France is city named Paris", 5.48 t/s eval) — the end-to-end
claim holds.

One real gap, not previously flagged: **`llama-cli` fails where
`llama-completion` succeeds**, same model/flags/env vars/`-ngl`, including
with `-fa 0` explicit on the CLI (not just the env var). `llama-cli` throws
"Shared memory size too small for matrix multiplication" at model load;
`llama-completion` does not. Not yet root-caused — worth understanding
before calling this generally "usable," since `llama-cli` is the standard
interactive entry point most people (and any future Bonbibi integration)
would reach for first.
