# llama.cpp on the Raspberry Pi 5 GPU (V3D)

Drop-in kit that builds GPU-accelerated llama.cpp for the Pi 5's V3D GPU.
Stock llama.cpp Vulkan either aborts ("Shared memory size too small") or
hangs on this device; the patch series here fixes both and adds an
FSM-optimized matvec shader. Every GPU-computed value is verified against
llama.cpp's CPU backend with its own NMSE thresholds (`test-backend-ops`).

## What you get

- `llama-completion`, `llama-bench`, `llama-server` with working V3D
  decode: ~5.5 t/s token generation on granite-4.0-1b Q4_0 at `-ngl 6`
  (Pi 5, Mesa 25.0.7). CPU-only decode on the same board is ~10.9 t/s, so
  the value is co-processing: the GPU decodes while the CPU does other
  work.
- `vkgemm_nmse` + `gemm.comp`: standalone 13.42 GFLOP/s SGEMM
  (43.7 GFLOP/s roofline) with a double-precision CPU-reference
  correctness gate.

## Install

On the Pi 5 (64-bit OS, `mesa-vulkan-drivers` installed):

```
sudo apt install git cmake g++ glslc glslang-tools libvulkan-dev
./setup-llama-v3d.sh ~/llama-v3d
```

Builds take 20-30 minutes. The script clones llama.cpp at the verified
commit, applies the three patches, builds, and prints usage.

## Run

Every GPU run needs:

```
export GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1
```

```
~/llama-v3d/llama.cpp/build-vulkan/bin/llama-completion \
  -m granite-4.0-1b-Q4_0.gguf -ngl 6 -c 4096 -fa 0 --temp 0 \
  -p "The capital of France is"
```

The first run compiles GPU pipelines through the driver's slow register
allocator (minutes); results are disk-cached afterwards.

## Verified envelope — read this before relying on output

- Coherent, NMSE-verified: `-ngl 6`, short prompts, via
  `llama-completion`, `llama-bench`, or `llama-server`.
- NOT verified / known broken (upstream ggml-vulkan defect, reproducible
  on llvmpipe, not V3D-specific): `-ngl` above 6, prompts beyond ~25
  tokens, `GGML_VK_ALLOW_MM=1` (GPU prompt processing; per-op correct,
  composed output wrong), and `llama-cli` (broken at this commit even on
  CPU — use `llama-completion`).

To re-verify numerics on your board:

```
bin/export-graph-ops -m model.gguf -fa off -o ops.txt
GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1 \
  bin/test-backend-ops test -b Vulkan0 --test-file ops.txt
```

Expect every listed op to pass or report "not supported" (deliberate CPU
fallback); any FAIL means the GPU computed wrong numbers on your setup.

## Files

- `setup-llama-v3d.sh` — one-shot build script
- `llama-cpp-v3d-fixes.patch` — cap all >256-invocation workgroups (V3D
  executes them silently wrong), gate the never-terminating multi-column
  matvec pipeline compiles, gate flash attention (driver compiler abort)
- `llama-cpp-v3d-mmv-deunroll.patch` — de-unrolled `mul_mat_vec` shader:
  +28% decode (4.32 to 5.55 t/s); manual unrolling forces the v3dv
  register allocator off its best strategy
- `llama-cpp-v3d-mm-path.patch` — opt-in tiled-matmul path
  (`GGML_VK_ALLOW_MM=1`), per-op verified, blocked for real use by the
  upstream defect above
- `gemm-best.comp`, `vkgemm_nmse.cpp` — standalone SGEMM demo
- `gemm.comp` — original 7.02 GFLOP/s baseline shader, kept for reference
