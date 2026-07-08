#!/bin/bash
# Build GPU-accelerated llama.cpp for the Raspberry Pi 5 V3D GPU.
#
# Clones llama.cpp at the verified commit, applies the Seppa V3D patch
# series, builds the trusted binaries, and (optionally, PREFETCHED=1 model
# path) runs the verification suite. Produces:
#   <target>/llama.cpp/build-vulkan/bin/{llama-completion,llama-bench,llama-server,test-backend-ops}
#   <target>/vkgemm_nmse + gemm.comp   (standalone 13.42 GFLOP/s SGEMM demo)
#
# Usage:
#   ./setup-llama-v3d.sh [target-dir]          # default ./llama-v3d
#
# Every GPU run needs this environment:
#   GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1
#
# Verified envelope (Pi 5, V3D 7.1.10.2, Mesa 25.0.7):
#   - decode (-ngl 6, short prompts): coherent, ~5.5 t/s on
#     granite-4.0-1b Q4_0; all GPU ops NMSE-checked vs the CPU backend
#   - long prompts / -ngl > 6 / GGML_VK_ALLOW_MM=1: NOT verified coherent
#     (upstream ggml-vulkan defect, reproducible on llvmpipe)
#   - use llama-completion / llama-bench / llama-server; llama-cli is
#     broken upstream at this commit even on CPU
set -euo pipefail

LLAMA_COMMIT=bb28c1fe246b72276ee1d00ce89306be7b865766
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-./llama-v3d}"

for tool in git cmake glslc glslangValidator g++; do
    command -v "$tool" > /dev/null || { echo "missing: $tool (apt install git cmake glslc glslang-tools g++ libvulkan-dev)"; exit 1; }
done
for p in llama-cpp-v3d-fixes.patch llama-cpp-v3d-mmv-deunroll.patch llama-cpp-v3d-mm-path.patch; do
    [ -f "$PATCH_DIR/$p" ] || { echo "missing patch: $PATCH_DIR/$p"; exit 1; }
done

mkdir -p "$TARGET"
TARGET="$(cd "$TARGET" && pwd)"

if [ ! -d "$TARGET/llama.cpp" ]; then
    mkdir -p "$TARGET/llama.cpp"
    cd "$TARGET/llama.cpp"
    git init -q
    git remote add origin https://github.com/ggml-org/llama.cpp.git
    git fetch --depth 1 origin "$LLAMA_COMMIT"
    git checkout -q FETCH_HEAD
fi
cd "$TARGET/llama.cpp"
if ! git log -1 --format=%s | grep -q "mm tile path"; then
    git -c user.name=seppa -c user.email=seppa@local am \
        "$PATCH_DIR/llama-cpp-v3d-fixes.patch" \
        "$PATCH_DIR/llama-cpp-v3d-mmv-deunroll.patch" \
        "$PATCH_DIR/llama-cpp-v3d-mm-path.patch"
fi

# LLAMA_BUILD_UI=OFF: the embedded server web UI needs a prebuilt bundle
# that is not in the source tree; llama-server's HTTP API works without it.
cmake -B build-vulkan -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF
cmake --build build-vulkan -j"$(nproc)" --target \
    llama-completion llama-bench llama-server test-backend-ops export-graph-ops

# Standalone SGEMM demo: the FSM-optimized 13.42 GFLOP/s shader plus its
# NMSE-gated benchmark harness (double-precision CPU reference).
cp "$PATCH_DIR/gemm-best.comp" "$TARGET/gemm.comp"
g++ -O3 -o "$TARGET/vkgemm_nmse" "$PATCH_DIR/vkgemm_nmse.cpp" -lvulkan
glslangValidator -V "$TARGET/gemm.comp" -o "$TARGET/gemm.spv"

cat << DONE

Build complete.

GEMM demo (expect ~13.4 GFLOP/s at SZ=512, correct=yes):
  cd $TARGET && ./vkgemm_nmse

LLM decode (first run compiles GPU pipelines, allow several minutes):
  export GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1
  $TARGET/llama.cpp/build-vulkan/bin/llama-completion \\
    -m <granite-4.0-1b Q4_0 gguf> -ngl 6 -c 4096 -fa 0 --temp 0 \\
    -p "The capital of France is"

Verify GPU numerics against the CPU oracle (model graph, NMSE):
  bin/export-graph-ops -m <model.gguf> -fa off -o ops.txt
  GGML_VK_MMV_MAX_COLS=1 GGML_VK_DISABLE_FLASH_ATTN=1 \\
    bin/test-backend-ops test -b Vulkan0 --test-file ops.txt
DONE
