# Draft upstream report (for ggml-org/llama.cpp, relates to #20029 / #20465)

Not yet posted anywhere. Post as a new issue or a comment on #20029 after
review; it contains only facts verified on our hardware this week.

---

**Title: Eval bug: Vulkan backend produces incoherent output on
non-coopmat devices while every op passes test-backend-ops — reproducible
on llvmpipe (no GPU hardware required)**

## Summary

On Vulkan devices without cooperative matrix support, model output is
incoherent at full offload even though every op the model graph executes
passes `test-backend-ops` NMSE against the CPU backend at the real model
shapes. The defect reproduces on **llvmpipe**, Mesa's software
rasterizer, which removes GPU hardware and vendor drivers from the
equation entirely: any Linux machine can reproduce this with two
commands and no GPU.

## Reproduction (llvmpipe, no GPU needed)

```
cmake -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-completion

GGML_VK_VISIBLE_DEVICES=0 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ./build/bin/llama-completion -m <model.gguf> -ngl 99 -c 2048 -fa 0 \
  -n 8 --temp 0 -p "The capital of France is"
```

- `-ngl 0` (pure CPU): " Paris." — correct.
- `-ngl 99` on llvmpipe: garbage. Observed with three models spanning
  three architectures: granite-4.0-1b Q4_0 ("__$$$____" /
  "GGGGGGGG"), LFM2-1.2B Q4_0, and SmolLM2-360M-Instruct Q8_0
  ("user user user assistant user assistant example").

Verified at commit bb28c1fe246b (May) and at master bec4772 (July) —
present in both, so not a recent regression.

## The same behavior on real hardware (Raspberry Pi 5, V3D/v3dv)

Identical signature on the Pi 5's V3D GPU (`fp16: 0 | int dot: 0 |
matrix cores: none`, the same feature profile as llvmpipe here). Output
quality degrades progressively with the amount of GPU work per graph:
coherent at `-ngl 6` with a short prompt, marginal at 8, garbage at 10+,
and garbage at `-ngl 6` once the prompt exceeds ~25 tokens. It scales
with tokens x layers, not layer count alone.

## Per-op correctness is green — this is a composition defect

Using `export-graph-ops` + `test-backend-ops --test-file` so the exact
model-graph shapes are tested: every op the graph runs on the Vulkan
backend passes NMSE vs the CPU backend (49/49 for the granite graph,
including n=512 batched matmuls, RMS_NORM, ROPE, SOFT_MAX, SET_ROWS,
GET_ROWS at real strides). The corruption only appears when the ops are
composed into a full forward pass.

## Falsified hypotheses (all produce the same garbage)

- `GGML_VK_DISABLE_FUSION=1`
- `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`
- `GGML_VK_PREFER_HOST_MEMORY=1`
- smaller `GGML_VK_SUBALLOCATION_BLOCK_SIZE`
- smaller command buffers (submit every 8 nodes instead of 100)
- flash attention off in all tests (`-fa 0`)
- on v3dv additionally: `V3D_DEBUG=sync` (serialize every job) and
  `V3D_DEBUG=tmu32` (force 32-bit TMU precision)

The output is deterministic and byte-identical across these knobs for a
given configuration, which suggests a deterministic addressing/layout or
graph-construction defect on this path rather than a race.

## Relation to #20029 / #20465

Those reports bisect to aa6f918c (Vulkan Scalar Flash Attention
Refactor). The behavior described here exists at bb28c1fe246b, which
predates aa6f918c, and occurs with flash attention disabled — so it is
either a distinct, older defect on the same non-coopmat path, or #20029's
bisect found a second, newer instance. The llvmpipe reproduction above
should let a maintainer debug either without any of the affected
hardware.
