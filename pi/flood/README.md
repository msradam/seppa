# Optimized V3D flood stencil (bonbibi's GPU flood sim)

Fused, strip-mined variant of bonbibi's two-pass WCA2D flood kernels for
the Pi 5 V3D GPU. **1.59x faster at 256², identical physics**, verified by
the same three gates as the original harness: NMSE vs a double-precision
CPU reference, mass conservation vs rain injected, and basin pooling.

## Results (Pi 5, V3D 7.1.10.2, 400 steps)

| grid  | original (flux+height) | fused2s | speedup |
|-------|------------------------|---------|---------|
| 256²  | 0.297 s (1.94 GF/s, 1,345 steps/s) | 0.187 s (3.08 GF/s, 2,140 steps/s) | 1.59x |
| 512²  | 1.128 s (2.04 GF/s) | 0.801 s (2.88 GF/s) | 1.41x |
| 1024² | 4.701 s (1.96 GF/s) | 3.976 s (2.32 GF/s) | 1.18x |

Long-horizon check: 4,000 steps at 256² — NMSE 1.3e-9, mass conserved,
pools in basin.

## What made it faster (measured, not assumed)

The original kernels are bound by fixed per-invocation cost, not by
memory-op count, bytes, dispatches, or barriers — each of those was
falsified by a variant in this directory:

- `flux2.comp`/`height2.comp` (packed (water, surface) vec2 state, half
  the flux-pass loads): 1.93 GF/s — no change. Op count is not the bound.
- `fused.comp` (one dispatch per step, flux buffer eliminated): 2.04 GF/s
  — +5%. Traffic and barriers are not the bound.
- `flux2s`/`height2s` (2 cells per invocation): 2.40 GF/s — +23%.
  Per-invocation overhead is the bound.
- `flux4s`/`height4s` (4 cells per invocation): 2.35 GF/s — register
  pressure eats the gain; 2 is the sweet spot.
- **`fused2s.comp` (fused + 2 cells per invocation): 3.08 GF/s** — the
  wins stack. This is the shipped kernel.

Vertical strips keep the 16 lanes of a V3D subgroup on adjacent x
addresses (lane contiguity matters more than load count on this GPU).

## Run

```
g++ -O3 -o vkflood2 vkflood2.cpp -lvulkan
glslangValidator -V fused2s.comp -o fused2s.spv
STRIP=2 FUSED=1 FLUX_SPV=fused2s.spv ./vkflood2 256 400
```

Expect all three gate lines to say yes. `DEM=<file>` loads real terrain
exactly as the original harness does; `sim` as the third argument skips
the CPU verification for production runs.

## Integrating into bonbibi

Three host-side changes relative to bonbibi's `vkflood.cpp` (all visible
in `vkflood2.cpp`):

1. Water state is a vec2 buffer `P = (W, H+W)` instead of a float buffer;
   initialize `P[i] = (0, H[i])` and read depth as `P[2*i]`.
2. One pipeline (`fused2s.spv`) and one dispatch+barrier per step instead
   of two; the flux buffer and its binding go away (binding 2 unused).
3. Dispatch y is halved: `Gy = (SZ + 31) / 32` with the same 16x16
   workgroup.

## Known headroom

`fused2s.comp` still compiles through the v3dv register-allocator fallback
ladder (the CSE'd neighbourhood loads hold ~18 vec2 values live). Trading
some reloads for registers, or a shared-memory tile, might buy more; the
seppa flood FSM (`v3d_flood_opt.py`, physics-gated) is the harness for
that search.
