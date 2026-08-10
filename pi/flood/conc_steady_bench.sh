#!/bin/bash
# Steady-state variant of concurrency_bench.sh. Every measured phase runs
# after a decode soak so the board reaches its thermally governed
# equilibrium before the window opens; decode windows are long enough for
# n >= 10 flood completions per kernel (llama-bench tg64, r=20). The alone
# flood runs stay cool-start (they finish before heat accumulates).
#
#   OUT=/tmp/conc_steady bash conc_steady_bench.sh
set -u
RES=/root/v3d-research
cd "$RES" || exit 1
BIN=$RES/llama.cpp/build-vulkan/bin
M=$(find "$RES/models" -iname 'granite-4.0-1b-Q4_0.gguf' | head -1)
OUT=${OUT:-/tmp/conc_steady}
STEPS=${STEPS:-4000}
SOAK_R=${SOAK_R:-8}
MEAS_R=${MEAS_R:-20}
mkdir -p "$OUT"

glslangValidator -V fused2s.comp -o fused2s.spv
glslangValidator -V flux2.comp -o flux2.spv
glslangValidator -V height2.comp -o height2.spv

flood_opt()  { STRIP=2 FUSED=1 FLUX_SPV=fused2s.spv ./vkflood2 256 "$STEPS" sim; }
flood_orig() { STRIP=1 FLUX_SPV=flux2.spv HEIGHT_SPV=height2.spv ./vkflood2 256 "$STEPS" sim; }
stamp() { date +%s.%N; }
therm() { echo "temp=$(vcgencmd measure_temp) throttled=$(vcgencmd get_throttled)"; }
cooldown() {
    local deadline=$((SECONDS + 240))
    while [ "$SECONDS" -lt "$deadline" ]; do
        t=$(vcgencmd measure_temp | grep -o '[0-9.]*' | head -1)
        awk -v t="$t" 'BEGIN{exit !(t<55)}' && break
        sleep 5
    done
}
sampler() { while :; do echo "$(stamp) $(therm)"; sleep 1; done >> "$1" 2>&1; }
bench() { GGML_VK_VISIBLE_DEVICES=99 "$BIN/llama-bench" -m "$M" -ngl 0 -t 4 -p 0 -n 64 -r "$1"; }

therm > "$OUT/thermal_start.log"

echo "== flood alone (3x each kernel, cool start)"
cooldown
for _ in 1 2 3; do echo "t=$(stamp)"; flood_opt;  done > "$OUT/flood_opt_alone.log" 2>&1
cooldown
for _ in 1 2 3; do echo "t=$(stamp)"; flood_orig; done > "$OUT/flood_orig_alone.log" 2>&1

echo "== decode alone, soaked then measured"
cooldown
( sampler "$OUT/thermal_llama_alone.log" ) & TS=$!
bench "$SOAK_R" > "$OUT/llama_alone_soak.log" 2>&1
bench "$MEAS_R" > "$OUT/llama_alone.log" 2>&1
kill "$TS" 2>/dev/null

run_concurrent() { # $1 = opt|orig
    local floodlog="$OUT/flood_$1_conc.log" llamalog="$OUT/llama_vs_$1.log"
    cooldown
    : > "$floodlog"
    ( sampler "$OUT/thermal_$1_conc.log" ) & local ts=$!
    ( while :; do echo "t=$(stamp)"; "flood_$1"; done >> "$floodlog" 2>&1 ) & local fp=$!
    sleep 2
    bench "$SOAK_R" > "$OUT/llama_vs_$1_soak.log" 2>&1
    echo "llama_start=$(stamp) $(therm)" >> "$floodlog"
    bench "$MEAS_R" > "$llamalog" 2>&1
    echo "llama_end=$(stamp) $(therm)" >> "$floodlog"
    kill "$fp" "$ts" 2>/dev/null
    wait "$fp" 2>/dev/null
    pkill -f 'vkfloo[d]2' 2>/dev/null
    true
}

echo "== concurrent optimized (soaked)"
run_concurrent opt
echo "== concurrent original (soaked)"
run_concurrent orig

therm > "$OUT/thermal_end.log"
echo "DONE: logs in $OUT"
