#!/bin/bash
# CPU/GPU interference measurement for the bonbibi split: GPU flood stencil
# vs CPU LLM decode (Granite 4.0 1B, llama-bench -ngl 0) and CPU router.
# Runs on the Pi in /root/v3d-research. Results land in $OUT as raw logs;
# every number in the paper's concurrency table comes from these files.
#
#   OUT=/tmp/conc_bench bash concurrency_bench.sh
set -u
RES=/root/v3d-research
cd "$RES" || exit 1
BIN=$RES/llama.cpp/build-vulkan/bin
M=$(find "$RES/models" -iname 'granite-4.0-1b-Q4_0.gguf' | head -1)
OUT=${OUT:-/tmp/conc_bench}
STEPS=${STEPS:-4000}
mkdir -p "$OUT"

glslangValidator -V fused2s.comp -o fused2s.spv
glslangValidator -V flux2.comp -o flux2.spv
glslangValidator -V height2.comp -o height2.spv

flood_opt()  { STRIP=2 FUSED=1 FLUX_SPV=fused2s.spv ./vkflood2 256 "$STEPS" sim; }
flood_orig() { STRIP=1 FLUX_SPV=flux2.spv HEIGHT_SPV=height2.spv ./vkflood2 256 "$STEPS" sim; }

stamp() { date +%s.%N; }
therm() { echo "temp=$(vcgencmd measure_temp) throttled=$(vcgencmd get_throttled)"; }

# The soft temperature limit engages well below the 80C hard throttle and
# quietly caps clocks, confounding contention with heat. Gate every phase on
# a cool start and sample temp+throttle at 1 Hz through the concurrent runs
# so tainted trials are identifiable (same policy as v3d_coproc.py).
cooldown() {
    local deadline=$((SECONDS + 180))
    while [ "$SECONDS" -lt "$deadline" ]; do
        t=$(vcgencmd measure_temp | grep -o '[0-9.]*' | head -1)
        awk -v t="$t" 'BEGIN{exit !(t<55)}' && break
        sleep 5
    done
    echo "cooldown_exit_temp=$t"
}

thermal_sampler() { # $1 = logfile; runs until killed
    while :; do echo "$(stamp) $(therm)"; sleep 1; done >> "$1" 2>&1
}

therm > "$OUT/thermal_start.log"

echo "== [1/6] flood alone (3x each kernel, $STEPS steps)"
cooldown
for _ in 1 2 3; do echo "t=$(stamp)"; flood_opt;  done > "$OUT/flood_opt_alone.log" 2>&1
for _ in 1 2 3; do echo "t=$(stamp)"; flood_orig; done > "$OUT/flood_orig_alone.log" 2>&1

echo "== [2/6] llama decode alone (tg64, 5 reps, 4 threads, CPU only)"
cooldown
( thermal_sampler "$OUT/thermal_llama_alone.log" ) &
TSPID=$!
"$BIN/llama-bench" -m "$M" -ngl 0 -t 4 -p 0 -n 64 -r 5 > "$OUT/llama_alone.log" 2>&1
kill "$TSPID" 2>/dev/null

echo "== [3/6] router alone"
if [ -f route.py ] && [ -f /tmp/flood_depth.txt ]; then
    { time python3 route.py 0.5 > /dev/null; } 2> "$OUT/route_alone.log"
fi

run_concurrent() { # $1 = kernel name (opt|orig)
    local floodlog="$OUT/flood_$1_conc.log" llamalog="$OUT/llama_vs_$1.log"
    cooldown
    : > "$floodlog"
    ( thermal_sampler "$OUT/thermal_$1_conc.log" ) &
    local tspid=$!
    ( while :; do echo "t=$(stamp)"; "flood_$1"; done >> "$floodlog" 2>&1 ) &
    local pid=$!
    sleep 3
    echo "llama_start=$(stamp) $(therm)" >> "$floodlog"
    "$BIN/llama-bench" -m "$M" -ngl 0 -t 4 -p 0 -n 64 -r 5 > "$llamalog" 2>&1
    echo "llama_end=$(stamp) $(therm)" >> "$floodlog"
    kill "$pid" "$tspid" 2>/dev/null
    wait "$pid" 2>/dev/null
    pkill -f 'vkfloo[d]2' 2>/dev/null
    true
}

echo "== [4/6] concurrent: optimized flood loop + llama decode"
run_concurrent opt
echo "== [5/6] concurrent: original flood loop + llama decode"
run_concurrent orig

echo "== [6/6] router under optimized flood load"
if [ -f route.py ] && [ -f /tmp/flood_depth.txt ]; then
    ( while :; do flood_opt > /dev/null 2>&1; done ) &
    pid=$!
    sleep 2
    { time python3 route.py 0.5 > /dev/null; } 2> "$OUT/route_conc.log"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    pkill -f 'vkfloo[d]2' 2>/dev/null
fi

therm > "$OUT/thermal_end.log"
echo "DONE: logs in $OUT"
grep -H 'time=' "$OUT"/flood_*_alone.log | tail -6
grep -H 'tg' "$OUT"/llama_*.log
