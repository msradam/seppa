#!/bin/bash
# Steady-state CPU-only counterfactual: same soak-then-measure protocol as
# conc_steady_bench.sh. Decode phases soak with r=8 then measure with r=20;
# CPU flood alones are cool-start windows.
#
#   OUT=/tmp/cpu_steady bash cpu_steady_bench.sh
set -u
RES=/root/v3d-research
cd "$RES" || exit 1
BIN=$RES/llama.cpp/build-vulkan/bin
M=$(find "$RES/models" -iname 'granite-4.0-1b-Q4_0.gguf' | head -1)
OUT=${OUT:-/tmp/cpu_steady}
STEPS=${STEPS:-4000}
SOAK_R=${SOAK_R:-8}
MEAS_R=${MEAS_R:-20}
mkdir -p "$OUT"

g++ -O3 -march=native -fopenmp -o cpuflood cpuflood.cpp 2> "$OUT/build.log" || true
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
bench() { # $1 reps, $2 threads, $3 cpuset
    GGML_VK_VISIBLE_DEVICES=99 taskset -c "$3" "$BIN/llama-bench" -m "$M" -ngl 0 -t "$2" -p 0 -n 64 -r "$1"
}

echo "== CPU flood alone, 1/2/4 threads (cool, 3 runs each)"
for T in 1 2 4; do
    cooldown
    for _ in 1 2 3; do echo "t=$(stamp)"; OMP_NUM_THREADS=$T ./cpuflood 256 "$STEPS" sim; done \
        > "$OUT/flood_cpu${T}t_alone.log" 2>&1
done

echo "== decode alone, 3 threads pinned, soaked"
cooldown
( sampler "$OUT/thermal_llama3t.log" ) & TS=$!
bench "$SOAK_R" 3 1-3 > "$OUT/llama_t3_soak.log" 2>&1
bench "$MEAS_R" 3 1-3 > "$OUT/llama_t3_alone.log" 2>&1
kill "$TS" 2>/dev/null

echo "== partitioned: flood 1t core 0 + decode 3t cores 1-3, soaked"
cooldown
( sampler "$OUT/thermal_part.log" ) & TS=$!
FLOG="$OUT/flood_cpu1part_conc.log"; : > "$FLOG"
( while :; do echo "t=$(stamp)"; OMP_NUM_THREADS=1 taskset -c 0 ./cpuflood 256 "$STEPS" sim; done >> "$FLOG" 2>&1 ) & FP=$!
sleep 2
bench "$SOAK_R" 3 1-3 > "$OUT/llama_vs_part_soak.log" 2>&1
echo "llama_start=$(stamp) $(therm)" >> "$FLOG"
bench "$MEAS_R" 3 1-3 > "$OUT/llama_vs_cpu1part.log" 2>&1
echo "llama_end=$(stamp) $(therm)" >> "$FLOG"
kill "$FP" "$TS" 2>/dev/null; wait "$FP" 2>/dev/null; pkill -f 'cpufloo[d]' 2>/dev/null

echo "== oversubscribed: flood 4t + decode 4t, soaked"
cooldown
( sampler "$OUT/thermal_over.log" ) & TS=$!
FLOG="$OUT/flood_cpu4over_conc.log"; : > "$FLOG"
( while :; do echo "t=$(stamp)"; OMP_NUM_THREADS=4 ./cpuflood 256 "$STEPS" sim; done >> "$FLOG" 2>&1 ) & FP=$!
sleep 2
bench "$SOAK_R" 4 0-3 > "$OUT/llama_vs_over_soak.log" 2>&1
echo "llama_start=$(stamp) $(therm)" >> "$FLOG"
bench "$MEAS_R" 4 0-3 > "$OUT/llama_vs_cpu4over.log" 2>&1
echo "llama_end=$(stamp) $(therm)" >> "$FLOG"
kill "$FP" "$TS" 2>/dev/null; wait "$FP" 2>/dev/null; pkill -f 'cpufloo[d]' 2>/dev/null

therm > "$OUT/thermal_end.log"
echo "DONE: logs in $OUT"
