#!/bin/bash
# The CPU-only counterfactual for the concurrency study: the same WCA2D
# flood update run on the Cortex-A76 cores (cpuflood.cpp, OpenMP) instead
# of the V3D GPU, alone and sharing the four cores with LLM decode two
# ways: oversubscribed (4 flood threads + 4 decode threads) and
# partitioned (flood pinned to core 0, decode pinned to cores 1-3).
# Same cooldown gates and 1 Hz thermal sampling as concurrency_bench.sh.
#
#   OUT=/tmp/cpu_flood_bench bash cpu_flood_bench.sh
set -u
RES=/root/v3d-research
cd "$RES" || exit 1
BIN=$RES/llama.cpp/build-vulkan/bin
M=$(find "$RES/models" -iname 'granite-4.0-1b-Q4_0.gguf' | head -1)
OUT=${OUT:-/tmp/cpu_flood_bench}
STEPS=${STEPS:-4000}
mkdir -p "$OUT"

g++ -O3 -march=native -fopenmp -o cpuflood cpuflood.cpp

stamp() { date +%s.%N; }
therm() { echo "temp=$(vcgencmd measure_temp) throttled=$(vcgencmd get_throttled)"; }

cooldown() {
    local deadline=$((SECONDS + 180))
    while [ "$SECONDS" -lt "$deadline" ]; do
        t=$(vcgencmd measure_temp | grep -o '[0-9.]*' | head -1)
        awk -v t="$t" 'BEGIN{exit !(t<55)}' && break
        sleep 5
    done
    echo "cooldown_exit_temp=$t"
}

thermal_sampler() {
    while :; do echo "$(stamp) $(therm)"; sleep 1; done >> "$1" 2>&1
}

therm > "$OUT/thermal_start.log"

echo "== [1/5] physics gates (float vs double reference, 400 steps, 1t and 4t)"
OMP_NUM_THREADS=1 ./cpuflood 256 400 > "$OUT/gates_1t.log" 2>&1
OMP_NUM_THREADS=4 ./cpuflood 256 400 > "$OUT/gates_4t.log" 2>&1
grep -h '=yes\|=NO\|=no' "$OUT"/gates_*.log

echo "== [2/5] CPU flood alone (1, 2, 4 threads; 3x each, $STEPS steps)"
for T in 1 2 4; do
    cooldown
    for _ in 1 2 3; do echo "t=$(stamp)"; OMP_NUM_THREADS=$T ./cpuflood 256 "$STEPS" sim; done \
        > "$OUT/flood_cpu${T}t_alone.log" 2>&1
done

echo "== [3/5] decode alone (4 threads and 3 threads)"
cooldown
( thermal_sampler "$OUT/thermal_llama_alone.log" ) &
TSPID=$!
"$BIN/llama-bench" -m "$M" -ngl 0 -t 4 -p 0 -n 64 -r 5 > "$OUT/llama_alone.log" 2>&1
kill "$TSPID" 2>/dev/null
cooldown
( thermal_sampler "$OUT/thermal_llama_t3_alone.log" ) &
TSPID=$!
taskset -c 1-3 "$BIN/llama-bench" -m "$M" -ngl 0 -t 3 -p 0 -n 64 -r 5 > "$OUT/llama_t3_alone.log" 2>&1
kill "$TSPID" 2>/dev/null

run_concurrent() { # $1 tag, $2 flood threads, $3 flood cpuset, $4 llama threads, $5 llama cpuset
    local floodlog="$OUT/flood_$1_conc.log" llamalog="$OUT/llama_vs_$1.log"
    cooldown
    : > "$floodlog"
    ( thermal_sampler "$OUT/thermal_$1_conc.log" ) &
    local tspid=$!
    ( while :; do echo "t=$(stamp)"; OMP_NUM_THREADS=$2 taskset -c "$3" ./cpuflood 256 "$STEPS" sim; done >> "$floodlog" 2>&1 ) &
    local pid=$!
    sleep 3
    echo "llama_start=$(stamp) $(therm)" >> "$floodlog"
    taskset -c "$5" "$BIN/llama-bench" -m "$M" -ngl 0 -t "$4" -p 0 -n 64 -r 5 > "$llamalog" 2>&1
    echo "llama_end=$(stamp) $(therm)" >> "$floodlog"
    kill "$pid" "$tspid" 2>/dev/null
    wait "$pid" 2>/dev/null
    pkill -f 'cpufloo[d]' 2>/dev/null
    true
}

echo "== [4/5] oversubscribed: flood 4t + decode 4t on 4 cores"
run_concurrent cpu4over 4 0-3 4 0-3

echo "== [5/5] partitioned: flood 1t on core 0 + decode 3t on cores 1-3"
run_concurrent cpu1part 1 0 3 1-3

therm > "$OUT/thermal_end.log"
echo "DONE: logs in $OUT"
grep -H 'time=' "$OUT"/flood_*_alone.log | tail -9
grep -H 'tg' "$OUT"/llama_*.log
