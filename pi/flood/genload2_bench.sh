#!/bin/bash
# GPU flood vs generic CPU loads, repaired: the GPU probe runs in sim mode
# (no CPU-side reference computation), probes loop continuously through the
# whole window, and both sides of every pairing are logged. Window
# boundaries are stamped into every log for offline windowing.
#
#   OUT=/tmp/genload2 bash genload2_bench.sh
set -u
RES=/root/v3d-research
cd "$RES" || exit 1
OUT=${OUT:-/tmp/genload2}
STEPS=${STEPS:-4000}
WINDOW=${WINDOW:-60}
mkdir -p "$OUT"

gpu() { STRIP=2 FUSED=1 FLUX_SPV=fused2s.spv ./vkflood2 256 "$STEPS" sim; }
cpuflood() { OMP_NUM_THREADS=4 ./cpuflood 256 "$STEPS" sim; }
stamp() { date +%s.%N; }
cooldown() {
    local deadline=$((SECONDS + 240))
    while [ "$SECONDS" -lt "$deadline" ]; do
        t=$(vcgencmd measure_temp | grep -o '[0-9.]*' | head -1)
        awk -v t="$t" 'BEGIN{exit !(t<55)}' && break
        sleep 5
    done
}
sampler() { while :; do echo "$(stamp) temp=$(vcgencmd measure_temp) throttled=$(vcgencmd get_throttled)"; sleep 1; done >> "$1" 2>&1; }

pair() { # $1 name, $2 cpu-load command line (empty for none)
    cooldown
    ( sampler "$OUT/$1_temp.log" ) & local ts=$!
    : > "$OUT/$1_gpu.log"
    if [ -n "$2" ]; then
        : > "$OUT/$1_cpu.log"
        ( while :; do echo "t=$(stamp)"; $2; done >> "$OUT/$1_cpu.log" 2>&1 ) & local cp=$!
    fi
    sleep 2
    echo "window_start=$(stamp)" >> "$OUT/$1_gpu.log"
    [ -n "$2" ] && echo "window_start=$(stamp)" >> "$OUT/$1_cpu.log"
    ( while :; do echo "t=$(stamp)"; gpu; done >> "$OUT/$1_gpu.log" 2>&1 ) & local gp=$!
    sleep "$WINDOW"
    echo "window_end=$(stamp)" >> "$OUT/$1_gpu.log"
    [ -n "$2" ] && echo "window_end=$(stamp)" >> "$OUT/$1_cpu.log"
    kill "$gp" 2>/dev/null; wait "$gp" 2>/dev/null
    [ -n "$2" ] && { kill "$cp" 2>/dev/null; wait "$cp" 2>/dev/null; }
    pkill -f 'vkfloo[d]2' 2>/dev/null
    pkill -f 'cpufloo[d]' 2>/dev/null
    kill "$ts" 2>/dev/null
    true
}

echo "== GPU alone"
pair gpu_alone ""

echo "== cpuflood alone (window of runs, no GPU)"
cooldown
( while :; do echo "t=$(stamp)"; cpuflood; done > "$OUT/cpuflood_alone.log" 2>&1 ) & CP=$!
sleep "$WINDOW"
kill "$CP" 2>/dev/null; wait "$CP" 2>/dev/null; pkill -f 'cpufloo[d]' 2>/dev/null

echo "== openssl alone"
cooldown
openssl speed -multi 4 -seconds 15 sha256 > "$OUT/openssl_alone.log" 2>&1

echo "== GPU + openssl"
cooldown
( sampler "$OUT/openssl_pair_temp.log" ) & TS=$!
: > "$OUT/openssl_pair_gpu.log"
openssl speed -multi 4 -seconds "$WINDOW" sha256 > "$OUT/openssl_pair_cpu.log" 2>&1 & OP=$!
sleep 2
echo "window_start=$(stamp)" >> "$OUT/openssl_pair_gpu.log"
( while :; do echo "t=$(stamp)"; gpu; done >> "$OUT/openssl_pair_gpu.log" 2>&1 ) & GP=$!
wait "$OP"
echo "window_end=$(stamp)" >> "$OUT/openssl_pair_gpu.log"
kill "$GP" "$TS" 2>/dev/null; wait "$GP" 2>/dev/null; pkill -f 'vkfloo[d]2' 2>/dev/null

echo "== GPU + cpuflood"
pair cpuflood_pair "cpuflood"

echo "DONE: logs in $OUT"
