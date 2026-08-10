#!/bin/bash
# GPU flood concurrency against generic CPU loads (no LLM): openssl sha256
# on 4 threads (compute-bound, cache-resident) and the OpenMP CPU flood on
# 4 threads (memory-bound). Cooldown below 55 C before each phase; 1 Hz
# temperature and throttle-register log per phase. Run ON the Pi:
#   OUT=/tmp/genload_bench bash genload_bench.sh
set -u
R=/root/v3d-research
OUT=${OUT:-/tmp/genload_bench}
STEPS=${STEPS:-4000}
mkdir -p "$OUT"

stamp() { date +%s.%N; }
cool() {
    while :; do
        t=$(vcgencmd measure_temp | grep -o '[0-9.]*')
        awk "BEGIN{exit !($t < 55)}" && break
        sleep 10
    done
}
tlog() {
    while :; do echo "$(date +%s) $(vcgencmd measure_temp) $(vcgencmd get_throttled)"; sleep 1; done
}
gpu_run() {
    (cd "$R" && STRIP=2 FUSED=1 FLUX_SPV=fused2s.spv ./vkflood2 256 "$STEPS")
}

phase() {
    cool
    tlog > "$OUT/$1_temp.log" & TPID=$!
}
endphase() { kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null; }

# 1. GPU alone
phase gpu_alone
for _ in 1 2 3; do echo "t=$(stamp)"; gpu_run; done > "$OUT/gpu_alone.log" 2>&1
endphase

# 2. openssl alone (4 threads, 20 s)
phase openssl_alone
openssl speed -multi 4 -seconds 20 sha256 > "$OUT/openssl_alone.log" 2>&1
endphase

# 3. openssl + GPU concurrent
phase openssl_conc
openssl speed -multi 4 -seconds 45 sha256 > "$OUT/openssl_conc.log" 2>&1 & OPID=$!
sleep 2
for _ in 1 2 3; do echo "t=$(stamp)"; gpu_run; done > "$OUT/gpu_during_openssl.log" 2>&1
wait "$OPID"
endphase

# 4. CPU flood alone (4 threads)
phase cpuflood_alone
(cd "$R" && for _ in 1 2 3; do echo "t=$(stamp)"; OMP_NUM_THREADS=4 ./cpuflood 256 "$STEPS" sim; done) \
    > "$OUT/cpuflood4t_alone.log" 2>&1
endphase

# 5. CPU flood + GPU concurrent
phase cpuflood_conc
(cd "$R" && while :; do echo "t=$(stamp)"; OMP_NUM_THREADS=4 ./cpuflood 256 "$STEPS" sim; done) \
    > "$OUT/cpuflood4t_conc.log" 2>&1 & CPID=$!
sleep 2
for _ in 1 2 3; do echo "t=$(stamp)"; gpu_run; done > "$OUT/gpu_during_cpuflood.log" 2>&1
kill "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null
pkill -f "\./cpuflood" 2>/dev/null
endphase

echo done
