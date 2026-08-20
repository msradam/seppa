#!/bin/bash
# GPU flood concurrency against generic CPU loads (no LLM): openssl sha256
# on 4 threads (compute-bound, cache-resident) and the OpenMP CPU flood on
# 4 threads (memory-bound). Cooldown below 55 C before each phase; 1 Hz
# temperature and throttle-register log per phase. Run ON the Pi:
#   OUT=/tmp/genload_bench bash genload_bench.sh
set -u
RES=/root/v3d-research
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
    (cd "$RES" && STRIP=2 FUSED=1 FLUX_SPV=fused2s.spv ./vkflood2 256 "$STEPS")
}

phase() {
    cool
    tlog > "$OUT/$1_temp.log" & TPID=$!
}
endphase() { kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null; }

phase gpu_alone
for _ in 1 2 3; do echo "t=$(stamp)"; gpu_run; done > "$OUT/gpu_alone.log" 2>&1
endphase

phase openssl_alone
openssl speed -multi 4 -seconds 20 sha256 > "$OUT/openssl_alone.log" 2>&1
endphase

phase openssl_conc
openssl speed -multi 4 -seconds 45 sha256 > "$OUT/openssl_conc.log" 2>&1 & OPID=$!
sleep 2
for _ in 1 2 3; do echo "t=$(stamp)"; gpu_run; done > "$OUT/gpu_during_openssl.log" 2>&1
wait "$OPID"
endphase

phase cpuflood_alone
(cd "$RES" && for _ in 1 2 3; do echo "t=$(stamp)"; OMP_NUM_THREADS=4 ./cpuflood 256 "$STEPS" sim; done) \
    > "$OUT/cpuflood4t_alone.log" 2>&1
endphase

phase cpuflood_conc
(cd "$RES" && while :; do echo "t=$(stamp)"; OMP_NUM_THREADS=4 ./cpuflood 256 "$STEPS" sim; done) \
    > "$OUT/cpuflood4t_conc.log" 2>&1 & CPID=$!
sleep 2
for _ in 1 2 3; do echo "t=$(stamp)"; gpu_run; done > "$OUT/gpu_during_cpuflood.log" 2>&1
kill "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null
pkill -f "\./cpuflood" 2>/dev/null
endphase

echo "done"
