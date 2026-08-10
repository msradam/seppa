#!/bin/bash
# Decode A/B: Vulkan device visible vs hidden (GGML_VK_VISIBLE_DEVICES=99),
# llama-bench tg64, r=5, each condition from a cool start below 55 C.
set -u
RES=/root/v3d-research
BIN=$RES/llama.cpp/build-vulkan/bin
M=$(find "$RES/models" -iname 'granite-4.0-1b-Q4_0.gguf' | head -1)
OUT=${OUT:-/tmp/ab_vkvisible}
mkdir -p "$OUT"
cool() {
    while :; do
        t=$(vcgencmd measure_temp | grep -o '[0-9.]*')
        awk "BEGIN{exit !($t < 55)}" && break
        sleep 10
    done
}
run() { # $1 tag, $2 env setting ("" or hidden)
    cool
    echo "start_temp=$(vcgencmd measure_temp) throttled=$(vcgencmd get_throttled)" > "$OUT/$1_meta.txt"
    if [ "$2" = hidden ]; then
        GGML_VK_VISIBLE_DEVICES=99 "$BIN/llama-bench" -m "$M" -ngl 0 -t 4 -p 0 -n 64 -r 5 > "$OUT/$1.log" 2>&1
    else
        "$BIN/llama-bench" -m "$M" -ngl 0 -t 4 -p 0 -n 64 -r 5 > "$OUT/$1.log" 2>&1
    fi
    echo "end_temp=$(vcgencmd measure_temp) throttled=$(vcgencmd get_throttled)" >> "$OUT/$1_meta.txt"
}
echo "model=$M" > "$OUT/meta.txt"
md5sum "$M" >> "$OUT/meta.txt"
run visible ""
run hidden hidden
echo done
