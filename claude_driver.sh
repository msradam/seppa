#!/usr/bin/env bash
# Drive a Seppa optimization session with Claude as the proposer.
#
# Usage: ./claude_driver.sh [mcp-url]
#   MODEL=claude-sonnet-5 EFFORT=high MAX_EXPERIMENTS=3 ./claude_driver.sh
#
# Uses the signed-in Claude Code session (subscription auth). The full
# tool-call transcript is written as stream-json for the artifact record.
set -euo pipefail
cd "$(dirname "$0")"

URL="${1:?usage: claude_driver.sh http://<pi>:8000/mcp}"
MODEL="${MODEL:-claude-sonnet-5}"
EFFORT="${EFFORT:-high}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-3}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="${OUT:-docs/paper/artifacts/claude_sessions/$STAMP}"

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY is set; unset it so the run uses the signed-in session." >&2
    exit 1
fi

mkdir -p "$OUT"
printf '{"mcpServers":{"seppa":{"type":"http","url":"%s"}}}\n' "$URL" > "$OUT/mcp.json"
{
    echo "date_utc=$STAMP"
    echo "mcp_url=$URL"
    echo "model=$MODEL"
    echo "effort=$EFFORT"
    echo "max_experiments=$MAX_EXPERIMENTS"
    echo "temperature=not exposed by the Claude Code client (server default)"
    echo "auth=signed-in Claude subscription session"
    echo "client=$(claude --version 2>/dev/null)"
} > "$OUT/run_meta.txt"

PROMPT="$(cat claude_driver_prompt.md)

Experiment budget for this session: $MAX_EXPERIMENTS."

claude -p "$PROMPT" \
    --model "$MODEL" \
    --effort "$EFFORT" \
    --mcp-config "$OUT/mcp.json" \
    --strict-mcp-config \
    --allowedTools "mcp__seppa__step" \
    --output-format stream-json \
    --verbose \
    > "$OUT/transcript.jsonl" 2> "$OUT/stderr.log"

echo "session artifacts in $OUT"
python3 - "$OUT/transcript.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    ev = json.loads(line)
    if ev.get("type") == "result":
        print(ev.get("result", "")[:4000])
PY
