"""Drive the flood2 FSM over the live MCP wire (not in-process like
replay_flood2.py): connect to Theodosia's streamable-http endpoint and call
the `step` tool through a full optimization cycle, then demonstrate the
correctness gate by submitting a mass-violating kernel and attempting to
benchmark it anyway.

    uv run python drive_flood2_mcp.py http://<pi>:8000/mcp

Prints a JSON-lines transcript; every line is {action, sent, result}.
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

HERE = os.path.dirname(os.path.abspath(__file__))
FUSED = open(os.path.join(HERE, "pi", "flood", "fused2s.comp")).read()
GOOD_LINE = "float wA = P[iA].x + pc.rain - outA + inA;"
BROKEN = FUSED.replace(GOOD_LINE, GOOD_LINE.replace("pc.rain", "2.0 * pc.rain"))
assert BROKEN != FUSED


async def step(session, action, inputs=None):
    res = await session.call_tool("step", {"action": action, "inputs": inputs or {}})
    body = "".join(c.text for c in res.content if getattr(c, "text", None))
    # Theodosia prefixes the JSON payload with "Step N: name ⊢ → next".
    try:
        parsed = json.loads(body[body.index("{") :])
    except ValueError:
        parsed = body
    print(json.dumps({"action": action, "is_error": bool(res.isError), "result": parsed}))
    return res.isError, parsed


async def main(url):
    async with streamablehttp_client(url) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            print(json.dumps({"connected": url, "tools": tools}))

            # Cycle 1: the machine independently re-judges the fused strip-2 kernel.
            for a in ("characterize", "baseline", "hypothesize"):
                await step(session, a)
            await step(session, "implement", {"shader": FUSED, "height_shader": "", "strip": 2})
            for a in ("compile_", "verify", "benchmark", "evaluate", "log_variant"):
                await step(session, a)

            # Cycle 2: a kernel that doubles rain injection. It compiles; the
            # physics gate must fail it, and the server must refuse benchmark.
            await step(session, "hypothesize")
            await step(session, "implement", {"shader": BROKEN, "height_shader": "", "strip": 2})
            await step(session, "compile_")
            err, res = await step(session, "verify")
            failed = err or (
                isinstance(res, dict) and res.get("result", {}).get("verify_ok") is False
            )
            assert failed, "gate did not fail the broken kernel"
            gate_err, gate_res = await step(session, "benchmark")
            refused = bool(gate_err) or (
                isinstance(gate_res, dict) and gate_res.get("error") == "invalid_transition"
            )
            print(
                json.dumps(
                    {
                        "gate_demo": "benchmark after failed verify",
                        "server_refused": refused,
                        "response": gate_res,
                    }
                )
            )
            await step(session, "log_variant")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://pi.local:8000/mcp"))
