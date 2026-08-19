# Archived artifacts

Raw logs, transcripts, and machine-collected system data behind the paper.
The measurement hardware is no longer live, so these files are the record.

| Directory | What it holds | Used by |
|---|---|---|
| `conc_steady_2026-08-10/` | Steady-state GPU/CPU concurrency campaign | Table IV |
| `cpu_steady_2026-08-10/` | CPU-only counterfactual, partitioned and oversubscribed | Table V |
| `passk_2026-08-09/` | Five scored reproductions, one JSONL per run plus `summary.json` | Section VI |
| `ab_vkvisible_2026-08-10/` | `GGML_VK_VISIBLE_DEVICES` A/B, with the run script | Section VII |
| `genload2_2026-08-10/` | GPU retention against generic CPU loads | Section VII |
| `specs_2026-08-10/` | `collect_specs.sh` output, `vulkaninfo` dump, `vcgencmd` | Tables I and II |
| `system_2026-08-02/` | Pre/post OS-update system records (kernel 6.12 to 6.18; Mesa unchanged at 25.0.7) | Section VIII |
| `mcp_flood2_recreation_2026-07-11.jsonl` | The machine-checked reproduction over MCP | Section VI |
| `mcp_flood2_recreation_run1_2026-07-11.jsonl` | Aborted first attempt at the same reproduction; the driver crashed on an assertion before the refusal cycle, and its fused benchmark (2105.3 steps/s, 1.563x) is superseded by the completed transcript above | not cited |
| `claude_sessions/` | Complete agent session transcripts | Section IV |
| `genload_bench_2026-08-09/` | Earlier generic-load run; source of the gate-log excerpt | Section IV |
| `conc_bench/`, `conc_bench2/`, `cpu_flood_bench/`, `cpu_flood_bench2/` | Superseded short-window campaigns; the paper cites `conc_bench2/` and `cpu_flood_bench/`, and the other two are kept as the full record | Sections VII and VIII |

## Reading the numbers

Flood throughput is not stored as steps/s. Each run prints `steps=N time=Ts`,
so steps/s is `N/T`. Concurrent phases additionally carry `llama_start=` and
`llama_end=` markers. A run counts only if it both began and finished
inside that window, so every counted run executed entirely under decode
load. This rule is why the concurrent row counts (n=28, n=12, n=25,
n=89) are smaller than the number of runs in the file. `notebooks/analysis_utils.py`
implements this; `notebooks/02_concurrency_envelope.ipynb` reproduces the tables.

## Known trap in the July transcript

`mcp_flood2_recreation_2026-07-11.jsonl` contains a record with
`"server_refused": false` sitting next to a response whose payload is
`{"error": "invalid_transition", ...}`. The server did refuse. The flag is
wrong because the driver of that era tested only the MCP `isError` bit, and
the server returns a refusal as a normal tool result with `isError` unset.
Both `drive_flood2_mcp.py` and `passk_flood2.py` now detect the refusal from
the payload instead, and carry a comment saying so. The `response` object is
authoritative; the flag is stale.
