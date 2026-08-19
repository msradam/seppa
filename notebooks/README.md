# Notebooks

Executed Jupyter notebooks that derive the Section VI and VII results in the paper
(`../docs/paper/paper.pdf`) from the raw artifacts committed under
`../docs/paper/artifacts/`. They run offline; no hardware is needed.
Shared log parsers live in `analysis_utils.py`. Each notebook opens with
its own description of the experiment, the provenance of its input data
(which script produced which directory, and when), and closes with the
commands to regenerate the raw data live and an environment record.

| Notebook | Paper elements | Raw data |
|---|---|---|
| `01_reproduction_and_repeatability.ipynb` | Section VI (ledger, gate refusal, pass^5, agent session) | `mcp_flood2_recreation_2026-07-11.jsonl`, `passk_2026-08-09/`, `claude_sessions/20260802T150755Z/` |
| `02_concurrency_envelope.ipynb` | Section VII, Tables IV-V, the campaign alignment, `gpu_retention.png` | `conc_steady_2026-08-10/`, `cpu_steady_2026-08-10/`, `genload2_2026-08-10/`, `ab_vkvisible_2026-08-10/`, `conc_bench2/` |

Run them with the project environment: `uv sync && uv run jupyter lab notebooks/`.
