# AutoKernel-to-FSM fidelity audit (2026-08-09)

Comparison of the upstream AutoKernel loop (RightNow-AI/autokernel,
`program.md` B4 and `bench.py`) against the implemented FSM target
(`v3d_flood2_opt.py`, the paper's subject). Line references are to
upstream `program.md`.

## Faithful

| AutoKernel rule | FSM realization |
|---|---|
| One focused change per experiment (l.228) | One `proposed_variant` per hypothesize/implement cycle; the graph cannot hold two |
| Record a baseline first (l.207) | `baseline` action measures and gates the unmodified two-pass kernel before any experiment |
| correctness FAIL: revert immediately, never keep (l.273) | `verify -> log_variant` guard; `benchmark` unreachable; verdict forced to revert |
| PASS + at least 1% gain: keep, new baseline (l.274, l.277) | `evaluate`: `sps > best * 1.01` updates `best_sps`/`best_variant`, persists winner |
| PASS + same or worse: revert (l.275) | `evaluate` else-branch |
| Correctness and timing from one execution (`bench.py`) | `run_gate` returns (gates, steps/s) from one run; `verify` returns that timing to the caller as `steps_per_sec_if_kept` whether or not the gates passed, and `benchmark` echoes it. What the green verify gates is entry into the ledger and the verdict, not the caller's sight of the number |
| Every experiment recorded (`results.tsv` via `orchestrate.py record`) | `variant_log` ledger entry per experiment with gate results and verdict |

## Deliberate deviations

1. **Keep/revert authority moved from agent to machine.** Upstream asks
   the agent to run `bench.py` and honestly `git reset --hard` on
   failure; nothing enforces it. The FSM makes the verdict a
   deterministic action the agent cannot influence. This inversion is
   the point of the port (`phase1_harness_fsm.md`).
2. **Stop rule added.** Upstream loops forever under human supervision
   (l.212). The FSM stops on 3 consecutive non-improvements on
   gate-passing variants, or the experiment budget.
3. **The simplicity tie-break is dropped.** Upstream allows keeping a
   tied variant if the code is simpler (l.281). "Simpler" is a judgment
   the machine cannot verify, and it reopens a self-reported keep path,
   so the FSM keeps only on measured gain.
4. **Git mechanics replaced by state.** Revert is bookkeeping
   (`best_variant` retained, proposal discarded); the kept winner is
   persisted to disk. Each `implement` fully overwrites the workspace,
   so a reverted shader cannot leak into the next experiment.
5. **Compile failures are ledgered.** The phase-1 design routed compile
   failure straight back to `hypothesize`; the implementation routes it
   through `log_variant` first so every attempt, including failures,
   appears in the ledger. The agent still pivots on the next
   `hypothesize`, which carries the compile error in the history.

## Drift found and fixed

`log_variant` counted compile/verify failures toward the 3-strike stop,
contradicting the phase-1 design ("consecutive_no_gain >= 3 on correct
variants; correctness fails don't count"). Three consecutive compile
errors would have ended a run instead of letting the agent pivot. Fixed
in `v3d_flood2_opt.py` and `v3d_llama_mmv.py` on 2026-08-09; the
experiment budget still bounds every run. No published measurement is
affected: the paper's runs contained no consecutive-failure sequences.
