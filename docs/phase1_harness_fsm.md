# Phase 1 — Agent Harness → FSM Mapping (single-kernel V3D optimization)

**Scope:** the smallest complete thing. One V3D kernel, optimized by an *agent* (Claude/Codex) driving AutoKernel's edit→verify→keep/revert loop through a Burr FSM served by Theodosia, correctness-gated by NMSE-vs-CPU. No concurrency, no multi-kernel orchestration, no offload scheduling. Those are later phases and are explicitly out of scope here.

**Why kernel optimization first:** until `mul_mat` runs inside the V3D envelope (256 invocations / 16 KB shared mem / no fp16 / no coop-matrix), *nothing* runs on the GPU. Stock llama.cpp Vulkan aborts on exactly this op. So the first enabling exploration is getting one core op to pass `test-backend-ops -b Vulkan0` and then optimizing it. This is the prerequisite for every downstream ambition.

---

## The one design decision that defines Phase 1

**In AutoKernel the agent decides keep/revert. In the FSM the machine does.**

AutoKernel's loop is agent-self-reported: `program.md` *asks* the agent to run `bench.py`, parse `run.log`, and honestly `git reset --hard HEAD~1` on failure. Nothing enforces it. That is precisely the surface every reward-hacking incident exploited (Sakana, CUDA-L1, Kevin) and precisely what produced the Session 10 mirage.

The port inverts this:

- **The agent owns the creative states** — `HYPOTHESIZE` (what to try) and `IMPLEMENT` (write the shader edit). This is where "bash head against the wall, read the cryptic error, pivot" lives. The agent is good at this.
- **The FSM owns the trust-critical states** — `COMPILE`, `VERIFY`, `BENCHMARK`, `EVALUATE`. These run as deterministic Burr actions on the Pi. The agent *cannot* influence them, cannot skip `VERIFY`, cannot read the timing before correctness passes, cannot self-report "keep." It proposes; the machine disposes and records.

Theodosia enforces this structurally: the agent only ever calls `step(action, inputs)`, `action` is constrained to a JSON-Schema enum, and `VERIFY → BENCHMARK` is unreachable unless the correctness guard holds. There is no tool the agent can call to fake a pass.

---

## File / concept mapping

| AutoKernel | Phase-1 realization | Owner |
|---|---|---|
| `program.md` loop discipline (prose) | Theodosia `DEFAULT_INSTRUCTIONS` + the action enum + `valid_next_actions` in every response | enforced by server |
| `program.md` playbook tiers | the V3D 6-tier playbook, given to the agent as context for `HYPOTHESIZE` | agent reads |
| agent hypothesizes | `step("hypothesize", {tier, param, rationale})` → records intent to ledger, transitions to IMPLEMENT | **agent** |
| edit `kernel.py` | `step("implement", {shader_diff, host_patch})` → writes `kernel.comp` + co-varying host constants on the Pi, `git commit` | **agent** |
| Triton/CUDA JIT (implicit) | `step("compile")` → `glslangValidator` + **offline SPIR-V repack** + relink; sets `compile_ok` / `compile_oom` | FSM |
| `bench.py` 5-stage correctness | `step("verify")` → `test-backend-ops -b Vulkan0 -o <OP>`, NMSE gate; sets `verify_ok`, `verify_complete` | FSM |
| `bench.py` performance | `step("benchmark")` → op timing / decode t/s; **unreachable unless verify passed** | FSM |
| agent's keep/revert decision | `step("evaluate")` → **deterministic** keep/revert from recorded verify + measured delta | FSM |
| `results.tsv` append + git commit | `step("log")` → append `results.tsv`, write git SHA into `variant_log`, ledger entry | FSM |
| `orchestrate.py next` | `valid_next_actions` + `should_stop` guard | FSM |
| `reference.py` (torch oracle) | the CPU backend of the same engine (`test-backend-ops` CPU path). No file. | FSM |
| `bench.py` (torch/CUDA) | replaced entirely by the `verify`/`benchmark` actions calling Pi tools over local subprocess | FSM |

`orchestrate.py`'s multi-kernel scheduling and Amdahl logic are **not used in Phase 1** (single kernel). Its per-kernel bookkeeping (baseline vs best, experiment counters) is reused inside Burr `State`.

---

## States, guards, stopping (Phase 1 subset of the existing FSM)

Reuse `~/v3d-pi-ai/burr_fsm/fsm.py`'s graph, trimmed to one kernel:

```
CHARACTERIZE → BASELINE → HYPOTHESIZE → IMPLEMENT → COMPILE → VERIFY → BENCHMARK → EVALUATE → LOG → (loop | STOP)
```

Guards:
- `COMPILE → HYPOTHESIZE` if `compile_oom` (agent reads the error, pivots)
- `VERIFY → LOG` (verdict=revert) if **not** (`verify_ok` and `verify_complete`) — bypasses BENCHMARK, wrong kernel never timed
- `VERIFY → BENCHMARK` only if `verify_ok and verify_complete`
- `BENCHMARK → EVALUATE → LOG`
- `LOG → HYPOTHESIZE` (loop) unless `should_stop`
- `should_stop`: `consecutive_no_gain ≥ 3` **on correct variants** (correctness fails don't count), or time budget, or `best ≥ target`

`verify_ok` = NMSE ≤ threshold (1e-7 default, 1e-6 fp16) vs CPU. `verify_complete` = full-output coverage + range/std/axes filters (the Session-10 guard). Both must hold. See `verification_design.md`.

---

## The agent's turn, from its side of the MCP wire

1. read `theodosia://state` + `valid_next_actions`
2. `step("hypothesize", {tier:1, param:"local_size_x=240", rationale:"..."})`
3. `step("implement", {shader_diff, host_patch})`
4. `step("compile")` → may **refuse** with `compile_oom` + the compiler error; agent pivots back to `hypothesize`
5. `step("verify")` → NMSE result; if it fails, EVALUATE will revert, agent cannot override
6. `step("benchmark")` → refused if verify didn't pass; otherwise returns t/s
7. `step("evaluate")` → FSM returns keep/revert (deterministic), agent does not decide
8. `step("log")` → recorded to git + ledger
9. loop or `STOP`

Every call, success or refusal, is one hash-chained ledger line. That transcript *is* the auditable, reproducible record.

---

## Deployment (Phase 1)

- Theodosia server runs **on the Pi** (aarch64 / py3.13, confirmed): `theodosia serve fsm:build_app --transport streamable-http --host 0.0.0.0 --port 8000`, factory mode.
- Agent runs on the **laptop**, MCP client → `http://pi.local:8000/mcp`.
- The FSM's `implement`/`compile`/`verify`/`benchmark` actions call Pi tools via **local subprocess** (as the existing burr_fsm already does). No SSH inside the loop.

---

## First target and definition of done

**Target op: `MUL_MAT`** (matmul). It is the op that aborts on the 16 KB limit and blocks the entire forward pass, so optimizing it into the envelope is the literal enabling work. `test-backend-ops -o MUL_MAT -b Vulkan0` is the gate. (The Session-10 `convolution1x1.comp` is the better *correctness-demo* kernel and comes second, once the loop itself is proven.)

**Phase 1 is done when:**
1. The agent, over MCP, runs ≥1 full `HYPOTHESIZE→…→LOG` cycle that is **kept** (correct + improved) and ≥1 that is **refused/reverted**, both recorded in the ledger.
2. A deliberately-broken variant (partial-output) is **refused at VERIFY** and never reaches BENCHMARK — the ungameability proof.
3. The run reproduces from the ledger + pinned Mesa build + logged seeds + git SHAs.

No concurrency, no second kernel, no offload. That is Phase 2+.

---

## Concrete build order (for a future implementation session, not now)

1. `verify` action wrapping `test-backend-ops -b Vulkan0` with NMSE parse + the `verify_complete` filters. *(highest risk, build first — it's the gate everything depends on)*
2. `compile` action: `glslangValidator` + offline repack + stale-bytecode assertion.
3. `kernel.comp` + `kernel.meta.json` for `MUL_MAT` (extracted from the patched llama.cpp shader).
4. Trim `burr_fsm/fsm.py` to the single-kernel graph above; wire the four FSM-owned actions to Pi subprocess calls.
5. `theodosia_server.py`: `mount(build_app).run(streamable-http)`.
6. Agent-side: system prompt = the V3D playbook; connect MCP client; run the loop.

Component (1) is the load-bearing one. Everything else is plumbing around it.
