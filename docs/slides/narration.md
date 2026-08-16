# Narration script for the three-minute video cut

The video (`defense.mp4`) is the twelve deck slides on a fixed schedule,
silent, meant to be spoken over. Cue times below match the slide changes.
The suggested lines run a little under the slide durations so there is
room to breathe. Total: 3:00.

| Cue | Slide | Suggested narration |
|---|---|---|
| 0:00 | Title | This is my advanced project: getting a language model to optimize GPU kernels on a Raspberry Pi, without ever trusting it. |
| 0:08 | The goal | The application needs a real-time flood simulation and a language model on one offline Pi. Both run continuously, and there are only four CPU cores. The way out is the GPU that every Pi 5 ships and almost nothing uses. Two questions: can it be made fast enough, and can it share the memory bus with a busy CPU? |
| 0:24 | The GPU | This GPU is a strange target. Two hundred fifty-six threads per workgroup, sixteen kilobytes of shared memory, no fp16. And the compiler never tells you when you have gone too far. It silently retries with slower strategies, so the performance cliffs come from the register allocator, not from your arithmetic. |
| 0:40 | The problem | Why not just ask a model to optimize it? Because the record says models cheat. CUDA-L1 found a third of its RL-generated kernels faked their speedups through timing loopholes. Every published fix is the same idea: verification the model cannot touch. |
| 0:54 | The FSM | So the harness is a state machine. The model owns two states: propose a hypothesis, write the kernel. The machine owns everything else: compile, verify against physics, benchmark, verdict. The benchmark state is only reachable through a green verify. Skipping verification is not a request the protocol can express. |
| 1:12 | Division of labor | Following Chip-Chat's disclosure convention: I built the harness and chose the gates, the model wrote kernel proposals and nothing else, and the machine did everything evidential. No proposed kernel was hand-edited. |
| 1:24 | The sweep | Five hypotheses, each priced by the loop. Halving memory traffic changed nothing, which killed the obvious theory. The real bound is fixed per-invocation cost: about two clock cycles per invocation, which strip-mining amortizes. Fused plus strip-two stacked to fifty-nine percent. |
| 1:44 | The ledger | Then the machine re-judged the result from its own baseline, over the network, from a second computer. Verified, benchmarked, kept, at 2,116 steps per second. This ledger line is the machine's, not mine. |
| 2:00 | The refusal | And when the driver cheats, doubling the rainfall inside the kernel, it compiles, the physics gate fails it, and the server refuses to benchmark it at all. That refusal is the point of the whole design. |
| 2:18 | It repeats | Five scored reproductions from cool starts: five passes, kept kernel identical to the timer's resolution every time. |
| 2:32 | Concurrency | The payoff, measured at thermal steady state: 883 simulation steps per second next to a language model keeping 84 percent of its speed, and the split beats the best CPU-only arrangement on both axes. |
| 2:52 | Close | Everything you just saw reproduces from the raw logs in the repository. The evidence is the gate ledger, not the model's account of its work. Thanks. |
