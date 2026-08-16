# Presentation kit

- `deck.md` — 12-slide Marp deck in assertion-evidence style (full-sentence
  claim per title, artifact evidence in the body). Render with:
  `npx @marp-team/marp-cli deck.md --html --pdf --allow-local-files -o deck.pdf`
  (the `--html` flag is required; the deck uses inline HTML for layout).
- `deck.pdf` — rendered deck.
- `defense.mp4` — the deck on a fixed 3:00 schedule, silent, for narrating
  over. Rebuild: render frames with `--images png -o frames/slide.png`, then
  `ffmpeg -f concat -safe 0 -i frames/concat.txt -t 180 ...` (schedule in
  `narration.md`).
- `narration.md` — per-slide cue times and suggested narration.
- `fsm.png`, `gpu_retention.png` — evidence images (the FSM diagram and the
  GPU-retention chart from notebook 02).

All numbers on the slides come from the archived artifacts under
`../paper/artifacts/`; the hardware is no longer live.
