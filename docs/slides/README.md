# Presentation kit

A 27-slide deck and a silent video cut for the oral
presentation that the NYU Tandon MS program requires alongside the project
report.

## Contents

- `deck.md`: the deck. Titles are full-sentence claims and the body carries
  the evidence for that claim (assertion-evidence style), inside the IEEE
  visual conventions.
- `deck.pdf`: rendered deck.
- `themes/ieee.src.css`: the Marp theme, editable source.
- `themes/ieee.css`: generated from the source with the fonts inlined. Do not
  edit directly; run `build_theme.py`.
- `fonts/`: Open Sans, vendored (SIL Open Font License, see
  `fonts/LICENSE-OpenSans.txt`).
- `narration.md`: per-slide cue times and suggested narration.
- `defense.mp4`: the slides on the `narration.md` schedule, silent, for
  narrating over.
- `fsm.png`: the state-machine figure, built from `fsm_standalone.tex`.
- `gpu_retention.png`: the retention chart, produced by notebook 02.

## The theme

IEEE publishes presentation templates for PowerPoint and Google Slides
only, so `themes/ieee.css` follows the corporate template's conventions:
IEEE Blue title slide and section dividers, a blue rule above every
content slide, and blue table headers. It carries no IEEE logo or wordmark, because the work is not an IEEE
publication.

- Palette: IEEE Blue is PMS 3015 C, `#00629b`, from the [IEEE brand colors
  chart](https://brand-experience.ieee.org/wp-content/uploads/2020/10/IEEE_Brand_Colors_Hex_Formulas_for_Solids_and_Tints.pdf).
  Accents are PMS 200 C red `#ba0c2f`, PMS 348 C green `#00843d`, PMS 295 C
  navy `#002855`, and Cool Grey 9 C `#75787b`.
- Typeface: Open Sans, which the IEEE visual identity guidelines name as the
  preferred web font. (Formata is the primary IEEE typeface but is not freely
  licensed; Calibri is the alternate for PowerPoint and Word.)

Marp inlines a theme's CSS into the output document, so relative `url()` paths
in the theme would resolve against the output location instead of the theme
file. `build_theme.py` embeds the fonts as data URIs to make the theme work
for any output path.

## Rebuilding

```sh
python3 build_theme.py                     # only after editing themes/ieee.src.css

npx @marp-team/marp-cli deck.md --html --theme themes/ieee.css \
  --allow-local-files --pdf -o deck.pdf

npx @marp-team/marp-cli deck.md --html --theme themes/ieee.css \
  --allow-local-files --images png --image-scale 3 -o frames/slide.png

ffmpeg -y -f concat -safe 0 -i frames/concat.txt -vf "format=yuv420p" \
  -c:v libx264 -crf 20 -r 30 -t <sum of durations> defense.mp4
```

`--html` is required: without it Marp strips the inline HTML the deck uses for
two-column layouts. `--image-scale 3` renders frames at 3840x2160. The deck is designed at
1280x720. Frames rendered smaller than 3840x2160 are upscaled by the player on
a high-density display and read as blurry even though the encode is lossless-looking. The frame
PNGs are generated output that git ignores. The per-slide durations are
tracked in `frames/concat.txt`. Set `-t` to their sum (847 today);
without it the concat demuxer holds the trailing entry past the end.

Rebuilding `fsm.png` needs `pdflatex`, `pdftoppm`, and Pillow:

```sh
pdflatex -interaction=nonstopmode fsm_standalone.tex
pdftoppm -r 600 -png fsm_standalone.pdf fsm_raw
```

then crop `fsm_raw-1.png` to its content box with a small margin. Rasterize
with `pdftoppm` at high dpi; `sips --resampleWidth` rasterizes the
PDF at its native size first and upscaling from there ships a blurry figure.

## Provenance

Every number in the scored campaigns comes from the archived artifacts under
`../paper/artifacts/`, and the two executed notebooks under `../../notebooks/`
re-derive the tables; the hand-timed and larger-grid figures live in the
dated running notes, as the slides that use them state. The measurement
hardware is no longer live.
