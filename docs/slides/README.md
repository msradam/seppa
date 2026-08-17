# Presentation kit

A 27-slide deck and a silent video cut, sized for the 15-minute oral
presentation that the NYU Tandon MS program requires alongside the project
report.

## Contents

- `deck.md` — the deck. Titles are full-sentence claims and the body carries
  the evidence for that claim (assertion-evidence style), inside the IEEE
  visual conventions.
- `deck.pdf` — rendered deck.
- `themes/ieee.src.css` — the Marp theme, editable source.
- `themes/ieee.css` — generated from the source with the fonts inlined. Do not
  edit directly; run `build_theme.py`.
- `fonts/` — Open Sans, vendored (SIL Open Font License, see
  `fonts/LICENSE-OpenSans.txt`).
- `narration.md` — per-slide cue times and suggested narration, 15:00 total.
- `defense.mp4` — the slides on the `narration.md` schedule, silent, for
  narrating over.
- `fsm.png` — the state-machine figure, built from `fsm_standalone.tex`.
- `gpu_retention.png` — the retention chart, produced by notebook 02.

## The theme

IEEE publishes presentation templates for PowerPoint and Google Slides, not
for Marp or LaTeX, so `themes/ieee.css` follows the corporate template's
conventions rather than being a conversion of it: IEEE Blue title slide and
section dividers, a blue rule above every content slide, and blue table
headers. It carries no IEEE logo or wordmark, because the work is not an IEEE
publication.

- Palette: IEEE Blue is PMS 3015 C, `#00629b`, from the [IEEE brand colors
  chart](https://brand-experience.ieee.org/wp-content/uploads/2020/10/IEEE_Brand_Colors_Hex_Formulas_for_Solids_and_Tints.pdf).
  Accents are PMS 200 C red `#ba0c2f`, PMS 348 C green `#00843d`, PMS 295 C
  navy `#002855`, and Cool Grey 9 C `#75787b`.
- Typeface: Open Sans, which the IEEE visual identity guidelines name as the
  preferred web font. (Formata is the primary IEEE typeface but is not freely
  licensed; Calibri is the alternate for PowerPoint and Word.)

Marp inlines a theme's CSS into the output document, so relative `url()` paths
in the theme would resolve against the output location rather than the theme
file. `build_theme.py` embeds the fonts as data URIs to make the theme work
for any output path.

## Rebuilding

```sh
python3 build_theme.py                     # only after editing themes/ieee.src.css

npx @marp-team/marp-cli deck.md --html --theme themes/ieee.css \
  --allow-local-files --pdf -o deck.pdf

npx @marp-team/marp-cli deck.md --html --theme themes/ieee.css \
  --allow-local-files --images png -o frames/slide.png

ffmpeg -y -f concat -safe 0 -i frames/concat.txt \
  -vf "scale=1920:1080:flags=lanczos,format=yuv420p" \
  -c:v libx264 -crf 20 -r 30 -t 900 defense.mp4
```

`--html` is required: without it Marp strips the inline HTML the deck uses for
two-column layouts. `frames/` is generated and not tracked; the per-slide
durations live in `frames/concat.txt`, and `-t 900` trims the trailing entry
that the concat demuxer would otherwise hold past the end.

Rebuilding `fsm.png` needs `pdflatex` and Pillow:

```sh
pdflatex -interaction=nonstopmode fsm_standalone.tex
sips -s format png --resampleWidth 3200 fsm_standalone.pdf --out /tmp/fsm_raw.png
```

then crop the result on its alpha channel and composite it onto white.

## Provenance

Every number on the slides comes from the archived artifacts under
`../paper/artifacts/`, and the two executed notebooks under `../../notebooks/`
re-derive the tables. The measurement hardware is no longer live.
