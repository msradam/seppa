# Paper build

`paper.md` is the source. `paper.pdf` is built from it with the official
IEEEtran class (vendored here, CTAN V1.8b).

```sh
cd docs/paper
python3 build.py            # writes paper_ieee.tex, compiles paper.pdf
python3 build.py --check    # validate paper.md without building
```

Requires `pandoc` and `pdflatex` (BasicTeX or MacTeX).

## Editing

Edit `paper.md` and rebuild. Nothing in `build.py` matches on prose, so
rewording any sentence is safe. Three markers in the source drive the parts
LaTeX needs and markdown cannot express:

| Marker | Effect |
|---|---|
| `<!--FIG:name-->` | inserts `name.tex` as a figure float at that point |
| `<!--TABLE:label\|Caption-->` | caption and label for the markdown table directly below it |
| `<!--SPECS-->` | two tables generated from `artifacts/specs_2026-08-10/specs.json` |

Write cross-references as raw LaTeX and pandoc passes them through:
`Fig.~\ref{fig:fsm}`, `Table~\ref{tab:conc}`. Section numbers written as
"Section 5" or "Section 5.2" convert to IEEE roman form using the actual
`## N.` headings, so renumbering sections needs no change to the build.

## What the build checks

It stops with a plain message, before writing anything, when:

- a section cross-reference points at a heading that does not exist
- a markdown table has no `<!--TABLE:-->` marker above it, or the counts disagree
- a `<!--FIG:name-->` names a `.tex` file that is not there
- an unrecognized `<!--MARKER:-->` appears

If `pdflatex` fails, the previous `paper.pdf` is left untouched rather than
replaced by a stale or partial file. A successful build reports page count,
table and reference counts, and the number of overfull boxes, which should
stay at zero.

## Layout notes

`\IEEEtriggeratref` in `build.py` (the `TRIGGER` constant, currently `None`,
so the macro is not emitted and the columns fill naturally) sets which
reference starts the second column on the last page; set it to a reference
number if the count changes and the final page looks lopsided. `fsm_fig.tex` holds the TikZ state-machine
figure. The two spec tables are generated from the archived capture rather
than typed, so they cannot drift from the measured hardware.
