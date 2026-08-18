"""Build paper.pdf from paper.md via the official IEEEtran LaTeX class.
Usage: python3 build.py  (writes paper_ieee.tex, compiles paper.pdf)
IEEEtran.cls is vendored in this directory (CTAN, V1.8b).
"""

import json
import re
import subprocess
from pathlib import Path

src = Path("paper.md").read_text()

title = re.match(r"# (.+)", src).group(1)
abstract_md = src[
    src.index("## Abstract") + len("## Abstract") : src.index("## 1. Introduction")
].strip()
body_md = src[src.index("## 1. Introduction") : src.index("## References")]
refs_md = src[src.index("## References") :].split("\n", 1)[1].strip()

# strip manual numbers; IEEEtran numbers headings IEEE-style (I., A.)
body_md = re.sub(r"^## \d+\. ", "# ", body_md, flags=re.M)
body_md = re.sub(r"^### \d+\.\d+ ", "## ", body_md, flags=re.M)
# textual cross-references -> IEEE roman style (longest first)
body_md = body_md.replace("Sections 5 and 6 answer", "Sections V and VI answer")
for a, b in [
    ("5.1", "V-A"),
    ("5.2", "V-B"),
    ("5.3", "V-C"),
    ("7.1", "VII-A"),
    ("10", "X"),
    ("11", "XI"),
    ("2", "II"),
    ("3", "III"),
    ("4", "IV"),
    ("5", "V"),
    ("6", "VI"),
    ("7", "VII"),
    ("8", "VIII"),
    ("9", "IX"),
]:
    body_md = body_md.replace(f"Section {a}", f"Section {b}")


def rep(old, new):
    global body_md
    assert old in body_md, f"missing: {old[:60]}"
    body_md = body_md.replace(old, new)


# the ASCII FSM sketch becomes the TikZ figure (raw latex passes through pandoc)
rep(
    """serves it from the Pi itself. The graph is:

```
characterize -> baseline -> hypothesize -> implement
  -> compile -> verify -> benchmark -> evaluate
  -> log_variant -> (hypothesize | stop)
```

with two guard edges to `log_variant`,""",
    """serves it from the Pi itself.

\\input{fsm_fig}

Fig.~\\ref{fig:fsm} shows the graph, with two guard edges to `log_variant`,""",
)
# tables become numbered floats; point the prose at them
rep(
    "every winning change in Table I lives outside the shader",
    "every winning change in Table~\\ref{tab:sweep} lives outside the shader",
)
rep(
    "so every comparison in this paper is within `vkflood2`.",
    "so every comparison in this paper is within `vkflood2`. Table~\\ref{tab:sweep} shows the sweep.",
)
rep(
    "Results from the steady-state run of 2026-08-10 (raw logs under",
    "Results from the steady-state run of 2026-08-10 are in Table~\\ref{tab:conc} (raw logs under",
)
rep("`conc_steady` in `docs/paper/artifacts/`):", "`conc_steady` in `docs/paper/artifacts/`).")
rep(
    "The same soak-and-measure protocol applied; raw logs are under `cpu_steady` in `docs/paper/artifacts/`.",
    "The same soak-and-measure protocol applied; raw logs are under `cpu_steady` in `docs/paper/artifacts/`. Table~\\ref{tab:cpu} presents the outcome.",
)


# spec tables are generated from the archived system capture, not typed in
_specs = json.loads(Path("artifacts/specs_2026-08-10/specs.json").read_text())
_plat, _gpu = _specs["platform"], _specs["gpu"]
_plat_rows = [
    ("Model", _plat["model"]),
    ("SoC", _plat["soc"]),
    ("CPU", f"{_plat['cpu']} @ {_plat['cpu_clock_mhz']} MHz"),
    ("RAM", f"{_plat['ram_gb']} GB LPDDR (shared with GPU)"),
    ("OS / kernel", f"{_plat['os']}, {_plat['kernel']}"),
    ("Firmware", _plat["firmware"]),
]
_gpu_rows = [
    ("Device", _gpu["device"]),
    ("Driver", _gpu["driver"] + " (v3dv)"),
    ("Vulkan API", _gpu["vulkan_api"]),
    ("Core clock", f"{_gpu['core_clock_mhz']} MHz"),
    ("Max invocations / workgroup", str(_gpu["max_workgroup_invocations"])),
    ("Shared memory / workgroup", f"{_gpu['max_shared_memory_bytes'] // 1024} KB"),
    ("Subgroup (SIMD) width", str(_gpu["subgroup_size"])),
    ("fp16 arithmetic", "yes" if _gpu["shader_float16"] else "no"),
    ("Cooperative matrix", "yes" if _gpu["cooperative_matrix"] else "no"),
]


def _md_table(rows):
    out = ["| | |", "|---------|---------|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


assert "<!--SPECS-->" in body_md
body_md = body_md.replace("<!--SPECS-->", _md_table(_plat_rows) + "\n\n" + _md_table(_gpu_rows))


def pandoc(text):
    return subprocess.run(
        [
            "pandoc",
            "-f",
            "markdown+autolink_bare_uris",
            "-t",
            "latex",
            "--syntax-highlighting=none",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


body = pandoc(body_md)
abstract = pandoc(abstract_md).strip()

# pandoc escapes the ~ in "Fig.~\ref{...}" while passing \ref through raw
body = body.replace("\\textasciitilde{}\\ref", "~\\ref")
# keep code blocks on one column
body = body.replace(
    "\\begin{verbatim}",
    "\\medskip\\noindent\\begin{minipage}{\\linewidth}\n"
    "\\begin{Verbatim}[frame=single,numbers=left,numbersep=3pt,framesep=1.6mm,fontsize=\\scriptsize]",
)
body = body.replace("\\end{verbatim}", "\\end{Verbatim}\n\\end{minipage}\\medskip")

# tables: drop pandoc's minipage header cells, rewrap longtable (illegal in
# two-column mode) as an IEEE table float with caption above
body = re.sub(
    r"\\begin\{minipage\}\[[bt]\]\{\\linewidth\}\\raggedright\s*(.*?)\s*\\end\{minipage\}",
    r"\1",
    body,
    flags=re.S,
)
captions = iter(
    [
        (
            "tab:platform",
            "Platform, collected from the running board by \\mbox{collect\\_specs.sh}",
        ),
        ("tab:gpu", "GPU compute limits, collected live (vulkaninfo)"),
        ("tab:sweep", "Falsification sweep: 400 steps at 256x256 in the \\mbox{vkflood2} harness"),
        ("tab:conc", "Concurrent GPU flood and CPU LLM decode"),
        ("tab:cpu", "The CPU-only counterfactual"),
    ]
)


def table_open(_m):
    lab, cap = next(captions)
    return (
        "\\begin{table}[!t]\\caption{%s}\\label{%s}\\centering\\footnotesize"
        "\\setlength{\\tabcolsep}{2.5pt}"
        "\\renewcommand{\\arraystretch}{1.15}\\begin{tabular}{" % (cap, lab)
    )


body = re.sub(r"\\begin\{longtable\}\[\]\{", table_open, body)
body = body.replace("\\end{longtable}", "\\bottomrule\n\\end{tabular}\\end{table}")
body = body.replace("\\noalign{}", "")
body = re.sub(r"^\\end(first)?head\n", "", body, flags=re.M)
body = re.sub(r"^\\end(last)?foot\n", "", body, flags=re.M)
# pandoc's longtable footer rule lands right under the header once the
# endhead/endlastfoot markers are stripped; the real bottom rule is added
# at \end{tabular} above
body = body.replace("\\midrule\n\\bottomrule", "\\midrule")
body = body.replace("\\toprule\n\\bottomrule", "\\toprule")

bibitems = []
for i, para in enumerate(re.split(r"\n\n+", pandoc(refs_md).strip()), 1):
    entry = re.sub(r"^\s*(\{\[\}|\[)\d+(\{\]\}|\])\s*", "", para.strip())
    bibitems.append(f"\\bibitem{{r{i}}} {entry}")

tex = r"""\documentclass[conference]{IEEEtran}
\usepackage{array}
\usepackage{booktabs}
\usepackage{calc}
\usepackage{url}
\usepackage{textcomp}
\usepackage{fancyvrb}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\newcommand{\real}[1]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\makeatletter
\def\verbatim@font{\ttfamily\scriptsize}
\makeatother
\begin{document}
\title{%s}
\author{\IEEEauthorblockN{Adam Munawar Rahman}
\IEEEauthorblockA{New York University \\ New York, NY, USA \\ msr541@nyu.edu \\ ECE-GY 9953 Advanced Project. Adviser: Prof.\ Brandon Reagen}}
\maketitle
\begin{abstract}
%s
\end{abstract}
\begin{IEEEkeywords}
GPU kernel optimization, LLM agents, correctness verification, edge computing, Vulkan
\end{IEEEkeywords}
%s
\IEEEtriggeratref{16}
\begin{thebibliography}{%d}
\scriptsize
%s
\end{thebibliography}
\end{document}
""" % (title, abstract, body, len(bibitems), "\n\n".join(bibitems))

Path("paper_ieee.tex").write_text(tex)
for _ in range(2):
    r = subprocess.run(
        ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode", "paper_ieee.tex"],
        capture_output=True,
        text=True,
    )
if r.returncode != 0:
    print("\n".join(ln for ln in r.stdout.splitlines() if ln.startswith("!") or "Error" in ln))
    raise SystemExit(1)
Path("paper_ieee.pdf").replace("paper.pdf")
print("built paper.pdf")
