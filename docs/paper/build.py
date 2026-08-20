"""Build paper.pdf from paper.md via the official IEEEtran LaTeX class.

Usage:
    python3 build.py            write paper_ieee.tex, compile paper.pdf
    python3 build.py --check    validate paper.md and stop, writing nothing

Edit paper.md freely; nothing here matches on prose. The build is driven by
markers in the source:

    <!--FIG:name-->             insert name.tex here as a figure float
    <!--TABLE:label|Caption-->  caption and label for the next markdown table
    <!--SPECS-->                two tables generated from the archived specs

Write cross-references as raw LaTeX (Fig.~\\ref{fig:fsm}, Table~\\ref{tab:conc});
pandoc passes them through. Section numbers written as "Section 5" or
"Section 5.2" are converted to IEEE roman form from the actual headings, so
renumbering a section needs no change here.

IEEEtran.cls is vendored in this directory (CTAN, V1.8b).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

CHECK_ONLY = "--check" in sys.argv
HERE = Path(__file__).parent
src = (HERE / "paper.md").read_text()


def die(msg):
    raise SystemExit(f"build.py: {msg}")


title_m = re.match(r"# (.+)", src)
if not title_m:
    die("paper.md must open with '# Title'")
title = title_m.group(1)

for marker in ("## Abstract", "## 1. Introduction", "## References"):
    if marker not in src:
        die(f"paper.md is missing the '{marker}' heading")

abstract_md = src[
    src.index("## Abstract") + len("## Abstract") : src.index("## 1. Introduction")
].strip()
body_md = src[src.index("## 1. Introduction") : src.index("## References")]
refs_md = src[src.index("## References") :].split("\n", 1)[1].strip()

# Section numbers come from the headings themselves, so renumbering the paper
# does not need an edit here.
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
sections = [int(n) for n in re.findall(r"^## (\d+)\. ", body_md, re.M)]
subsections = re.findall(r"^### (\d+)\.(\d+) ", body_md, re.M)
if not sections:
    die("no '## N. Title' section headings found")
xref = {f"{a}.{b}": f"{ROMAN[int(a)]}-{chr(64 + int(b))}" for a, b in subsections}
xref.update({str(n): ROMAN[n] for n in sections})

body_md = re.sub(r"^## \d+\. ", "# ", body_md, flags=re.M)
body_md = re.sub(r"^### \d+\.\d+ ", "## ", body_md, flags=re.M)


def to_roman(m):
    plural, num = m.group(1), m.group(2)
    if num not in xref:
        die(f"cross-reference to a section that does not exist: 'Section{plural} {num}'")
    return f"Section{plural} {xref[num]}"


body_md = re.sub(
    r"Sections (\d+) and (\d+)",
    lambda m: f"Sections {xref[m.group(1)]} and {xref[m.group(2)]}",
    body_md,
)
body_md = re.sub(r"Section(s?) (\d+(?:\.\d+)?)", to_roman, body_md)
# a hyphenated roman subsection must not break across lines
body_md = re.sub(r"Section (I?[VX]?I*-[A-Z])", r"\\mbox{Section \1}", body_md)


# <!--FIG:name--> becomes \input{name}
def fig_sub(m):
    name = m.group(1)
    if not (HERE / f"{name}.tex").exists():
        die(f"<!--FIG:{name}--> refers to {name}.tex, which does not exist")
    return f"\\input{{{name}}}"


body_md = re.sub(r"<!--FIG:([\w-]+)-->", fig_sub, body_md)

# spec tables are generated from the archived capture, never typed in
if "<!--SPECS-->" in body_md:
    specs = json.loads((HERE / "artifacts/specs_2026-08-10/specs.json").read_text())
    plat, gpu = specs["platform"], specs["gpu"]
    rows = [
        [
            ("Model", plat["model"]),
            ("SoC", plat["soc"]),
            ("CPU", f"{plat['cpu']} @ {plat['cpu_clock_mhz']} MHz"),
            ("RAM", f"{plat['ram_gb']} GB LPDDR (shared with GPU)"),
            ("OS / kernel", f"{plat['os']}, {plat['kernel']}"),
            ("Firmware", plat["firmware"]),
        ],
        [
            ("Device", gpu["device"]),
            ("Driver", gpu["driver"] + " (v3dv)"),
            ("Vulkan API", gpu["vulkan_api"]),
            ("Core clock", f"{gpu['core_clock_mhz']} MHz"),
            ("Max invocations / workgroup", str(gpu["max_workgroup_invocations"])),
            ("Shared memory / workgroup", f"{gpu['max_shared_memory_bytes'] // 1024} KB"),
            ("Subgroup (SIMD) width", str(gpu["subgroup_size"])),
            ("fp16 arithmetic", "yes" if gpu["shader_float16"] else "no"),
            ("Cooperative matrix", "yes" if gpu["cooperative_matrix"] else "no"),
        ],
    ]
    tables = "\n\n".join(
        "\n".join(["| | |", "|---------|---------|"] + [f"| {k} | {v} |" for k, v in group])
        for group in rows
    )
    body_md = body_md.replace("<!--SPECS-->", tables)

# each <!--TABLE:label|caption--> binds to the table that follows it
captions = re.findall(r"<!--TABLE:([\w:]+)\|(.+?)-->", body_md)
body_md = re.sub(r"<!--TABLE:[\w:]+\|.+?-->\n?", "", body_md)

if leftover := re.findall(r"<!--(\w+)[:>]", body_md):
    die(f"unrecognized marker(s) in paper.md: {sorted(set(leftover))}")


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

n_tables = len(re.findall(r"\\begin\{longtable\}", body))
if n_tables != len(captions):
    die(
        f"{n_tables} table(s) in the body but {len(captions)} <!--TABLE:...--> marker(s); "
        "each markdown table needs one marker directly above it"
    )

# pandoc escapes the ~ in "Fig.~\ref{...}" while passing \ref through raw
body = body.replace("\\textasciitilde{}\\ref", "~\\ref")
# keep code blocks inside one column
body = body.replace(
    "\\begin{verbatim}",
    "\\medskip\\noindent\\begin{minipage}{\\linewidth}\n"
    "\\begin{Verbatim}[frame=single,numbers=left,numbersep=3pt,framesep=1.6mm,fontsize=\\scriptsize]",
)
body = body.replace("\\end{verbatim}", "\\end{Verbatim}\n\\end{minipage}\\medskip")

# tables: drop pandoc's minipage header cells, rewrap longtable (illegal in
# two-column mode) as an IEEE table float with the caption above
body = re.sub(
    r"\\begin\{minipage\}\[[bt]\]\{\\linewidth\}\\raggedright\s*(.*?)\s*\\end\{minipage\}",
    r"\1",
    body,
    flags=re.S,
)
caption_iter = iter(captions)


def table_open(_m):
    lab, cap = next(caption_iter)
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
# pandoc's longtable footer rule lands under the header once those markers are
# stripped; the real bottom rule is added at \end{tabular} above
body = body.replace("\\midrule\n\\bottomrule", "\\midrule")
body = body.replace("\\toprule\n\\bottomrule", "\\toprule")

bibitems = []
for i, para in enumerate(re.split(r"\n\n+", pandoc(refs_md).strip()), 1):
    entry = re.sub(r"^\s*(\{\[\}|\[)\d+(\{\]\}|\])\s*", "", para.strip())
    bibitems.append(f"\\bibitem{{r{i}}} {entry}")

# balance the final page's two reference columns; None when they fill naturally
TRIGGER = None

if CHECK_ONLY:
    print(
        f"build.py --check: {len(sections)} sections, {n_tables} tables, "
        f"{len(bibitems)} references, markers resolved. paper.md is buildable."
    )
    raise SystemExit(0)

tex = r"""\documentclass[conference]{IEEEtran}
\usepackage{array}
\usepackage{booktabs}
\usepackage{calc}
\usepackage{url}
\Urlmuskip=0mu\relax
\hyphenation{off-line}
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
\IEEEauthorblockA{New York University \\ New York, NY, USA \\ msr541@nyu.edu \\ ECE-GY 9953 Advanced Project. Advisor: Prof.\ Brandon Reagen}}
\maketitle
\begin{abstract}
%s
\end{abstract}
\begin{IEEEkeywords}
GPU kernel optimization, LLM agents, correctness verification, edge computing, Vulkan
\end{IEEEkeywords}
%s
%s
\begin{thebibliography}{%d}
\scriptsize
%s
\end{thebibliography}
\end{document}
""" % (
    title,
    abstract,
    body,
    f"\\IEEEtriggeratref{{{TRIGGER}}}" if TRIGGER else "%",
    len(bibitems),
    "\n\n".join(bibitems),
)

(HERE / "paper_ieee.tex").write_text(tex)
for _ in range(2):
    # pdflatex logs are not UTF-8 when a warning echoes a multibyte source line
    r = subprocess.run(
        ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode", "paper_ieee.tex"],
        cwd=HERE,
        capture_output=True,
        text=True,
        errors="replace",
    )
if r.returncode != 0:
    print("\n".join(ln for ln in r.stdout.splitlines() if ln.startswith("!") or "Error" in ln))
    die("pdflatex failed; paper.pdf left unchanged")

overfull = (HERE / "paper_ieee.log").read_text(errors="replace").count("Overfull \\hbox")
(HERE / "paper_ieee.pdf").replace(HERE / "paper.pdf")
pages = subprocess.run(["pdfinfo", str(HERE / "paper.pdf")], capture_output=True, text=True).stdout
pages = next((ln.split()[-1] for ln in pages.splitlines() if ln.startswith("Pages")), "?")
print(
    f"built paper.pdf: {pages} pages, {n_tables} tables, {len(bibitems)} references, "
    f"{overfull} overfull boxes"
)
