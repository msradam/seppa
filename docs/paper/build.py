"""Build paper.pdf from paper.md via the official IEEEtran LaTeX class.
Usage: python3 build.py  (writes paper_ieee.tex, compiles paper.pdf)
IEEEtran.cls is vendored in this directory (CTAN, V1.8b).
"""

import re
import subprocess
from pathlib import Path

src = Path("paper.md").read_text()

title = re.match(r"# (.+)", src).group(1)
abstract_md = src[src.index("## Abstract") + len("## Abstract") : src.index("## 1. Introduction")].strip()
body_md = src[src.index("## 1. Introduction") : src.index("## References")]
refs_md = src[src.index("## References") :].split("\n", 1)[1].strip()

# strip manual numbers; IEEEtran numbers headings IEEE-style (I., A.)
body_md = re.sub(r"^## \d+\. ", "# ", body_md, flags=re.M)
body_md = re.sub(r"^### \d+\.\d+ ", "## ", body_md, flags=re.M)
# textual cross-references -> IEEE roman style (longest first)
for a, b in [("4.1", "IV-A"), ("4.2", "IV-B"), ("4.3", "IV-C"), ("6.1", "VI-A"),
             ("10", "X"), ("2", "II"), ("3", "III"), ("4", "IV"), ("5", "V"),
             ("6", "VI"), ("7", "VII"), ("8", "VIII"), ("9", "IX")]:
    body_md = body_md.replace(f"Section {a}", f"Section {b}")


def pandoc(text):
    return subprocess.run(
        ["pandoc", "-f", "markdown+autolink_bare_uris", "-t", "latex",
         "--syntax-highlighting=none"],
        input=text, capture_output=True, text=True, check=True,
    ).stdout


body = pandoc(body_md)
abstract = pandoc(abstract_md).strip()

# longtable cannot appear in two-column mode; rewrap as plain tabular
body = body.replace("\\begin{longtable}[]{", "\\begin{center}\\footnotesize\\begin{tabular}{")
body = body.replace("\\end{longtable}", "\\end{tabular}\\end{center}")
body = body.replace("\\noalign{}", "")
body = re.sub(r"^\\end(first)?head\n", "", body, flags=re.M)
body = re.sub(r"^\\end(last)?foot\n", "", body, flags=re.M)

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
\newcommand{\real}[1]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\makeatletter
\def\verbatim@font{\ttfamily\scriptsize}
\makeatother
\begin{document}
\title{%s}
\author{\IEEEauthorblockN{Adam Munawar Rahman}
\IEEEauthorblockA{New York University \\ New York, NY, USA \\ msr541@nyu.edu}}
\maketitle
\begin{abstract}
%s
\end{abstract}
\begin{IEEEkeywords}
GPU kernel optimization, LLM agents, correctness verification, edge computing, Vulkan
\end{IEEEkeywords}
%s
\begin{thebibliography}{%d}
\footnotesize
%s
\end{thebibliography}
\end{document}
""" % (title, abstract, body, len(bibitems), "\n\n".join(bibitems))

Path("paper_ieee.tex").write_text(tex)
for _ in range(2):
    r = subprocess.run(
        ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode", "paper_ieee.tex"],
        capture_output=True, text=True,
    )
if r.returncode != 0:
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("!") or "Error" in l))
    raise SystemExit(1)
Path("paper_ieee.pdf").replace("paper.pdf")
print("built paper.pdf")
