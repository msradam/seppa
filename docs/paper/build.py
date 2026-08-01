"""Build paper.pdf from paper.md via the charged-ieee typst template.
Usage: python3 build.py  (writes paper_ieee.typ, compiles paper.pdf)
"""

import re
import subprocess

src = open("paper.md").read()

title = re.match(r"# (.+)", src).group(1)
abstract = src[src.index("## Abstract") + len("## Abstract") : src.index("## 1. Introduction")].strip()
body_md = src[src.index("## 1. Introduction") :]

# strip manual numbers; charged-ieee numbers headings IEEE-style (I., A.)
body_md = re.sub(r"^## \d+\. ", "# ", body_md, flags=re.M)
body_md = re.sub(r"^### \d+\.\d+ ", "## ", body_md, flags=re.M)
# textual cross-references -> IEEE roman style (longest first)
for a, b in [("4.1", "IV-A"), ("4.2", "IV-B"), ("4.3", "IV-C"), ("6.1", "VI-A"),
             ("2", "II"), ("3", "III"), ("4", "IV"), ("5", "V"), ("6", "VI"),
             ("7", "VII"), ("8", "VIII"), ("9", "IX"), ("10", "X")]:
    body_md = body_md.replace(f"Section {a}", f"Section {b}")

open("/tmp/_body.md", "w").write(body_md)
subprocess.run(["pandoc", "/tmp/_body.md", "-t", "typst", "-o", "/tmp/_body.typ"], check=True)
body = open("/tmp/_body.typ").read()
# headings: pandoc H2 -> "= "; References must be unnumbered
body = body.replace("= References", '#heading(numbering: none)[References]\n#set text(size: 0.82em)')

subprocess.run(["pandoc", "-t", "typst", "-o", "/tmp/_abs.typ"], input=abstract, text=True, check=True)
abs_typ = open("/tmp/_abs.typ").read().strip()

wrapper = f'''#import "@preview/charged-ieee:0.1.4": ieee
#show: ieee.with(
  title: [{title}],
  abstract: [{abs_typ}],
  authors: ((
    name: "Adam Munawar Rahman",
    organization: [New York University],
    location: [New York, NY, USA],
    email: "msr541@nyu.edu",
  ),),
  index-terms: ("GPU kernel optimization", "LLM agents", "correctness verification", "edge computing", "Vulkan"),
)
#show table: set text(size: 0.82em)
#show raw.where(block: true): set text(size: 0.78em)

{body}
'''
open("paper_ieee.typ", "w").write(wrapper)
subprocess.run(["typst", "compile", "paper_ieee.typ", "paper.pdf"], check=True)
print("built paper.pdf")
