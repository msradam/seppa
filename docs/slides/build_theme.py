#!/usr/bin/env python3
"""Generate themes/ieee.css from themes/ieee.src.css with fonts inlined.

Marp inlines a theme's CSS into the output document, so relative `url()` paths
in the theme resolve against the output location rather than the theme file.
Embedding the fonts as data URIs makes the theme work for any output path.
"""

import base64
import re
from pathlib import Path

HERE = Path(__file__).parent
FONTS = HERE / "fonts"


def data_uri(name: str) -> str:
    payload = base64.b64encode((FONTS / name).read_bytes()).decode()
    return f"data:font/ttf;base64,{payload}"


src = (HERE / "themes" / "ieee.src.css").read_text()
out = re.sub(
    r"url\('\.\./fonts/([^']+)'\)",
    lambda m: f"url('{data_uri(m.group(1))}')",
    src,
)
(HERE / "themes" / "ieee.css").write_text(out)
print(f"themes/ieee.css: {len(out) / 1024:.0f} KB")
