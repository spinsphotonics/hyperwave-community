#!/usr/bin/env python3
"""Convert the proofread reading texts and the apparatus from Markdown to LaTeX.

Reads tex/manifest.json, which lists the texts in book order:
  {"chapters": [{"slug": "01-luther-1524-letter", "title": "...", "short": "...",
                 "headnote": "heads/01-luther-1524-letter.md",
                 "text": "../texts/01-luther-1524-letter.md",
                 "sourceline": "..."}, ...]}
For each chapter it converts the headnote and the text with pandoc (LaTeX output, no wrapping),
drops the text file's metadata block (everything up to and including the first line consisting
of three or more dashes, if the file begins with such a block), turns the omission marker lines
into \\cut, and writes texts.tex which \\inputs them all in order.
"""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
OUT_TEXTS = HERE / "texts_tex"
OUT_HEADS = HERE / "heads_tex"
OMISSION = re.compile(r"^\s*(\[\s*\.\.\.\s*\]|\[…\]|\.\s*\.\s*\.|…|·\s*·\s*·|\*\s*\*\s*\*)\s*$")


def strip_metadata(md: str) -> str:
    lines = md.splitlines()
    # a metadata block is a leading run of non-blank lines ending at a '---' line,
    # or a leading YAML-style block delimited by '---' lines
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1:])
    for i, line in enumerate(lines[:40]):
        if re.match(r"^-{3,}\s*$", line):
            return "\n".join(lines[i + 1:])
    return md


def mark_cuts(md: str) -> str:
    out = []
    for line in md.splitlines():
        out.append("\\cut" if OMISSION.match(line) else line)
    return "\n".join(out)


def pandoc(md: str, top_level: str = "section") -> str:
    r = subprocess.run(
        ["pandoc", "-f", "markdown+smart-auto_identifiers", "-t", "latex", "--wrap=none",
         f"--top-level-division={top_level}"],
        input=md, capture_output=True, text=True, check=True)
    tex = r.stdout
    # headings inside a text are unnumbered subsections; keep pandoc's \section for the
    # text's own internal divisions but strip the numbering
    tex = tex.replace("\\section{", "\\section*{").replace("\\subsection{", "\\subsection*{")
    tex = tex.replace("\\section*{", "\\section*{").replace("\\subsubsection{", "\\subsubsection*{")
    return tex


def main():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_TEXTS.mkdir(exist_ok=True)
    OUT_HEADS.mkdir(exist_ok=True)
    parts = []
    for ch in m["chapters"]:
        slug = ch["slug"]
        text_md = (HERE / ch["text"]).read_text(encoding="utf-8")
        body = mark_cuts(strip_metadata(text_md))
        (OUT_TEXTS / f"{slug}.tex").write_text(pandoc(body), encoding="utf-8")
        head_path = HERE / ch["headnote"]
        head_tex = pandoc(head_path.read_text(encoding="utf-8")) if head_path.exists() else "\\emph{Headnote to come.}"
        (OUT_HEADS / f"{slug}.tex").write_text(head_tex, encoding="utf-8")
        parts.append(
            f"\\chapter[{ch.get('short', ch['title'])}]{{{ch['title']}}}\n"
            f"\\sourceline{{{ch.get('sourceline', '')}}}\n"
            f"\\begin{{headnote}}\n\\input{{heads_tex/{slug}}}\n\\end{{headnote}}\n"
            f"\\input{{texts_tex/{slug}}}\n")
    (HERE / "texts.tex").write_text("\n".join(parts), encoding="utf-8")
    print(f"converted {len(parts)} texts")


if __name__ == "__main__":
    main()
