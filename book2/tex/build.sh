#!/usr/bin/env bash
# Build The Reformed School: convert the proofread Markdown texts and apparatus to LaTeX
# with pandoc, then typeset with LuaLaTeX via latexmk. Run from book2/tex/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build texts_tex heads_tex front back

# Convert each reading text. The metadata block at the top of a text file (everything before
# the first blank-line-separated body, delimited by a line of three or more dashes) is dropped;
# the book's own headnote carries the edition statement instead.
python3 md2tex.py

latexmk -lualatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >build/latexmk.log 2>&1 \
  || { echo "latexmk failed; see build/latexmk.log"; grep -nE "^!|Error" build/main.log | head -20; exit 1; }
cp build/main.pdf "The_Reformed_School.pdf"
pdfinfo The_Reformed_School.pdf | grep -E "Pages|Page size"
