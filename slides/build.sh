#!/bin/sh
# Build main.pdf. latexmk reruns pdflatex and bibtex as needed; -f carries on past errors so
# one broken slide doesn't stop the rest of the deck building.
cd "$(dirname "$0")" || exit 1
latexmk -pdf -interaction=nonstopmode -f main.tex
