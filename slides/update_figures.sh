#!/bin/sh
# Refresh slides/figures from the repo's generated figures.
#
# Mirrors ../figures/<path>.svg (via Inkscape) and ../figures/<path>.pdf to figures/<path>.pdf,
# cropping each to its ink bounding box so beamer does not need per-figure trim= values. Only
# sources newer than their copy are converted. Figures from elsewhere (figures/Template/,
# <source>__<name> files taken from papers) are left alone.
#
# Run after regenerating the repo's figures. Requires inkscape and pdfcrop.
set -eu

cd "$(dirname "$0")"
src=../figures
dst=figures

find "$src" \( -name '*.svg' -o -name '*.pdf' \) | while read -r fig; do
    out="$dst/${fig#"$src"/}"
    out="${out%.*}.pdf"
    [ "$out" -nt "$fig" ] && continue
    mkdir -p "$(dirname "$out")"
    case "$fig" in
        *.svg) inkscape --export-type=pdf --export-filename="$out" "$fig" >/dev/null 2>&1 ;;
        *.pdf) cp "$fig" "$out" ;;
    esac
    pdfcrop --margins 3 "$out" "$out.crop" >/dev/null
    mv "$out.crop" "$out"
    echo "$out"
done
