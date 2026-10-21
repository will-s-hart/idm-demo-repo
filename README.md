# idm-demo-repo

Slides and an example repository skeleton for a talk to the Oxford Maths infectious disease
modelling group on software engineering practices and AI tool use in research code. The
repository itself illustrates the layout and tooling set-up the talk describes; the slides are
in [`slides/`](slides/).

This project is at an early stage: the skeleton contains no analyses yet.

## Installation

We recommend using the [pixi](https://pixi.prefix.dev/) package manager. Once pixi is
installed, running

```bash
pixi install --locked
```

creates a local environment (under `.pixi/`) containing the exact dependency versions
recorded in `pixi.lock`. Prefixing a command with `pixi run` executes it in that environment.

## Reproducing the results

Individual analyses are run as modules of the [`scripts`](scripts/) package from the
repository root, e.g.

```bash
pixi run python -m scripts.<name>
```

Alternatively, the full pipeline for all results and figures can be run with
[Snakemake](https://snakemake.readthedocs.io/):

```bash
pixi run snakemake --cores 1 --forceall
```

## Slides

The talk is a Beamer deck, [`slides/slides.tex`](slides/slides.tex), and the compiled
[`slides/slides.pdf`](slides/slides.pdf) is committed alongside it. Rebuilding it needs
`latexmk` and `pdflatex` from a TeX distribution (MacTeX, TeX Live), which pixi does not
provide:

```bash
latexmk -pdf -cd slides/slides.tex
```
