# idm-demo-repo

Slides and an example repository skeleton for a talk to the Oxford Maths infectious disease
modelling group on software engineering practices and AI tool use in research code. The
repository itself illustrates the layout and tooling set-up the talk describes; the slides are
in [`slides/`](slides/).

## Analyses

- **`generation_time`**: an independent reproduction of Fig. 1 of Hart et al., "Inference of
  the SARS-CoV-2 generation time using UK household data", *eLife* 11:e70767 (2022),
  [doi:10.7554/eLife.70767](https://doi.org/10.7554/eLife.70767). It fits the paper's two
  models of infectiousness (independent transmission and symptoms, and a mechanistic
  latent/presymptomatic/symptomatic stage model) to the UK household study data in
  [`data/`](data/). It uses the paper's likelihood and priors, but a different
  (data-augmentation MCMC) sampler written in [JAX](https://docs.jax.dev/). Posterior
  summaries, set beside the paper's estimates, are in
  [`results/generation_time/summary.csv`](results/generation_time/summary.csv) and the figure
  is [`figures/generation_time/posterior_violins.svg`](figures/generation_time/posterior_violins.svg).

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

This takes under a minute on a 10-core laptop, mostly MCMC (chains run in parallel, one per
core).

## Slides

The talk is a Beamer deck in the Oxford Maths theme, [`slides/main.tex`](slides/main.tex), and
the compiled [`slides/main.pdf`](slides/main.pdf) is committed alongside it. Rebuilding it
needs `latexmk`, `pdflatex` and `bibtex` from a TeX distribution (MacTeX, TeX Live), which
pixi does not provide:

```bash
slides/build.sh
```
