# AGENTS.md

Project conventions for coding agents in this repo (Claude Code reads it via `CLAUDE.md`).
Personal preferences live in the gitignored `CLAUDE.local.md`.

## Project

Slides and an example repo skeleton for a talk to the Oxford Maths infectious disease modelling
group on software engineering practices and AI tool use. The repo illustrates the layout and
tooling for research code: a library package drives a Snakemake workflow that produces the
numerical results and figures. The Beamer deck in `slides/` is the talk itself.

**Status: skeleton only.** There is no library code, analysis or Snakemake rule yet. The deck
is drafted (see `slides/AGENTS.md`). Keep this file in step as code lands: describe what
exists, not what is planned.

## Layout

- `src/idmdemo/` — library package (src layout, installed editable by pixi, so edits take
  effect immediately). `__init__.__all__` is the public API. Modules lead with their public
  functions; private `_helpers` sit below them. Keep imports between modules one-way, with no
  cycles.
- `scripts/` — analysis + plot script pairs plus shared `inputs.py` (see below). A package at
  the repo root, **not** under `src/`.
- `tests/` — top-level pytest suite (not nested under the package).
- `Snakefile` — orchestrates the full results-and-figures pipeline.
- `data/` — inputs. `results/`, `figures/` — generated outputs, committed.
- `slides/` — the talk: a Beamer deck in the amended Oxford Maths theme (`bill`), with its own
  `AGENTS.md` for slide conventions. `main.tex` and the compiled `main.pdf` are committed;
  `slides/.gitignore` ignores the build files. Built by hand with system TeX, not by pixi or
  Snakemake (see Commands). Rebuild the PDF after editing the deck.

## Running scripts

Scripts are run **as modules from the repo root**: `pixi run python -m scripts.<name>`. Never
`python scripts/<name>.py`: with the src layout, the editable install puts only `src/` on
`sys.path`, so `from scripts.inputs import ...` fails when a script is run by path. pytest gets
the root via `pythonpath = ["."]` in `pyproject.toml`.

## Snakemake workflow

The Snakefile is the single source of truth for reproduction. **Don't add ad-hoc reproduction
commands** — wire them in as rules.

- `all` (default) builds every figure; `results` builds every results file only.
- Each analysis gets a `<name>_results` rule running `pixi run python -m scripts.<name> -r` and a
  `<name>_plots` rule running `pixi run python -m scripts.<name>_plots`.
- Every rule lists `shared_input_paths` (all package modules, globbed) plus its own script(s),
  data files and upstream results as inputs, so a change to shared code triggers a rebuild.
  Once `scripts/inputs.py` exists, add it to `shared_input_paths`.
- Add each analysis's output paths to the `results_paths` / `figure_paths` aggregates.

## Analysis vs plotting convention

Every analysis is a **script pair**: `scripts/<name>.py` does the heavy compute,
`scripts/<name>_plots.py` makes the figures.

- `<name>.py` always takes `-r/--results-only` (skip plotting). Without `-r` it also calls the
  plot script's `make_plots`, which is handy for one-shot local iteration but never the
  Snakemake path.
- **All input/parameter assembly lives in `scripts/inputs.py`** as `get_inputs_<name>(...)`,
  returning a dict that includes `results_paths`, `fig_paths` and the prepared
  arrays/parameters. Both scripts in a pair call it; never duplicate path construction.
- Outputs follow `results/<name>/<output>.csv` and `figures/<name>/<output>.svg`. Snakemake
  `output:` lists assume this.
- Each script has a single public entry point (`run_analyses` / `make_plots`) at the top; every
  other function is `_private` and sits below it. A helper that is a general reusable task (not
  one analysis's step) and is used across scripts belongs in the package instead. Don't factor
  a one-liner into a helper just because it repeats.
- If paper figures are compiled from per-analysis panels, figure numbers live only in that
  compile script and the Snakefile. Don't put figure or panel numbers in other scripts'
  docstrings or comments; they go stale on any renumbering.

To add an analysis: write the script pair, add `get_inputs_<name>()` to `inputs.py`, and add
`<name>_results` + `<name>_plots` rules to the Snakefile.

## Tests

- `pixi run test` (all tests); `pixi run pytest tests/test_x.py -k name` (a single test).
- `tests/test_public_api.py` asserts the exact set of names in `idmdemo.__all__`. **If the
  exports change, update the package `__init__` and this test in the same change.**
- Test files written with AI assistance start with the line
  `# Note that AI tools were used to generate tests`. Keep or add it when generating or
  substantially modifying a test file.
- Tests can and should be edited to enable code improvements. Don't preserve a test, or contort
  the code, just to avoid touching it; keep the suite sensible and green.

## Commands

| Task | Command |
|------|---------|
| Tests | `pixi run test` |
| Format | `pixi run format` (ruff import sort + format) |
| Lint | `pixi run lint` (ruff check) |
| Type check | `pixi run typecheck` (ty check) |
| Full reproduction | `pixi run snakemake --cores N` |
| Results only | `pixi run snakemake --cores N results` |
| Slides | `slides/build.sh` (system TeX, outside pixi) |

Run format, lint and typecheck after code changes.

## Tooling

- The only deliberate ruff config is import sorting (`extend-select = ["I"]`); the `format` task
  runs `ruff check --select I --fix && ruff format`. Otherwise ruff and ty use their defaults on
  purpose, so don't add other config.
- New dependencies go in `pyproject.toml` (conda-forge first, `pypi-dependencies` only if
  needed; dev-only tools in the `dev` feature). Commit the updated `pixi.lock`.
- matplotlib SVG output is non-deterministic (run-varying IDs, float noise), so `git diff
  figures/` can't show whether a plotting change preserved the output. Check by running the
  plot script and looking at the figure.

## Repo

- GitHub: `will-s-hart/idm-demo-repo` (private). Main branch is `main`.
