# AGENTS.md

This file provides guidance to coding agents working on this presentation (Claude Code reads
it via `CLAUDE.md`).

> **This is a live document.** Whenever the talk's content, structure, figures or workflow
> change, update the matching section here as part of the same task.

## What this is

A LaTeX Beamer presentation: *"Software engineering practices and AI tools for research code"*
(William S Hart, Infectious Disease Modelling Group meeting, Mathematical Institute, Oxford,
29 October 2026). The deliverable is [main.pdf](main.pdf), built from [main.tex](main.tex).

A ~60-minute group-meeting talk, in the formal structure: two sections, each opened by a
contents divider, and a Summary (no Acknowledgements). There are no demo slides.

1. **Practices and tooling:**
   - "Organising code", introducing the package / analysis / results / figures workflow;
   - a "Nine practices" overview, with the key to the adoption-priority meter and tool chips;
   - one slide per practice, ordered by priority score;
   - "What goes in a repo" (colours show essential / good practice; asterisks mark choices
     of format, tool or layout);
   - "Documenting code";
   - "Additional resources" (papers, a handbook and Oxford RSE training).
2. **AI tools:**
   - chat vs coding agents;
   - how much to delegate (you write on the left, full vibe on the right);
   - plan the science before the code (make scientific choices explicit and record an agreed
     specification; AI can help develop it);
   - uses;
   - prompting, planning with an agent, then sessions and context;
   - instruction files and skills;
   - other useful features (an optional reference slide);
   - two risk slides;
   - a closing infographic with no frame title (it carries its own headline).

The final **Summary** is a single 2:1 illustrated workflow: plan, implement, validate and
reproduce, supported by version control, a recorded environment, automated checks, and README
plus agent instructions. Validation shows code matching known answers. It has one headline,
"You own every line, whoever wrote it", two advice lines per stage, and no bottom sentence. It
fills the slide body, with no frame title or separate "Questions?" text.
There is no separate ownership slide. "This talk" includes an explicit AI-assistance
disclaimer asking the audience to check advice against documentation and their project's needs.

The running example varies R0 and computes and plots the final size (fraction ever infected)
in an SIR model. The AI checking and testing slide shows two example agent requests: review
an implementation against a model description and report concrete findings without editing;
inspect an existing module and suite, add missing unit tests, run them and show the results.
The prompting example specifies the symptom, file, expected display, unchanged saved units
and regenerated figure, following the linked best-practices guide. Workflow advice focuses on
explicit dependencies, rerunning affected steps and running jobs on HPC clusters. Documentation advice
explicitly discourages commented-out code. The correctness risks include weak tests and gaming
the checks.

Scores describe personal adoption priorities for a typical analysis repo. Formal testing
stays at 4: basic scientific checks are essential. Its slide focuses on known answers, edge
cases, important rules and tests that prevent fixed bugs from returning; RNG details and TDD
are in the speaker notes. Editor and debugger is 5, immediately
before formal testing. Its score is for using a capable editor; learning debugger features is
a useful next step. The editor slide also covers reviewing and editing agents' changes.
An environment specification is essential;
a lockfile is good practice, not mandatory. CI and archiving have context-specific ratings.
Tool panels retain options across Python, R and Julia; outlined chips include complementary
tools as well as alternatives. Repo-layout alternatives are in the speaker notes.

**Section 2 teaches general principles, not Claude Code instructions** (the speaker's
preference). Tool-specific commands, key presses and file-loading rules go in the speaker
notes. Claude Code docs links stay as small references (`\docref`, `\doclink`).

The repo around this folder is the worked example, shown live rather than excerpted or linked
on the slides. Content is drafted and has been through a content and a design review. The
title is a working title, and some speaker notes are empty stubs.

## Building

```bash
./build.sh   # latexmk -pdf -interaction=nonstopmode -f main.tex
```

- The engine is pdflatex (`% !TEX program = pdflatex` at the top of main.tex).
- The bibliography is biblatex with the bibtex backend, and latexmk runs bibtex itself.
  `main-blx.bib`, `main.bbl`, `main.aux`, `main.nav` and the like are generated, so don't
  edit them.
- After a build, look at the changed slides. In Claude Code, `Read main.pdf pages=N` renders
  them, which needs poppler (`pdftoppm`, from Homebrew in `/opt/homebrew/bin`) on the harness
  PATH. Also run `grep -n Overfull main.log`.

## Figures

[figures/](figures/) is on `\graphicspath`. Figures taken from papers or other outside sources
are named `<source>__<name>.<ext>`, where the prefix records provenance (for example
`thompson__fig1.pdf` for a figure from the paper cited as `thompson2024ebola`). List the
prefixes here as they are added:

| Prefix | Source |
| ------ | ------ |

The Summary infographic is [figures/summary.png](figures/summary.png), generated with the
built-in `image_gen` tool. The v1--v4 generation/edit prompts and local refinement prompts are
kept in [figures/summary.prompt.txt](figures/summary.prompt.txt).

Figures from this repo's analysis are not copied by hand.
[update_figures.sh](update_figures.sh) mirrors `../figures/**.{svg,pdf}` into `figures/` as
cropped PDFs, keeping their paths. Re-run it after the figures are regenerated. Don't run the
analysis pipeline just to refresh the slides.

[figures/Template/](figures/Template/) holds the theme's logos and title backgrounds, not
content. Leave it alone.

## Slide conventions

- Theme `bill` (amended Oxford Maths theme, [beamerthemebill.sty](beamerthemebill.sty)), 16:9,
  charter font. `\usetheme[wellcome]{bill}` puts the Wellcome logo on the title slide.
- The footer shows the short title and short date: `\title[Short]{Full}`,
  `\date[1 January 2026]{Venue, 1 January 2026}`.
- `\section{Title}` takes no manual "N." prefix, because the theme numbers sections. Every
  section, the first included, opens with a contents divider (`\AtBeginSection` has no
  section-number guard). Dividers are numbered like any frame (no `[noframenumbering]`, which
  would repeat the previous slide's number in the footer).
- Left-align text. Centre images, but keep a caption that follows them left-aligned.
- For a full-width image at frame level, use `\centerline{\includegraphics[...]{f}}`, never
  `{\centering ...\par}`, which silently drops the image in a vertically centred frame.
- A lead-in paragraph at the start of a vertically centred frame should use a `minipage`;
  a font/colour group ending in `\par` can also silently disappear there.
- Size figures by height (`height=0.7\textheight,keepaspectratio`) and crop with
  `trim=l b r t,clip` (units are bp).
- Put a slide's paper reference on its own line at the bottom-left with `\sourceref{key}`.
  Cite inline with `\citewithjournalurl{key}`, never plain `\cite`. Add sources to
  [references.bib](references.bib) with a doi.
- Acknowledgements: one flat list, grant PIs first, then everyone else alphabetical by
  surname, with no titles.

## This deck's macros and patterns

- **Palette** (defined in the preamble):
  - `oxtint` for panels and block bodies;
  - `oxteal` for the accent: preferred tools, `\key{}` words and links;
  - `oxamber` for the importance meter and for "you" in diagrams, against `oxfordblue` for
    the agent; keep this pairing consistent, as in the delegation axis and the plan loop;
  - `oxambertext` is the darker orange for text and essential repo entries; essential
    filenames are also bold, so colour is not their only distinction;
  - `oxred` for risks, `curvegreen` for mitigations, `oxgrey` for secondary text.
- **Practice slides** are `[t]` frames:
  - `\practicehead{fa-icon}{score}{one-line pitch}`;
  - `columns[t]` 0.54/0.42, each column starting with `\vspace{0pt}`: three or four bullets
    on the left, and a `toolpanel` of `\toolrow{label}{chips}` on the right;
  - chips are `\pick{}` (my choice, filled teal) and `\opt{}` (other tools, outlined);
    `\alt` is taken by beamer;
  - `\practicetracker{k}` at the end of the frame.
  - The tracker's icon list and the "Nine practices" tiles must match the order of the
    slides (by score). Testing is labelled "Formal testing" in the overview and its slide.
- **Other patterns:**
  - `\meter{n}` draws the importance meter on its own.
  - `riskcard{title}` holds an `\eg{}` example line and then `\fix` for the mitigation, in a
    `tcbraster`.
  - `infopanel{colour}{title}` is the general content panel (`\small`, coloured left edge and
    title), used on the plan slides.
  - `promptbox{colour}{title}` is used for prompt examples and `scopecard` for the
    instruction-file scopes. `demobox` with `\demoitem` (a dark "terminal" panel) is kept
    for demo slides but currently unused.
  - `\tileshead{colour}{title}{caption}` plus `\tilestack{colour}{icon/text, ...}` give a
    column of fixed-height, one-line icon tiles (the "Uses: grunt work" slide). Keep each
    text to one line, and brace any text that contains commas.
- **Boxes:** tcolorbox's `frame hidden` only works together with `enhanced`.
- **Docs links:** `\docref{page}` pins "Docs: code.claude.com/docs/en/<page>" bottom-left
  (like `\sourceref`); `\doclink{page}{text}` is the inline form. Check that each page
  resolves when adding one.
- **Link colours:** set them with `\hypersetup` after the theme. `\PassOptionsToPackage` is
  too late, because beamer loads hyperref first.
- **Tables** use the ragged-right `L{width}` column type. Keep the widths plus the column gaps
  under `\textwidth`.
- **Repo tree:**
  - It uses `dirtree`, with entries coloured by `\ess` / `\good` according to the importance
    of the practice, comments in `\cmt`, and `\choice` asterisks for specific formats, tools
    or layouts that can be replaced. Personal choice is not a colour category.
  - Its line spacing is set inside `\DTstyle`, because dirtree applies that style to each
    entry.
- **Bold:** `\renewcommand{\bfdefault}{bx}` after `charter` is deliberate. Without it,
  `\textbf` renders in medium weight, because beamer's sans body font has no `b` series.
