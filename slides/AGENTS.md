# AGENTS.md

This file provides guidance to coding agents working on this presentation (Claude Code reads
it via `CLAUDE.md`).

> **This is a live document.** Whenever the talk's content, structure, figures or workflow
> change, update the matching section here as part of the same task.

## What this is

A LaTeX Beamer presentation: *"Software engineering practices and AI tools for research code"*
(William S Hart, Infectious Disease Modelling Group meeting, Mathematical Institute, Oxford,
29 October 2026). The deliverable is [main.pdf](main.pdf), built from [main.tex](main.tex).

A ~60-minute group-meeting talk including two live demos, in the formal structure: two
sections with a contents divider before the second, and a Summary (no Acknowledgements).

1. **Practices and tooling:**
   - a "Nine practices" overview;
   - one slide per practice, ordered by importance score;
   - "What goes in a repo" (a colour-coded tree of essential / good practice / my choice);
   - "Organising code" and "Documenting code";
   - the repo demo and further reading.
2. **AI tools:**
   - chat vs coding agents;
   - how much to delegate;
   - uses;
   - working with an agent (prompting, sessions);
   - plan the science before the code (what you decide, and AI's help with the maths);
   - plan first (the agent's plan-review-implement loop), instruction files and skills;
   - other useful features;
   - the Claude Code demo;
   - two risk slides, and "You own every line, whoever wrote it";
   - a closing infographic with no frame title (it carries its own headline).

The final **Summary** is a single 2:1 illustrated workflow: plan, implement, validate and
reproduce, supported by version control, a locked environment, tests and automated checks,
and README plus agent instructions. Validation shows code matching known answers. It fills
the slide body, with no frame title or separate "Questions?" text.

**Section 2 teaches general principles, not Claude Code instructions** (the speaker's
preference). Tool-specific commands, key presses and file-loading rules go in the speaker
notes. Claude Code docs links stay as small references (`\docref`, `\doclink`).

The repo around this folder is the worked example, shown live rather than excerpted or linked
on the slides. Content is drafted and has been through a content and a design review. The
title is a working title, the demo slides are checklists, and some speaker notes are empty
stubs.

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
built-in `image_gen` tool. The v1 and v2 generation prompts and local refinement prompts are
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
  section after the first opens with a contents divider.
- Left-align text. Centre images, but keep a caption that follows them left-aligned.
- For a full-width image at frame level, use `\centerline{\includegraphics[...]{f}}`, never
  `{\centering ...\par}`, which silently drops the image in a vertically centred frame.
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
  - `oxred` for risks, `curvegreen` for mitigations, `oxgrey` for secondary text.
- **Practice slides** are `[t]` frames:
  - `\practicehead{fa-icon}{score}{one-line pitch}`;
  - `columns[t]` 0.54/0.42, each column starting with `\vspace{0pt}`: three or four bullets
    on the left, and a `toolpanel` of `\toolrow{label}{chips}` on the right;
  - chips are `\pick{}` (my choice, filled teal) and `\opt{}` (alternative, outlined);
    `\alt` is taken by beamer;
  - `\practicetracker{k}` at the end of the frame.
  - The tracker's icon list and the "Nine practices" tiles must match the order of the
    slides (by score).
- **Other patterns:**
  - `\meter{n}` draws the importance meter on its own.
  - `riskcard{title}` holds an `\eg{}` example line and then `\fix` for the mitigation, in a
    `tcbraster`.
  - `infopanel{colour}{title}` is the general content panel (`\small`, coloured left edge and
    title), used on the plan slides.
  - `promptbox{colour}{title}` is used for prompt examples, `scopecard` for the
    instruction-file scopes, and `demobox` with `\demoitem` for the demo slides.
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
  - It uses `dirtree`, with entries coloured by `\ess` / `\good` / `\mine` and comments in
    `\cmt`.
  - Its line spacing is set inside `\DTstyle`, because dirtree applies that style to each
    entry.
- **Bold:** `\renewcommand{\bfdefault}{bx}` after `charter` is deliberate. Without it,
  `\textbf` renders in medium weight, because beamer's sans body font has no `b` series.
