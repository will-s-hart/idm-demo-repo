import glob

# Paths mirror get_inputs_generation_time() in scripts/inputs.py.
generation_time_results = [
    "results/generation_time/posterior_independent.csv",
    "results/generation_time/posterior_mechanistic.csv",
    "results/generation_time/summary.csv",
    "results/generation_time/diagnostics.csv",
]
generation_time_figures = ["figures/generation_time/posterior_violins.svg"]

results_paths = generation_time_results
figure_paths = generation_time_figures

# Globbed rather than listed by hand: every package module is an input to every analysis, and
# an omission here silently stops results being rebuilt when that module changes.
package_paths = sorted(glob.glob("src/idmdemo/*.py"))
shared_input_paths = package_paths + ["scripts/inputs.py"]


rule all:
    input:
        figure_paths,


rule results:
    input:
        results_paths,


# Fits the independent and mechanistic infectiousness models to the UK household study data
# (Hart et al., eLife 2022, Fig. 1). Takes about 2 minutes.
rule generation_time_results:
    input:
        shared_input_paths,
        "scripts/generation_time.py",
        "data/hart_elife_data.xlsx",
        "data/hart2022_estimates.csv",
    output:
        generation_time_results,
    shell:
        "pixi run python -m scripts.generation_time -r"


rule generation_time_plots:
    input:
        shared_input_paths,
        "scripts/generation_time_plots.py",
        generation_time_results,
    output:
        generation_time_figures,
    shell:
        "pixi run python -m scripts.generation_time_plots"
