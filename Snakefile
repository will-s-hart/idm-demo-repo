import glob

results_paths = []
figure_paths = []

# Globbed rather than listed by hand: every package module is an input to every analysis, and
# an omission here silently stops results being rebuilt when that module changes. Add
# "scripts/inputs.py" once it exists, since every analysis/plot script will import from it.
package_paths = sorted(glob.glob("src/idmdemo/*.py"))
shared_input_paths = package_paths


rule all:
    input:
        figure_paths,


rule results:
    input:
        results_paths,
