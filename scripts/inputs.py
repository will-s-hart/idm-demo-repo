"""Inputs for each analysis: data, model settings and output paths."""

from pathlib import Path

import pandas as pd

from idmdemo import (
    independent_model,
    load_household_data,
    mechanistic_model,
    prepare_household_data,
)

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def get_inputs_generation_time() -> dict:
    """Inputs for fitting the two infectiousness models to the UK household study data.

    Exclusions, fixed parameter values and priors follow Hart et al. (eLife 2022): Methods and
    Appendix 1, tables 1-3. The reference estimates are that paper's point estimates and 95%
    credible intervals.
    """
    household_df = load_household_data(
        DATA_DIR / "hart_elife_data.xlsx", max_onset_gap=30
    )
    models = {
        "independent": independent_model(
            incubation_log_mean=1.63,
            incubation_log_sd=0.50,
            alpha_asymptomatic=0.35,
            mean_prior=(1.6, 0.35),
            sd_prior=(0.7, 0.65),
            beta0_prior=(0.7, 0.8),
        ),
        "mechanistic": mechanistic_model(
            incubation_shape=3.5,
            incubation_mean=5.8,
            alpha_asymptomatic=0.35,
            latent_fraction_prior=(2.1, 2.1),
            symptomatic_period_prior=(1.6, 0.8),
            alpha_presymptomatic_prior=(0.0, 0.8),
            beta0_prior=(0.7, 0.8),
        ),
    }
    results_dir = RESULTS_DIR / "generation_time"
    return {
        "household_data": prepare_household_data(household_df),
        "models": models,
        "mcmc_settings": {
            "seed": 2022,
            "n_chains": 10,
            "n_warmup": 2000,
            "n_draws": 400,
            "thin": 25,
            "n_param_steps": 5,
        },
        "paper_estimates": pd.read_csv(DATA_DIR / "hart2022_estimates.csv"),
        "results_paths": {
            "posterior": {
                name: results_dir / f"posterior_{name}.csv" for name in models
            },
            "summary": results_dir / "summary.csv",
            "diagnostics": results_dir / "diagnostics.csv",
        },
        "fig_paths": {
            "posterior_violins": FIGURES_DIR
            / "generation_time"
            / "posterior_violins.svg",
        },
    }
