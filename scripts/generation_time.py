"""Fit the two infectiousness models to the UK household study data.

An independent reproduction of the generation time estimates of Hart et al. (eLife 2022). It
uses the paper's models, likelihood and priors, with a different (vectorised, adaptive)
data-augmentation MCMC scheme. Posterior summaries are written next to the paper's estimates.
"""

import argparse
import os

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from idmdemo import MCMCResult, TransmissionModel, fit_household_model
from scripts.generation_time_plots import make_plots
from scripts.inputs import get_inputs_generation_time


def run_analyses(results_only: bool = False) -> None:
    """Fit both models and write posterior draws, summaries and convergence diagnostics."""
    inputs = get_inputs_generation_time()
    models = inputs["models"]
    paths = inputs["results_paths"]
    settings = dict(inputs["mcmc_settings"])
    keys = jax.random.split(jax.random.key(settings.pop("seed")), len(models))

    posteriors = {}
    for key, (name, model) in zip(keys, models.items()):
        print(f"Fitting {name} model...", flush=True)
        result = fit_household_model(
            model, inputs["household_data"], key, progress=True, **settings
        )
        print(
            "  Acceptance rates:",
            {
                move: float(rate.mean().round(2))
                for move, rate in result.acceptance.items()
            },
        )
        posteriors[name] = _posterior_table(model, result)

    summary = _summarise(posteriors, models, inputs["paper_estimates"])
    diagnostics = _diagnose(posteriors)

    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    for name, posterior in posteriors.items():
        posterior.to_csv(paths["posterior"][name], index=False, float_format="%.6g")
    summary.to_csv(paths["summary"], index=False, float_format="%.4g")
    diagnostics.to_csv(paths["diagnostics"], index=False, float_format="%.4g")
    print(summary.round(2).to_string(index=False))
    print(f"Worst R-hat {diagnostics['r_hat'].max():.3f}, ", end="")
    print(f"lowest bulk ESS {diagnostics['ess_bulk'].min():.0f}")

    if not results_only:
        make_plots()


def _posterior_table(model: TransmissionModel, result: MCMCResult) -> pd.DataFrame:
    params = {name: jnp.asarray(draws) for name, draws in result.params.items()}
    summaries = jax.vmap(jax.vmap(model.summaries))(params)
    columns = {
        **result.params,
        **{name: np.asarray(values) for name, values in summaries.items()},
        "log_likelihood": result.log_likelihood,
    }
    n_chains, n_draws = result.log_likelihood.shape
    chain, draw = np.meshgrid(np.arange(n_chains), np.arange(n_draws), indexing="ij")
    return pd.DataFrame(
        {
            "chain": chain.ravel(),
            "draw": draw.ravel(),
            **{name: np.ravel(values) for name, values in columns.items()},
        }
    )


def _summarise(
    posteriors: dict[str, pd.DataFrame],
    models: dict[str, TransmissionModel],
    paper_estimates: pd.DataFrame,
) -> pd.DataFrame:
    # As in the paper, point estimates of derived quantities are evaluated at the posterior
    # mean parameters (for fitted parameters, they are just posterior means).
    rows = []
    for name, posterior in posteriors.items():
        model = models[name]
        mean_params = {
            param: jnp.asarray(posterior[param].mean()) for param in model.param_names
        }
        point_estimates = {
            **{param: posterior[param].mean() for param in model.param_names},
            **{
                quantity: float(value)
                for quantity, value in model.summaries(mean_params).items()
            },
        }
        for quantity, point_estimate in point_estimates.items():
            values = posterior[quantity]
            rows.append(
                {
                    "model": name,
                    "quantity": quantity,
                    "point_estimate": point_estimate,
                    "posterior_mean": values.mean(),
                    "lower": values.quantile(0.025),
                    "upper": values.quantile(0.975),
                }
            )
    paper = paper_estimates.rename(
        columns={
            "estimate": "paper_estimate",
            "lower": "paper_lower",
            "upper": "paper_upper",
        }
    ).drop(columns="source")
    return pd.DataFrame(rows).merge(paper, on=["model", "quantity"], how="left")


def _diagnose(posteriors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tables = []
    for name, posterior in posteriors.items():
        n_chains = posterior["chain"].nunique()
        draws = {
            quantity: posterior[quantity].to_numpy().reshape(n_chains, -1)
            for quantity in posterior.columns.drop(["chain", "draw"])
        }
        table = az.summary(
            az.from_dict({"posterior": draws}), kind="diagnostics", round_to="none"
        )
        tables.append(
            table[["ess_bulk", "ess_tail", "r_hat"]]
            .rename_axis("quantity")
            .reset_index()
            .assign(model=name)
        )
    return pd.concat(tables)[["model", "quantity", "ess_bulk", "ess_tail", "r_hat"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--results-only", action="store_true", help="skip making the figures"
    )
    args = parser.parse_args()
    # One JAX device per CPU core, so MCMC chains run in parallel. This must be set before
    # JAX first computes anything.
    jax.config.update("jax_num_cpu_devices", os.cpu_count())
    run_analyses(results_only=args.results_only)
