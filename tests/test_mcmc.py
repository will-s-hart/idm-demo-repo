# Note that AI tools were used to generate tests
import jax
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from idmdemo import fit_household_model, load_household_data, prepare_household_data


@pytest.mark.parametrize("model_name", ["independent", "mechanistic"])
def test_recovers_prior_without_transmission_data(model_name, request):
    # Single-person households carry no information about transmission, so the posterior of
    # the parameters is the prior. This checks the transforms, Jacobians and acceptance step.
    model = request.getfixturevalue(model_name)
    n = 20
    df = pd.DataFrame(
        {
            "household": np.arange(n),
            "index_case": True,
            "household_size": 1,
            "status": "infected",
            "symptomatic": True,
            "onset_day": -3.0,
            "infection_time_max": 1.0,
        }
    )
    result = fit_household_model(
        model,
        prepare_household_data(df),
        jax.random.key(0),
        n_chains=4,
        n_warmup=1000,
        n_draws=1000,
        thin=5,
        n_param_steps=3,
    )
    priors = {
        "independent": {
            "generation_time_mean": stats.lognorm(s=0.35, scale=np.exp(1.6)),
            "generation_time_sd": stats.lognorm(s=0.65, scale=np.exp(0.7)),
            "beta0": stats.lognorm(s=0.8, scale=np.exp(0.7)),
        },
        "mechanistic": {
            "latent_fraction": stats.beta(2.1, 2.1),
            "symptomatic_period": stats.lognorm(s=0.8, scale=np.exp(1.6)),
            "alpha_p": stats.lognorm(s=0.8, scale=np.exp(0.0)),
            "beta0": stats.lognorm(s=0.8, scale=np.exp(0.7)),
        },
    }[model_name]
    for name, prior in priors.items():
        draws = result.params[name].ravel()
        quantiles = np.quantile(draws, [0.1, 0.5, 0.9])
        # Tolerance allows for Monte Carlo error with a few hundred effective draws.
        np.testing.assert_allclose(prior.cdf(quantiles), [0.1, 0.5, 0.9], atol=0.05)


def test_runs_on_study_data(independent):
    data = prepare_household_data(
        load_household_data("data/hart_elife_data.xlsx", max_onset_gap=30)
    )
    result = fit_household_model(
        independent,
        data,
        jax.random.key(1),
        n_chains=2,
        n_warmup=20,
        n_draws=5,
        thin=2,
        n_param_steps=2,
    )
    assert set(result.params) == set(independent.param_names)
    for draws in [*result.params.values(), result.log_likelihood]:
        assert draws.shape == (2, 5)
        assert np.isfinite(draws).all()
    assert set(result.acceptance) == {"params", "latent_times"}
