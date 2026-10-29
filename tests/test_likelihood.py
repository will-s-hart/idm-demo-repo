# Note that AI tools were used to generate tests
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from idmdemo import (
    HouseholdData,
    LatentTimes,
    household_log_likelihood,
    prepare_household_data,
)

INDEPENDENT_PARAMS = {
    "generation_time_mean": 4.2,
    "generation_time_sd": 4.9,
    "beta0": 1.7,
}
MECHANISTIC_PARAMS = {
    "latent_fraction": 0.2,
    "symptomatic_period": 5.0,
    "alpha_p": 3.1,
    "beta0": 1.8,
}
ALPHA_A = 0.35


@pytest.fixture
def households():
    # Household 1 (size 5): symptomatic index, symptomatic and asymptomatic infected contacts,
    # an uninfected contact and one of unknown status. Household 2 (size 3): symptomatic
    # index and two uninfected contacts.
    df = pd.DataFrame(
        {
            "household": [1, 1, 1, 1, 1, 2, 2, 2],
            "index_case": [True, False, False, False, False, True, False, False],
            "household_size": [5, 5, 5, 5, 5, 3, 3, 3],
            "status": ["infected", "infected", "infected", "uninfected", "unknown"]
            + ["infected", "uninfected", "uninfected"],
            "symptomatic": [True, True, False, False, True, True, False, False],
            "onset_day": [-2.0, 3.0, np.nan, np.nan, np.nan, -1.0, np.nan, np.nan],
            "infection_time_max": [
                1.0,
                np.inf,
                9.0,
                np.inf,
                np.inf,
                1.0,
                np.inf,
                np.inf,
            ],
        }
    )
    return prepare_household_data(df)


# Infected individuals in order: household 1 index, symptomatic contact, asymptomatic
# contact; household 2 index. The second set makes the asymptomatic contact the primary case.
LATENT_TIMES = [
    LatentTimes(np.array([-6.3, 0.4, 1.7, -4.1]), np.array([-1.5, 3.2, 6.0, -0.6])),
    LatentTimes(np.array([-6.3, 0.4, -8.0, -4.1]), np.array([-1.5, 3.2, -2.5, -0.6])),
]


@pytest.mark.parametrize("latent", LATENT_TIMES)
@pytest.mark.parametrize("model_name", ["independent", "mechanistic"])
def test_matches_reference(households, latent, model_name, request):
    model = request.getfixturevalue(model_name)
    params = INDEPENDENT_PARAMS if model_name == "independent" else MECHANISTIC_PARAMS
    expected = _reference_log_likelihood(households, latent, model_name, params)
    log_lik = household_log_likelihood(
        _as_jax(params), _as_jax(latent), households, model
    )
    np.testing.assert_allclose(log_lik, expected, rtol=1e-8)


@pytest.mark.parametrize("model_name", ["independent", "mechanistic"])
def test_invariant_to_member_order(households, model_name, request):
    model = request.getfixturevalue(model_name)
    params = _as_jax(
        INDEPENDENT_PARAMS if model_name == "independent" else MECHANISTIC_PARAMS
    )
    latent = LATENT_TIMES[1]
    perm = np.array([3, 2, 0, 1])
    inverse = np.argsort(perm)
    permuted = households._replace(
        **{
            field: getattr(households, field)[perm]
            for field in (
                "household",
                "slot",
                "symptomatic",
                "onset_day",
                "infection_time_max",
            )
        },
        pair_i=inverse[households.pair_i],
        pair_j=inverse[households.pair_j],
    )
    permuted_latent = LatentTimes(latent.infection_time[perm], latent.onset_time[perm])
    np.testing.assert_allclose(
        household_log_likelihood(params, _as_jax(permuted_latent), permuted, model),
        household_log_likelihood(params, _as_jax(latent), households, model),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("index", "infection_time", "onset_time"),
    [
        (2, 9.5, 12.0),  # infected after a positive test
        (0, -6.3, -0.5),  # onset outside the recorded day
        (1, 3.5, 3.2),  # infected after symptom onset
    ],
)
def test_inconsistent_latent_times(
    households, independent, index, infection_time, onset_time
):
    t, t_onset = (np.array(times) for times in LATENT_TIMES[0])
    t[index], t_onset[index] = infection_time, onset_time
    log_lik = household_log_likelihood(
        _as_jax(INDEPENDENT_PARAMS),
        _as_jax(LatentTimes(t, t_onset)),
        households,
        independent,
    )
    assert np.isneginf(log_lik[0])
    assert np.isfinite(log_lik[1])


def _as_jax(tree):
    if isinstance(tree, dict):
        return {name: jnp.asarray(value) for name, value in tree.items()}
    return type(tree)(*(jnp.asarray(value) for value in tree))


def _reference_log_likelihood(data: HouseholdData, latent, model_name, params):
    """Direct, loop-based implementation of the likelihood in Hart et al. (eLife 2022)."""
    if model_name == "independent":
        mean, sd = params["generation_time_mean"], params["generation_time_sd"]
        log_sd = np.sqrt(np.log1p((sd / mean) ** 2))
        generation_time = stats.lognorm(s=log_sd, scale=mean * np.exp(-(log_sd**2) / 2))
        incubation = stats.lognorm(s=0.50, scale=np.exp(1.63))

        def infectiousness(tau, tau_inc):
            return generation_time.pdf(tau) if tau > 0 else 0.0

        def cumulative(tau, tau_inc):
            return generation_time.cdf(tau) if tau > 0 else 0.0

        has_incubation = data.symptomatic
    else:
        k_inc, gamma = 3.5, 1 / 5.8
        k_e = params["latent_fraction"] * k_inc
        k_p = k_inc - k_e
        mu = 1 / params["symptomatic_period"]
        alpha_p = params["alpha_p"]
        c = k_inc * gamma * mu / (alpha_p * k_p * mu + k_inc * gamma)
        incubation = stats.gamma(a=k_inc, scale=1 / (k_inc * gamma))

        def infectiousness(tau, tau_inc):
            if tau <= 0:
                return 0.0
            if tau < tau_inc:
                return alpha_p * c * (1 - special.betainc(k_p, k_e, 1 - tau / tau_inc))
            return c * (1 - stats.expon(scale=1 / mu).cdf(tau - tau_inc))

        def cumulative(tau, tau_inc):
            # Integrate the presymptomatic and symptomatic parts separately (the
            # infectiousness jumps at symptom onset).
            if tau <= 0:
                return 0.0
            before_onset = integrate.quad(
                infectiousness, 0, min(tau, tau_inc), (tau_inc,)
            )
            if tau <= tau_inc:
                return before_onset[0]
            return (
                before_onset[0]
                + integrate.quad(infectiousness, tau_inc, tau, (tau_inc,))[0]
            )

        has_incubation = np.ones_like(data.symptomatic)

    t, t_onset = latent
    log_lik = []
    for m, n in enumerate(data.household_size):
        members = np.flatnonzero(data.household == m)
        weight = {
            j: (1.0 if data.symptomatic[j] else ALPHA_A) * params["beta0"] / n
            for j in members
        }
        total = 0.0
        primary = members[np.argmin(t[members])]
        for k in members:
            if has_incubation[k]:
                total += incubation.logpdf(t_onset[k] - t[k])
            if k == primary:
                continue
            sources = [j for j in members if j != k]
            force = sum(
                weight[j] * infectiousness(t[k] - t[j], t_onset[j] - t[j])
                for j in sources
            )
            cumulative_force = sum(
                weight[j] * cumulative(t[k] - t[j], t_onset[j] - t[j]) for j in sources
            )
            total += np.log(force) - cumulative_force
        escape = sum(weight[j] * cumulative(np.inf, t_onset[j] - t[j]) for j in members)
        log_lik.append(total - data.n_uninfected[m] * escape)
    return np.array(log_lik)
