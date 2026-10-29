# Note that AI tools were used to generate tests
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

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


def _as_jax(params):
    return {name: jnp.asarray(value) for name, value in params.items()}


def _scalar_fn(fn, params, tau_inc=None):
    """Wrap fn(tau, tau_inc, params) as a jitted float -> float function, for quadrature."""
    jitted = jax.jit(fn)
    params = _as_jax(params)
    tau_inc = jnp.asarray(np.nan if tau_inc is None else tau_inc)
    return lambda tau: float(jitted(jnp.asarray(tau), tau_inc, params))


def test_independent_generation_time_moments(independent):
    g = _scalar_fn(independent.infectiousness, INDEPENDENT_PARAMS)
    total = integrate.quad(g, 0, np.inf)[0]
    mean = integrate.quad(lambda tau: tau * g(tau), 0, np.inf)[0]
    second_moment = integrate.quad(lambda tau: tau**2 * g(tau), 0, np.inf)[0]
    assert total == pytest.approx(1, rel=1e-6)
    assert mean == pytest.approx(4.2, rel=1e-6)
    assert np.sqrt(second_moment - mean**2) == pytest.approx(4.9, rel=1e-6)


def test_independent_cumulative_infectiousness(independent):
    g = _scalar_fn(independent.infectiousness, INDEPENDENT_PARAMS)
    big_g = _scalar_fn(independent.cumulative_infectiousness, INDEPENDENT_PARAMS)
    for tau in [-1.0, 0.5, 3.0, 12.0]:
        expected = integrate.quad(g, 0, tau)[0] if tau > 0 else 0.0
        assert big_g(tau) == pytest.approx(expected, rel=1e-8, abs=1e-12)


def test_independent_presymptomatic_proportion(independent):
    g = _scalar_fn(independent.infectiousness, INDEPENDENT_PARAMS)
    incubation = stats.lognorm(s=0.50, scale=np.exp(1.63))
    expected = integrate.quad(lambda tau: g(tau) * incubation.sf(tau), 0, np.inf)[0]
    summaries = independent.summaries(_as_jax(INDEPENDENT_PARAMS))
    assert float(summaries["presymptomatic_proportion"]) == pytest.approx(
        expected, rel=1e-6
    )
    # Point estimate reported by Hart et al. (eLife 2022) at these parameter values.
    assert float(summaries["presymptomatic_proportion"]) == pytest.approx(
        0.72, abs=0.005
    )


@pytest.mark.parametrize("tau_inc", [1.0, 5.8, 14.0])
def test_mechanistic_cumulative_infectiousness(mechanistic, tau_inc):
    g = _scalar_fn(mechanistic.infectiousness, MECHANISTIC_PARAMS, tau_inc)
    big_g = _scalar_fn(
        mechanistic.cumulative_infectiousness, MECHANISTIC_PARAMS, tau_inc
    )
    for tau in [-1.0, 0.2 * tau_inc, 0.9 * tau_inc, tau_inc + 3.0]:
        expected = integrate.quad(g, 0, tau, points=[tau_inc])[0] if tau > 0 else 0.0
        assert big_g(tau) == pytest.approx(expected, rel=1e-7, abs=1e-12)
    total = float(
        mechanistic.total_infectiousness(
            jnp.asarray(tau_inc), _as_jax(MECHANISTIC_PARAMS)
        )
    )
    expected = integrate.quad(g, 0, tau_inc)[0] + integrate.quad(g, tau_inc, np.inf)[0]
    assert total == pytest.approx(expected)


def test_mechanistic_incubation_period(mechanistic):
    incubation = stats.gamma(a=3.5, scale=5.8 / 3.5)
    tau_inc = np.array([0.5, 4.0, 10.0])
    np.testing.assert_allclose(
        mechanistic.log_incubation_density(jnp.asarray(tau_inc)),
        incubation.logpdf(tau_inc),
    )
    assert np.isneginf(mechanistic.log_incubation_density(jnp.asarray(-1.0)))

    # Averaged over hosts, infectiousness integrates to one (per unit beta0/n).
    params = _as_jax(MECHANISTIC_PARAMS)
    total = jax.jit(mechanistic.total_infectiousness)
    mean_total = integrate.quad(
        lambda t: float(total(jnp.asarray(t), params)) * incubation.pdf(t), 0, np.inf
    )[0]
    assert mean_total == pytest.approx(1, rel=1e-8)


def test_mechanistic_summaries_match_simulation(mechanistic):
    # Simulate the stage model directly: hosts infect in proportion to their total
    # infectiousness (alpha_P y_P + y_I, with y_I exponential), at a uniformly distributed
    # time within the P stage or the I stage (weighted by alpha_P and 1 respectively).
    rng = np.random.default_rng(1)
    n = 2_000_000
    shape, rate = 3.5, 3.5 / 5.8
    k_e = MECHANISTIC_PARAMS["latent_fraction"] * shape
    alpha_p = MECHANISTIC_PARAMS["alpha_p"]
    y_e = rng.gamma(k_e, 1 / rate, n)
    y_p = rng.gamma(shape - k_e, 1 / rate, n)
    y_i = rng.exponential(MECHANISTIC_PARAMS["symptomatic_period"], n)
    weight = alpha_p * y_p + y_i
    infectors = rng.choice(n, size=n, p=weight / weight.sum())
    y_e, y_p, y_i, weight = (
        y_e[infectors],
        y_p[infectors],
        y_i[infectors],
        weight[infectors],
    )
    presymptomatic = rng.random(n) < alpha_p * y_p / weight
    u = rng.random(n)
    generation_time = y_e + np.where(presymptomatic, u * y_p, y_p + u * y_i)

    summaries = mechanistic.summaries(_as_jax(MECHANISTIC_PARAMS))
    assert float(summaries["generation_time_mean"]) == pytest.approx(
        generation_time.mean(), rel=0.01
    )
    assert float(summaries["generation_time_sd"]) == pytest.approx(
        generation_time.std(), rel=0.01
    )
    assert float(summaries["presymptomatic_proportion"]) == pytest.approx(
        presymptomatic.mean(), abs=0.003
    )


def test_constrain_round_trip(independent, mechanistic):
    for model in (independent, mechanistic):
        u = jnp.asarray(model.initial_u)
        params, log_jac = model.constrain(u)
        assert tuple(params) == model.param_names
        assert np.isfinite(log_jac)
        assert np.isfinite(model.log_prior(params))
    params, _ = mechanistic.constrain(jnp.asarray(mechanistic.initial_u))
    assert float(params["latent_fraction"]) == pytest.approx(0.5)
    assert float(params["alpha_p"]) == pytest.approx(1.0)
