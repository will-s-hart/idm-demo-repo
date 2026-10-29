"""Models of infectiousness for SARS-CoV-2 household transmission (Hart et al., eLife 2022).

Each model gives the expected infectiousness of an infected host at time `tau` since infection,
per unit of beta0/n (beta0 being the overall infectiousness and n the household size), so that
on average over hosts it integrates to one. In the mechanistic model this depends on the host's
incubation period, `tau_inc`; the independent transmission and symptoms model ignores it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln, gammaln, ndtr, xlog1py, xlogy

Params = dict[str, jax.Array]


@dataclass(frozen=True, eq=False)
class TransmissionModel:
    """An infectiousness model, with its priors and derived summaries.

    Parameters are sampled on an unconstrained scale: `constrain` maps a vector `u` to named
    parameters and also returns the log-Jacobian of that map. `summaries` gives the mean and
    standard deviation of the generation time, the proportion of transmissions before symptom
    onset (among symptomatic infectors) and beta0. Asymptomatic hosts are `alpha_asymptomatic`
    times as infectious as symptomatic ones, with the same profile; if `asymptomatic_incubation`
    is true, that profile depends on an (unobserved) incubation period that has the same
    distribution as for symptomatic hosts.
    """

    param_names: tuple[str, ...]
    initial_u: tuple[float, ...]
    constrain: Callable[[jax.Array], tuple[Params, jax.Array]]
    log_prior: Callable[[Params], jax.Array]
    infectiousness: Callable[[jax.Array, jax.Array, Params], jax.Array]
    cumulative_infectiousness: Callable[[jax.Array, jax.Array, Params], jax.Array]
    total_infectiousness: Callable[[jax.Array, Params], jax.Array]
    log_incubation_density: Callable[[jax.Array], jax.Array]
    summaries: Callable[[Params], Params]
    alpha_asymptomatic: float
    asymptomatic_incubation: bool


def independent_model(
    *,
    incubation_log_mean: float,
    incubation_log_sd: float,
    alpha_asymptomatic: float,
    mean_prior: tuple[float, float],
    sd_prior: tuple[float, float],
    beta0_prior: tuple[float, float],
) -> TransmissionModel:
    """Model in which the generation time is independent of the incubation period.

    The generation time and incubation period are both lognormal; the generation time is
    parameterised by its mean and standard deviation. Priors are lognormal, each given as the
    (mean, sd) of the parameter's logarithm.
    """

    def constrain(u):
        params = dict(
            zip(("generation_time_mean", "generation_time_sd", "beta0"), jnp.exp(u))
        )
        return params, jnp.sum(u)

    def log_prior(params):
        return (
            _lognormal_logpdf(params["generation_time_mean"], *mean_prior)
            + _lognormal_logpdf(params["generation_time_sd"], *sd_prior)
            + _lognormal_logpdf(params["beta0"], *beta0_prior)
        )

    def infectiousness(tau, tau_inc, params):
        return jnp.exp(_lognormal_logpdf(tau, *_generation_time_log_params(params)))

    def cumulative_infectiousness(tau, tau_inc, params):
        return _lognormal_cdf(tau, *_generation_time_log_params(params))

    def total_infectiousness(tau_inc, params):
        return jnp.ones_like(tau_inc)

    def log_incubation_density(tau_inc):
        return _lognormal_logpdf(tau_inc, incubation_log_mean, incubation_log_sd)

    def summaries(params):
        # Transmission is presymptomatic when the generation time is shorter than the
        # (independent) incubation period; for lognormals the log-difference is normal.
        log_mean, log_sd = _generation_time_log_params(params)
        presymptomatic = ndtr(
            (incubation_log_mean - log_mean) / jnp.hypot(log_sd, incubation_log_sd)
        )
        return {
            "generation_time_mean": params["generation_time_mean"],
            "generation_time_sd": params["generation_time_sd"],
            "presymptomatic_proportion": presymptomatic,
            "beta0": params["beta0"],
        }

    return TransmissionModel(
        param_names=("generation_time_mean", "generation_time_sd", "beta0"),
        initial_u=(mean_prior[0], sd_prior[0], beta0_prior[0]),
        constrain=constrain,
        log_prior=log_prior,
        infectiousness=infectiousness,
        cumulative_infectiousness=cumulative_infectiousness,
        total_infectiousness=total_infectiousness,
        log_incubation_density=log_incubation_density,
        summaries=summaries,
        alpha_asymptomatic=alpha_asymptomatic,
        asymptomatic_incubation=False,
    )


def mechanistic_model(
    *,
    incubation_shape: float,
    incubation_mean: float,
    alpha_asymptomatic: float,
    latent_fraction_prior: tuple[float, float],
    symptomatic_period_prior: tuple[float, float],
    alpha_presymptomatic_prior: tuple[float, float],
    beta0_prior: tuple[float, float],
) -> TransmissionModel:
    """Model in which hosts pass through latent (E), presymptomatic (P) and symptomatic (I) stages.

    Stage durations are gamma distributed, with the E and P stages (together the incubation
    period, gamma with the given shape and mean) sharing a rate, and an exponential I stage.
    Infectiousness is constant within each stage, with P-stage infectiousness `alpha_p` times
    I-stage infectiousness. Fitted parameters are `latent_fraction` (k_E/k_inc, the ratio of
    mean E to mean incubation period; beta prior given as (a, b)), the mean I stage duration
    `symptomatic_period`, `alpha_p` and beta0 (lognormal priors, as in `independent_model`).
    """
    incubation_rate = incubation_shape / incubation_mean

    def stage_params(params):
        k_e = params["latent_fraction"] * incubation_shape
        k_p = incubation_shape - k_e
        mu = 1 / params["symptomatic_period"]
        c = incubation_rate * mu / (params["alpha_p"] * k_p * mu + incubation_rate)
        return k_e, k_p, mu, c

    def constrain(u):
        params = {
            "latent_fraction": jax.nn.sigmoid(u[0]),
            "symptomatic_period": jnp.exp(u[1]),
            "alpha_p": jnp.exp(u[2]),
            "beta0": jnp.exp(u[3]),
        }
        log_jac = jax.nn.log_sigmoid(u[0]) + jax.nn.log_sigmoid(-u[0]) + jnp.sum(u[1:])
        return params, log_jac

    def log_prior(params):
        a, b = latent_fraction_prior
        x = params["latent_fraction"]
        return (
            (a - 1) * jnp.log(x)
            + (b - 1) * jnp.log1p(-x)
            - betaln(a, b)
            + _lognormal_logpdf(params["symptomatic_period"], *symptomatic_period_prior)
            + _lognormal_logpdf(params["alpha_p"], *alpha_presymptomatic_prior)
            + _lognormal_logpdf(params["beta0"], *beta0_prior)
        )

    def infectiousness(tau, tau_inc, params):
        # Given the incubation period, the fraction of it spent latent is Beta(k_E, k_P).
        k_e, k_p, mu, c = stage_params(params)
        x = jnp.clip(tau / jnp.where(tau_inc > 0, tau_inc, 1.0), 0, 1)
        presymptomatic = params["alpha_p"] * c * betainc(k_e, k_p, x)
        symptomatic = c * jnp.exp(-mu * jnp.maximum(tau - tau_inc, 0))
        return jnp.where(
            tau <= 0, 0.0, jnp.where(tau < tau_inc, presymptomatic, symptomatic)
        )

    def cumulative_infectiousness(tau, tau_inc, params):
        k_e, k_p, mu, c = stage_params(params)
        x = jnp.clip(tau / jnp.where(tau_inc > 0, tau_inc, 1.0), 0, 1)
        # The integral of the Beta(k_E, k_P) CDF needs I_x(k_E + 1, k_P), obtained from
        # I_x(k_E, k_P) by recurrence to save a second (costly) betainc evaluation.
        cdf = betainc(k_e, k_p, x)
        cdf_shifted = cdf - jnp.exp(
            xlogy(k_e, x) + xlog1py(k_p, -x) - jnp.log(k_e) - betaln(k_e, k_p)
        )
        presymptomatic = (
            params["alpha_p"]
            * c
            * (tau * cdf - tau_inc * (k_e / incubation_shape) * cdf_shifted)
        )
        symptomatic = params[
            "alpha_p"
        ] * c * tau_inc * k_p / incubation_shape + c / mu * (
            -jnp.expm1(-mu * jnp.maximum(tau - tau_inc, 0))
        )
        return jnp.where(
            tau <= 0, 0.0, jnp.where(tau < tau_inc, presymptomatic, symptomatic)
        )

    def total_infectiousness(tau_inc, params):
        _, k_p, mu, c = stage_params(params)
        return params["alpha_p"] * c * tau_inc * k_p / incubation_shape + c / mu

    def log_incubation_density(tau_inc):
        tau_safe = jnp.where(tau_inc > 0, tau_inc, 1.0)
        log_density = (
            incubation_shape * jnp.log(incubation_rate)
            + (incubation_shape - 1) * jnp.log(tau_safe)
            - incubation_rate * tau_safe
            - gammaln(incubation_shape)
        )
        return jnp.where(tau_inc > 0, log_density, -jnp.inf)

    def summaries(params):
        # The generation time is y_E + y*, where y* (the time from entering P to transmitting)
        # is independent of y_E and has moments
        # E[y*^m] = C (alpha_P E[y_P^(m+1)] + E[(y_P + y_I)^(m+1) - y_P^(m+1)]) / (m + 1).
        k_e, k_p, mu, c = stage_params(params)
        alpha_p = params["alpha_p"]
        p1, p2, p3 = (_gamma_raw_moment(k_p, incubation_rate, m) for m in (1, 2, 3))
        i1, i2, i3 = 1 / mu, 2 / mu**2, 6 / mu**3
        y1 = c / 2 * (alpha_p * p2 + 2 * p1 * i1 + i2)
        y2 = c / 3 * (alpha_p * p3 + 3 * p2 * i1 + 3 * p1 * i2 + i3)
        presymptomatic_rate = alpha_p * k_p * mu
        return {
            "generation_time_mean": k_e / incubation_rate + y1,
            "generation_time_sd": jnp.sqrt(k_e / incubation_rate**2 + y2 - y1**2),
            "presymptomatic_proportion": presymptomatic_rate
            / (presymptomatic_rate + incubation_rate),
            "beta0": params["beta0"],
        }

    a, b = latent_fraction_prior
    return TransmissionModel(
        param_names=("latent_fraction", "symptomatic_period", "alpha_p", "beta0"),
        initial_u=(
            math.log(a / b),  # logit of the prior mean
            symptomatic_period_prior[0],
            alpha_presymptomatic_prior[0],
            beta0_prior[0],
        ),
        constrain=constrain,
        log_prior=log_prior,
        infectiousness=infectiousness,
        cumulative_infectiousness=cumulative_infectiousness,
        total_infectiousness=total_infectiousness,
        log_incubation_density=log_incubation_density,
        summaries=summaries,
        alpha_asymptomatic=alpha_asymptomatic,
        asymptomatic_incubation=True,
    )


def _generation_time_log_params(params):
    mean, sd = params["generation_time_mean"], params["generation_time_sd"]
    log_var = jnp.log1p((sd / mean) ** 2)
    return jnp.log(mean) - log_var / 2, jnp.sqrt(log_var)


def _gamma_raw_moment(shape, rate, m):
    return jnp.exp(gammaln(shape + m) - gammaln(shape)) / rate**m


def _lognormal_logpdf(x, log_mean, log_sd):
    x_safe = jnp.where(x > 0, x, 1.0)
    z = (jnp.log(x_safe) - log_mean) / log_sd
    log_density = -(z**2) / 2 - jnp.log(x_safe * log_sd) - jnp.log(2 * jnp.pi) / 2
    return jnp.where(x > 0, log_density, -jnp.inf)


def _lognormal_cdf(x, log_mean, log_sd):
    x_safe = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, ndtr((jnp.log(x_safe) - log_mean) / log_sd), 0.0)
