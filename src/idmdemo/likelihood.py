"""Household transmission likelihood, given augmented infection and onset times."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.ops import segment_min, segment_sum

from idmdemo.households import AnyArray, HouseholdData
from idmdemo.models import Params, TransmissionModel


class LatentTimes(NamedTuple):
    """Augmented data, with one entry per infected individual (as in `HouseholdData`).

    `onset_time` is the precise time of symptom onset for symptomatic hosts. For asymptomatic
    hosts it is the end of the (unobserved) incubation period if the model uses one, and is
    otherwise ignored.
    """

    infection_time: AnyArray
    onset_time: AnyArray


def household_log_likelihood(
    params: Params, latent: LatentTimes, data: HouseholdData, model: TransmissionModel
) -> jax.Array:
    """Log-likelihood contribution of each household, given the augmented data.

    Each household's earliest infection is its primary case. Every later infection k
    contributes lambda(t_k) exp(-Lambda(t_k)), where lambda is the force of infection from the
    household's infected members and Lambda its integral, and every uninfected member
    contributes exp(-Lambda(inf)). Symptomatic hosts (and, where the model uses one, asymptomatic
    hosts) also contribute the density of their incubation period. Returns -inf for households
    whose augmented data are inconsistent with the observations.
    """
    t, t_onset = jnp.asarray(latent.infection_time), jnp.asarray(latent.onset_time)
    n_infected = t.shape[0]
    n_households = data.household_size.shape[0]
    tau_inc = t_onset - t
    weight = jnp.where(data.symptomatic, 1.0, model.alpha_asymptomatic)
    scale = weight * params["beta0"] / data.household_size[data.household]

    # Only the member of each pair infected first can have infected the other.
    i_first = t[data.pair_i] <= t[data.pair_j]
    source = jnp.where(i_first, data.pair_i, data.pair_j)
    target = jnp.where(i_first, data.pair_j, data.pair_i)
    lag = t[target] - t[source]
    force = segment_sum(
        scale[source] * model.infectiousness(lag, tau_inc[source], params),
        target,
        num_segments=n_infected,
    )
    cumulative_force = segment_sum(
        scale[source] * model.cumulative_infectiousness(lag, tau_inc[source], params),
        target,
        num_segments=n_infected,
    )
    is_primary = (
        t == segment_min(t, data.household, num_segments=n_households)[data.household]
    )
    infection_terms = jnp.where(is_primary, 0.0, jnp.log(force) - cumulative_force)

    has_incubation = data.symptomatic | model.asymptomatic_incubation
    incubation_terms = jnp.where(
        has_incubation, model.log_incubation_density(tau_inc), 0.0
    )
    onset_in_day = (t_onset >= data.onset_day) & (t_onset < data.onset_day + 1)
    consistent = (t < data.infection_time_max) & (onset_in_day | ~data.symptomatic)

    total_force = segment_sum(
        scale * model.total_infectiousness(tau_inc, params),
        data.household,
        num_segments=n_households,
    )
    return (
        segment_sum(
            jnp.where(consistent, infection_terms + incubation_terms, -jnp.inf),
            data.household,
            num_segments=n_households,
        )
        - data.n_uninfected * total_force
    )
