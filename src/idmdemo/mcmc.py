"""Data-augmentation MCMC for household transmission models.

The sampler is Metropolis-within-Gibbs over the model parameters and the augmented data
(infection times and precise onset times). Parameters are updated by adaptive random-walk
Metropolis on the unconstrained scale. Augmented data are updated one household member slot at
a time, with a joint proposal of each member's infection and onset times; since households are
independent given the parameters, each slot's proposals are accepted or rejected household by
household, all households at once. Chains run in parallel
(vectorised) and proposal scales adapt during warm-up only.
"""

from functools import cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_min
from tqdm import tqdm

from idmdemo.households import HouseholdData
from idmdemo.likelihood import LatentTimes, household_log_likelihood
from idmdemo.models import TransmissionModel


class MCMCResult(NamedTuple):
    """Posterior draws, each of shape (n_chains, n_draws), and post-warm-up acceptance rates."""

    params: dict[str, np.ndarray]
    log_likelihood: np.ndarray
    acceptance: dict[str, np.ndarray]


def fit_household_model(
    model: TransmissionModel,
    data: HouseholdData,
    key: jax.Array,
    *,
    n_chains: int,
    n_warmup: int,
    n_draws: int,
    thin: int,
    n_param_steps: int,
    progress: bool = False,
) -> MCMCResult:
    """Sample the posterior of `model`'s parameters given household data.

    Each chain runs `n_warmup` adaptive sweeps and then `n_draws * thin` sweeps, keeping every
    `thin`-th. A sweep is `n_param_steps` parameter updates and one augmented-data update for
    every household member slot, each costing one likelihood evaluation. Several parameter
    updates per sweep offset the parameters' slow random-walk mixing given the augmented data. Chains run in parallel
    across the available devices. If `progress` is true, a progress bar of sweeps completed is
    shown.
    """
    data = HouseholdData(*(jnp.asarray(field) for field in data))
    sampler = _Sampler(
        model, data, n_slots=int(data.slot.max()) + 1, n_param_steps=n_param_steps
    )
    init_key, warmup_key, sample_key = jax.random.split(key, 3)

    state = jax.vmap(sampler.initial_state)(jax.random.split(init_key, n_chains))
    if not np.all(np.isfinite(state.log_lik)):
        raise ValueError(
            "Initial augmented data are inconsistent with the observations"
        )

    # Chains are split across as many devices as divide them evenly (on CPU, the number of
    # devices is set by the `jax_num_cpu_devices` option, before JAX first computes).
    n_devices = max(d for d in range(1, len(jax.devices()) + 1) if n_chains % d == 0)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:n_devices]), ("chains",))

    def over_chains(fn):
        # Chains never communicate, so shard_map's check of how values vary across devices
        # (which would flag the per-chain loop counters) isn't needed.
        chains = jax.sharding.PartitionSpec("chains")
        return jax.jit(
            jax.shard_map(
                jax.vmap(fn),
                mesh=mesh,
                in_specs=chains,
                out_specs=chains,
                check_vma=False,
            )
        )

    # Run in chunks of about _SWEEPS_PER_CHUNK sweeps, so progress can be reported. Each
    # distinct chunk size compiles once.
    @cache
    def warmup_chunk(n_sweeps):
        return over_chains(partial(sampler.run_warmup, n_sweeps=n_sweeps))

    @cache
    def sampling_chunk(n_draws):
        return over_chains(partial(sampler.run_sampling, n_draws=n_draws, thin=thin))

    outputs = []
    with tqdm(
        total=n_warmup + n_draws * thin, unit="sweep", disable=not progress
    ) as bar:
        for i, n_sweeps in enumerate(_chunk_sizes(n_warmup, _SWEEPS_PER_CHUNK)):
            chain_keys = jax.random.split(jax.random.fold_in(warmup_key, i), n_chains)
            state = jax.block_until_ready(warmup_chunk(n_sweeps)(state, chain_keys))
            bar.update(n_sweeps)
        draws_per_chunk = max(1, _SWEEPS_PER_CHUNK // thin)
        for i, n_chunk_draws in enumerate(_chunk_sizes(n_draws, draws_per_chunk)):
            chain_keys = jax.random.split(jax.random.fold_in(sample_key, i), n_chains)
            state, chunk_outputs = sampling_chunk(n_chunk_draws)(state, chain_keys)
            outputs.append(jax.block_until_ready(chunk_outputs))
            bar.update(n_chunk_draws * thin)
    u, log_lik, stats = jax.tree.map(lambda *x: jnp.concatenate(x, axis=1), *outputs)

    params = jax.vmap(jax.vmap(lambda u_draw: model.constrain(u_draw)[0]))(u)
    return MCMCResult(
        params={name: np.asarray(params[name]) for name in model.param_names},
        log_likelihood=np.asarray(log_lik),
        acceptance={
            name: np.asarray(rate.mean(axis=1)) for name, rate in stats.items()
        },
    )


class _State(NamedTuple):
    u: jax.Array
    latent: LatentTimes
    log_lik: jax.Array  # per household, for the current u and latent times
    # Adaptation: log proposal scales (for latent times, per infected individual) and a
    # running estimate of the posterior covariance of u.
    u_log_scale: jax.Array
    u_mean: jax.Array
    u_cov: jax.Array
    latent_log_scale: jax.Array
    n_adapted: jax.Array


_SWEEPS_PER_CHUNK = 100
# Targets near the optima for random-walk proposals in several dimensions and in one.
_TARGET_ACCEPTANCE = {"params": 0.3, "latent_times": 0.44}


class _Sampler:
    def __init__(
        self,
        model: TransmissionModel,
        data: HouseholdData,
        n_slots: int,
        n_param_steps: int,
    ):
        self.model = model
        self.data = data
        self.n_slots = n_slots
        self.n_param_steps = n_param_steps

    def log_lik(self, u, latent):
        params, _ = self.model.constrain(u)
        return household_log_likelihood(params, latent, self.data, self.model)

    def log_prior(self, u):
        params, log_jac = self.model.constrain(u)
        return self.model.log_prior(params) + log_jac

    def initial_state(self, key):
        data = self.data
        n_households = data.household_size.shape[0]
        keys = jax.random.split(key, 6)
        u = jnp.array(self.model.initial_u) + 0.1 * jax.random.normal(
            keys[0], (len(self.model.initial_u),)
        )

        # Symptomatic hosts: onset within the recorded day, a plausible incubation period, and
        # infection before any positive test. Asymptomatic hosts: infected a few days after
        # their household's first symptomatic host (and before any positive test).
        shape = data.symptomatic.shape
        before_test = data.infection_time_max - jax.random.uniform(keys[1], shape)
        onset = data.onset_day + jax.random.uniform(keys[2], shape)
        t_symptomatic = jnp.minimum(
            onset - jax.random.uniform(keys[3], shape, minval=2, maxval=8), before_test
        )
        first_symptomatic = segment_min(
            jnp.where(data.symptomatic, t_symptomatic, jnp.inf),
            data.household,
            num_segments=n_households,
        )[data.household]
        t_asymptomatic = jnp.minimum(
            jnp.where(jnp.isfinite(first_symptomatic), first_symptomatic, 0.0)
            + jax.random.uniform(keys[4], shape, minval=1, maxval=6),
            before_test,
        )
        t = jnp.where(data.symptomatic, t_symptomatic, t_asymptomatic)
        onset = jnp.where(
            data.symptomatic,
            onset,
            t + jax.random.uniform(keys[5], shape, minval=3, maxval=9),
        )
        latent = LatentTimes(t, onset)

        n_params = len(self.model.initial_u)
        return _State(
            u=u,
            latent=latent,
            log_lik=self.log_lik(u, latent),
            u_log_scale=jnp.array(0.0),
            u_mean=u,
            u_cov=0.01 * jnp.eye(n_params),
            latent_log_scale=jnp.zeros(shape),
            n_adapted=jnp.array(10.0),  # weight of the initial covariance guess
        )

    def run_warmup(self, state, key, *, n_sweeps):
        def step(state, key):
            state, _ = self.sweep(state, key, adapt=True)
            return state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(key, n_sweeps))
        return state

    def run_sampling(self, state, key, *, n_draws, thin):
        def draw(state, key):
            def thinned_sweep(i, carry):
                state, stats = carry
                state, sweep_stats = self.sweep(
                    state, jax.random.fold_in(key, i), adapt=False
                )
                return state, jax.tree.map(
                    lambda a, b: a + b / thin, stats, sweep_stats
                )

            zero_stats = {name: jnp.array(0.0) for name in _TARGET_ACCEPTANCE}
            state, stats = jax.lax.fori_loop(
                0, thin, thinned_sweep, (state, zero_stats)
            )
            return state, (state.u, state.log_lik.sum(), stats)

        return jax.lax.scan(draw, state, jax.random.split(key, n_draws))

    def sweep(self, state, key, *, adapt):
        params_key, latent_key = jax.random.split(key)

        def update_params(i, carry):
            state, n_accepted = carry
            state, accepted = self.update_params(
                state, jax.random.fold_in(params_key, i)
            )
            return state, n_accepted + accepted

        state, n_params_accepted = jax.lax.fori_loop(
            0, self.n_param_steps, update_params, (state, 0)
        )
        params_acceptance = n_params_accepted / self.n_param_steps
        if adapt:
            state = self.adapt_params(state, params_acceptance)

        def update_slot(slot, carry):
            state, n_latent_accepted = carry
            state, accepted = self.update_latent_times(
                state,
                jax.random.fold_in(latent_key, slot),
                self.data.slot == slot,
                adapt=adapt,
            )
            return state, n_latent_accepted + accepted.sum()

        state, n_latent_accepted = jax.lax.fori_loop(
            0, self.n_slots, update_slot, (state, 0)
        )
        if adapt:
            state = state._replace(n_adapted=state.n_adapted + 1)
        stats = {
            "params": params_acceptance,
            "latent_times": n_latent_accepted / self.data.slot.shape[0],
        }
        return state, stats

    def update_params(self, state, key):
        normal_key, accept_key = jax.random.split(key)
        n_params = state.u.shape[0]
        chol = jnp.linalg.cholesky(state.u_cov + 1e-8 * jnp.eye(n_params))
        step = jnp.exp(state.u_log_scale) * 2.38 / jnp.sqrt(n_params)
        u_prop = state.u + step * chol @ jax.random.normal(normal_key, (n_params,))
        log_lik_prop = self.log_lik(u_prop, state.latent)
        log_ratio = (
            log_lik_prop.sum()
            + self.log_prior(u_prop)
            - state.log_lik.sum()
            - self.log_prior(state.u)
        )
        accepted = jnp.log(jax.random.uniform(accept_key)) < log_ratio
        state = state._replace(
            u=jnp.where(accepted, u_prop, state.u),
            log_lik=jnp.where(accepted, log_lik_prop, state.log_lik),
        )
        return state, accepted

    def adapt_params(self, state, acceptance):
        # Robbins-Monro on the proposal scale, and a running estimate of the posterior
        # covariance of u from one draw per sweep.
        rate = _adaptation_rate(state.n_adapted)
        delta = state.u - state.u_mean
        u_mean = state.u_mean + delta / (state.n_adapted + 1)
        return state._replace(
            u_log_scale=state.u_log_scale
            + rate * (acceptance - _TARGET_ACCEPTANCE["params"]),
            u_mean=u_mean,
            u_cov=state.u_cov
            + (jnp.outer(delta, state.u - u_mean) - state.u_cov)
            / (state.n_adapted + 1),
        )

    def update_latent_times(self, state, key, in_slot, *, adapt):
        # Random-walk infection times, with a fresh onset time within the recorded day for
        # symptomatic hosts. Asymptomatic hosts with an incubation period either keep the end of
        # incubation or shift it along with infection, chosen at random (a mixture of two
        # symmetric proposals).
        normal_key, uniform_key, shift_key, accept_key = jax.random.split(key, 4)
        t, t_onset = state.latent
        step = jnp.exp(state.latent_log_scale) * jax.random.normal(normal_key, t.shape)
        shift = (
            ~self.data.symptomatic
            & self.model.asymptomatic_incubation
            & jax.random.bernoulli(shift_key, shape=t.shape)
        )
        latent_prop = LatentTimes(
            infection_time=jnp.where(in_slot, t + step, t),
            onset_time=jnp.where(
                in_slot & self.data.symptomatic,
                self.data.onset_day + jax.random.uniform(uniform_key, t.shape),
                jnp.where(in_slot & shift, t_onset + step, t_onset),
            ),
        )
        state, accepted = self._accept_per_household(
            state, accept_key, latent_prop, in_slot
        )
        if adapt:
            rate = _adaptation_rate(state.n_adapted)
            state = state._replace(
                latent_log_scale=state.latent_log_scale
                + in_slot * rate * (accepted - _TARGET_ACCEPTANCE["latent_times"])
            )
        return state, accepted

    def _accept_per_household(self, state, key, latent_prop, proposed):
        log_lik_prop = self.log_lik(state.u, latent_prop)
        n_households = state.log_lik.shape[0]
        log_uniform = jnp.log(jax.random.uniform(key, (n_households,)))
        household_accepted = log_uniform < log_lik_prop - state.log_lik
        accepted = household_accepted[self.data.household] & proposed
        latent = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), latent_prop, state.latent
        )
        log_lik = jnp.where(household_accepted, log_lik_prop, state.log_lik)
        return state._replace(latent=latent, log_lik=log_lik), accepted


def _chunk_sizes(total, size):
    return [size] * (total // size) + ([total % size] if total % size else [])


def _adaptation_rate(n_adapted):
    return (n_adapted + 1) ** -0.6
