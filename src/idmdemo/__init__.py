import jax

# Likelihoods sum many small terms over households and use distribution tails, so the package
# works in double precision throughout. This sets a global JAX option on import.
jax.config.update("jax_enable_x64", True)

from idmdemo.households import (
    HouseholdData,
    load_household_data,
    prepare_household_data,
)
from idmdemo.likelihood import LatentTimes, household_log_likelihood
from idmdemo.mcmc import MCMCResult, fit_household_model
from idmdemo.models import (
    TransmissionModel,
    independent_model,
    mechanistic_model,
)

__all__: list[str] = [
    "HouseholdData",
    "LatentTimes",
    "MCMCResult",
    "TransmissionModel",
    "fit_household_model",
    "household_log_likelihood",
    "independent_model",
    "load_household_data",
    "mechanistic_model",
    "prepare_household_data",
]
