# Note that AI tools were used to generate tests
import pytest

from idmdemo import independent_model, mechanistic_model


@pytest.fixture(scope="session")
def independent():
    return independent_model(
        incubation_log_mean=1.63,
        incubation_log_sd=0.50,
        alpha_asymptomatic=0.35,
        mean_prior=(1.6, 0.35),
        sd_prior=(0.7, 0.65),
        beta0_prior=(0.7, 0.8),
    )


@pytest.fixture(scope="session")
def mechanistic():
    return mechanistic_model(
        incubation_shape=3.5,
        incubation_mean=5.8,
        alpha_asymptomatic=0.35,
        latent_fraction_prior=(2.1, 2.1),
        symptomatic_period_prior=(1.6, 0.8),
        alpha_presymptomatic_prior=(0.0, 0.8),
        beta0_prior=(0.7, 0.8),
    )
