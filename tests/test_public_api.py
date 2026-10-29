# Note that AI tools were used to generate tests
import idmdemo


def test_public_api():
    assert set(idmdemo.__all__) == {
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
    }
