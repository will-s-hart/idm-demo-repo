# Note that AI tools were used to generate tests
import numpy as np
import pandas as pd
import pytest

from idmdemo import load_household_data, prepare_household_data


@pytest.fixture(scope="module")
def study_data():
    return load_household_data("data/hart_elife_data.xlsx", max_onset_gap=30)


def test_study_cohort_matches_paper(study_data):
    # Methods of Hart et al. (eLife 2022): after excluding 5 of 172 households, 167 remain,
    # comprising 587 individuals, of whom 330 became infected and developed symptoms, 27
    # became infected but remained asymptomatic, 200 remained uninfected and 30 were of
    # unknown infection status (two of them don't appear in the data, but are counted in
    # their household's size).
    households = study_data.groupby("household")
    infected = study_data["status"] == "infected"
    n_missing = (households["household_size"].first() - households.size()).sum()
    assert study_data["household"].nunique() == 167
    assert households["household_size"].first().sum() == 587
    assert (infected & study_data["symptomatic"]).sum() == 330
    assert (infected & ~study_data["symptomatic"]).sum() == 27
    assert (study_data["status"] == "uninfected").sum() == 200
    assert (study_data["status"] == "unknown").sum() + n_missing == 30


def test_study_exclusions(study_data):
    # Three households in which all contacts are of unknown status, and two in which onsets
    # 55-56 days before the index case suggest multiple introductions.
    excluded = {48, 82, 84, 121, 122}
    raw = pd.read_excel("data/hart_elife_data.xlsx", sheet_name="data")
    assert set(raw["hoconumber"]) - set(study_data["household"]) == excluded


def test_infection_time_bounds(study_data):
    index_cases = study_data[study_data["index_case"]]
    assert (index_cases["infection_time_max"] == 1).all()
    uninfected = study_data[study_data["status"] == "uninfected"]
    assert np.isinf(uninfected["infection_time_max"]).all()


def test_prepare_household_data():
    df = pd.DataFrame(
        {
            "household": [7, 7, 7, 7, 3, 3],
            "index_case": [True, False, False, False, True, False],
            "household_size": [5, 5, 5, 5, 2, 2],
            "status": ["infected", "infected", "uninfected", "infected"]
            + ["infected", "unknown"],
            "symptomatic": [True, True, True, False, True, False],
            "onset_day": [-2.0, 3.0, np.nan, np.nan, -1.0, np.nan],
            "infection_time_max": [1.0, np.inf, np.inf, 9.0, 1.0, np.inf],
        }
    )
    data = prepare_household_data(df)
    np.testing.assert_array_equal(data.household_id, [3, 7])
    np.testing.assert_array_equal(data.household_size, [2, 5])
    np.testing.assert_array_equal(data.n_uninfected, [0, 1])
    # Infected grouped by household, symptomatic first in onset order.
    np.testing.assert_array_equal(data.household, [0, 1, 1, 1])
    np.testing.assert_array_equal(data.slot, [0, 0, 1, 2])
    np.testing.assert_array_equal(data.symptomatic, [True, True, True, False])
    np.testing.assert_array_equal(data.onset_day, [-1.0, -2.0, 3.0, np.nan])
    np.testing.assert_array_equal(data.infection_time_max, [1.0, 1.0, np.inf, 9.0])
    pairs = set(zip(data.pair_source, data.pair_target))
    assert pairs == {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}
