"""Loading and preparing household transmission study data.

Times are in days relative to the date the household's index case was swabbed (and found
positive), matching the `case_swab_*` columns of the study data.
"""

from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np
import pandas as pd

# Array fields hold NumPy arrays when prepared, and JAX arrays once passed to the sampler.
AnyArray = np.ndarray | jax.Array


class HouseholdData(NamedTuple):
    """Household study data as flat arrays, ready for the likelihood.

    Arrays prefixed per-infected have one entry per infected individual, grouped by household
    and ordered by `slot` within it. Per-household arrays have one entry per household. Pair
    arrays list every ordered (source, target) pair of distinct infected individuals in the
    same household, as indices into the per-infected arrays.
    """

    household: AnyArray  # per-infected: household index (0, ..., M - 1)
    slot: AnyArray  # per-infected: position among the household's infected members
    symptomatic: AnyArray  # per-infected: whether symptoms developed
    onset_day: AnyArray  # per-infected: day of symptom onset (NaN if asymptomatic)
    # per-infected: upper bound on infection time (or inf)
    infection_time_max: AnyArray
    # per-household: number of members, including those of unknown status
    household_size: AnyArray
    n_uninfected: AnyArray  # per-household: members known to have remained uninfected
    household_id: AnyArray  # per-household: identifier in the study data
    pair_source: AnyArray  # per-pair: index of potential infector
    pair_target: AnyArray  # per-pair: index of potential infectee


def load_household_data(path: str | Path, max_onset_gap: float) -> pd.DataFrame:
    """Load the household study spreadsheet as a tidy table, applying study exclusions.

    Each household member's `status` is "infected" (positive PCR or antibody test),
    "uninfected" (negative antibody test, no positive PCR) or "unknown" (no antibody test, no
    positive PCR). Two kinds of household are excluded: those in which no contact has a known
    status, and those with a gap between successive symptom onsets exceeding `max_onset_gap`
    days (taken to indicate multiple introductions). Unknown-status members are kept here and
    counted in `household_size`, but don't enter the likelihood.

    `infection_time_max` is the end of the day of an individual's first positive PCR test
    (for index cases, the recruiting test on day 0), or inf if they never tested positive.
    """
    raw = pd.read_excel(path, sheet_name="data")
    positive_test_days = pd.concat(
        [
            pd.Series(np.where(raw["status"] == "CASE", 0.0, np.inf)),
            raw["case_swab_swab1"].where(raw["swab1"] == 1, np.inf),
            raw["case_swab_swab2"].where(raw["swab2"] == 1, np.inf),
        ],
        axis=1,
    )
    df = pd.DataFrame(
        {
            "household": raw["hoconumber"],
            "index_case": raw["status"] == "CASE",
            "household_size": raw["householdsize"],
            "status": raw["infected"].map(
                {1: "infected", 0: "uninfected", 9: "unknown"}
            ),
            "symptomatic": raw["symptoms"] == 1,
            "onset_day": raw["case_swab_ill"],
            "infection_time_max": positive_test_days.min(axis=1) + 1.0,
            "recruitment_month": raw["month_case_swab"],
        }
    )
    _validate(df)
    kept_households = [
        household_id
        for household_id, household in df.groupby("household")
        if _has_known_contact(household) and _max_onset_gap(household) <= max_onset_gap
    ]
    return df[df["household"].isin(kept_households)].reset_index(drop=True)


def prepare_household_data(df: pd.DataFrame) -> HouseholdData:
    """Convert a table from `load_household_data` into flat arrays for the likelihood."""
    households = df.groupby("household", sort=True)
    household_id = np.array(list(households.groups))
    infected = df[df["status"] == "infected"].sort_values(["household", "onset_day"])
    household = np.searchsorted(household_id, infected["household"].to_numpy())
    slot = infected.groupby("household").cumcount().to_numpy()
    symptomatic = infected["symptomatic"].to_numpy()

    sources, targets = [], []
    for members in np.split(
        np.arange(len(infected)), np.flatnonzero(np.diff(household)) + 1
    ):
        for j in members:
            for k in members:
                if j != k:
                    sources.append(j)
                    targets.append(k)

    return HouseholdData(
        household=household,
        slot=slot,
        symptomatic=symptomatic,
        onset_day=np.where(symptomatic, infected["onset_day"].to_numpy(), np.nan),
        infection_time_max=infected["infection_time_max"].to_numpy(),
        household_size=households["household_size"].first().to_numpy(dtype=float),
        n_uninfected=households["status"]
        .agg(lambda s: (s == "uninfected").sum())
        .to_numpy(dtype=float),
        household_id=household_id,
        pair_source=np.array(sources, dtype=int),
        pair_target=np.array(targets, dtype=int),
    )


def _validate(df: pd.DataFrame) -> None:
    if df["status"].isna().any():
        raise ValueError("Unrecognised infection status code")
    if (df.groupby("household")["index_case"].sum() != 1).any():
        raise ValueError("Each household must have exactly one index case")
    if (df.groupby("household")["household_size"].nunique() != 1).any():
        raise ValueError("Household size must be constant within a household")
    if (
        df.groupby("household").size()
        > df.groupby("household")["household_size"].first()
    ).any():
        raise ValueError("A household has more members listed than its size")
    symptomatic_infected = (df["status"] == "infected") & df["symptomatic"]
    if (symptomatic_infected != df["onset_day"].notna()).any():
        raise ValueError(
            "Onset dates must be given for exactly the symptomatic infected"
        )
    if not df.loc[df["index_case"], "status"].eq("infected").all():
        raise ValueError("Every index case must be infected")


def _has_known_contact(household: pd.DataFrame) -> bool:
    return bool((~household["index_case"] & (household["status"] != "unknown")).any())


def _max_onset_gap(household: pd.DataFrame) -> float:
    onsets = np.sort(household["onset_day"].dropna().to_numpy())
    return float(np.diff(onsets).max()) if len(onsets) > 1 else 0.0
