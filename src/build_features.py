"""
Phase 3 — Feature Engineering

Combines everything collected so far into ONE table: one row per
(driver, race), with every column knowable before Sunday's race —
Group A (rolling form, computed from PRIOR races only), Group B
(qualifying, known Saturday evening), and Group C (circuit characteristics,
static).

LEAKAGE DISCIPLINE: every rolling/cumulative stat is shifted so that the
row for race N only ever sees data from races BEFORE race N. This is the
single most important thing to get right in this script — double check
any new feature you add follows the same pattern.

Run:
    python src/build_features.py

Reads:  data/raw/race_results.parquet, quali_results.parquet, schedule.parquet
        data/processed/circuit_table.parquet
Writes: data/processed/feature_table.parquet
"""

import os
import re
import pandas as pd
import numpy as np

from f1_utils import is_dnf

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

ROLLING_WINDOWS = [3, 5]  # races


def _time_to_seconds(t) -> float:
    """Converts a lap time string like '1:23.456' or '23.456' to seconds.
    Returns NaN for missing/None."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return np.nan
    t = str(t)
    if ":" in t:
        minutes, rest = t.split(":")
        return int(minutes) * 60 + float(rest)
    try:
        return float(t)
    except ValueError:
        return np.nan


def load_raw():
    race = pd.read_parquet(os.path.join(RAW_DIR, "race_results.parquet"))
    quali = pd.read_parquet(os.path.join(RAW_DIR, "quali_results.parquet"))
    schedule = pd.read_parquet(os.path.join(RAW_DIR, "schedule.parquet"))
    circuit_table = pd.read_parquet(os.path.join(PROCESSED_DIR, "circuit_table.parquet"))
    return race, quali, schedule, circuit_table


def add_race_date_and_sort(race: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    race = race.merge(schedule[["year", "round", "date", "circuit_id"]], on=["year", "round"], how="left")
    race["date"] = pd.to_datetime(race["date"])
    race["is_dnf"] = race["status"].apply(is_dnf)
    return race.sort_values(["date", "driver_id"]).reset_index(drop=True)


def add_rolling_form_features(race: pd.DataFrame) -> pd.DataFrame:
    """
    Driver and constructor rolling average finish position, over the last
    N races, PLUS a separate rolling DNF rate. Uses .shift(1) before
    rolling so that the value attached to race N is computed ONLY from
    races strictly before N — this is what prevents leakage.

    Design choice: finish-position averages and DNF rate are kept as
    SEPARATE features rather than folding DNFs into the average (e.g. by
    treating a DNF as "last place"). A driver who's fast-but-crash-prone
    and a driver who's just slow would otherwise look identical on a
    blended average — keeping them separate lets the model learn each
    pattern independently.
    """
    race = race.sort_values(["driver_id", "date"]).reset_index(drop=True)

    for window in ROLLING_WINDOWS:
        race[f"driver_form_last{window}"] = (
            race.groupby("driver_id")["finish_position"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        race[f"driver_dnf_rate_last{window}"] = (
            race.groupby("driver_id")["is_dnf"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    race = race.sort_values(["constructor_id", "date"]).reset_index(drop=True)
    for window in ROLLING_WINDOWS:
        race[f"constructor_form_last{window}"] = (
            race.groupby("constructor_id")["finish_position"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        race[f"constructor_dnf_rate_last{window}"] = (
            race.groupby("constructor_id")["is_dnf"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    return race


def add_season_standing_features(race: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative points BEFORE this race, within the season — a proxy for
    championship standing. Shifted the same way as rolling form.
    """
    race = race.sort_values(["driver_id", "year", "date"]).reset_index(drop=True)
    race["driver_points_before_race"] = (
        race.groupby(["driver_id", "year"])["points"]
        .transform(lambda s: s.shift(1).cumsum().fillna(0))
    )

    race = race.sort_values(["constructor_id", "year", "date"]).reset_index(drop=True)
    race["constructor_points_before_race"] = (
        race.groupby(["constructor_id", "year"])["points"]
        .transform(lambda s: s.shift(1).cumsum().fillna(0))
    )

    return race


def add_driver_at_circuit_history(race: pd.DataFrame) -> pd.DataFrame:
    """
    Driver's average finish position at THIS circuit, across all prior
    years (not this year's race, since that would leak the outcome).
    """
    race = race.sort_values(["driver_id", "circuit_id", "date"]).reset_index(drop=True)
    race["driver_circuit_history_avg_finish"] = (
        race.groupby(["driver_id", "circuit_id"])["finish_position"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return race


def add_qualifying_features(race: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """
    Grid position (post-penalty, from race results) plus pure qualifying
    pace: quali_position (pre-penalty) and gap to pole in seconds, derived
    from Q1/Q2/Q3 times.
    """
    quali = quali.copy()
    # Best (minimum) non-null time across Q1/Q2/Q3 for each driver
    times = quali[["q1", "q2", "q3"]].apply(lambda col: col.map(_time_to_seconds))
    quali["best_time_sec"] = times.min(axis=1, skipna=True)

    # Pole time per race = fastest best_time_sec in that (year, round)
    pole_time = quali.groupby(["year", "round"])["best_time_sec"].transform("min")
    quali["gap_to_pole_sec"] = quali["best_time_sec"] - pole_time
    quali["gap_to_pole_pct"] = quali["gap_to_pole_sec"] / pole_time

    quali_features = quali[[
        "year", "round", "driver_id", "quali_position", "best_time_sec",
        "gap_to_pole_sec", "gap_to_pole_pct"
    ]]

    race = race.merge(quali_features, on=["year", "round", "driver_id"], how="left")
    return race


def add_circuit_features(race: pd.DataFrame, circuit_table: pd.DataFrame) -> pd.DataFrame:
    circuit_cols = ["circuit_id", "dnf_rate", "avg_position_change", "rain_probability", "is_street_circuit"]
    race = race.merge(
        circuit_table[circuit_cols].rename(columns={
            "dnf_rate": "circuit_dnf_rate",
            "avg_position_change": "circuit_avg_position_change",
            "rain_probability": "circuit_rain_probability",
        }),
        on="circuit_id", how="left"
    )
    return race


def add_targets(race: pd.DataFrame) -> pd.DataFrame:
    race["is_podium"] = ((race["finish_position"].notna()) & (race["finish_position"] <= 3)).astype(int)

    # Ranking-friendly target: classified finishers keep their real position;
    # DNFs get ranked after the last classified finisher in that race, so a
    # ranking-loss model has a well-defined (if crude) ordering to learn from.
    max_classified = race.groupby(["year", "round"])["finish_position"].transform("max")
    race["rank_target"] = race["finish_position"].fillna(max_classified + 1)

    return race


def build_feature_table() -> pd.DataFrame:
    race, quali, schedule, circuit_table = load_raw()
    race = add_race_date_and_sort(race, schedule)
    race = add_rolling_form_features(race)
    race = add_season_standing_features(race)
    race = add_driver_at_circuit_history(race)
    race = add_qualifying_features(race, quali)
    race = add_circuit_features(race, circuit_table)
    race = add_targets(race)

    race = race.sort_values(["date", "driver_id"]).reset_index(drop=True)
    return race


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feature_table = build_feature_table()

    out_path = os.path.join(PROCESSED_DIR, "feature_table.parquet")
    feature_table.to_parquet(out_path)

    print(f"Built feature table: {feature_table.shape[0]} rows, {feature_table.shape[1]} columns")
    print("\nColumns:")
    print(feature_table.columns.tolist())
    print("\nSample row:")
    print(feature_table.iloc[len(feature_table) // 2].to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
