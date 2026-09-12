"""
Phase 2 — Circuit Reference Table

Builds one row per circuit summarizing characteristics that stay fairly
stable across regulation eras: DNF/attrition rate, street-circuit flag,
how much finishing order shuffles relative to the grid, and historical
rain likelihood.

This is a STATIC table (recomputed occasionally, not per-race) that gets
joined onto every driver-race row later in feature engineering.

Run:
    python src/build_circuit_table.py

Reads:  data/raw/race_results.parquet, schedule.parquet, weather.parquet
Writes: data/processed/circuit_table.parquet
"""

import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Circuits that are run on closed public roads / temporary street layouts.
# Overtaking is harder and DNF-by-barrier-contact is more common here.
# circuit_id values match Jolpica's naming.
STREET_CIRCUITS = {
    "monaco",       # Monaco GP
    "baku",         # Azerbaijan GP
    "marina_bay",   # Singapore GP
    "jeddah",       # Saudi Arabian GP
    "miami",        # Miami GP (semi-permanent street-style circuit)
    "vegas",        # Las Vegas GP
}

# Jolpica's `status` field uses a fixed small set of values. A driver is
# a DNF (did not finish/classify normally) unless they're "Finished" or
# "Lapped" (still classified, just laps down — NOT a retirement).
CLASSIFIED_FINISH_STATUSES = {"Finished", "Lapped"}

def _is_dnf(status: str) -> bool:
    return status not in CLASSIFIED_FINISH_STATUSES


def build_circuit_table() -> pd.DataFrame:
    race_results = pd.read_parquet(os.path.join(RAW_DIR, "race_results.parquet"))
    schedule = pd.read_parquet(os.path.join(RAW_DIR, "schedule.parquet"))
    weather = pd.read_parquet(os.path.join(RAW_DIR, "weather.parquet"))

    # Attach circuit_id to every race result row via (year, round)
    race_results = race_results.merge(
        schedule[["year", "round", "circuit_id", "circuit_name"]],
        on=["year", "round"], how="left"
    )

    race_results["is_dnf"] = race_results["status"].apply(_is_dnf)

    # --- DNF rate per circuit ---
    dnf_stats = (
        race_results.groupby("circuit_id")["is_dnf"]
        .mean()
        .rename("dnf_rate")
        .reset_index()
    )

    # --- Position-shuffle proxy: average |grid - finish| per circuit ---
    # Only meaningful for classified finishers (DNFs distort raw position math)
    finishers = race_results[~race_results["is_dnf"]].copy()
    finishers["position_change"] = (finishers["grid"] - finishers["finish_position"]).abs()
    shuffle_stats = (
        finishers.groupby("circuit_id")["position_change"]
        .mean()
        .rename("avg_position_change")
        .reset_index()
    )

    # --- Number of races observed per circuit (so downstream code can
    #     judge how reliable each stat is — 1 race vs 10 races matters) ---
    race_counts = (
        race_results.groupby("circuit_id")["round"]
        .nunique()
        .rename("races_observed")
        .reset_index()
    )

    # --- Rain history per circuit ---
    weather_with_circuit = weather.merge(
        schedule[["year", "round", "circuit_id"]], on=["year", "round"], how="left"
    )
    weather_with_circuit["had_rain"] = weather_with_circuit["precipitation_mm"] > 1.0
    rain_stats = (
        weather_with_circuit.groupby("circuit_id")["had_rain"]
        .mean()
        .rename("rain_probability")
        .reset_index()
    )

    # --- Combine everything ---
    circuit_table = schedule[["circuit_id", "circuit_name", "lat", "long", "country"]].drop_duplicates("circuit_id")
    circuit_table = circuit_table.merge(dnf_stats, on="circuit_id", how="left")
    circuit_table = circuit_table.merge(shuffle_stats, on="circuit_id", how="left")
    circuit_table = circuit_table.merge(race_counts, on="circuit_id", how="left")
    circuit_table = circuit_table.merge(rain_stats, on="circuit_id", how="left")

    circuit_table["is_street_circuit"] = circuit_table["circuit_id"].isin(STREET_CIRCUITS).astype(int)

    return circuit_table.sort_values("circuit_id").reset_index(drop=True)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    circuit_table = build_circuit_table()

    out_path = os.path.join(PROCESSED_DIR, "circuit_table.parquet")
    circuit_table.to_parquet(out_path)

    print(f"Built circuit table: {len(circuit_table)} circuits")
    print(circuit_table[[
        "circuit_id", "races_observed", "dnf_rate",
        "avg_position_change", "rain_probability", "is_street_circuit"
    ]].to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
