"""
Phase 1 — Data Collection

Pulls historical race results, qualifying results, and weather data
using the FastF1 API and saves them as raw parquet files.

Run this on your own machine (needs internet access):
    pip install -r requirements.txt
    python src/collect_data.py --start_year 2018 --end_year 2025

FastF1 caches every API response locally, so re-running this script
is cheap after the first pull.
"""

import argparse
import os
import time
import pandas as pd
import fastf1

CACHE_DIR = "data/raw/fastf1_cache"
RAW_DIR = "data/raw"


def setup_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


def get_season_schedule(year: int) -> pd.DataFrame:
    """Returns the race calendar for a season (event names, rounds, dates)."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule


def collect_race_weekend(year: int, round_num: int) -> dict:
    """
    Pulls race results, qualifying results, and weather for a single
    race weekend. Returns a dict of DataFrames (or None if the session
    doesn't exist / hasn't happened yet).
    """
    out = {"race_results": None, "quali_results": None, "weather": None}

    # --- Race session ---
    try:
        race = fastf1.get_session(year, round_num, "R")
        race.load(laps=False, telemetry=False, weather=True, messages=False)

        results = race.results.copy()
        results["year"] = year
        results["round"] = round_num
        results["circuit"] = race.event["EventName"]
        out["race_results"] = results

        if race.weather_data is not None and len(race.weather_data) > 0:
            w = race.weather_data.copy()
            weather_summary = pd.DataFrame([{
                "year": year,
                "round": round_num,
                "air_temp_mean": w["AirTemp"].mean(),
                "track_temp_mean": w["TrackTemp"].mean(),
                "humidity_mean": w["Humidity"].mean(),
                "rainfall": bool(w["Rainfall"].any()),
                "wind_speed_mean": w["WindSpeed"].mean(),
            }])
            out["weather"] = weather_summary
    except Exception as e:
        print(f"  [race] skipped {year} round {round_num}: {e}")

    # --- Qualifying session ---
    try:
        quali = fastf1.get_session(year, round_num, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)

        q_results = quali.results.copy()
        q_results["year"] = year
        q_results["round"] = round_num
        out["quali_results"] = q_results
    except Exception as e:
        print(f"  [quali] skipped {year} round {round_num}: {e}")

    return out


def collect_season(year: int) -> dict:
    """Collects all race weekends for a given season."""
    schedule = get_season_schedule(year)
    all_race, all_quali, all_weather = [], [], []

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        if round_num == 0:
            continue  # pre-season testing, already excluded but just in case

        print(f"Collecting {year} round {round_num}: {event['EventName']}")
        data = collect_race_weekend(year, round_num)

        if data["race_results"] is not None:
            all_race.append(data["race_results"])
        if data["quali_results"] is not None:
            all_quali.append(data["quali_results"])
        if data["weather"] is not None:
            all_weather.append(data["weather"])

        time.sleep(0.5)  # be polite to the API

    return {
        "race_results": pd.concat(all_race, ignore_index=True) if all_race else pd.DataFrame(),
        "quali_results": pd.concat(all_quali, ignore_index=True) if all_quali else pd.DataFrame(),
        "weather": pd.concat(all_weather, ignore_index=True) if all_weather else pd.DataFrame(),
    }


def main(start_year: int, end_year: int):
    setup_cache()
    os.makedirs(RAW_DIR, exist_ok=True)

    all_race, all_quali, all_weather = [], [], []

    for year in range(start_year, end_year + 1):
        print(f"\n=== Season {year} ===")
        season_data = collect_season(year)
        all_race.append(season_data["race_results"])
        all_quali.append(season_data["quali_results"])
        all_weather.append(season_data["weather"])

    race_df = pd.concat(all_race, ignore_index=True)
    quali_df = pd.concat(all_quali, ignore_index=True)
    weather_df = pd.concat(all_weather, ignore_index=True)

    race_df.to_parquet(os.path.join(RAW_DIR, "race_results.parquet"))
    quali_df.to_parquet(os.path.join(RAW_DIR, "quali_results.parquet"))
    weather_df.to_parquet(os.path.join(RAW_DIR, "weather.parquet"))

    print(f"\nSaved: {len(race_df)} race result rows, "
          f"{len(quali_df)} quali result rows, "
          f"{len(weather_df)} race-weather rows")
    print(f"Files written to {RAW_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2018,
                         help="FastF1 has reliable data from ~2018 onward")
    parser.add_argument("--end_year", type=int, default=2025)
    args = parser.parse_args()

    main(args.start_year, args.end_year)
