"""
Phase 1 — Data Collection (v3)

Two reliability fixes over v2:
  1. EVERY API call (not just weather) now retries with backoff. Free
     APIs under load occasionally time out — that's not a "your network
     is blocked" problem like the earlier 403 issue, it's just normal
     flakiness that a retry absorbs.
  2. Per-season checkpointing. Each season's data is saved to disk the
     moment it's collected (data/raw/by_season/). If the script crashes
     on season 6 of 8, seasons 1-5 are NOT lost — re-running the script
     skips any season that already has a complete checkpoint file and
     picks up where it left off. At the end, all season files are
     combined into the final data/raw/*.parquet outputs.

Run:
    pip install -r requirements.txt
    python src/collect_data_v2.py --start_year 2018 --end_year 2025

Safe to re-run after any crash — it resumes automatically.
"""

import argparse
import os
import time
import requests
import pandas as pd

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = "data/raw"
BY_SEASON_DIR = os.path.join(RAW_DIR, "by_season")
REQUEST_DELAY = 0.3  # be polite to the free APIs


def _get_json(url: str, params: dict | None = None, timeout: int = 25, max_retries: int = 4) -> dict:
    """GET a URL as JSON, retrying with backoff on any request failure.
    Used by every API call in this file — this is the single fix that
    makes the whole pipeline resilient to free-tier API flakiness."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 * attempt
                print(f"  [retry] attempt {attempt} failed for {url}: {e} — retrying in {wait}s")
                time.sleep(wait)
    raise last_error


def get_season_schedule(year: int) -> pd.DataFrame:
    data = _get_json(f"{JOLPICA_BASE}/{year}.json", params={"limit": 40})
    races = data["MRData"]["RaceTable"]["Races"]

    rows = []
    for r in races:
        circuit = r["Circuit"]
        rows.append({
            "year": year,
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "date": r["date"],
            "circuit_id": circuit["circuitId"],
            "circuit_name": circuit["circuitName"],
            "lat": float(circuit["Location"]["lat"]),
            "long": float(circuit["Location"]["long"]),
            "locality": circuit["Location"]["locality"],
            "country": circuit["Location"]["country"],
        })
    return pd.DataFrame(rows)


def get_race_results(year: int, round_num: int) -> pd.DataFrame:
    data = _get_json(f"{JOLPICA_BASE}/{year}/{round_num}/results.json", params={"limit": 40})
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()

    rows = []
    for res in races[0]["Results"]:
        rows.append({
            "year": year,
            "round": round_num,
            "driver_id": res["Driver"]["driverId"],
            "driver_code": res["Driver"].get("code"),
            "constructor_id": res["Constructor"]["constructorId"],
            "grid": int(res["grid"]),
            "finish_position": int(res["position"]) if res["positionText"].isdigit() else None,
            "position_text": res["positionText"],
            "status": res["status"],
            "points": float(res["points"]),
            "laps": int(res["laps"]),
        })
    return pd.DataFrame(rows)


def get_qualifying_results(year: int, round_num: int) -> pd.DataFrame:
    data = _get_json(f"{JOLPICA_BASE}/{year}/{round_num}/qualifying.json", params={"limit": 40})
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()

    rows = []
    for res in races[0]["QualifyingResults"]:
        rows.append({
            "year": year,
            "round": round_num,
            "driver_id": res["Driver"]["driverId"],
            "constructor_id": res["Constructor"]["constructorId"],
            "quali_position": int(res["position"]),
            "q1": res.get("Q1"),
            "q2": res.get("Q2"),
            "q3": res.get("Q3"),
        })
    return pd.DataFrame(rows)


def get_weather(lat: float, lon: float, date: str) -> dict | None:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "UTC",
    }
    try:
        data = _get_json(OPEN_METEO_ARCHIVE, params=params, timeout=40)
        daily = data.get("daily", {})
        if not daily.get("time"):
            return None
        return {
            "temp_max": daily["temperature_2m_max"][0],
            "temp_min": daily["temperature_2m_min"][0],
            "precipitation_mm": daily["precipitation_sum"][0],
            "wind_speed_max": daily["windspeed_10m_max"][0],
        }
    except Exception as e:
        print(f"  [weather] gave up on ({lat},{lon}) on {date}: {e}")
        return None


def season_checkpoint_paths(year: int) -> dict:
    return {
        "schedule": os.path.join(BY_SEASON_DIR, f"schedule_{year}.parquet"),
        "race_results": os.path.join(BY_SEASON_DIR, f"race_results_{year}.parquet"),
        "quali_results": os.path.join(BY_SEASON_DIR, f"quali_results_{year}.parquet"),
        "weather": os.path.join(BY_SEASON_DIR, f"weather_{year}.parquet"),
    }


def season_already_done(year: int) -> bool:
    paths = season_checkpoint_paths(year)
    return all(os.path.exists(p) for p in paths.values())


def collect_season(year: int) -> dict:
    schedule = get_season_schedule(year)
    all_race, all_quali, all_weather = [], [], []

    for _, event in schedule.iterrows():
        round_num = event["round"]
        print(f"Collecting {year} round {round_num}: {event['race_name']}")

        race_df = get_race_results(year, round_num)
        if not race_df.empty:
            all_race.append(race_df)
        time.sleep(REQUEST_DELAY)

        quali_df = get_qualifying_results(year, round_num)
        if not quali_df.empty:
            all_quali.append(quali_df)
        time.sleep(REQUEST_DELAY)

        weather = get_weather(event["lat"], event["long"], event["date"])
        if weather is not None:
            weather["year"] = year
            weather["round"] = round_num
            all_weather.append(pd.DataFrame([weather]))
        time.sleep(REQUEST_DELAY)

    return {
        "schedule": schedule,
        "race_results": pd.concat(all_race, ignore_index=True) if all_race else pd.DataFrame(),
        "quali_results": pd.concat(all_quali, ignore_index=True) if all_quali else pd.DataFrame(),
        "weather": pd.concat(all_weather, ignore_index=True) if all_weather else pd.DataFrame(),
    }


def save_season_checkpoint(year: int, season_data: dict):
    os.makedirs(BY_SEASON_DIR, exist_ok=True)
    paths = season_checkpoint_paths(year)
    for key, path in paths.items():
        season_data[key].to_parquet(path)
    print(f"  Checkpoint saved for {year}")


def combine_all_seasons(start_year: int, end_year: int):
    """Reads every per-season checkpoint file and combines into the
    final combined outputs used by later pipeline stages."""
    all_schedule, all_race, all_quali, all_weather = [], [], [], []

    for year in range(start_year, end_year + 1):
        paths = season_checkpoint_paths(year)
        if not season_already_done(year):
            print(f"WARNING: {year} has no complete checkpoint — skipping from combined output")
            continue
        all_schedule.append(pd.read_parquet(paths["schedule"]))
        all_race.append(pd.read_parquet(paths["race_results"]))
        all_quali.append(pd.read_parquet(paths["quali_results"]))
        all_weather.append(pd.read_parquet(paths["weather"]))

    schedule_df = pd.concat(all_schedule, ignore_index=True)
    race_df = pd.concat(all_race, ignore_index=True)
    quali_df = pd.concat(all_quali, ignore_index=True)
    weather_df = pd.concat(all_weather, ignore_index=True)

    schedule_df.to_parquet(os.path.join(RAW_DIR, "schedule.parquet"))
    race_df.to_parquet(os.path.join(RAW_DIR, "race_results.parquet"))
    quali_df.to_parquet(os.path.join(RAW_DIR, "quali_results.parquet"))
    weather_df.to_parquet(os.path.join(RAW_DIR, "weather.parquet"))

    print(f"\nCombined: {len(schedule_df)} races, {len(race_df)} race result rows, "
          f"{len(quali_df)} quali result rows, {len(weather_df)} weather rows")
    print(f"Files written to {RAW_DIR}/")


def main(start_year: int, end_year: int):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(BY_SEASON_DIR, exist_ok=True)

    for year in range(start_year, end_year + 1):
        if season_already_done(year):
            print(f"\n=== Season {year}: already collected, skipping ===")
            continue

        print(f"\n=== Season {year} ===")
        season_data = collect_season(year)
        save_season_checkpoint(year, season_data)

    print("\nAll seasons collected. Combining into final output files...")
    combine_all_seasons(start_year, end_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2025)
    args = parser.parse_args()

    main(args.start_year, args.end_year)
