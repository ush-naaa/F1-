"""
Phase 1 — Data Collection (v2)

FastF1's session.load() depends on livetiming.formula1.com, which returns
403 Forbidden from cloud/datacenter IPs (e.g. GitHub Codespaces). This
version bypasses that entirely and pulls data directly from:

  - Jolpica (https://api.jolpi.ca/ergast) — the actively maintained
    Ergast-API-compatible replacement. Gives us race results, qualifying
    results, grid positions, points, and finish status (DNF etc).
  - Open-Meteo (https://open-meteo.com) — free historical + forecast
    weather API, keyed off each circuit's lat/long. Also usable later
    for forecasting weather on an upcoming race weekend.

Run:
    pip install -r requirements.txt
    python src/collect_data_v2.py --start_year 2018 --end_year 2025
"""

import argparse
import os
import time
import requests
import pandas as pd

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = "data/raw"
REQUEST_DELAY = 0.3  # be polite to the free API


def _get_json(url: str, params: dict | None = None, timeout: int = 20) -> dict:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_season_schedule(year: int) -> pd.DataFrame:
    """Returns one row per race: round, name, date, circuit id, lat/long."""
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
    """Race classification: finishing position, grid, status, points, driver/constructor."""
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
            "position_text": res["positionText"],   # e.g. "R" for retired
            "status": res["status"],                 # e.g. "Finished", "Accident", "+1 Lap"
            "points": float(res["points"]),
            "laps": int(res["laps"]),
        })
    return pd.DataFrame(rows)


def get_qualifying_results(year: int, round_num: int) -> pd.DataFrame:
    """Qualifying classification: grid-deciding position and Q1/Q2/Q3 times."""
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


def get_weather(lat: float, lon: float, date: str, max_retries: int = 3) -> dict | None:
    """Daily weather summary for a circuit on race day, via Open-Meteo archive API.
    Retries with backoff since this free endpoint is occasionally slow, not blocked."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "UTC",
    }
    for attempt in range(1, max_retries + 1):
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
            if attempt < max_retries:
                wait = 2 * attempt
                print(f"  [weather] attempt {attempt} failed for ({lat},{lon}) on {date}: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  [weather] gave up on ({lat},{lon}) on {date} after {max_retries} attempts: {e}")
                return None


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


def main(start_year: int, end_year: int):
    os.makedirs(RAW_DIR, exist_ok=True)

    all_schedule, all_race, all_quali, all_weather = [], [], [], []

    for year in range(start_year, end_year + 1):
        print(f"\n=== Season {year} ===")
        season_data = collect_season(year)
        all_schedule.append(season_data["schedule"])
        all_race.append(season_data["race_results"])
        all_quali.append(season_data["quali_results"])
        all_weather.append(season_data["weather"])

    schedule_df = pd.concat(all_schedule, ignore_index=True)
    race_df = pd.concat(all_race, ignore_index=True)
    quali_df = pd.concat(all_quali, ignore_index=True)
    weather_df = pd.concat(all_weather, ignore_index=True)

    schedule_df.to_parquet(os.path.join(RAW_DIR, "schedule.parquet"))
    race_df.to_parquet(os.path.join(RAW_DIR, "race_results.parquet"))
    quali_df.to_parquet(os.path.join(RAW_DIR, "quali_results.parquet"))
    weather_df.to_parquet(os.path.join(RAW_DIR, "weather.parquet"))

    print(f"\nSaved: {len(schedule_df)} races, {len(race_df)} race result rows, "
          f"{len(quali_df)} quali result rows, {len(weather_df)} weather rows")
    print(f"Files written to {RAW_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2018)
    parser.add_argument("--end_year", type=int, default=2025)
    args = parser.parse_args()

    main(args.start_year, args.end_year)
