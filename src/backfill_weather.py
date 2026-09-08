"""
Backfill missing weather rows only — doesn't touch race_results.parquet
or quali_results.parquet, which already pulled successfully.

Run:
    python src/backfill_weather.py --start_year 2024 --end_year 2024
"""
import argparse
import os
import pandas as pd

from collect_data_v2 import get_season_schedule, get_weather, RAW_DIR


def main(start_year: int, end_year: int):
    weather_path = os.path.join(RAW_DIR, "weather.parquet")
    existing = pd.read_parquet(weather_path) if os.path.exists(weather_path) else pd.DataFrame(columns=["year", "round"])

    new_rows = []
    for year in range(start_year, end_year + 1):
        schedule = get_season_schedule(year)
        for _, event in schedule.iterrows():
            round_num = event["round"]
            already_have = ((existing["year"] == year) & (existing["round"] == round_num)).any()
            if already_have:
                continue

            print(f"Fetching weather: {year} round {round_num} ({event['race_name']})")
            weather = get_weather(event["lat"], event["long"], event["date"])
            if weather is not None:
                weather["year"] = year
                weather["round"] = round_num
                new_rows.append(weather)

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["year", "round"])
        combined.to_parquet(weather_path)
        print(f"\nAdded {len(new_rows)} new weather rows. Total now: {len(combined)}")
    else:
        print("\nNo new weather rows added (all already present or all retries failed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2024)
    parser.add_argument("--end_year", type=int, default=2024)
    args = parser.parse_args()
    main(args.start_year, args.end_year)
