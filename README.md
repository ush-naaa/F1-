# F1 Podium Predictor

Predicts race finishing order (and derived podium/top-3) using pre-race
features: qualifying results, rolling driver/constructor form, and
circuit-specific characteristics.

## Project structure

```
f1_podium_predictor/
├── data/
│   ├── raw/            # raw pulls from FastF1 (race results, quali, weather)
│   └── processed/       # engineered feature tables, ready for modeling
├── src/
│   ├── collect_data.py  # Phase 1: pull historical data
│   ├── build_circuit_table.py   # Phase 2: circuit reference table (next)
│   ├── build_features.py        # Phase 3: feature engineering (next)
│   └── train_model.py           # Phase 5: model training (next)
├── notebooks/            # exploratory analysis / error analysis
└── requirements.txt
```

## Setup (run on your own machine — needs internet access)

```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run order

1. **Collect raw data**
   ```bash
   python src/collect_data.py --start_year 2018 --end_year 2025
   ```
   This pulls race results, qualifying results, and weather summaries for
   every race weekend in the range, and caches raw API responses locally
   (`data/raw/fastf1_cache/`) so re-runs are fast. First run for the full
   range will take a while — FastF1 hits the live timing API per session.

   Output: `data/raw/race_results.parquet`, `quali_results.parquet`,
   `weather.parquet`.

2. *(next)* Build circuit reference table
3. *(next)* Build the full feature table (driver form + qualifying + track features)
4. *(next)* Train the ranking model
5. *(next)* Validate with chronological splits
6. *(next)* Build inference pipeline for upcoming races

## Notes on data range

FastF1 has full session data (laps, telemetry, weather) reliably from
**2018 onward**. If you want older history for career-long driver stats,
supplement with the Ergast API or a Kaggle F1 dataset, but that will need
separate cleaning since schemas differ.

## Design principles this project follows

- **No random train/test splits** — all validation is chronological
  (train on past seasons, test on later ones) to avoid leaking future
  information.
- **Relative/recent-form features**, not raw team/driver identity, are
  the primary signal — this keeps the model robust across regulation eras.
- **Recency-weighted training samples** — more recent races count more.
- **Every feature must be knowable at prediction time** (Saturday evening,
  post-qualifying, pre-race).
