"""
Diagnostic script — figures out exactly where the FastF1 data fetch is failing.
"""
import requests

endpoints = {
    "F1 livetiming (schedule index)": "https://livetiming.formula1.com/static/2024/Index.json",
    "F1 livetiming mirror": "https://livetiming-mirror.fastf1.dev/static/2024/Index.json",
    "Jolpica (Ergast replacement)": "https://api.jolpi.ca/ergast/f1/2024/1/results.json",
    "General internet check": "https://www.google.com",
}

for name, url in endpoints.items():
    try:
        r = requests.get(url, timeout=10)
        print(f"[OK]   {name}: status {r.status_code}, {len(r.content)} bytes")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")