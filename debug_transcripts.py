"""Step 1 debug: hit API Ninjas transcript endpoint directly, no app code."""
import os
from dotenv import load_dotenv
load_dotenv()

import httpx

key = os.environ.get("API_NINJAS_KEY", "")
print(f"API_NINJAS_KEY loaded: length={len(key)}, starts={key[:4]}...\n")

BASE = "https://api.api-ninjas.com/v1/earningstranscript"
HEADERS = {"X-Api-Key": key}

# Matrix: (year, quarter) — 4 most recent calendar quarters + fiscal assumptions for AAPL
# Calendar: 2024 Q3, 2024 Q2, 2024 Q1, 2023 Q4
# Apple fiscal: FY ends Sep — FY2024 Q4 = Jul-Sep 2024, FY2024 Q3 = Apr-Jun 2024, etc.
# API Ninjas docs say quarter is 1-4 fiscal quarter of the company
# AAPL FY2024: Q4=Jul-Sep24, Q3=Apr-Jun24, Q2=Jan-Mar24, Q1=Oct-Dec23
# AAPL FY2025: Q1=Oct-Dec24 (most recent as of Aug 2026)

attempts = [
    # Most recent Apple fiscal quarters (FY2025)
    {"ticker": "AAPL", "year": 2025, "quarter": 3, "label": "AAPL FY2025 Q3 (Apr-Jun25)"},
    {"ticker": "AAPL", "year": 2025, "quarter": 2, "label": "AAPL FY2025 Q2 (Jan-Mar25)"},
    {"ticker": "AAPL", "year": 2025, "quarter": 1, "label": "AAPL FY2025 Q1 (Oct-Dec24)"},
    {"ticker": "AAPL", "year": 2024, "quarter": 4, "label": "AAPL FY2024 Q4 (Jul-Sep24)"},
    {"ticker": "AAPL", "year": 2024, "quarter": 3, "label": "AAPL FY2024 Q3 (Apr-Jun24)"},
    # Calendar-year framing in case API uses that
    {"ticker": "AAPL", "year": 2026, "quarter": 1, "label": "AAPL CY2026 Q1 calendar"},
    {"ticker": "AAPL", "year": 2025, "quarter": 4, "label": "AAPL CY2025 Q4 calendar"},
    {"ticker": "AAPL", "year": 2024, "quarter": 4, "label": "AAPL CY2024 Q4 calendar (dup check)"},
]

for a in attempts:
    params = {"ticker": a["ticker"], "year": a["year"], "quarter": a["quarter"]}
    url = BASE
    print(f"--- {a['label']} ---")
    print(f"  URL:    {url}")
    print(f"  Params: {params}")
    try:
        r = httpx.get(url, params=params, headers=HEADERS, timeout=10)
        print(f"  Status: {r.status_code}")
        body = r.text
        print(f"  Body (first 500): {body[:500]!r}")
        # Try parse
        try:
            import json
            parsed = json.loads(body)
            print(f"  Parsed type: {type(parsed).__name__}, ", end="")
            if isinstance(parsed, dict):
                print(f"keys={list(parsed.keys())[:8]}")
            elif isinstance(parsed, list):
                print(f"len={len(parsed)}, first_keys={list(parsed[0].keys()) if parsed else 'empty'}")
            else:
                print(f"value={parsed!r}")
        except Exception as pe:
            print(f"  JSON parse error: {pe}")
    except Exception as e:
        print(f"  REQUEST FAILED: {e}")
    print()
