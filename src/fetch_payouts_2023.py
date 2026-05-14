"""2023年の実払いデータをバックグラウンドで取得"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from race_scraper import fetch_market_data_for_year

print(">>> Fetching 2023 payout data...", flush=True)
result = fetch_market_data_for_year(
    year=2023,
    fetch_payouts=True,
    fetch_odds=False,
)
print(f"Done: {result}", flush=True)
