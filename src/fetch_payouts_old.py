"""2017〜2022年の実払い戻しデータを年ごとに取得する"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from race_scraper import fetch_market_data_for_year

years = [2022, 2021, 2020, 2019, 2018, 2017]

for year in years:
    print(f"\n>>> Fetching {year} payout data...", flush=True)
    result = fetch_market_data_for_year(
        year=year,
        fetch_payouts=True,
        fetch_odds=False,
    )
    ok   = result.get("payout_ok", 0)
    skip = result.get("payout_skip", 0)
    fail = result.get("payout_fail", 0)
    print(f"   {year}: 取得{ok} 既存{skip} 失敗{fail}", flush=True)

print("\nAll done.", flush=True)
