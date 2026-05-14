"""
80%以上のレースで賭けるスタイルのROI調査。
単一スタイルをconf>=0で全レース対象に走らせ、ROIと賭けたレース数を確認する。
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_app import simulate_period

STYLES = [
    "roi_focus",
    "hit_focus",
    "fukusho_roi",
    "diverse",
    "maxroi",
]

PERIODS = [
    ("2025-01-01", "2025-12-31", "2025"),
    ("2024-01-01", "2024-12-31", "2024"),
]

for start, end, label in PERIODS:
    print(f"\n=== {label} ===")
    for style in STYLES:
        try:
            r = simulate_period(
                start_date=start,
                end_date=end,
                budget=3000,
                style=style,
                model_variant="no_market",
                min_confidence=0.0,
            )
            bought = r.get("bought", 0)
            roi = r.get("roi", 0)
            profit = r.get("profit", 0)
            actual_rate = r.get("actual_payout_rate", 0)
            print(
                f"  {style:<16} ROI={roi:6.1f}%  bought={bought:4d}"
                f"  profit={profit:+10,}  実払={actual_rate:.0f}%"
            )
        except Exception as e:
            print(f"  {style:<16} ERROR: {e}")
