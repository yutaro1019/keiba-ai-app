"""Quick style comparison: 2025 only, print after each result."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_app import simulate_period

STYLES = ["roi_focus", "hit_focus", "fukusho_roi", "diverse", "maxroi"]
TOTAL = 3455  # 2025 race count

print("=== 2025 conf>=0 style comparison ===", flush=True)
for style in STYLES:
    try:
        r = simulate_period("2025-01-01", "2025-12-31", 3000, style, "no_market", 0.0)
        bought = r.get("bought", 0)
        roi = r.get("roi", 0)
        profit = r.get("profit", 0)
        coverage = bought / TOTAL * 100
        print(
            f"  {style:<16} ROI={roi:6.1f}%  bought={bought:4d} ({coverage:4.1f}%)  profit={profit:+10,}",
            flush=True,
        )
    except Exception as e:
        print(f"  {style:<16} ERROR: {e}", flush=True)

print("done", flush=True)
