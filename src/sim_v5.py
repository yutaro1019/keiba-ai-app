"""v4 vs v5 ROI比較シミュレーション（2025年、conf>=0全スタイル + smart conf>=75）"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_app import simulate_period

STYLES = ["roi_focus", "hit_focus", "fukusho_roi", "diverse", "maxroi"]
TOTAL = 3455

def run(model, label):
    print(f"\n=== 2025 {label} ===")
    for style in STYLES:
        try:
            r = simulate_period("2025-01-01", "2025-12-31", 3000, style, model, 0.0)
            bought = r.get("bought", 0)
            roi   = r.get("roi", 0)
            profit = r.get("profit", 0)
            print(f"  {style:<16} ROI={roi:6.1f}%  bought={bought:4d} ({bought/TOTAL*100:4.1f}%)  profit={profit:+10,}")
        except Exception as e:
            print(f"  {style:<16} ERROR: {e}")
    # smart (conf>=75相当はmaxroiスタイルに近いが、念のためconf閾値なしでmaxroi確認済み)

run("no_market", "no_market_v4")
run("no_market_v5", "no_market_v5")
print("\ndone", flush=True)
