"""
🏇 KEIBA AI デモスクリプト
- 学習済モデルで2025年の有名レースを予想
- 4つの賭け方スタイル全部のサンプルを表示
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from predictor import KeibaPredictor, load_race_from_data, list_recent_races
from betting import suggest, format_suggestion
from keiba_ai import show_race_info, show_prediction, BANNER, VENUE_NAMES


def run_demo():
    print(BANNER)
    print("⚙️  モデルをロード中...\n", flush=True)
    p = KeibaPredictor()
    metrics = p.meta.get("metrics", {})
    print(f"  ✓ TOP3 アンサンブル AUC: {metrics.get('ens_top3_auc', 0):.4f}")
    print(f"  ✓ WIN  アンサンブル AUC: {metrics.get('ens_win_auc', 0):.4f}\n")

    # 2025年の最新レースから1つピック
    races = list_recent_races(2025, limit=10)
    rid = int(races.iloc[0]["race_id"])

    print("=" * 70)
    print("📋 デモ: 2025年 最新レースで AI 予想 + 賭け方4スタイル提案")
    print("=" * 70)

    df = load_race_from_data(rid)
    show_race_info(df)

    pred = p.predict_race(df)
    show_prediction(pred, top_n=min(10, len(pred)), show_actual=True)

    # 4スタイル × 予算3000円
    for style in ["kenjitsu", "balance", "ippatsu", "ai"]:
        result = suggest(pred, budget=3000, style=style)
        print(format_suggestion(result))


if __name__ == "__main__":
    run_demo()
