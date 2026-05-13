import argparse
import gzip
import pickle
import sys

import pandas as pd

sys.path.insert(0, "src")

from betting import fukusho_odds_est, odds_est_for
from keiba_ai import race_confidence
from predictor import DATA_PKL, KeibaPredictor


def settle(kind, horses, amount, odds, pred):
    ranks = {
        int(r["horse_no"]): int(r["rank"])
        for _, r in pred.iterrows()
        if pd.notna(r["horse_no"]) and pd.notna(r["rank"])
    }
    if any(h not in ranks for h in horses):
        return 0
    if kind == "tansho":
        hit = ranks[horses[0]] == 1
    elif kind == "fukusho":
        hit = ranks[horses[0]] <= 3
    elif kind == "wide":
        hit = all(ranks[h] <= 3 for h in horses)
    else:
        hit = False
    return int(amount * odds) if hit else 0


def add_result(results, name, bet, payout):
    item = results.setdefault(name, {"races": 0, "bet": 0, "payout": 0, "hits": 0})
    if bet > 0:
        item["races"] += 1
        item["bet"] += bet
        item["payout"] += payout
        item["hits"] += int(payout > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    predictor = KeibaPredictor()
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    year_df = df[df["date"].dt.year == args.year].copy()
    results = {}

    for _, race_df in year_df.groupby("race_id", sort=False):
        if race_df["rank"].isna().all():
            continue
        pred = predictor.predict_race(race_df.reset_index(drop=True))
        conf = race_confidence(pred)
        if conf["score"] < args.min_confidence:
            continue

        top = pred.iloc[0]
        top1 = int(top["horse_no"])
        top2 = int(pred.iloc[1]["horse_no"]) if len(pred) > 1 else None
        top3 = int(pred.iloc[2]["horse_no"]) if len(pred) > 2 else None

        win_odds = float(top["odds"]) if "odds" in top and pd.notna(top["odds"]) else max(1.1, 0.8 / max(float(top["p_win"]), 1e-4))
        fuku_odds = fukusho_odds_est(float(top["p_top3"]), win_odds)
        payout = settle("fukusho", [top1], args.budget, fuku_odds, pred)
        add_result(results, "top1_fukusho_all", args.budget, payout)

        payout = settle("tansho", [top1], args.budget, win_odds, pred)
        add_result(results, "top1_tansho_all", args.budget, payout)

        if float(top["p_top3"]) >= 0.35:
            payout = settle("fukusho", [top1], args.budget, fuku_odds, pred)
            add_result(results, "top1_fukusho_pTop3_35", args.budget, payout)

        if top2 is not None:
            p = min(0.95, float(pred.iloc[0]["p_top3"]) * float(pred.iloc[1]["p_top3"]) * 1.5)
            wide_odds = odds_est_for("wide", [p])
            payout = settle("wide", sorted([top1, top2]), args.budget, wide_odds, pred)
            add_result(results, "wide_top1_top2", args.budget, payout)

        if top2 is not None and top3 is not None:
            bets = [
                sorted([top1, top2]),
                sorted([top1, top3]),
                sorted([top2, top3]),
            ]
            amount = args.budget // 3
            total_bet = 0
            total_payout = 0
            for pair in bets:
                rows = pred[pred["horse_no"].astype(int).isin(pair)]
                p = min(0.95, float(rows.iloc[0]["p_top3"]) * float(rows.iloc[1]["p_top3"]) * 1.5)
                wide_odds = odds_est_for("wide", [p])
                total_bet += amount
                total_payout += settle("wide", pair, amount, wide_odds, pred)
            add_result(results, "wide_top3_box", total_bet, total_payout)

    print(f"year={args.year} min_confidence={args.min_confidence}")
    print("strategy                  races      hit%          bet       payout     ROI")
    for name, item in sorted(results.items()):
        roi = item["payout"] / item["bet"] * 100 if item["bet"] else 0
        hit = item["hits"] / item["races"] * 100 if item["races"] else 0
        print(f"{name:<24} {item['races']:>6,}  {hit:>7.1f}%  {item['bet']:>11,} {item['payout']:>12,} {roi:>7.1f}%")


if __name__ == "__main__":
    main()
