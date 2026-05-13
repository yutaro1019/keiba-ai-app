import argparse
import gzip
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from betting import generate_candidates
from keiba_ai import race_confidence
from predictor import DATA_PKL, KeibaPredictor


def rank_prediction(pred: pd.DataFrame, win_weight: float) -> pd.DataFrame:
    out = pred.copy()
    out["score"] = win_weight * out["p_win"] + (1.0 - win_weight) * out["p_top3"]
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["pred_rank"] = np.arange(1, len(out) + 1)
    return out


def build_races(year: int, limit: int = None):
    predictor = KeibaPredictor()
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    year_df = df[df["date"].dt.year == year].copy()
    year_df = year_df.sort_values(["date", "race_id"]).reset_index(drop=True)

    groups = list(year_df.groupby("race_id", sort=False))
    if limit:
        groups = groups[:limit]

    races = []
    for idx, (race_id, race_df) in enumerate(groups, start=1):
        if race_df["rank"].isna().all():
            continue
        pred = predictor.predict_race(race_df.reset_index(drop=True))
        races.append((int(race_id), pred))
        if idx % 500 == 0 or idx == len(groups):
            print(f"predicted {idx}/{len(groups)} races", flush=True)
    return races


def accuracy_for(races, win_weight: float):
    n = 0
    top1_win = 0
    top1_top3 = 0
    top3_hits = 0
    for _, pred0 in races:
        pred = rank_prediction(pred0, win_weight)
        n += 1
        top = pred.iloc[0]
        top1_win += int(top["rank"] == 1)
        top1_top3 += int(top["rank"] <= 3)
        pred_top3 = set(pred.head(3)["horse_no"].astype(int))
        actual_top3 = set(pred[pred["rank"] <= 3]["horse_no"].astype(int))
        top3_hits += len(pred_top3 & actual_top3)
    return {
        "top1_win": top1_win / n * 100 if n else 0.0,
        "top1_top3": top1_top3 / n * 100 if n else 0.0,
        "top3_hits": top3_hits / n if n else 0.0,
    }


def candidate_payout_multiplier(candidate: dict, pred: pd.DataFrame) -> float:
    ranks = {
        int(row["horse_no"]): int(row["rank"])
        for _, row in pred.iterrows()
        if pd.notna(row.get("horse_no")) and pd.notna(row.get("rank"))
    }
    horses = [int(h) for h in candidate["horses"]]
    if any(h not in ranks for h in horses):
        return 0.0

    kind = candidate["ticket_kind"]
    if kind == "tansho":
        hit = ranks[horses[0]] == 1
    elif kind == "fukusho":
        hit = ranks[horses[0]] <= 3
    elif kind == "wide":
        hit = all(ranks[h] <= 3 for h in horses)
    elif kind == "umaren":
        hit = set(ranks[h] for h in horses) == {1, 2}
    elif kind == "umatan":
        hit = ranks[horses[0]] == 1 and ranks[horses[1]] == 2
    elif kind == "sanrenpuku":
        hit = set(ranks[h] for h in horses) == {1, 2, 3}
    elif kind == "sanrentan":
        hit = ranks[horses[0]] == 1 and ranks[horses[1]] == 2 and ranks[horses[2]] == 3
    else:
        hit = False
    return float(candidate["odds_est"]) if hit else 0.0


def build_cache_for_weight(races, win_weight: float):
    cache = []
    for race_id, pred0 in races:
        pred = rank_prediction(pred0, win_weight)
        conf = race_confidence(pred)
        candidates = generate_candidates(pred, "ai")
        compact = []
        for c in candidates:
            compact.append({
                "kind": c["ticket_kind"],
                "ev": float(c["ev"]),
                "p_hit": float(c["p_hit"]),
                "payout_mult": candidate_payout_multiplier(c, pred),
            })
        cache.append({
            "race_id": race_id,
            "confidence": conf["score"],
            "candidates": compact,
        })
    return cache


def allocate(candidates, budget: int, ev_min: float, max_tickets: int, kinds: set, stake_mode: str):
    selected = [
        c for c in candidates
        if c["ev"] >= ev_min and c["kind"] in kinds
    ]
    if not selected:
        return []

    selected = sorted(selected, key=lambda c: (c["ev"], c["p_hit"]), reverse=True)[:max_tickets]
    if stake_mode == "top1":
        return [(selected[0], budget)]

    if stake_mode == "ev_prop":
        weights = np.array([max(0.01, c["ev"] - ev_min + 0.05) for c in selected])
        weights = weights / weights.sum()
        amounts = np.floor((weights * budget) / 100) * 100
        leftover = budget - int(amounts.sum())
        if leftover >= 100:
            amounts[0] += (leftover // 100) * 100
    else:
        unit = max(100, (budget // len(selected)) // 100 * 100)
        amounts = np.array([unit] * len(selected), dtype=float)
        leftover = budget - int(amounts.sum())
        if leftover >= 100:
            amounts[0] += (leftover // 100) * 100

    bets = []
    spent = 0
    for c, amount in zip(selected, amounts):
        amount = int(amount)
        if amount < 100 or spent + amount > budget:
            continue
        bets.append((c, amount))
        spent += amount
    return bets


def backtest_cache(cache, budget: int, min_conf: float, ev_min: float, max_tickets: int, kinds: set, stake_mode: str):
    total_bet = 0
    total_payout = 0
    bought = 0
    hits = 0
    for row in cache:
        if row["confidence"] < min_conf:
            continue
        bets = allocate(row["candidates"], budget, ev_min, max_tickets, kinds, stake_mode)
        race_bet = sum(amount for _, amount in bets)
        if race_bet <= 0:
            continue
        payout = int(sum(amount * c["payout_mult"] for c, amount in bets))
        total_bet += race_bet
        total_payout += payout
        bought += 1
        hits += int(payout > 0)

    roi = total_payout / total_bet * 100 if total_bet else 0.0
    return total_bet, total_payout, bought, hits, roi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-races", type=int, default=100)
    parser.add_argument("--weights", default="0.0,0.2,0.4,0.6,0.8,1.0")
    args = parser.parse_args()

    races = build_races(args.year, args.limit)

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    print("\nAccuracy by ranking weight")
    for w in weights:
        a = accuracy_for(races, w)
        print(
            f"win_weight={w:.1f}  top1_win={a['top1_win']:.2f}%  "
            f"top1_top3={a['top1_top3']:.2f}%  top3_hits={a['top3_hits']:.3f}"
        )

    kind_sets = [
        {"fukusho"},
        {"wide"},
        {"umaren"},
        {"tansho", "fukusho"},
        {"fukusho", "wide"},
        {"tansho", "fukusho", "wide"},
        {"tansho", "fukusho", "wide", "umaren"},
        {"fukusho", "wide", "umaren", "sanrenpuku"},
        {"tansho", "fukusho", "wide", "umaren", "sanrenpuku", "umatan", "sanrentan"},
    ]
    min_confs = [0, 55, 65, 75, 85, 92, 97]
    ev_mins = [0.90, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50, 2.00, 3.00]
    max_ticket_grid = [1, 2, 3, 5]
    stake_modes = ["top1", "equal", "ev_prop"]

    results = []
    for w in weights:
        print(f"\nsearching win_weight={w:.1f}", flush=True)
        cache = build_cache_for_weight(races, w)
        for min_conf in min_confs:
            for ev_min in ev_mins:
                for max_tickets in max_ticket_grid:
                    for kinds in kind_sets:
                        for stake_mode in stake_modes:
                            bet, payout, bought, hits, roi = backtest_cache(
                                cache,
                                budget=args.budget,
                                min_conf=float(min_conf),
                                ev_min=float(ev_min),
                                max_tickets=max_tickets,
                                kinds=kinds,
                                stake_mode=stake_mode,
                            )
                            if bought >= args.min_races:
                                results.append({
                                    "roi": roi,
                                    "profit": payout - bet,
                                    "bought": bought,
                                    "hit_rate": hits / bought * 100 if bought else 0.0,
                                    "bet": bet,
                                    "payout": payout,
                                    "win_weight": w,
                                    "min_conf": min_conf,
                                    "ev_min": ev_min,
                                    "max_tickets": max_tickets,
                                    "stake": stake_mode,
                                    "kinds": ",".join(sorted(kinds)),
                                })

    out = pd.DataFrame(results).sort_values(["roi", "profit"], ascending=False)
    print("\nTop ROI strategies")
    cols = [
        "roi", "profit", "bought", "hit_rate", "bet", "payout",
        "win_weight", "min_conf", "ev_min", "max_tickets", "stake", "kinds",
    ]
    print(out[cols].head(40).to_string(index=False))

    print("\nTop Profit strategies")
    print(out.sort_values(["profit", "roi"], ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
