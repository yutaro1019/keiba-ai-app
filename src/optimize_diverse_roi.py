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


def payout_multiplier(candidate: dict, pred: pd.DataFrame) -> float:
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


def build_cache(year: int, win_weight: float, limit: int = None):
    predictor = KeibaPredictor()
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    year_df = df[df["date"].dt.year == year].copy()
    year_df = year_df.sort_values(["date", "race_id"]).reset_index(drop=True)
    groups = list(year_df.groupby("race_id", sort=False))
    if limit:
        groups = groups[:limit]

    rows = []
    for idx, (race_id, race_df) in enumerate(groups, start=1):
        if race_df["rank"].isna().all():
            continue
        pred = rank_prediction(predictor.predict_race(race_df.reset_index(drop=True)), win_weight)
        conf = race_confidence(pred)
        candidates = []
        for c in generate_candidates(pred, "ai"):
            candidates.append({
                "kind": c["ticket_kind"],
                "ev": float(c["ev"]),
                "p_hit": float(c["p_hit"]),
                "payout_mult": payout_multiplier(c, pred),
            })
        rows.append({"race_id": int(race_id), "confidence": conf["score"], "candidates": candidates})
        if idx % 500 == 0 or idx == len(groups):
            print(f"cached {idx}/{len(groups)} races", flush=True)
    return rows


def allocate_diverse(candidates, budget, ev_min, max_tickets, min_tickets, min_kinds, kinds, stake):
    pool = [c for c in candidates if c["kind"] in kinds and c["ev"] >= ev_min]
    if len(pool) < min_tickets:
        return []

    by_kind = {}
    for c in sorted(pool, key=lambda x: (x["ev"], x["p_hit"]), reverse=True):
        by_kind.setdefault(c["kind"], []).append(c)
    if len(by_kind) < min_kinds:
        return []

    selected = []
    for kind, items in sorted(by_kind.items(), key=lambda kv: kv[1][0]["ev"], reverse=True):
        if len(selected) >= min_kinds:
            break
        selected.append(items[0])

    used = {id(c) for c in selected}
    rest = [c for c in sorted(pool, key=lambda x: (x["ev"], x["p_hit"]), reverse=True) if id(c) not in used]
    for c in rest:
        if len(selected) >= max_tickets:
            break
        selected.append(c)

    if len(selected) < min_tickets:
        return []

    if stake == "ev_prop":
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


def backtest(cache, budget, min_conf, ev_min, max_tickets, min_tickets, min_kinds, kinds, stake):
    total_bet = total_payout = bought = hits = 0
    for row in cache:
        if row["confidence"] < min_conf:
            continue
        bets = allocate_diverse(row["candidates"], budget, ev_min, max_tickets, min_tickets, min_kinds, kinds, stake)
        race_bet = sum(amount for _, amount in bets)
        if race_bet <= 0:
            continue
        payout = int(sum(amount * c["payout_mult"] for c, amount in bets))
        total_bet += race_bet
        total_payout += payout
        bought += 1
        hits += int(payout > 0)
    return {
        "roi": total_payout / total_bet * 100 if total_bet else 0.0,
        "profit": total_payout - total_bet,
        "bought": bought,
        "hit_rate": hits / bought * 100 if bought else 0.0,
        "bet": total_bet,
        "payout": total_payout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-races", type=int, default=100)
    parser.add_argument("--win-weight", type=float, default=0.0)
    args = parser.parse_args()

    cache = build_cache(args.year, args.win_weight, args.limit)
    kind_sets = [
        {"fukusho", "wide"},
        {"fukusho", "wide", "umaren"},
        {"fukusho", "wide", "sanrenpuku"},
        {"fukusho", "wide", "umaren", "sanrenpuku"},
        {"tansho", "fukusho", "wide", "umaren", "sanrenpuku"},
    ]

    results = []
    for min_conf in [0, 55, 65, 75, 85, 92]:
        for ev_min in [0.85, 0.90, 1.00, 1.05, 1.10, 1.20]:
            for max_tickets in [4, 5, 6, 8]:
                for min_tickets in [4, 5, 6]:
                    for min_kinds in [2, 3]:
                        for kinds in kind_sets:
                            for stake in ["equal", "ev_prop"]:
                                r = backtest(
                                    cache,
                                    args.budget,
                                    min_conf,
                                    ev_min,
                                    max_tickets,
                                    min_tickets,
                                    min_kinds,
                                    kinds,
                                    stake,
                                )
                                if r["bought"] >= args.min_races:
                                    r.update({
                                        "min_conf": min_conf,
                                        "ev_min": ev_min,
                                        "max_tickets": max_tickets,
                                        "min_tickets": min_tickets,
                                        "min_kinds": min_kinds,
                                        "stake": stake,
                                        "kinds": ",".join(sorted(kinds)),
                                    })
                                    results.append(r)

    out = pd.DataFrame(results).sort_values(["roi", "profit"], ascending=False)
    cols = [
        "roi", "profit", "bought", "hit_rate", "bet", "payout",
        "min_conf", "ev_min", "max_tickets", "min_tickets", "min_kinds", "stake", "kinds",
    ]
    print("\nTop diversified ROI strategies")
    print(out[cols].head(40).to_string(index=False))
    print("\nTop diversified profit strategies")
    print(out.sort_values(["profit", "roi"], ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
