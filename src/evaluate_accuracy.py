import argparse
import gzip
import pickle
import sys

sys.path.insert(0, "src")

from predictor import DATA_PKL, KeibaPredictor
from betting import rank_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    predictor = KeibaPredictor()
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    year_df = df[df["date"].dt.year == args.year].copy()
    races = 0
    top1_win = 0
    top1_top3 = 0
    pred_top3_hits = 0
    pred_top3_at_least_one = 0
    pred_top3_all_three = 0

    for _, race_df in year_df.groupby("race_id", sort=False):
        if race_df["rank"].isna().all():
            continue
        race_df = race_df[race_df["rank"].notna()].reset_index(drop=True)
        if len(race_df) < 3:
            continue

        pred = predictor.predict_race(race_df.reset_index(drop=True))
        pred = rank_predictions(pred, "profitmax")
        races += 1

        top = pred.iloc[0]
        top1_win += int(top["rank"] == 1)
        top1_top3 += int(top["rank"] <= 3)

        pred_top3 = set(pred.head(3)["horse_no"].astype(int))
        actual_top3 = set(pred[pred["rank"] <= 3]["horse_no"].astype(int))
        hits = len(pred_top3 & actual_top3)
        pred_top3_hits += hits
        pred_top3_at_least_one += int(hits >= 1)
        pred_top3_all_three += int(hits == 3)

    print(f"year: {args.year}")
    print(f"races: {races:,}")
    print(f"features: {len(predictor.feats)}")
    print(f"top1 win hit rate: {top1_win / races * 100:.2f}% ({top1_win:,}/{races:,})")
    print(f"top1 top3 hit rate: {top1_top3 / races * 100:.2f}% ({top1_top3:,}/{races:,})")
    print(f"avg actual top3 in predicted top3: {pred_top3_hits / races:.3f} horses/race")
    print(f"predicted top3 has >=1 actual top3: {pred_top3_at_least_one / races * 100:.2f}%")
    print(f"predicted top3 exactly matches actual top3: {pred_top3_all_three / races * 100:.2f}%")


if __name__ == "__main__":
    main()
