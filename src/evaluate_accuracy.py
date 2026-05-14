import argparse
import gzip
import pickle
import sys

sys.path.insert(0, "src")

from predictor import DATA_PKL, DEFAULT_MODEL_VARIANT, MODEL_VARIANTS, KeibaPredictor
from betting import rank_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--model-variant", default=DEFAULT_MODEL_VARIANT, choices=MODEL_VARIANTS.keys())
    args = parser.parse_args()

    predictor = KeibaPredictor(model_variant=args.model_variant)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    year_df = df[df["date"].dt.year == args.year].copy()
    races = 0
    top1_win = 0
    top1_top3 = 0
    pred_top3_hits = 0
    pred_top3_at_least_one = 0
    pred_top3_all_three = 0

    predicted = predictor.predict_frame(year_df.reset_index(drop=True))
    predicted = rank_predictions(predicted, "profitmax")

    for _, race_df in predicted.groupby("race_id", sort=False):
        if race_df["rank"].isna().all():
            continue
        race_df = race_df[race_df["rank"].notna()].reset_index(drop=True)
        if len(race_df) < 3:
            continue

        pred = race_df.sort_values("pred_rank").reset_index(drop=True)
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
    print(f"model_variant: {args.model_variant}")
    print(f"races: {races:,}")
    print(f"features: {len(predictor.feats)}")
    print(f"top1 win hit rate: {top1_win / races * 100:.2f}% ({top1_win:,}/{races:,})")
    print(f"top1 top3 hit rate: {top1_top3 / races * 100:.2f}% ({top1_top3:,}/{races:,})")
    print(f"avg actual top3 in predicted top3: {pred_top3_hits / races:.3f} horses/race")
    print(f"predicted top3 has >=1 actual top3: {pred_top3_at_least_one / races * 100:.2f}%")
    print(f"predicted top3 exactly matches actual top3: {pred_top3_all_three / races * 100:.2f}%")


if __name__ == "__main__":
    main()
