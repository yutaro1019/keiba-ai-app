import argparse
import gc
import gzip
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from predictor import DATA_PKL, DEFAULT_MODEL_VARIANT, MODEL_VARIANTS, KeibaPredictor
from betting import rank_predictions


def _run_full_pipeline(df: pd.DataFrame, fe: str) -> pd.DataFrame:
    """訓練時と同じ特徴量エンジニアリングパイプラインを全データに適用する。"""
    if fe in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"):
        from train_no_market_v5 import (
            add_sire_features, add_horse_rolling_features, add_jockey_trainer_rolling,
            add_field_strength, add_weight_trend, add_lap_rolling_features, add_field_pace_features,
        )
        for label, fn in [
            ("sire/class",    add_sire_features),
            ("horse rolling", add_horse_rolling_features),
            ("jockey/trainer",add_jockey_trainer_rolling),
            ("field strength",add_field_strength),
            ("weight trend",  add_weight_trend),
            ("lap rolling",   add_lap_rolling_features),
            ("field pace",    add_field_pace_features),
        ]:
            print(f"  {label}...", flush=True)
            df = fn(df); gc.collect()

    if fe in ("v6", "v7", "v8", "v9", "v10", "v11"):
        from train_no_market_v6 import add_v6_features
        print("  v6...", flush=True); df = add_v6_features(df); gc.collect()

    if fe in ("v7", "v8", "v9", "v10", "v11"):
        from train_no_market_v7 import add_v7_features
        print("  v7...", flush=True); df = add_v7_features(df); gc.collect()

    if fe in ("v8", "v9", "v10", "v11"):
        from train_no_market_v8 import add_v8_features
        print("  v8...", flush=True); df = add_v8_features(df); gc.collect()

    if fe in ("v9", "v10", "v11"):
        from train_no_market_v10 import add_v10_features
        print("  v9/v10...", flush=True); df = add_v10_features(df); gc.collect()

    if fe in ("v11",):
        from train_no_market_v11 import add_v11_features
        print("  v11...", flush=True); df = add_v11_features(df); gc.collect()

    from feature_engineering import add_model_features
    print("  vs_field...", flush=True); df = add_model_features(df, race_col="race_id"); gc.collect()
    return df


def _add_jh_combo(df: pd.DataFrame) -> pd.DataFrame:
    """jockey×horse コンボ特徴量（lambdarank 訓練スクリプトと同一ロジック）。"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"] = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]     = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_t"], inplace=True)
    return df


def _calc_hit_rates(predicted: pd.DataFrame):
    races = top1_win = top1_top3 = pred_top3_hits = pred_top3_at_least_one = pred_top3_all_three = 0
    for _, race_df in predicted.groupby("race_id", sort=False):
        if race_df["rank"].isna().all():
            continue
        race_df = race_df[race_df["rank"].notna()].reset_index(drop=True)
        if len(race_df) < 3:
            continue
        pred = race_df.sort_values("pred_rank").reset_index(drop=True)
        races += 1
        top = pred.iloc[0]
        top1_win  += int(top["rank"] == 1)
        top1_top3 += int(top["rank"] <= 3)
        pred_top3  = set(pred.head(3)["horse_no"].astype(int))
        actual_top3 = set(pred[pred["rank"] <= 3]["horse_no"].astype(int))
        hits = len(pred_top3 & actual_top3)
        pred_top3_hits         += hits
        pred_top3_at_least_one += int(hits >= 1)
        pred_top3_all_three    += int(hits == 3)
    return races, top1_win, top1_top3, pred_top3_hits, pred_top3_at_least_one, pred_top3_all_three


def _print_results(label, year, variant, races, feats_n, top1_win, top1_top3,
                   pred_top3_hits, pred_top3_at_least_one, pred_top3_all_three):
    print(f"\n{'='*55}")
    print(f"  mode         : {label}")
    print(f"  year         : {year}")
    print(f"  model_variant: {variant}")
    print(f"  races        : {races:,}")
    print(f"  features     : {feats_n}")
    print(f"  top1 win     : {top1_win/races*100:.2f}%  ({top1_win:,}/{races:,})")
    print(f"  top1 top3    : {top1_top3/races*100:.2f}%  ({top1_top3:,}/{races:,})")
    print(f"  avg top3 hits: {pred_top3_hits/races:.3f} horses/race")
    print(f"  >=1 top3 hit : {pred_top3_at_least_one/races*100:.2f}%")
    print(f"  all-3 top3   : {pred_top3_all_three/races*100:.2f}%")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--model-variant", default=DEFAULT_MODEL_VARIANT, choices=MODEL_VARIANTS.keys())
    parser.add_argument(
        "--full-pipeline", action=argparse.BooleanOptionalAction, default=None,
        help="訓練時と同じフルパイプラインで特徴量を計算する（lambdarank モデルはデフォルト True）",
    )
    args = parser.parse_args()

    predictor = KeibaPredictor(model_variant=args.model_variant)

    # lambdarank モデルはデフォルトでフルパイプラインを使う
    use_full = args.full_pipeline
    if use_full is None:
        use_full = (predictor.model_type == "lambdarank")

    print(f"Loading data from {DATA_PKL} ...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if use_full:
        fe = predictor.meta.get("feature_engineering", "")
        print(f"Full-pipeline mode (feature_engineering={fe})", flush=True)

        df = _run_full_pipeline(df, fe)
        if predictor.model_type == "lambdarank":
            df = _add_jh_combo(df)

        year_df = df[df["date"].dt.year == args.year].dropna(subset=["rank"]).copy()
        feats = predictor.feats
        X = year_df.reindex(columns=feats)

        if predictor.model_type == "lambdarank":
            raw = np.mean(
                [m.predict(X, num_iteration=m.best_iteration) for m in predictor.lr_models], axis=0
            )
            year_df = year_df.copy()
            year_df["_raw"] = raw
            def _softmax(g):
                e = np.exp(g["_raw"] - g["_raw"].max()); return e / e.sum()
            year_df["score"] = year_df.groupby("race_id", group_keys=False).apply(_softmax)
            year_df["pred_rank"] = (
                year_df.groupby("race_id")["score"].rank(ascending=False, method="min").astype(int)
            )
            predicted = year_df
        else:
            # binary モデルでもフルパイプライン評価を通す
            p_top3 = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in predictor.top3_models], axis=0)
            p_win  = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in predictor.win_models],  axis=0)
            year_df = year_df.copy()
            year_df["p_top3"] = p_top3
            year_df["p_win"]  = p_win
            year_df["score"]  = year_df["p_win"] * predictor.score_win_weight + year_df["p_top3"] * (1 - predictor.score_win_weight)
            year_df["pred_rank"] = (
                year_df.groupby("race_id")["score"].rank(ascending=False, method="min").astype(int)
            )
            predicted = year_df

        label = "full-pipeline"
    else:
        year_df = df[df["date"].dt.year == args.year].copy()
        predicted = predictor.predict_frame(year_df.reset_index(drop=True))
        predicted = rank_predictions(predicted, "profitmax")
        label = "lookup-table"

    races, top1_win, top1_top3, pred_top3_hits, pred_top3_at_least_one, pred_top3_all_three = _calc_hit_rates(predicted)
    _print_results(label, args.year, args.model_variant, races, len(predictor.feats),
                   top1_win, top1_top3, pred_top3_hits, pred_top3_at_least_one, pred_top3_all_three)


if __name__ == "__main__":
    main()
