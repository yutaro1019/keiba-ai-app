import gzip
import pickle

import numpy as np
import pandas as pd

from predictor import DATA_PKL


def add_recent_style_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    hg = df["horse_id"]
    df["field_size"] = df.groupby("race_id")["horse_no"].transform("count").astype("float32")

    for lag in (1, 2, 3):
        df[f"horse_rank_lag{lag}"] = df.groupby(hg)["rank"].shift(lag).astype("float32")
        if "time_index" in df.columns:
            df[f"horse_time_idx_lag{lag}"] = df.groupby(hg)["time_index"].shift(lag).astype("float32")
        df[f"horse_agari_lag{lag}"] = df.groupby(hg)["agari"].shift(lag).astype("float32")
        df[f"horse_passing_first_lag{lag}"] = df.groupby(hg)["passing_first"].shift(lag).astype("float32")
        df[f"horse_passing_last_lag{lag}"] = df.groupby(hg)["passing_last"].shift(lag).astype("float32")
        df[f"horse_field_size_lag{lag}"] = df.groupby(hg)["field_size"].shift(lag).astype("float32")
        df[f"horse_distance_lag{lag}"] = df.groupby(hg)["distance"].shift(lag).astype("float32")

        denom = (df[f"horse_field_size_lag{lag}"] - 1).replace(0, np.nan)
        df[f"horse_passing_first_rate_lag{lag}"] = ((df[f"horse_passing_first_lag{lag}"] - 1) / denom).astype("float32")
        df[f"horse_passing_last_rate_lag{lag}"] = ((df[f"horse_passing_last_lag{lag}"] - 1) / denom).astype("float32")
        df[f"horse_passing_gain_lag{lag}"] = (df[f"horse_passing_first_lag{lag}"] - df[f"horse_passing_last_lag{lag}"]).astype("float32")
        df[f"horse_distance_diff_lag{lag}"] = (df["distance"].astype("float32") - df[f"horse_distance_lag{lag}"]).astype("float32")

    rank_lags = ["horse_rank_lag1", "horse_rank_lag2", "horse_rank_lag3"]
    df["horse_recent_avg_rank3"] = df[rank_lags].mean(axis=1).astype("float32")
    df["horse_recent_top3_rate3"] = (df[rank_lags] <= 3).mean(axis=1).astype("float32")
    df["horse_front_style"] = df[[
        "horse_passing_first_rate_lag1",
        "horse_passing_first_rate_lag2",
        "horse_passing_first_rate_lag3",
    ]].mean(axis=1).astype("float32")
    df["horse_closing_style"] = df[[
        "horse_passing_gain_lag1",
        "horse_passing_gain_lag2",
        "horse_passing_gain_lag3",
    ]].mean(axis=1).astype("float32")
    df["horse_shorter_than_last"] = (df["horse_distance_diff_lag1"] < 0).astype("float32")
    df["horse_longer_than_last"] = (df["horse_distance_diff_lag1"] > 0).astype("float32")
    df["closer_short_distance_risk"] = (
        df["horse_closing_style"].clip(lower=0) * (df["distance"] <= 1400).astype("float32")
    ).astype("float32")
    df["front_short_distance_fit"] = (
        (1 - df["horse_front_style"].clip(lower=0, upper=1)) * (df["distance"] <= 1400).astype("float32")
    ).astype("float32")
    return df


def main():
    print(f"Loading {DATA_PKL}")
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    before_cols = len(df.columns)
    df = add_recent_style_features(df)
    with gzip.open(DATA_PKL, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {DATA_PKL}")
    print(f"Columns: {before_cols} -> {len(df.columns)}")


if __name__ == "__main__":
    main()
