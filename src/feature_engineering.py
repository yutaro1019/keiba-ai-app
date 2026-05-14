"""
追加特徴量エンジニアリング
- 学習時(train.py)と予測時(predictor.py)の両方から呼ばれる
- 既存の processed_all.pkl.gz の列から派生特徴量を計算
- オッズ・人気は使わない
"""
import numpy as np
import pandas as pd

# within-race 正規化する特徴量 (col, higher_is_better)
WITHIN_RACE_COLS = [
    ("horse_avg_rank",         False),   # 低いほど良い(1位の馬が最良)
    ("horse_recent_avg_rank3", False),
    ("horse_top3_rate",        True),
    ("horse_win_rate",         True),
    ("horse_recent_top3_rate3", True),
    ("jockey_win_rate",        True),
    ("jockey_top3_rate",       True),
    ("trainer_top3_rate",      True),
    ("horse_agari_lag1",       False),   # 上り低い=速い=良い
    ("horse_dist_top3_rate",   True),
    ("horse_dist_win_rate",    True),
    ("horse_rank_pct_lag1",    False),   # 0=1着相当、1=最下位相当
]


def _within_race_pct_groupby(df: pd.DataFrame, col: str, higher_is_better: bool,
                              race_col: str = "race_id") -> pd.Series:
    """各馬が同じレース内で何パーセンタイルか計算。1.0=最良、0.0=最悪。"""
    asc = not higher_is_better
    ranked = df.groupby(race_col, observed=True, sort=False)[col].rank(
        method="average", ascending=asc, na_option="keep"
    )
    n_valid = df.groupby(race_col, observed=True, sort=False)[col].transform(
        lambda x: float(x.notna().sum())
    )
    pct = 1.0 - (ranked - 1) / (n_valid - 1).clip(lower=1)
    return pct.where(df[col].notna()).astype("float32")


def _within_race_pct_single(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """単一レース(全行が同一レース)のパーセンタイル計算。"""
    valid = s.notna()
    n = int(valid.sum())
    if n <= 1:
        return pd.Series(np.nan, index=s.index, dtype="float32")
    asc = not higher_is_better
    ranked = s.rank(method="average", ascending=asc, na_option="keep")
    pct = (1.0 - (ranked - 1) / (n - 1)).where(valid)
    return pct.astype("float32")


def add_model_features(df: pd.DataFrame, race_col: str = "race_id") -> pd.DataFrame:
    """
    既存列から派生特徴量を追加して返す。
    - 学習時: race_col="race_id" でグループ化してレース内正規化
    - 予測時: 1レース分の行が渡される(全行が同一レース)
    """
    df = df.copy()

    # 1. 前走着順の相対順位 (0=1着、1=最下位)
    for lag in (1, 2, 3):
        r_col = f"horse_rank_lag{lag}"
        s_col = f"horse_field_size_lag{lag}"
        if r_col in df.columns and s_col in df.columns:
            denom = (df[s_col] - 1).replace(0, np.nan)
            df[f"horse_rank_pct_lag{lag}"] = (
                (df[r_col] - 1) / denom
            ).clip(0, 1).astype("float32")

    # 2. 上り傾向
    if "horse_agari_lag1" in df.columns and "horse_agari_lag2" in df.columns:
        df["horse_agari_diff_12"] = (
            df["horse_agari_lag1"] - df["horse_agari_lag2"]
        ).astype("float32")
    if "horse_agari_lag1" in df.columns and "horse_avg_agari" in df.columns:
        df["horse_agari_vs_avg"] = (
            df["horse_agari_lag1"] - df["horse_avg_agari"]
        ).astype("float32")

    # 3. 着順トレンド (正=改善、負=悪化)
    if "horse_rank_lag1" in df.columns and "horse_rank_lag2" in df.columns:
        df["horse_rank_trend"] = (
            df["horse_rank_lag2"] - df["horse_rank_lag1"]
        ).astype("float32")

    # 4. 斤量負担率
    if "weight_carry" in df.columns and "body_weight" in df.columns:
        bw = df["body_weight"].replace(0, np.nan)
        df["weight_burden_ratio"] = (df["weight_carry"] / bw).astype("float32")

    # 5. レース内相対パーセンタイル
    has_multi_race = (
        race_col in df.columns
        and df[race_col].nunique(dropna=False) > 1
    )

    for col, higher_is_better in WITHIN_RACE_COLS:
        if col not in df.columns:
            continue
        new_col = f"{col}_vs_field"
        if has_multi_race:
            df[new_col] = _within_race_pct_groupby(df, col, higher_is_better, race_col)
        else:
            df[new_col] = _within_race_pct_single(df[col], higher_is_better)

    return df


# train.py の NUMERIC_FEATS に追記する列名リスト
NEW_NUMERIC_FEATS = [
    # 前走の出走頭数(データ内にあるが未使用)
    "horse_field_size_lag1",
    "horse_field_size_lag2",
    "horse_field_size_lag3",
    # 相対順位
    "horse_rank_pct_lag1",
    "horse_rank_pct_lag2",
    "horse_rank_pct_lag3",
    # 上り傾向
    "horse_agari_diff_12",
    "horse_agari_vs_avg",
    # 着順トレンド
    "horse_rank_trend",
    # 斤量負担率
    "weight_burden_ratio",
    # レース内相対パーセンタイル
    "horse_avg_rank_vs_field",
    "horse_recent_avg_rank3_vs_field",
    "horse_top3_rate_vs_field",
    "horse_win_rate_vs_field",
    "horse_recent_top3_rate3_vs_field",
    "jockey_win_rate_vs_field",
    "jockey_top3_rate_vs_field",
    "trainer_top3_rate_vs_field",
    "horse_agari_lag1_vs_field",
    "horse_dist_top3_rate_vs_field",
    "horse_dist_win_rate_vs_field",
    "horse_rank_pct_lag1_vs_field",
]
