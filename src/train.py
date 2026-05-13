"""
予想モデル学習(LightGBM 複数シード アンサンブル + XGBoost軽量補助 + タイム指数回帰)
- メモリ制約(1GB)環境で動くストリーム的処理
"""
import os
import json
import gzip
import pickle
import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PKL = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
os.makedirs(MODEL_DIR, exist_ok=True)


NUMERIC_FEATS = [
    "frame_no", "horse_no", "age", "weight_carry",
    "body_weight", "body_weight_diff",
    "distance", "round_no",
    "horse_runs", "horse_avg_rank", "horse_win_rate", "horse_top3_rate",
    "horse_avg_time_idx", "horse_best_time_idx", "horse_avg_agari",
    "days_since_last",
    "field_size",
    "horse_rank_lag1", "horse_rank_lag2", "horse_rank_lag3",
    "horse_recent_avg_rank3", "horse_recent_top3_rate3",
    "horse_time_idx_lag1", "horse_time_idx_lag2", "horse_time_idx_lag3",
    "horse_agari_lag1", "horse_agari_lag2", "horse_agari_lag3",
    "horse_passing_first_lag1", "horse_passing_first_lag2", "horse_passing_first_lag3",
    "horse_passing_last_lag1", "horse_passing_last_lag2", "horse_passing_last_lag3",
    "horse_passing_first_rate_lag1", "horse_passing_first_rate_lag2", "horse_passing_first_rate_lag3",
    "horse_passing_last_rate_lag1", "horse_passing_last_rate_lag2", "horse_passing_last_rate_lag3",
    "horse_passing_gain_lag1", "horse_passing_gain_lag2", "horse_passing_gain_lag3",
    "horse_front_style", "horse_closing_style",
    "horse_distance_lag1", "horse_distance_lag2", "horse_distance_lag3",
    "horse_distance_diff_lag1", "horse_distance_diff_lag2", "horse_distance_diff_lag3",
    "horse_shorter_than_last", "horse_longer_than_last",
    "closer_short_distance_risk", "front_short_distance_fit",
    "closer_long_distance_fit", "front_long_distance_risk",
    "jockey_runs", "jockey_win_rate", "jockey_top3_rate",
    "trainer_win_rate", "trainer_top3_rate",
    "horse_dist_runs", "horse_dist_win_rate", "horse_dist_top3_rate",
    "horse_surface_runs", "horse_surface_win_rate", "horse_surface_top3_rate",
    "horse_venue_runs", "horse_venue_win_rate", "horse_venue_top3_rate",
    "horse_course_runs", "horse_course_win_rate", "horse_course_top3_rate",
    "jockey_venue_runs", "jockey_venue_win_rate", "jockey_venue_top3_rate",
    "jockey_course_runs", "jockey_course_win_rate", "jockey_course_top3_rate",
    "trainer_venue_runs", "trainer_venue_win_rate", "trainer_venue_top3_rate",
    "trainer_course_runs", "trainer_course_win_rate", "trainer_course_top3_rate",
]
MARKET_FEATS = ["odds", "popularity"]
CATEGORICAL_FEATS = [
    "sex", "surface", "direction", "weather", "going",
    "race_class", "venue", "sire", "broodmare_sire",
]


def load_split():
    print(">>> Loading processed data...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"   rows={len(df):,}, cols={len(df.columns)}", flush=True)

    keep_cols = ["race_id", "rank", "date", "time_index"] + NUMERIC_FEATS + CATEGORICAL_FEATS
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df = df.dropna(subset=["rank"]).reset_index(drop=True)
    df["target_top3"] = (df["rank"] <= 3).astype("int8")
    df["target_win"] = (df["rank"] == 1).astype("int8")
    df["__year"] = df["date"].dt.year

    tr_mask = df["__year"] < 2025
    va_mask = df["__year"] == 2025

    cat_meta = {}
    for c in CATEGORICAL_FEATS:
        if c in df.columns:
            df[c] = df[c].astype("category")
            tr_cats = df.loc[tr_mask, c].cat.categories
            cat_meta[c] = list(tr_cats.astype(str))
            df[c] = pd.Categorical(df[c], categories=tr_cats)

    feats = [c for c in NUMERIC_FEATS + CATEGORICAL_FEATS if c in df.columns]

    tr_df = df.loc[tr_mask].reset_index(drop=True)
    va_df = df.loc[va_mask].reset_index(drop=True)
    del df
    gc.collect()
    return tr_df, va_df, feats, cat_meta


def train_lgb_binary(X_tr, y_tr, X_va, y_va, cat_feats, seed=42, num_leaves=63, num_boost_round=1500):
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": num_leaves,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 50, "verbosity": -1, "seed": seed,
        "num_threads": 2,
    }
    cat_in = [c for c in cat_feats if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_in)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_in, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=num_boost_round,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
    )
    del dtr, dva
    gc.collect()
    return model


def train_lgb_regression(X_tr, y_tr, X_va, y_va, cat_feats, num_boost_round=1500):
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.05, "num_leaves": 63,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 50, "verbosity": -1, "seed": 42,
        "num_threads": 2,
    }
    cat_in = [c for c in cat_feats if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_in)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_in, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=num_boost_round,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
    )
    del dtr, dva
    gc.collect()
    return model


def main():
    tr_df, va_df, feats, cat_meta = load_split()
    print(f"\nFeatures ({len(feats)}) / Train: {len(tr_df):,} / Valid: {len(va_df):,}\n", flush=True)

    X_tr = tr_df[feats]
    X_va = va_df[feats]
    y_tr_top3 = tr_df["target_top3"].values
    y_va_top3 = va_df["target_top3"].values
    y_tr_win = tr_df["target_win"].values
    y_va_win = va_df["target_win"].values
    y_tr_time = tr_df["time_index"].values if "time_index" in tr_df.columns else None
    y_va_time = va_df["time_index"].values if "time_index" in va_df.columns else None
    del tr_df, va_df
    gc.collect()

    metrics = {}

    # ---- top3: 3シードのLGB アンサンブル ----
    print(">>> [1/3] LGB top3 (seed=42, leaves=63)...", flush=True)
    m1 = train_lgb_binary(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=42, num_leaves=63)
    p1 = m1.predict(X_va, num_iteration=m1.best_iteration)
    auc1 = float(roc_auc_score(y_va_top3, p1))
    metrics["lgb_top3_seed42"] = auc1
    m1.save_model(os.path.join(MODEL_DIR, "lgb_top3_s42.txt"))
    print(f"   AUC = {auc1:.4f}", flush=True)
    del m1
    gc.collect()

    print(">>> [2/3] LGB top3 (seed=7, leaves=95)...", flush=True)
    m2 = train_lgb_binary(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=7, num_leaves=95)
    p2 = m2.predict(X_va, num_iteration=m2.best_iteration)
    auc2 = float(roc_auc_score(y_va_top3, p2))
    metrics["lgb_top3_seed7"] = auc2
    m2.save_model(os.path.join(MODEL_DIR, "lgb_top3_s7.txt"))
    print(f"   AUC = {auc2:.4f}", flush=True)
    del m2
    gc.collect()

    print(">>> [3/3] LGB top3 (seed=2024, leaves=47)...", flush=True)
    m3 = train_lgb_binary(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=2024, num_leaves=47)
    p3 = m3.predict(X_va, num_iteration=m3.best_iteration)
    auc3 = float(roc_auc_score(y_va_top3, p3))
    metrics["lgb_top3_seed2024"] = auc3
    m3.save_model(os.path.join(MODEL_DIR, "lgb_top3_s2024.txt"))
    print(f"   AUC = {auc3:.4f}", flush=True)
    del m3
    gc.collect()

    p_ens = (p1 + p2 + p3) / 3
    metrics["ens_top3_auc"] = float(roc_auc_score(y_va_top3, p_ens))
    print(f"   ★ Ensemble TOP3 AUC = {metrics['ens_top3_auc']:.4f}\n", flush=True)
    del p1, p2, p3
    gc.collect()

    # ---- win: 2シードLGB アンサンブル ----
    print(">>> LGB win (seed=42)...", flush=True)
    mw1 = train_lgb_binary(X_tr, y_tr_win, X_va, y_va_win, CATEGORICAL_FEATS, seed=42, num_leaves=63)
    pw1 = mw1.predict(X_va, num_iteration=mw1.best_iteration)
    metrics["lgb_win_seed42"] = float(roc_auc_score(y_va_win, pw1))
    mw1.save_model(os.path.join(MODEL_DIR, "lgb_win_s42.txt"))
    print(f"   AUC = {metrics['lgb_win_seed42']:.4f}", flush=True)
    del mw1
    gc.collect()

    print(">>> LGB win (seed=7)...", flush=True)
    mw2 = train_lgb_binary(X_tr, y_tr_win, X_va, y_va_win, CATEGORICAL_FEATS, seed=7, num_leaves=95)
    pw2 = mw2.predict(X_va, num_iteration=mw2.best_iteration)
    metrics["lgb_win_seed7"] = float(roc_auc_score(y_va_win, pw2))
    mw2.save_model(os.path.join(MODEL_DIR, "lgb_win_s7.txt"))
    print(f"   AUC = {metrics['lgb_win_seed7']:.4f}", flush=True)
    pw_ens = (pw1 + pw2) / 2
    metrics["ens_win_auc"] = float(roc_auc_score(y_va_win, pw_ens))
    print(f"   ★ Ensemble WIN AUC = {metrics['ens_win_auc']:.4f}\n", flush=True)
    del mw2, pw1, pw2
    gc.collect()

    # ---- 補助: タイム指数回帰 ----
    if y_tr_time is not None:
        print(">>> LGB time_index regression...", flush=True)
        m_tr = ~np.isnan(y_tr_time)
        m_va = ~np.isnan(y_va_time)
        if m_tr.sum() > 1000 and m_va.sum() > 100:
            mt = train_lgb_regression(
                X_tr.loc[m_tr], y_tr_time[m_tr],
                X_va.loc[m_va], y_va_time[m_va],
                CATEGORICAL_FEATS,
            )
            mt.save_model(os.path.join(MODEL_DIR, "lgb_time.txt"))
            pt = mt.predict(X_va.loc[m_va], num_iteration=mt.best_iteration)
            rmse = float(np.sqrt(np.mean((pt - y_va_time[m_va]) ** 2)))
            metrics["lgb_time_rmse"] = rmse
            print(f"   RMSE = {rmse:.3f}", flush=True)
            del mt
            gc.collect()

    # ---- メタ ----
    meta = {
        "features": feats,
        "numeric": NUMERIC_FEATS,
        "excluded_market_features": MARKET_FEATS,
        "categorical": CATEGORICAL_FEATS,
        "cat_categories": cat_meta,
        "metrics": metrics,
        "ensemble_top3": ["lgb_top3_s42.txt", "lgb_top3_s7.txt", "lgb_top3_s2024.txt"],
        "ensemble_win": ["lgb_win_s42.txt", "lgb_win_s7.txt"],
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n✓ Saved to", MODEL_DIR, flush=True)
    print("Metrics:", json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
