"""
no_market_v2 モデル学習スクリプト
- オッズ・人気を除外
- feature_engineering.py の派生特徴量(レース内正規化など)を追加
- 学習済モデルを models/no_market_v2/ に保存
"""
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import add_model_features, NEW_NUMERIC_FEATS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
OUT_DIR = os.path.join(BASE_DIR, "models", "no_market_v2")
os.makedirs(OUT_DIR, exist_ok=True)

NUMERIC_FEATS = [
    "frame_no", "horse_no", "age", "weight_carry",
    "body_weight", "body_weight_diff",
    "distance", "round_no",
    "horse_runs", "horse_avg_rank", "horse_win_rate", "horse_top3_rate",
    "horse_avg_agari",
    "days_since_last",
    "field_size",
    "horse_rank_lag1", "horse_rank_lag2", "horse_rank_lag3",
    "horse_recent_avg_rank3", "horse_recent_top3_rate3",
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
CATEGORICAL_FEATS = [
    "sex", "surface", "direction", "weather", "going",
    "race_class", "venue", "sire", "broodmare_sire",
]


def load_and_prepare():
    print(">>> Loading processed data...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"   rows={len(df):,}, cols={len(df.columns)}", flush=True)

    # race_id を残す(レース内正規化に必要)
    keep_cols = (
        ["race_id", "rank", "date", "field_size"]
        + NUMERIC_FEATS
        + NEW_NUMERIC_FEATS
        + CATEGORICAL_FEATS
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    keep_cols = list(dict.fromkeys(keep_cols))  # 重複除去
    df = df[keep_cols].copy()
    df = df.dropna(subset=["rank"]).reset_index(drop=True)
    df["target_top3"] = (df["rank"] <= 3).astype("int8")
    df["target_win"] = (df["rank"] == 1).astype("int8")
    df["__year"] = df["date"].dt.year

    print(">>> Adding engineered features...", flush=True)
    df = add_model_features(df, race_col="race_id")
    print(f"   cols after engineering: {len(df.columns)}", flush=True)
    gc.collect()

    tr_mask = df["__year"] < 2025
    va_mask = df["__year"] == 2025

    cat_meta = {}
    for c in CATEGORICAL_FEATS:
        if c in df.columns:
            df[c] = df[c].astype("category")
            tr_cats = df.loc[tr_mask, c].cat.categories
            cat_meta[c] = list(tr_cats.astype(str))
            df[c] = pd.Categorical(df[c], categories=tr_cats)

    all_feats = NUMERIC_FEATS + NEW_NUMERIC_FEATS + CATEGORICAL_FEATS
    feats = [c for c in all_feats if c in df.columns]
    feats = list(dict.fromkeys(feats))  # 重複除去

    # LambdaRank用ラベル: winner=2, 2-3着=1, それ以外=0
    df["target_rank_label"] = np.where(df["rank"] == 1, 2,
                               np.where(df["rank"] <= 3, 1, 0)).astype("int8")

    tr_df = df.loc[tr_mask].sort_values(["date", "race_id"]).reset_index(drop=True)
    va_df = df.loc[va_mask].sort_values(["date", "race_id"]).reset_index(drop=True)
    del df
    gc.collect()
    return tr_df, va_df, feats, cat_meta


def train_lgb(X_tr, y_tr, X_va, y_va, cat_feats, seed=42, num_leaves=63, boost_rounds=1500):
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": num_leaves,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 50, "verbosity": -1, "seed": seed,
        "num_threads": 4,
    }
    cat_in = [c for c in cat_feats if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_in)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_in, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=boost_rounds,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
    )
    del dtr, dva
    gc.collect()
    return model


def make_groups(df: pd.DataFrame) -> list:
    """レースごとの馬数リスト(LambdaRank group parameter用)"""
    return df.groupby("race_id", sort=False)["race_id"].count().tolist()


def train_lambdarank(X_tr, y_tr, g_tr, X_va, y_va, g_va, cat_feats,
                     seed=42, num_leaves=63, boost_rounds=1000):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "label_gain": [0, 1, 3],   # label 0,1,2 のゲイン
        "learning_rate": 0.05, "num_leaves": num_leaves,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 20, "verbosity": -1, "seed": seed,
        "num_threads": 4,
    }
    cat_in = [c for c in cat_feats if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, categorical_feature=cat_in)
    dva = lgb.Dataset(X_va, label=y_va, group=g_va, categorical_feature=cat_in, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=boost_rounds,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(200)],
    )
    del dtr, dva
    gc.collect()
    return model


def main():
    tr_df, va_df, feats, cat_meta = load_and_prepare()
    print(f"\nFeatures: {len(feats)}  Train: {len(tr_df):,}  Val: {len(va_df):,}\n", flush=True)

    X_tr = tr_df[feats]
    X_va = va_df[feats]
    y_tr_top3 = tr_df["target_top3"].values
    y_va_top3 = va_df["target_top3"].values
    y_tr_win = tr_df["target_win"].values
    y_va_win = va_df["target_win"].values
    y_tr_rank = tr_df["target_rank_label"].values
    y_va_rank = va_df["target_rank_label"].values
    g_tr = make_groups(tr_df)
    g_va = make_groups(va_df)
    del tr_df, va_df
    gc.collect()

    metrics = {}

    print(">>> [1/5] LGB top3 seed=42 leaves=63 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=42, num_leaves=63)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s42"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s42.txt"))
    print(f"   AUC={metrics['lgb_top3_s42']:.4f}", flush=True)
    p1 = p; del m; gc.collect()

    print(">>> [2/5] LGB top3 seed=7 leaves=95 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=7, num_leaves=95)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s7"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s7.txt"))
    print(f"   AUC={metrics['lgb_top3_s7']:.4f}", flush=True)
    p2 = p; del m; gc.collect()

    print(">>> [3/5] LGB top3 seed=2024 leaves=47 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=2024, num_leaves=47)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s2024"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s2024.txt"))
    print(f"   AUC={metrics['lgb_top3_s2024']:.4f}", flush=True)
    p3 = p; del m; gc.collect()

    ens_auc = float(roc_auc_score(y_va_top3, (p1 + p2 + p3) / 3))
    metrics["ens_top3_auc"] = ens_auc
    print(f"   Ensemble TOP3 AUC={ens_auc:.4f}\n", flush=True)
    del p1, p2, p3; gc.collect()

    print(">>> [4/5] LGB win seed=42 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, CATEGORICAL_FEATS, seed=42, num_leaves=63)
    pw1 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s42"] = float(roc_auc_score(y_va_win, pw1))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s42.txt"))
    print(f"   AUC={metrics['lgb_win_s42']:.4f}", flush=True)
    del m; gc.collect()

    print(">>> [5/5] LGB win seed=7 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, CATEGORICAL_FEATS, seed=7, num_leaves=95)
    pw2 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s7"] = float(roc_auc_score(y_va_win, pw2))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s7.txt"))
    ens_win = float(roc_auc_score(y_va_win, (pw1 + pw2) / 2))
    metrics["ens_win_auc"] = ens_win
    print(f"   AUC={metrics['lgb_win_s7']:.4f}  Ensemble WIN AUC={ens_win:.4f}\n", flush=True)
    del m, pw1, pw2; gc.collect()

    print(">>> [6/6] LambdaRank (ranking直接最適化) ...", flush=True)
    mr = train_lambdarank(X_tr, y_tr_rank, g_tr, X_va, y_va_rank, g_va, CATEGORICAL_FEATS)
    mr.save_model(os.path.join(OUT_DIR, "lgb_rank.txt"))
    rank_score_va = mr.predict(X_va, num_iteration=mr.best_iteration)
    # NDCG@1 を手動計算(TOP1的中率の代理)
    # グループごとに最高スコアの馬が実際のWINNERか確認
    idx = 0
    ndcg1_hits = 0
    ndcg1_total = 0
    for gsize in g_va:
        chunk_scores = rank_score_va[idx:idx+gsize]
        chunk_labels = y_va_rank[idx:idx+gsize]
        best_pred = np.argmax(chunk_scores)
        ndcg1_hits += int(chunk_labels[best_pred] == 2)
        ndcg1_total += 1
        idx += gsize
    metrics["lgb_rank_ndcg1_hit"] = ndcg1_hits / ndcg1_total if ndcg1_total else 0
    print(f"   LambdaRank top-1 hit rate={metrics['lgb_rank_ndcg1_hit']*100:.1f}%  ({ndcg1_hits}/{ndcg1_total})\n", flush=True)
    del mr; gc.collect()

    meta = {
        "features": feats,
        "numeric": NUMERIC_FEATS + NEW_NUMERIC_FEATS,
        "excluded_market_features": ["odds", "popularity"],
        "categorical": CATEGORICAL_FEATS,
        "cat_categories": cat_meta,
        "metrics": metrics,
        "ensemble_top3": ["lgb_top3_s42.txt", "lgb_top3_s7.txt", "lgb_top3_s2024.txt"],
        "ensemble_win": ["lgb_win_s42.txt", "lgb_win_s7.txt"],
        "rank_model": "lgb_rank.txt",
        "rank_blend_weight": 0.35,
        "score_win_weight": 0.6,
        "feature_engineering": "v2",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUT_DIR}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
