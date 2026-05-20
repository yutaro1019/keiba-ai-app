"""
no_market_v3 モデル学習スクリプト
変更点 (vs v2):
1. 特徴量選択: 重要度 >= 0.0015 の47特徴量のみ使用（v2の112→47）
2. 新コンボ特徴量: 騎手×馬・調教師×馬の組み合わせ実績
3. 新距離カテゴリ特徴量: スプリント/マイル/中距離/長距離別の通算成績
4. カテゴリ特徴量: race_class, sire, broodmare_sire のみ（weather/going等は除外）
5. LambdaRank再挑戦（特徴量が少ないのでノイズが減った状態で試す）
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
OUT_DIR = os.path.join(BASE_DIR, "models", "no_market_v3")
os.makedirs(OUT_DIR, exist_ok=True)

# 重要度 >= 0.0015 で選択した特徴量 (v2の112→47)
SELECTED_NUMERIC_FEATS = [
    "horse_rank_lag1",
    "field_size",
    "jockey_top3_rate",
    "horse_rank_pct_lag1",
    "horse_recent_avg_rank3",
    "days_since_last",
    "jockey_venue_top3_rate",
    "body_weight",
    "horse_rank_pct_lag2",
    "jockey_win_rate",
    "weight_burden_ratio",
    "horse_avg_rank",
    "horse_front_style",
    "horse_avg_agari",
    "age",
    "trainer_win_rate",
    "trainer_top3_rate",
    "horse_passing_first_rate_lag1",
    "horse_distance_diff_lag1",
    "horse_rank_lag2",
    "horse_passing_last_rate_lag1",
    "jockey_venue_win_rate",
    "round_no",
    "trainer_venue_win_rate",
    "trainer_venue_top3_rate",
    "horse_surface_top3_rate",
    "horse_surface_win_rate",
    "horse_agari_lag2",
    "horse_agari_diff_12",
    "horse_agari_vs_avg",
    "jockey_runs",
    "horse_passing_last_rate_lag2",
    "jockey_course_top3_rate",
]

# feature_engineering.py が生成するvs_field特徴量から選択
SELECTED_ENGR_FEATS = [
    "horse_avg_rank_vs_field",
    "horse_recent_avg_rank3_vs_field",
    "jockey_top3_rate_vs_field",
    "horse_rank_pct_lag1_vs_field",
    "horse_top3_rate_vs_field",
    "horse_recent_top3_rate3_vs_field",
    "jockey_win_rate_vs_field",
    "horse_agari_lag1_vs_field",
    "trainer_top3_rate_vs_field",
    "horse_win_rate_vs_field",
    "horse_dist_top3_rate_vs_field",
    "horse_rank_pct_lag1",  # already above but keep for dedup
    "horse_rank_pct_lag2",  # already above
]

# カテゴリ特徴量: 重要度上位3つのみ
CATEGORICAL_FEATS = [
    "race_class",
    "sire",
    "broodmare_sire",
]

# 新コンボ特徴量 (この関数で計算)
COMBO_FEATS = [
    "jh_runs", "jh_win_rate", "jh_top3_rate",
    "th_runs", "th_win_rate", "th_top3_rate",
    "dist_bin_runs", "dist_bin_win_rate", "dist_bin_top3_rate",
]


def add_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手×馬・調教師×馬・距離カテゴリ別の累積実績を追加（データリーク防止）"""
    print("   Computing combo features...", flush=True)
    sort_cols = ["date", "race_id"] + (["horse_no"] if "horse_no" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    win_col = (df["rank"] == 1).astype(float)
    top3_col = (df["rank"] <= 3).astype(float)

    # ---- 騎手×馬 ----
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp_jh = df.groupby(["jockey_id", "horse_id"], sort=False)
        jh_runs = grp_jh.cumcount()                          # 過去レース数
        jh_wins = grp_jh[win_col.name if hasattr(win_col, 'name') else "rank"].transform(
            lambda x: x.values  # placeholder
        )
        # cumsum then subtract current
        df["_win_tmp"] = win_col
        df["_top3_tmp"] = top3_col
        grp_jh2 = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_wins_cum"]  = grp_jh2["_win_tmp"].cumsum()  - df["_win_tmp"]
        df["jh_top3_cum"]  = grp_jh2["_top3_tmp"].cumsum() - df["_top3_tmp"]
        df["jh_runs"]      = jh_runs.astype("float32")
        df["jh_win_rate"]  = (df["jh_wins_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_wins_cum", "jh_top3_cum"], inplace=True)
    else:
        df["jh_runs"] = np.nan
        df["jh_win_rate"] = np.nan
        df["jh_top3_rate"] = np.nan

    # ---- 調教師×馬 ----
    if "trainer_id" in df.columns and "horse_id" in df.columns:
        grp_th = df.groupby(["trainer_id", "horse_id"], sort=False)
        th_runs = grp_th.cumcount()
        df["th_wins_cum"]  = grp_th["_win_tmp"].cumsum()  - df["_win_tmp"]
        df["th_top3_cum"]  = grp_th["_top3_tmp"].cumsum() - df["_top3_tmp"]
        df["th_runs"]      = th_runs.astype("float32")
        df["th_win_rate"]  = (df["th_wins_cum"] / df["th_runs"].replace(0, np.nan)).astype("float32")
        df["th_top3_rate"] = (df["th_top3_cum"] / df["th_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["th_wins_cum", "th_top3_cum"], inplace=True)
    else:
        df["th_runs"] = np.nan
        df["th_win_rate"] = np.nan
        df["th_top3_rate"] = np.nan

    # ---- 距離カテゴリ別 ----
    if "distance" in df.columns and "horse_id" in df.columns:
        bins = [0, 1400, 1700, 2100, 9999]
        df["_dist_bin"] = pd.cut(df["distance"], bins=bins, labels=False)
        grp_db = df.groupby(["horse_id", "_dist_bin"], sort=False)
        db_runs = grp_db.cumcount()
        df["db_wins_cum"]  = grp_db["_win_tmp"].cumsum()  - df["_win_tmp"]
        df["db_top3_cum"]  = grp_db["_top3_tmp"].cumsum() - df["_top3_tmp"]
        df["dist_bin_runs"]      = db_runs.astype("float32")
        df["dist_bin_win_rate"]  = (df["db_wins_cum"] / df["dist_bin_runs"].replace(0, np.nan)).astype("float32")
        df["dist_bin_top3_rate"] = (df["db_top3_cum"] / df["dist_bin_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["db_wins_cum", "db_top3_cum", "_dist_bin"], inplace=True)
    else:
        df["dist_bin_runs"] = np.nan
        df["dist_bin_win_rate"] = np.nan
        df["dist_bin_top3_rate"] = np.nan

    df.drop(columns=["_win_tmp", "_top3_tmp"], inplace=True)
    print(f"   Combo features added.", flush=True)
    return df


def load_and_prepare():
    print(">>> Loading processed data...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"   rows={len(df):,}, cols={len(df.columns)}", flush=True)

    keep_cols = (
        ["race_id", "horse_id", "jockey_id", "trainer_id", "rank", "date", "distance", "field_size"]
        + SELECTED_NUMERIC_FEATS
        + CATEGORICAL_FEATS
        + ["horse_avg_rank", "horse_recent_avg_rank3", "horse_top3_rate", "horse_win_rate",
           "horse_recent_top3_rate3", "jockey_win_rate", "jockey_top3_rate", "trainer_top3_rate",
           "horse_agari_lag1", "horse_dist_top3_rate", "horse_dist_win_rate",
           "horse_rank_pct_lag1", "horse_rank_pct_lag2",
           "horse_rank_lag1", "horse_rank_lag2", "horse_rank_lag3",
           "horse_field_size_lag1", "horse_field_size_lag2", "horse_field_size_lag3",
           "weight_carry", "body_weight", "weight_burden_ratio"]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    df = df[keep_cols].copy()
    df = df.dropna(subset=["rank"]).reset_index(drop=True)
    df["target_top3"] = (df["rank"] <= 3).astype("int8")
    df["target_win"]  = (df["rank"] == 1).astype("int8")
    df["__year"]      = df["date"].dt.year

    print(">>> Adding combo features...", flush=True)
    df = add_combo_features(df)

    print(">>> Adding within-race engineered features...", flush=True)
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

    # 最終的な特徴量リスト（dedup）
    all_feats = list(dict.fromkeys(
        SELECTED_NUMERIC_FEATS
        + SELECTED_ENGR_FEATS
        + COMBO_FEATS
        + CATEGORICAL_FEATS
    ))
    feats = [c for c in all_feats if c in df.columns]
    feats = list(dict.fromkeys(feats))
    print(f"   Final feature count: {len(feats)}", flush=True)

    df["target_rank_label"] = np.where(df["rank"] == 1, 2,
                               np.where(df["rank"] <= 3, 1, 0)).astype("int8")

    tr_df = df.loc[tr_mask].sort_values(["date", "race_id"]).reset_index(drop=True)
    va_df = df.loc[va_mask].sort_values(["date", "race_id"]).reset_index(drop=True)
    del df
    gc.collect()
    return tr_df, va_df, feats, cat_meta


def train_lgb(X_tr, y_tr, X_va, y_va, cat_feats, seed=42, num_leaves=63, boost_rounds=2000):
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03,   # v2より低め: 収束を丁寧に
        "num_leaves": num_leaves,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 80,   # v2(50)より大きく: 過学習防止
        "lambda_l1": 0.1,         # L1正則化
        "lambda_l2": 0.5,         # L2正則化
        "verbosity": -1, "seed": seed,
        "num_threads": 4,
    }
    cat_in = [c for c in cat_feats if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_in)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_in, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=boost_rounds,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)],
    )
    del dtr, dva
    gc.collect()
    return model


def make_groups(df: pd.DataFrame) -> list:
    return df.groupby("race_id", sort=False)["race_id"].count().tolist()


def train_lambdarank(X_tr, y_tr, g_tr, X_va, y_va, g_va, cat_feats,
                     seed=42, num_leaves=63, boost_rounds=1500):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "label_gain": [0, 1, 3],
        "learning_rate": 0.03,
        "num_leaves": num_leaves,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 30,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "verbosity": -1, "seed": seed,
        "num_threads": 4,
    }
    cat_in = [c for c in cat_feats if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, group=g_tr, categorical_feature=cat_in)
    dva = lgb.Dataset(X_va, label=y_va, group=g_va, categorical_feature=cat_in, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=boost_rounds,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(300)],
    )
    del dtr, dva
    gc.collect()
    return model


def eval_hit_rates(scores, groups, labels):
    """top-1的中率・複勝率(top3)・予想top3の実top3的中率を計算"""
    idx = 0
    win_hits = 0
    top3_hits = 0
    pred_top3_real_top3 = 0
    pred_top3_count = 0
    total = 0
    for gsize in groups:
        chunk_s = scores[idx:idx+gsize]
        chunk_l = labels[idx:idx+gsize]
        # 予想1位
        best = np.argmax(chunk_s)
        win_hits += int(chunk_l[best] == 2)
        # 実際の3着以内
        real_top3 = (chunk_l >= 1)
        top3_hits += int(real_top3[best])
        # 予想top3 → 実際top3的中率
        top3_idx = np.argsort(chunk_s)[-3:]
        pred_top3_real_top3 += int(real_top3[top3_idx].any())
        pred_top3_count += 1
        total += 1
        idx += gsize
    return {
        "win_rate": win_hits / total,
        "top3_rate_of_win_pred": top3_hits / total,
        "pred_top3_hit_rate": pred_top3_real_top3 / pred_top3_count,
        "n_races": total,
    }


def main():
    tr_df, va_df, feats, cat_meta = load_and_prepare()
    print(f"\nFeatures: {len(feats)}  Train: {len(tr_df):,}  Val: {len(va_df):,}\n", flush=True)
    print("Feature list:", feats, flush=True)

    X_tr = tr_df[feats]
    X_va = va_df[feats]
    y_tr_top3 = tr_df["target_top3"].values
    y_va_top3 = va_df["target_top3"].values
    y_tr_win  = tr_df["target_win"].values
    y_va_win  = va_df["target_win"].values
    y_tr_rank = tr_df["target_rank_label"].values
    y_va_rank = va_df["target_rank_label"].values
    g_tr = make_groups(tr_df)
    g_va = make_groups(va_df)

    metrics = {}

    print(">>> [1/6] LGB top3 seed=42 leaves=63 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=42, num_leaves=63)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s42"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s42.txt"))
    print(f"   AUC={metrics['lgb_top3_s42']:.4f}", flush=True)
    p1 = p; del m; gc.collect()

    print(">>> [2/6] LGB top3 seed=7 leaves=95 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=7, num_leaves=95)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s7"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s7.txt"))
    print(f"   AUC={metrics['lgb_top3_s7']:.4f}", flush=True)
    p2 = p; del m; gc.collect()

    print(">>> [3/6] LGB top3 seed=2024 leaves=47 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, CATEGORICAL_FEATS, seed=2024, num_leaves=47)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s2024"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s2024.txt"))
    print(f"   AUC={metrics['lgb_top3_s2024']:.4f}", flush=True)
    p3 = p; del m; gc.collect()

    ens_top3 = (p1 + p2 + p3) / 3
    metrics["ens_top3_auc"] = float(roc_auc_score(y_va_top3, ens_top3))
    print(f"   Ensemble TOP3 AUC={metrics['ens_top3_auc']:.4f}\n", flush=True)
    del p1, p2, p3; gc.collect()

    print(">>> [4/6] LGB win seed=42 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, CATEGORICAL_FEATS, seed=42, num_leaves=63)
    pw1 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s42"] = float(roc_auc_score(y_va_win, pw1))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s42.txt"))
    print(f"   AUC={metrics['lgb_win_s42']:.4f}", flush=True)
    del m; gc.collect()

    print(">>> [5/6] LGB win seed=7 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, CATEGORICAL_FEATS, seed=7, num_leaves=95)
    pw2 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s7"] = float(roc_auc_score(y_va_win, pw2))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s7.txt"))
    ens_win = (pw1 + pw2) / 2
    metrics["ens_win_auc"] = float(roc_auc_score(y_va_win, ens_win))
    print(f"   AUC={metrics['lgb_win_s7']:.4f}  Ensemble WIN AUC={metrics['ens_win_auc']:.4f}\n", flush=True)
    del m; gc.collect()

    # 的中率を計算（バイナリモデル）
    score_binary = 0.6 * ens_win + 0.4 * ens_top3
    hit_binary = eval_hit_rates(score_binary, g_va, y_va_rank)
    metrics.update({f"binary_{k}": v for k, v in hit_binary.items()})
    print(f">>> Binary ensemble hit rates:", flush=True)
    print(f"   1位的中率: {hit_binary['win_rate']*100:.1f}%  複勝率: {hit_binary['top3_rate_of_win_pred']*100:.1f}%  予想3頭的中: {hit_binary['pred_top3_hit_rate']*100:.1f}%  ({hit_binary['n_races']}R)", flush=True)

    print(">>> [6/6] LambdaRank (ranking直接最適化) ...", flush=True)
    mr = train_lambdarank(X_tr, y_tr_rank, g_tr, X_va, y_va_rank, g_va, CATEGORICAL_FEATS)
    mr.save_model(os.path.join(OUT_DIR, "lgb_rank.txt"))
    rank_raw = mr.predict(X_va, num_iteration=mr.best_iteration)

    # LambdaRankをブレンドして評価（複数ウェイトで試す）
    best_hit = 0
    best_bw = 0.0
    for bw in [0.0, 0.15, 0.25, 0.35, 0.50]:
        rank_shifted = rank_raw - rank_raw.max()
        rank_prob_all = np.exp(rank_shifted)
        # グループ内softmax正規化
        rank_norm = np.zeros_like(rank_raw)
        idx = 0
        for gsize in g_va:
            chunk = rank_prob_all[idx:idx+gsize]
            rank_norm[idx:idx+gsize] = chunk / chunk.sum()
            idx += gsize
        win_norm = np.zeros_like(ens_win)
        idx = 0
        for gsize in g_va:
            chunk = ens_win[idx:idx+gsize]
            s = chunk.sum()
            win_norm[idx:idx+gsize] = chunk / s if s > 0 else chunk
            idx += gsize
        blended = (1 - bw) * win_norm + bw * rank_norm
        score_blend = 0.6 * blended + 0.4 * ens_top3
        hit_blend = eval_hit_rates(score_blend, g_va, y_va_rank)
        print(f"   bw={bw:.2f}: 1位={hit_blend['win_rate']*100:.1f}%  複勝={hit_blend['top3_rate_of_win_pred']*100:.1f}%  予3={hit_blend['pred_top3_hit_rate']*100:.1f}%", flush=True)
        if hit_blend["win_rate"] > best_hit:
            best_hit = hit_blend["win_rate"]
            best_bw = bw

    print(f"\n   Best rank_blend_weight={best_bw} (win_rate={best_hit*100:.1f}%)", flush=True)
    del mr; gc.collect()

    # LambdaRankのNDCG@1 hit rateも記録
    idx = 0
    ndcg1_hits = 0
    ndcg1_total = 0
    for gsize in g_va:
        chunk_s = rank_raw[idx:idx+gsize]
        chunk_l = y_va_rank[idx:idx+gsize]
        best_pred = np.argmax(chunk_s)
        ndcg1_hits += int(chunk_l[best_pred] == 2)
        ndcg1_total += 1
        idx += gsize
    metrics["lgb_rank_ndcg1_hit"] = ndcg1_hits / ndcg1_total if ndcg1_total else 0
    print(f"   LambdaRank単体 top-1 hit rate={metrics['lgb_rank_ndcg1_hit']*100:.1f}%", flush=True)

    meta = {
        "features": feats,
        "numeric": SELECTED_NUMERIC_FEATS + SELECTED_ENGR_FEATS + COMBO_FEATS,
        "excluded_market_features": ["odds", "popularity"],
        "categorical": CATEGORICAL_FEATS,
        "cat_categories": cat_meta,
        "metrics": metrics,
        "ensemble_top3": ["lgb_top3_s42.txt", "lgb_top3_s7.txt", "lgb_top3_s2024.txt"],
        "ensemble_win": ["lgb_win_s42.txt", "lgb_win_s7.txt"],
        "rank_model": "lgb_rank.txt",
        "rank_blend_weight": best_bw,
        "score_win_weight": 0.6,
        "feature_engineering": "v3",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUT_DIR}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
