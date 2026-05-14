"""
no_market_v5 モデル学習スクリプト
v4 からの主な改善:
1. ラップタイム特徴量（race_front_pace / race_back_pace / race_pace_diff）
2. 馬ごとの過去レースのラップペース（lag 1/2/3）
3. フィールド期待ペース（同レース他馬の前走前半ペース平均）
4. 馬と対戦フィールドのペース差（速い馬か遅い馬か）

事前準備: python add_lap_features.py でpickleにラップ列を追加しておく
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
from feature_engineering import add_model_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
OUT_DIR  = os.path.join(BASE_DIR, "models", "no_market_v5")
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================
# v4の特徴量エンジニアリング関数（そのまま流用）
# ================================================================

def add_sire_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_win"]  = (df["rank"] == 1).astype(float)
    df["_top3"] = (df["rank"] <= 3).astype(float)
    if "distance" in df.columns:
        df["_dist_bin"] = pd.cut(
            df["distance"], bins=[0, 1400, 1700, 2100, 9999], labels=False
        ).astype("float32")
    for entity in ["sire", "broodmare_sire"]:
        if entity not in df.columns:
            continue
        grp = df.groupby(entity, sort=False, observed=True)
        runs = grp.cumcount()
        wins = grp["_win"].cumsum()  - df["_win"]
        top3 = grp["_top3"].cumsum() - df["_top3"]
        df[f"{entity}_hist_runs"]      = runs.astype("float32")
        df[f"{entity}_hist_win_rate"]  = (wins / runs.replace(0, np.nan)).astype("float32")
        df[f"{entity}_hist_top3_rate"] = (top3 / runs.replace(0, np.nan)).astype("float32")
        if "_dist_bin" in df.columns:
            grp2 = df.groupby([entity, "_dist_bin"], sort=False, observed=True)
            runs2 = grp2.cumcount()
            wins2 = grp2["_win"].cumsum() - df["_win"]
            df[f"{entity}_dist_win_rate"] = (wins2 / runs2.replace(0, np.nan)).astype("float32")
    if "race_class" in df.columns:
        grp_rc = df.groupby("race_class", sort=False, observed=True)
        rc_runs = grp_rc.cumcount()
        rc_wins = grp_rc["_win"].cumsum()  - df["_win"]
        rc_top3 = grp_rc["_top3"].cumsum() - df["_top3"]
        df["class_hist_win_rate"]  = (rc_wins / rc_runs.replace(0, np.nan)).astype("float32")
        df["class_hist_top3_rate"] = (rc_top3 / rc_runs.replace(0, np.nan)).astype("float32")
        df["class_hist_runs"]      = rc_runs.astype("float32")
    df.drop(columns=["_win", "_top3"] + (["_dist_bin"] if "_dist_bin" in df.columns else []),
            inplace=True)
    return df


def add_horse_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_rank_f"] = df["rank"].astype(float)
    df["_top3_f"] = (df["rank"] <= 3).astype(float)
    df["_win_f"]  = (df["rank"] == 1).astype(float)
    grp = df.groupby("horse_id", sort=False)
    for window in [5, 10]:
        df[f"horse_recent{window}_avg_rank"]   = grp["_rank_f"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()).astype("float32")
        df[f"horse_recent{window}_top3_rate"]  = grp["_top3_f"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()).astype("float32")
        df[f"horse_recent{window}_win_rate"]   = grp["_win_f"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()).astype("float32")
    df.drop(columns=["_rank_f", "_top3_f", "_win_f"], inplace=True)
    return df


def add_jockey_trainer_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_win_f"]  = (df["rank"] == 1).astype(float)
    df["_top3_f"] = (df["rank"] <= 3).astype(float)
    for entity in ["jockey_id", "trainer_id"]:
        if entity not in df.columns:
            continue
        prefix = "jockey" if "jockey" in entity else "trainer"
        grp = df.groupby(entity, sort=False)
        for window in [20, 50]:
            df[f"{prefix}_recent{window}_win_rate"]  = grp["_win_f"].transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean()).astype("float32")
            df[f"{prefix}_recent{window}_top3_rate"] = grp["_top3_f"].transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean()).astype("float32")
    df.drop(columns=["_win_f", "_top3_f"], inplace=True)
    return df


def add_field_strength(df: pd.DataFrame) -> pd.DataFrame:
    race_col = "race_id"
    if race_col not in df.columns:
        return df
    for col, new_col in [
        ("jockey_win_rate",   "field_avg_jockey_win_rate"),
        ("horse_avg_rank",    "field_avg_horse_rank"),
        ("horse_top3_rate",   "field_avg_horse_top3_rate"),
        ("trainer_top3_rate", "field_avg_trainer_top3_rate"),
    ]:
        if col not in df.columns:
            continue
        race_sum = df.groupby(race_col, sort=False)[col].transform("sum")
        race_cnt = df.groupby(race_col, sort=False)[col].transform("count")
        df[new_col] = ((race_sum - df[col]) / (race_cnt - 1).replace(0, np.nan)).astype("float32")
    return df


def add_weight_trend(df: pd.DataFrame) -> pd.DataFrame:
    if "body_weight" not in df.columns or "horse_id" not in df.columns:
        return df
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    grp = df.groupby("horse_id", sort=False)
    df["horse_bw_lag3"]    = grp["body_weight"].transform(lambda x: x.shift(3)).astype("float32")
    df["horse_bw_trend_3"] = (df["body_weight"] - df["horse_bw_lag3"]).astype("float32")
    df.drop(columns=["horse_bw_lag3"], inplace=True)
    return df


# ================================================================
# v5新規: ラップタイム特徴量
# ================================================================

def add_lap_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとの過去ラップペース（lag 1/2/3）を計算。
    race_front_pace / race_back_pace / race_pace_diff が pickle に追加済みであること。
    """
    lap_cols = ["race_front_pace", "race_back_pace", "race_pace_diff"]
    missing = [c for c in lap_cols if c not in df.columns]
    if missing:
        print(f"  [WARNING] ラップ列なし: {missing}  →  add_lap_features.pyを先に実行してください")
        return df

    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    grp = df.groupby("horse_id", sort=False)
    for lag in [1, 2, 3]:
        for src in lap_cols:
            tag = src.replace("race_", "")  # front_pace / back_pace / pace_diff
            new_col = f"horse_prev_{tag}{lag}"
            df[new_col] = grp[src].transform(lambda x, l=lag: x.shift(l)).astype("float32")
    return df


def add_field_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    同レース内の他馬の前走ペースの平均（自分を除く）。
    馬が他馬より速いペースを経験してきたかを示す。
    """
    race_col = "race_id"
    if race_col not in df.columns:
        return df

    src_cols = ["horse_prev_front_pace1", "horse_prev_back_pace1", "horse_prev_pace_diff1"]
    field_names = ["field_avg_front_pace", "field_avg_back_pace", "field_avg_pace_diff"]

    for src, fname in zip(src_cols, field_names):
        if src not in df.columns:
            continue
        race_sum = df.groupby(race_col, sort=False)[src].transform("sum")
        race_cnt = df.groupby(race_col, sort=False)[src].transform("count")
        df[fname] = ((race_sum - df[src]) / (race_cnt - 1).replace(0, np.nan)).astype("float32")
        df[f"{src}_vs_field"] = (df[src] - df[fname]).astype("float32")

    return df


# ================================================================
# 特徴量リスト
# ================================================================

BASE_NUMERIC = [
    "horse_rank_lag1", "field_size",
    "jockey_top3_rate", "horse_rank_pct_lag1", "horse_recent_avg_rank3",
    "days_since_last", "jockey_venue_top3_rate", "body_weight",
    "horse_rank_pct_lag2", "jockey_win_rate", "weight_burden_ratio",
    "horse_avg_rank", "horse_front_style", "horse_avg_agari", "age",
    "trainer_win_rate", "trainer_top3_rate",
    "horse_passing_first_rate_lag1", "horse_distance_diff_lag1",
    "horse_rank_lag2", "horse_passing_last_rate_lag1",
    "jockey_venue_win_rate", "round_no",
    "trainer_venue_win_rate", "trainer_venue_top3_rate",
    "horse_surface_top3_rate", "horse_surface_win_rate",
    "horse_agari_lag2", "horse_agari_diff_12", "horse_agari_vs_avg",
    "jockey_runs", "horse_passing_last_rate_lag2", "jockey_course_top3_rate",
    "jh_top3_rate", "th_runs",
]

NEW_SIRE_FEATS = [
    "sire_hist_runs", "sire_hist_win_rate", "sire_hist_top3_rate", "sire_dist_win_rate",
    "broodmare_sire_hist_runs", "broodmare_sire_hist_win_rate", "broodmare_sire_hist_top3_rate",
    "broodmare_sire_dist_win_rate",
    "class_hist_win_rate", "class_hist_top3_rate", "class_hist_runs",
]

NEW_ROLLING_FEATS = [
    "horse_recent5_avg_rank", "horse_recent10_avg_rank",
    "horse_recent5_top3_rate", "horse_recent10_top3_rate",
    "horse_recent5_win_rate",
    "jockey_recent20_win_rate", "jockey_recent50_win_rate",
    "jockey_recent20_top3_rate",
    "trainer_recent20_win_rate", "trainer_recent20_top3_rate",
    "horse_bw_trend_3",
]

NEW_FIELD_FEATS = [
    "field_avg_jockey_win_rate", "field_avg_horse_rank",
    "field_avg_horse_top3_rate", "field_avg_trainer_top3_rate",
]

ENGR_FEATS = [
    "horse_avg_rank_vs_field", "horse_recent_avg_rank3_vs_field",
    "jockey_top3_rate_vs_field", "horse_rank_pct_lag1_vs_field",
    "horse_top3_rate_vs_field", "horse_recent_top3_rate3_vs_field",
    "jockey_win_rate_vs_field", "horse_agari_lag1_vs_field",
    "trainer_top3_rate_vs_field", "horse_win_rate_vs_field",
    "horse_dist_top3_rate_vs_field",
]

# v5新規ラップ特徴量
LAP_RACE_FEATS = [
    "race_front_pace", "race_back_pace", "race_pace_diff",
]

LAP_HORSE_FEATS = [
    f"horse_prev_{p}{lag}"
    for lag in [1, 2, 3]
    for p in ["front_pace", "back_pace", "pace_diff"]
]

LAP_FIELD_FEATS = [
    "field_avg_front_pace", "field_avg_back_pace", "field_avg_pace_diff",
    "horse_prev_front_pace1_vs_field",
    "horse_prev_back_pace1_vs_field",
    "horse_prev_pace_diff1_vs_field",
]


def load_and_prepare():
    print(">>> Loading processed data...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"   rows={len(df):,}, cols={len(df.columns)}")

    lap_present = [c for c in ["race_front_pace", "race_back_pace", "race_pace_diff"] if c in df.columns]
    print(f"   ラップ列: {lap_present}")

    df = df.dropna(subset=["rank"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["target_top3"] = (df["rank"] <= 3).astype("int8")
    df["target_win"]  = (df["rank"] == 1).astype("int8")
    df["__year"]      = df["date"].dt.year

    print(">>> Adding sire/class features...", flush=True)
    df = add_sire_features(df); gc.collect()

    print(">>> Adding horse rolling features...", flush=True)
    df = add_horse_rolling_features(df); gc.collect()

    print(">>> Adding jockey/trainer rolling features...", flush=True)
    df = add_jockey_trainer_rolling(df); gc.collect()

    print(">>> Adding field strength features...", flush=True)
    df = add_field_strength(df); gc.collect()

    print(">>> Adding weight trend features...", flush=True)
    df = add_weight_trend(df); gc.collect()

    print(">>> Adding lap rolling features (v5)...", flush=True)
    df = add_lap_rolling_features(df); gc.collect()

    print(">>> Adding field pace features (v5)...", flush=True)
    df = add_field_pace_features(df); gc.collect()

    print(">>> Adding within-race features (vs_field)...", flush=True)
    df = add_model_features(df, race_col="race_id"); gc.collect()

    print(">>> Adding combo features...", flush=True)
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_win_tmp"]  = df["target_win"].astype(float)
    df["_top3_tmp"] = df["target_top3"].astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp_jh = df.groupby(["jockey_id", "horse_id"], sort=False)
        jh_runs = grp_jh.cumcount()
        df["jh_top3_cum"] = grp_jh["_top3_tmp"].cumsum() - df["_top3_tmp"]
        df["jh_runs"]     = jh_runs.astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    if "trainer_id" in df.columns and "horse_id" in df.columns:
        grp_th = df.groupby(["trainer_id", "horse_id"], sort=False)
        df["th_runs"] = grp_th.cumcount().astype("float32")
    df.drop(columns=["_win_tmp", "_top3_tmp"], inplace=True)

    tr_mask = df["__year"] < 2025
    va_mask = df["__year"] == 2025

    all_feats = list(dict.fromkeys(
        BASE_NUMERIC + NEW_SIRE_FEATS + NEW_ROLLING_FEATS + NEW_FIELD_FEATS +
        ENGR_FEATS + LAP_RACE_FEATS + LAP_HORSE_FEATS + LAP_FIELD_FEATS
    ))
    feats = [c for c in all_feats if c in df.columns]

    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    df["target_rank_label"] = np.where(df["rank"] == 1, 2,
                               np.where(df["rank"] <= 3, 1, 0)).astype("int8")

    tr_df = df.loc[tr_mask].sort_values(["date", "race_id"]).reset_index(drop=True)
    va_df = df.loc[va_mask].sort_values(["date", "race_id"]).reset_index(drop=True)
    del df; gc.collect()
    print(f"   Final feature count: {len(feats)}  (v5 lap feats: {sum(1 for f in feats if 'pace' in f or 'front' in f or 'back' in f)})")
    return tr_df, va_df, feats


def train_lgb(X_tr, y_tr, X_va, y_va, seed=42, num_leaves=63, boost_rounds=2000):
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": num_leaves,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 80,
        "lambda_l1": 0.1, "lambda_l2": 0.5,
        "verbosity": -1, "seed": seed,
        "num_threads": 4,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=boost_rounds,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)],
    )
    del dtr, dva; gc.collect()
    return model


def make_groups(df):
    return df.groupby("race_id", sort=False)["race_id"].count().tolist()


def eval_hit_rates(scores, groups, labels):
    idx = 0
    win_hits = top3_hits = pred3_hits = pred3_count = n_races = 0
    for gsize in groups:
        chunk_s = scores[idx:idx+gsize]
        chunk_l = labels[idx:idx+gsize]
        real_top3 = (chunk_l >= 1)
        best = np.argmax(chunk_s)
        win_hits  += int(chunk_l[best] == 2)
        top3_hits += int(real_top3[best])
        top3_idx = np.argsort(chunk_s)[-3:]
        pred3_hits  += int(real_top3[top3_idx].sum())
        pred3_count += 3
        n_races += 1
        idx += gsize
    return {
        "win_rate":  win_hits / n_races,
        "top3_rate": top3_hits / n_races,
        "pred3_precision": pred3_hits / pred3_count,
        "n_races": n_races,
    }


def main():
    tr_df, va_df, feats = load_and_prepare()
    print(f"\nFeatures: {len(feats)}  Train: {len(tr_df):,}  Val: {len(va_df):,}\n", flush=True)

    X_tr = tr_df[feats]
    X_va = va_df[feats]
    y_tr_top3 = tr_df["target_top3"].values
    y_va_top3 = va_df["target_top3"].values
    y_tr_win  = tr_df["target_win"].values
    y_va_win  = va_df["target_win"].values
    y_va_rank = va_df["target_rank_label"].values
    g_va = make_groups(va_df)
    metrics = {}

    print(">>> [1/5] LGB top3 seed=42 leaves=63 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, seed=42, num_leaves=63)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s42"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s42.txt"))
    print(f"   AUC={metrics['lgb_top3_s42']:.4f}", flush=True)
    p1 = p; del m; gc.collect()

    print(">>> [2/5] LGB top3 seed=7 leaves=95 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, seed=7, num_leaves=95)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s7"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s7.txt"))
    print(f"   AUC={metrics['lgb_top3_s7']:.4f}", flush=True)
    p2 = p; del m; gc.collect()

    print(">>> [3/5] LGB top3 seed=2024 leaves=47 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, seed=2024, num_leaves=47)
    p = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_top3_s2024"] = float(roc_auc_score(y_va_top3, p))
    m.save_model(os.path.join(OUT_DIR, "lgb_top3_s2024.txt"))
    print(f"   AUC={metrics['lgb_top3_s2024']:.4f}", flush=True)
    p3 = p; del m; gc.collect()

    ens_top3 = (p1 + p2 + p3) / 3
    metrics["ens_top3_auc"] = float(roc_auc_score(y_va_top3, ens_top3))
    print(f"   Ensemble TOP3 AUC={metrics['ens_top3_auc']:.4f}\n", flush=True)
    del p1, p2, p3; gc.collect()

    print(">>> [4/5] LGB win seed=42 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, seed=42, num_leaves=63)
    pw1 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s42"] = float(roc_auc_score(y_va_win, pw1))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s42.txt"))
    print(f"   AUC={metrics['lgb_win_s42']:.4f}", flush=True)
    del m; gc.collect()

    print(">>> [5/5] LGB win seed=7 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, seed=7, num_leaves=95)
    pw2 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s7"] = float(roc_auc_score(y_va_win, pw2))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s7.txt"))
    ens_win = (pw1 + pw2) / 2
    metrics["ens_win_auc"] = float(roc_auc_score(y_va_win, ens_win))
    print(f"   AUC={metrics['lgb_win_s7']:.4f}  Ensemble WIN AUC={metrics['ens_win_auc']:.4f}\n", flush=True)
    del m; gc.collect()

    score = 0.6 * ens_win + 0.4 * ens_top3
    hit = eval_hit_rates(score, g_va, y_va_rank)
    metrics.update({f"v5_{k}": v for k, v in hit.items()})
    print(f">>> v5 Hit rates:")
    print(f"   1位的中率: {hit['win_rate']*100:.1f}%  複勝率: {hit['top3_rate']*100:.1f}%  予想3頭精度: {hit['pred3_precision']*100:.1f}%  ({hit['n_races']}R)", flush=True)

    m_check = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_win_s42.txt"))
    fi = pd.Series(m_check.feature_importance(importance_type="gain"), index=feats)
    fi = fi / fi.sum()
    print("\n>>> Top 20 feature importance:")
    v5_lap = set(LAP_RACE_FEATS + LAP_HORSE_FEATS + LAP_FIELD_FEATS)
    for f, v in fi.sort_values(ascending=False).head(20).items():
        tag = " [LAP]" if f in v5_lap else ""
        print(f"   {f:<50s} {v:.4f}{tag}")

    meta = {
        "features": feats,
        "categorical": [],
        "cat_categories": {},
        "metrics": metrics,
        "ensemble_top3": ["lgb_top3_s42.txt", "lgb_top3_s7.txt", "lgb_top3_s2024.txt"],
        "ensemble_win": ["lgb_win_s42.txt", "lgb_win_s7.txt"],
        "rank_model": None,
        "rank_blend_weight": 0.0,
        "score_win_weight": 0.6,
        "feature_engineering": "v5",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUT_DIR}")

    print("\n>>> Regenerating lookup tables for v5...", flush=True)
    from no_market_v4_lookups import regenerate_v4_lookups
    lmeta = regenerate_v4_lookups(model_dir=OUT_DIR)
    for name, cnt in lmeta.get("lookup_counts", {}).items():
        print(f"   {name}: {cnt:,}")
    print("Lookup tables done.", flush=True)


if __name__ == "__main__":
    main()
