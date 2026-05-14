"""
no_market_v7 モデル学習スクリプト
v6 からの追加:
【追加】
  - horse_going_top3_rate   : 馬場状態（良/稍重/重/不良）別の複勝率
  - horse_going_win_rate    : 馬場状態別の勝率
  - horse_going_vs_field    : 馬場適性 vs 同レース他馬
  - horse_dist_cat_top3_rate: 距離カテゴリ（短/中/中長/長）別の複勝率
  - horse_dist_cat_win_rate : 距離カテゴリ別の勝率
  - expected_pace_fit       : 今回の期待ペース差(field_avg_pace_diff) - 馬の前走ペース差
                              → 正=今回の方がスローペース、負=今回の方がハイペース
                              → 後半加速型の馬が今回もスローペースなら有利
"""
import os, sys, json, gzip, pickle, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import add_model_features
from train_no_market_v5 import (
    add_sire_features, add_horse_rolling_features, add_jockey_trainer_rolling,
    add_field_strength, add_weight_trend, add_lap_rolling_features, add_field_pace_features,
)
from train_no_market_v6 import add_v6_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL  = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
OUT_DIR   = os.path.join(BASE_DIR, "models", "no_market_v7")
os.makedirs(OUT_DIR, exist_ok=True)

# 距離カテゴリ
DIST_BINS   = [0, 1400, 1700, 2100, 9999]
DIST_LABELS = [0, 1, 2, 3]  # 短距離/マイル/中距離/長距離


def add_v7_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # ── 馬場状態別成績 ──────────────────────────────────────────
    if "going" in df.columns and "horse_id" in df.columns:
        df["_win"]  = (df["rank"] == 1).astype(float)
        df["_top3"] = (df["rank"] <= 3).astype(float)

        grp = df.groupby(["horse_id", "going"], sort=False, observed=True)
        runs  = grp.cumcount()
        wins  = grp["_win"].cumsum()  - df["_win"]
        top3s = grp["_top3"].cumsum() - df["_top3"]
        df["horse_going_win_rate"]  = (wins  / runs.replace(0, np.nan)).astype("float32")
        df["horse_going_top3_rate"] = (top3s / runs.replace(0, np.nan)).astype("float32")

        df.drop(columns=["_win", "_top3"], inplace=True)

        # vs_field: 馬場適性 vs 同レース他馬平均
        for col, new_col in [
            ("horse_going_top3_rate", "horse_going_top3_rate_vs_field"),
            ("horse_going_win_rate",  "horse_going_win_rate_vs_field"),
        ]:
            race_sum = df.groupby("race_id", sort=False)[col].transform("sum")
            race_cnt = df.groupby("race_id", sort=False)[col].transform("count")
            df[new_col] = ((race_sum - df[col]) / (race_cnt - 1).replace(0, np.nan)).astype("float32")
            # vs_field差分
            df[col + "_vs_field_diff"] = (df[col] - df[new_col]).astype("float32")

    # ── 距離カテゴリ別成績 ────────────────────────────────────────
    if "distance" in df.columns and "horse_id" in df.columns:
        df["_dist_cat"] = pd.cut(
            df["distance"], bins=DIST_BINS, labels=DIST_LABELS
        ).astype("float32")
        df["_win2"]  = (df["rank"] == 1).astype(float)
        df["_top3_2"] = (df["rank"] <= 3).astype(float)

        grp2 = df.groupby(["horse_id", "_dist_cat"], sort=False, observed=True)
        runs2  = grp2.cumcount()
        wins2  = grp2["_win2"].cumsum()  - df["_win2"]
        top3s2 = grp2["_top3_2"].cumsum() - df["_top3_2"]
        df["horse_dist_cat_win_rate"]  = (wins2  / runs2.replace(0, np.nan)).astype("float32")
        df["horse_dist_cat_top3_rate"] = (top3s2 / runs2.replace(0, np.nan)).astype("float32")

        df.drop(columns=["_dist_cat", "_win2", "_top3_2"], inplace=True)

    # ── 展開適性：今回期待ペースと馬の前走ペースのずれ ──────────────
    # field_avg_pace_diff = 同レース他馬の前走ペース差平均（今回レースの期待ペース）
    # horse_prev_pace_diff1 = 馬の前走ペース差
    # 差が正 → 今回の方がスローペース（後半加速型に有利）
    # 差が負 → 今回の方がハイペース（前半型に有利）
    if "field_avg_pace_diff" in df.columns and "horse_prev_pace_diff1" in df.columns:
        df["expected_pace_fit"] = (
            df["field_avg_pace_diff"] - df["horse_prev_pace_diff1"]
        ).astype("float32")

    return df


# ================================================================
# 特徴量リスト（v6 + v7新規）
# ================================================================

BASE_NUMERIC = [
    "horse_rank_lag1", "field_size",
    "jockey_top3_rate", "horse_rank_pct_lag1", "horse_recent_avg_rank3",
    "days_since_last", "jockey_venue_top3_rate",
    "body_weight_diff", "body_weight_diff_abs",
    "horse_rank_pct_lag2", "jockey_win_rate", "weight_burden_ratio",
    "horse_front_style", "horse_closing_style", "horse_avg_agari", "age",
    "trainer_win_rate", "trainer_top3_rate",
    "horse_passing_first_rate_lag1", "horse_distance_diff_lag1",
    "horse_passing_last_rate_lag1",
    "jockey_venue_win_rate",
    "trainer_venue_win_rate", "trainer_venue_top3_rate",
    "horse_surface_top3_rate", "horse_surface_win_rate",
    "horse_agari_lag2", "horse_agari_diff_12", "horse_agari_vs_avg",
    "jockey_runs", "horse_passing_last_rate_lag2", "jockey_course_top3_rate",
    "jh_top3_rate",
    # v6
    "jockey_change", "closing_style_x_dist_diff", "horse_agari_rank_pct_lag1",
    # v7新規
    "horse_going_top3_rate", "horse_going_win_rate",
    "horse_going_top3_rate_vs_field_diff",
    "horse_dist_cat_top3_rate", "horse_dist_cat_win_rate",
    "expected_pace_fit",
]

NEW_SIRE_FEATS = [
    "sire_hist_win_rate", "sire_hist_top3_rate", "sire_dist_win_rate",
    "broodmare_sire_hist_win_rate", "broodmare_sire_hist_top3_rate",
    "broodmare_sire_dist_win_rate",
    "class_hist_win_rate", "class_hist_top3_rate", "class_hist_runs",
]

NEW_ROLLING_FEATS = [
    "horse_recent5_avg_rank", "horse_recent10_avg_rank",
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

LAP_RACE_FEATS = ["race_front_pace", "race_back_pace", "race_pace_diff"]
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

    df = df.dropna(subset=["rank"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["target_top3"] = (df["rank"] <= 3).astype("int8")
    df["target_win"]  = (df["rank"] == 1).astype("int8")
    df["__year"]      = df["date"].dt.year

    steps = [
        ("sire/class",      add_sire_features),
        ("horse rolling",   add_horse_rolling_features),
        ("jockey/trainer",  add_jockey_trainer_rolling),
        ("field strength",  add_field_strength),
        ("weight trend",    add_weight_trend),
        ("lap rolling",     add_lap_rolling_features),
        ("field pace",      add_field_pace_features),
        ("v6 features",     add_v6_features),
        ("v7 features",     add_v7_features),
    ]
    for label, fn in steps:
        print(f">>> Adding {label}...", flush=True)
        df = fn(df); gc.collect()

    print(">>> Adding vs_field features...", flush=True)
    df = add_model_features(df, race_col="race_id"); gc.collect()

    print(">>> Adding combo features...", flush=True)
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_w"] = df["target_win"].astype(float)
    df["_t"] = df["target_top3"].astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"] = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]     = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_w", "_t"], inplace=True)

    all_feats = list(dict.fromkeys(
        BASE_NUMERIC + NEW_SIRE_FEATS + NEW_ROLLING_FEATS + NEW_FIELD_FEATS +
        ENGR_FEATS + LAP_RACE_FEATS + LAP_HORSE_FEATS + LAP_FIELD_FEATS
    ))
    feats = [c for c in all_feats if c in df.columns]
    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    df["target_rank_label"] = np.where(df["rank"] == 1, 2, np.where(df["rank"] <= 3, 1, 0)).astype("int8")
    tr_df = df[df["__year"] < 2025].sort_values(["date", "race_id"]).reset_index(drop=True)
    va_df = df[df["__year"] == 2025].sort_values(["date", "race_id"]).reset_index(drop=True)
    del df; gc.collect()
    print(f"   Features: {len(feats)}  Train: {len(tr_df):,}  Val: {len(va_df):,}")
    return tr_df, va_df, feats


def train_lgb(X_tr, y_tr, X_va, y_va, seed=42, num_leaves=63):
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.03,
        "num_leaves": num_leaves, "feature_fraction": 0.80,
        "bagging_fraction": 0.85, "bagging_freq": 5,
        "min_data_in_leaf": 80, "lambda_l1": 0.1, "lambda_l2": 0.5,
        "verbosity": -1, "seed": seed, "num_threads": 4,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)])
    del dtr, dva; gc.collect()
    return m


def eval_hit_rates(scores, groups, labels):
    idx = win_hits = top3_hits = pred3_hits = pred3_cnt = n = 0
    for g in groups:
        s, l = scores[idx:idx+g], labels[idx:idx+g]
        best = np.argmax(s)
        win_hits  += int(l[best] == 2)
        top3_hits += int(l[best] >= 1)
        top3i = np.argsort(s)[-3:]
        pred3_hits += int((l[top3i] >= 1).sum())
        pred3_cnt += 3; n += 1; idx += g
    return {"win": win_hits/n, "top3": top3_hits/n, "pred3": pred3_hits/pred3_cnt, "n": n}


def main():
    tr_df, va_df, feats = load_and_prepare()
    print(f"\nFeatures: {len(feats)}  Train: {len(tr_df):,}  Val: {len(va_df):,}\n", flush=True)

    X_tr = tr_df[feats]; X_va = va_df[feats]
    y_tr_top3 = tr_df["target_top3"].values; y_va_top3 = va_df["target_top3"].values
    y_tr_win  = tr_df["target_win"].values;  y_va_win  = va_df["target_win"].values
    y_va_rank = va_df["target_rank_label"].values
    g_va = va_df.groupby("race_id", sort=False)["race_id"].count().tolist()
    metrics = {}

    for name, target_tr, target_va, tag in [
        ("[1/5] top3 s42 l63",  y_tr_top3, y_va_top3, "lgb_top3_s42"),
        ("[2/5] top3 s7 l95",   y_tr_top3, y_va_top3, "lgb_top3_s7"),
        ("[3/5] top3 s2024 l47",y_tr_top3, y_va_top3, "lgb_top3_s2024"),
    ]:
        seed = int(tag.split("s")[-1]) if "s" in tag else 42
        leaves = 95 if "l95" in name else (47 if "l47" in name else 63)
        print(f">>> {name} ...", flush=True)
        m = train_lgb(X_tr, target_tr, X_va, target_va, seed=seed, num_leaves=leaves)
        p = m.predict(X_va, num_iteration=m.best_iteration)
        metrics[tag] = float(roc_auc_score(target_va, p))
        m.save_model(os.path.join(OUT_DIR, f"{tag}.txt"))
        print(f"   AUC={metrics[tag]:.4f}", flush=True)
        if tag == "lgb_top3_s42": p1 = p
        elif tag == "lgb_top3_s7": p2 = p
        else: p3 = p
        del m; gc.collect()

    ens_top3 = (p1 + p2 + p3) / 3
    metrics["ens_top3_auc"] = float(roc_auc_score(y_va_top3, ens_top3))
    print(f"   Ensemble TOP3 AUC={metrics['ens_top3_auc']:.4f}\n", flush=True)
    del p1, p2, p3; gc.collect()

    print(">>> [4/5] win s42 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, seed=42)
    pw1 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s42"] = float(roc_auc_score(y_va_win, pw1))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s42.txt"))
    print(f"   AUC={metrics['lgb_win_s42']:.4f}", flush=True)
    del m; gc.collect()

    print(">>> [5/5] win s7 ...", flush=True)
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, seed=7, num_leaves=95)
    pw2 = m.predict(X_va, num_iteration=m.best_iteration)
    metrics["lgb_win_s7"] = float(roc_auc_score(y_va_win, pw2))
    m.save_model(os.path.join(OUT_DIR, "lgb_win_s7.txt"))
    ens_win = (pw1 + pw2) / 2
    metrics["ens_win_auc"] = float(roc_auc_score(y_va_win, ens_win))
    print(f"   AUC={metrics['lgb_win_s7']:.4f}  Ensemble WIN={metrics['ens_win_auc']:.4f}\n", flush=True)
    del m; gc.collect()

    score = 0.6 * ens_win + 0.4 * ens_top3
    hit = eval_hit_rates(score, g_va, y_va_rank)
    metrics.update({f"v7_{k}": v for k, v in hit.items()})

    print("\n" + "=" * 65)
    print(f"              v6(86個)    v7({len(feats)}個)")
    print(f"  1位的中率:    33.5%       {hit['win']*100:.1f}%")
    print(f"  複勝率:       62.6%       {hit['top3']*100:.1f}%")
    print(f"  予想3頭精度:   50.2%       {hit['pred3']*100:.1f}%")
    print(f"  TOP3 AUC:    0.7934      {metrics['ens_top3_auc']:.4f}")
    print(f"  WIN AUC:     0.8252      {metrics['ens_win_auc']:.4f}")
    print("=" * 65)

    # 特徴量重要度
    m_w = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_win_s42.txt"))
    m_t = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_top3_s42.txt"))
    fi = ((pd.Series(m_w.feature_importance("gain"), index=feats) / m_w.feature_importance("gain").sum() +
           pd.Series(m_t.feature_importance("gain"), index=feats) / m_t.feature_importance("gain").sum()) / 2 * 100).round(2)
    fi = fi.sort_values(ascending=False)

    NEW_V7 = {"horse_going_top3_rate", "horse_going_win_rate",
               "horse_going_top3_rate_vs_field_diff",
               "horse_dist_cat_top3_rate", "horse_dist_cat_win_rate",
               "expected_pace_fit"}

    print("\n特徴量重要度:")
    for rank, (feat, val) in enumerate(fi.items(), 1):
        tag = " [NEW]" if feat in NEW_V7 else ""
        print(f"{rank:>3}. {feat:<48}  {val:>5.2f}%{tag}")

    meta = {
        "features": feats, "categorical": [], "cat_categories": {},
        "metrics": metrics,
        "ensemble_top3": ["lgb_top3_s42.txt", "lgb_top3_s7.txt", "lgb_top3_s2024.txt"],
        "ensemble_win":  ["lgb_win_s42.txt", "lgb_win_s7.txt"],
        "rank_model": None, "rank_blend_weight": 0.0,
        "score_win_weight": 0.6, "feature_engineering": "v7",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}")

    print("\n>>> Regenerating lookup tables for v7...", flush=True)
    from no_market_v4_lookups import regenerate_v4_lookups
    lmeta = regenerate_v4_lookups(model_dir=OUT_DIR)
    for name, cnt in lmeta.get("lookup_counts", {}).items():
        print(f"   {name}: {cnt:,}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
