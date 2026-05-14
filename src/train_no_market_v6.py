"""
no_market_v6 モデル学習スクリプト
v5 からの主な変更:
【追加】
  - body_weight_diff      : 前走比体重変化（体重より意味がある）
  - body_weight_diff_abs  : 体重変化の絶対値（大きな変化は不調サイン）
  - horse_closing_style   : 差し度（前走3走で何頭抜いてきたか）
  - closing_style_x_dist_diff : 差し度 × 距離変化（1800得意馬が1600走ると不利、等）
  - horse_agari_rank_pct_lag1 : 前走の上がり3F順位パーセンタイル（0=最速, 1=最遅）
  - jockey_change         : 乗り替わりフラグ（前走と騎手が変わったら1）
【削除】(重複・低重要度)
  - body_weight           → body_weight_diff に置き換え
  - horse_avg_rank        → vs_field版が1位なので単体不要
  - horse_recent5_win_rate / horse_recent5_top3_rate / horse_recent10_top3_rate
  - jockey_recent20_win_rate / trainer_recent20_win_rate  → top3_rateで代替
  - sire_hist_runs / broodmare_sire_hist_runs             → 学習量の代理変数
  - th_runs                                               → コンビ出走数は低寄与
  - horse_rank_lag2       → horse_rank_pct_lag2 と重複
  - round_no              → 重要度低
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL  = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
OUT_DIR   = os.path.join(BASE_DIR, "models", "no_market_v6")
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================
# v6 新規特徴量エンジニアリング
# ================================================================

def add_v6_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # 体重変化の絶対値
    if "body_weight_diff" in df.columns:
        df["body_weight_diff_abs"] = df["body_weight_diff"].abs().astype("float32")

    # 乗り替わりフラグ（前走と今走で騎手が異なる=1, 初出走/データなし=NaN）
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        df["_jockey_lag1"] = df.groupby("horse_id", sort=False)["jockey_id"].shift(1)
        df["jockey_change"] = np.where(
            df["_jockey_lag1"].isna(), np.nan,
            (df["jockey_id"] != df["_jockey_lag1"]).astype(float)
        )
        df["jockey_change"] = df["jockey_change"].astype("float32")
        df.drop(columns=["_jockey_lag1"], inplace=True)

    # 脚質 × 距離変化（差し馬が距離短縮=不利、逃げ馬が距離延長=不利）
    if "horse_closing_style" in df.columns and "horse_distance_diff_lag1" in df.columns:
        df["closing_style_x_dist_diff"] = (
            df["horse_closing_style"] * df["horse_distance_diff_lag1"]
        ).astype("float32")

    # 前走の上がり3F 順位パーセンタイル（0=最速=強い、1=最遅）
    if "agari" in df.columns and "field_size" in df.columns:
        df["_agari_rank"] = df.groupby("race_id", sort=False)["agari"].rank(
            ascending=True, method="min", na_option="keep"
        )
        denom = (df["field_size"] - 1).replace(0, np.nan)
        df["_agari_rank_pct"] = ((df["_agari_rank"] - 1) / denom).clip(0, 1).astype("float32")
        df["horse_agari_rank_pct_lag1"] = (
            df.groupby("horse_id", sort=False)["_agari_rank_pct"].shift(1).astype("float32")
        )
        df.drop(columns=["_agari_rank", "_agari_rank_pct"], inplace=True)

    return df


# ================================================================
# 特徴量リスト（v5から整理）
# ================================================================

BASE_NUMERIC = [
    "horse_rank_lag1", "field_size",
    "jockey_top3_rate", "horse_rank_pct_lag1", "horse_recent_avg_rank3",
    "days_since_last", "jockey_venue_top3_rate",
    # body_weight → body_weight_diff に置き換え
    "body_weight_diff", "body_weight_diff_abs",
    "horse_rank_pct_lag2", "jockey_win_rate", "weight_burden_ratio",
    # horse_avg_rank 削除（vs_field版が1位なので不要）
    "horse_front_style", "horse_closing_style", "horse_avg_agari", "age",
    "trainer_win_rate", "trainer_top3_rate",
    "horse_passing_first_rate_lag1", "horse_distance_diff_lag1",
    # horse_rank_lag2 削除（rank_pct_lag2と重複）
    "horse_passing_last_rate_lag1",
    "jockey_venue_win_rate",
    # round_no 削除
    "trainer_venue_win_rate", "trainer_venue_top3_rate",
    "horse_surface_top3_rate", "horse_surface_win_rate",
    "horse_agari_lag2", "horse_agari_diff_12", "horse_agari_vs_avg",
    "jockey_runs", "horse_passing_last_rate_lag2", "jockey_course_top3_rate",
    "jh_top3_rate",
    # th_runs 削除
    # v6新規
    "jockey_change", "closing_style_x_dist_diff", "horse_agari_rank_pct_lag1",
]

NEW_SIRE_FEATS = [
    # sire_hist_runs / broodmare_sire_hist_runs 削除
    "sire_hist_win_rate", "sire_hist_top3_rate", "sire_dist_win_rate",
    "broodmare_sire_hist_win_rate", "broodmare_sire_hist_top3_rate",
    "broodmare_sire_dist_win_rate",
    "class_hist_win_rate", "class_hist_top3_rate", "class_hist_runs",
]

NEW_ROLLING_FEATS = [
    "horse_recent5_avg_rank", "horse_recent10_avg_rank",
    # horse_recent5_top3_rate / horse_recent5_win_rate / horse_recent10_top3_rate 削除
    "jockey_recent20_win_rate", "jockey_recent50_win_rate",
    "jockey_recent20_top3_rate",
    "trainer_recent20_win_rate", "trainer_recent20_top3_rate",
    "horse_bw_trend_3",
]

NEW_FIELD_FEATS = [
    "field_avg_jockey_win_rate",
    "field_avg_horse_rank",
    "field_avg_horse_top3_rate",
    "field_avg_trainer_top3_rate",
]

ENGR_FEATS = [
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
        ("sire/class features",        add_sire_features),
        ("horse rolling features",      add_horse_rolling_features),
        ("jockey/trainer rolling",      add_jockey_trainer_rolling),
        ("field strength",              add_field_strength),
        ("weight trend",                add_weight_trend),
        ("lap rolling (v5)",            add_lap_rolling_features),
        ("field pace (v5)",             add_field_pace_features),
        ("v6 new features",             add_v6_features),
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

    print(">>> [1/5] LGB top3 seed=42 leaves=63 ...", flush=True)
    m = train_lgb(X_tr, y_tr_top3, X_va, y_va_top3, seed=42)
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
    m = train_lgb(X_tr, y_tr_win, X_va, y_va_win, seed=42)
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
    metrics.update({f"v6_{k}": v for k, v in hit.items()})

    # ============================================================
    # 結果サマリ
    # ============================================================
    print("\n" + "=" * 60)
    print(f"             v5(90個)    v6({len(feats)}個)")
    print(f"  1位的中率:   32.9%       {hit['win']*100:.1f}%")
    print(f"  複勝率:      62.5%       {hit['top3']*100:.1f}%")
    print(f"  予想3頭精度:  49.9%       {hit['pred3']*100:.1f}%")
    print(f"  TOP3 AUC:   0.7925      {metrics['ens_top3_auc']:.4f}")
    print(f"  WIN AUC:    0.8247      {metrics['ens_win_auc']:.4f}")
    print("=" * 60)

    # 特徴量重要度
    m_check = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_win_s42.txt"))
    m3_check = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_top3_s42.txt"))
    fi_win  = pd.Series(m_check.feature_importance(importance_type="gain"),  index=feats)
    fi_top3 = pd.Series(m3_check.feature_importance(importance_type="gain"), index=feats)
    fi = ((fi_win / fi_win.sum() + fi_top3 / fi_top3.sum()) / 2 * 100).round(2)
    fi = fi.sort_values(ascending=False)

    NEW_V6 = {"body_weight_diff", "body_weight_diff_abs", "horse_closing_style",
               "closing_style_x_dist_diff", "horse_agari_rank_pct_lag1", "jockey_change"}

    print("\n特徴量重要度ランキング (win+top3平均):")
    print(f"{'順位':>3}  {'特徴量':<46}  {'重要度':>6}")
    print("-" * 60)
    for rank, (feat, val) in enumerate(fi.items(), 1):
        tag = " [NEW]" if feat in NEW_V6 else ""
        print(f"{rank:>3}. {feat:<46}  {val:>5.2f}%{tag}")

    # メタ保存
    meta = {
        "features": feats, "categorical": [], "cat_categories": {},
        "metrics": metrics,
        "ensemble_top3": ["lgb_top3_s42.txt", "lgb_top3_s7.txt", "lgb_top3_s2024.txt"],
        "ensemble_win":  ["lgb_win_s42.txt", "lgb_win_s7.txt"],
        "rank_model": None, "rank_blend_weight": 0.0,
        "score_win_weight": 0.6, "feature_engineering": "v6",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}")

    print("\n>>> Regenerating lookup tables for v6...", flush=True)
    from no_market_v4_lookups import regenerate_v4_lookups
    lmeta = regenerate_v4_lookups(model_dir=OUT_DIR)
    for name, cnt in lmeta.get("lookup_counts", {}).items():
        print(f"   {name}: {cnt:,}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
