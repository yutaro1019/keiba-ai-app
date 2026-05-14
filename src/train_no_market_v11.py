"""
no_market_v11 モデル学習スクリプト
v10 からの変更（既存の未使用カラムを活用）:
  - sex_encoded:               牝馬フラグ (牝=1, 牡/セ=0)
  - going_encoded:             馬場状態数値 (良=0, 稍重=1, 重=2, 不良=3)
  - direction_encoded:         回り方向 (右=0, 左=1)
  - horse_venue_top3_rate_vs_field: 馬の競馬場別複勝率のフィールド内順位
  - horse_venue_win_rate_vs_field:  馬の競馬場別勝率のフィールド内順位
  - horse_course_top3_rate_vs_field: 馬のコース別複勝率のフィールド内順位
  - trainer_course_top3_rate_vs_field: 調教師コース別複勝率のフィールド内順位
  - horse_passing_gain_lag1/2: 前走・前々走の通過順改善 (後半追い込み力)
  - jockey_course_win_rate:    騎手コース別勝率（既存カラム）
"""
import os, sys, json, re, gzip, pickle, gc, warnings
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
from train_no_market_v7 import add_v7_features, DIST_BINS, DIST_LABELS
from train_no_market_v10 import (
    add_v10_features, SIRE_LINE_MAP, LINE_NAMES, N_LINES,
    normalize_sire_name, get_sire_line,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL  = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
OUT_DIR   = os.path.join(BASE_DIR, "models", "no_market_v11")
os.makedirs(OUT_DIR, exist_ok=True)


def _within_race_pct(df: pd.DataFrame, col: str, higher_is_better: bool = True,
                     race_col: str = "race_id") -> pd.Series:
    """レース内パーセンタイル: 1.0=最良, 0.0=最悪"""
    asc = not higher_is_better
    ranked = df.groupby(race_col, sort=False, observed=True)[col].rank(
        method="average", ascending=asc, na_option="keep"
    )
    n_valid = df.groupby(race_col, sort=False, observed=True)[col].transform(
        lambda x: float(x.notna().sum())
    )
    return (1.0 - (ranked - 1) / (n_valid - 1).clip(lower=1)).where(df[col].notna()).astype("float32")


def add_v11_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # ── 性別エンコード ─────────────────────────────────────────────
    if "sex" in df.columns:
        df["sex_encoded"] = (df["sex"] == "牝").astype("float32")  # 牝=1

    # ── 馬場状態エンコード ──────────────────────────────────────────
    if "going" in df.columns:
        going_map = {"良": 0.0, "稍重": 1.0, "稍": 1.0, "重": 2.0, "不良": 3.0, "不": 3.0}
        df["going_encoded"] = df["going"].map(going_map).astype("float32")

    # ── コース方向エンコード ────────────────────────────────────────
    if "direction" in df.columns:
        dir_map = {"右": 0.0, "左": 1.0}
        df["direction_encoded"] = df["direction"].map(dir_map).astype("float32")

    # ── 馬の競馬場別成績 → フィールド内パーセンタイル ───────────────
    for col in ["horse_venue_top3_rate", "horse_venue_win_rate"]:
        if col in df.columns:
            df[f"{col}_vs_field"] = _within_race_pct(df, col, higher_is_better=True)

    # ── 馬のコース別成績 → フィールド内パーセンタイル ─────────────────
    if "horse_course_top3_rate" in df.columns:
        df["horse_course_top3_rate_vs_field"] = _within_race_pct(df, "horse_course_top3_rate")

    # ── 調教師のコース別成績 → フィールド内パーセンタイル ───────────────
    if "trainer_course_top3_rate" in df.columns:
        df["trainer_course_top3_rate_vs_field"] = _within_race_pct(df, "trainer_course_top3_rate")

    # ── 騎手コース別勝率（既存カラム、フィールド内パーセンタイル） ─────────
    if "jockey_course_win_rate" in df.columns:
        df["jockey_course_win_rate_vs_field"] = _within_race_pct(df, "jockey_course_win_rate")

    # ── 通過順改善 (horse_passing_gain_lag1/2) ─────────────────────
    # passing_gain = passing_last - passing_first （正=後半追い上げ、負=後退）
    for lag in [1, 2]:
        col = f"horse_passing_gain_lag{lag}"
        if col in df.columns:
            df[col] = df[col].astype("float32")

    return df


FEATURE_COLS_V11 = [
    # v10 全特徴量 (107個)
    "horse_avg_rank_vs_field","horse_recent_avg_rank3_vs_field","class_hist_win_rate",
    "class_hist_top3_rate","field_size","horse_rank_lag1","jockey_top3_rate_vs_field",
    "horse_recent_top3_rate3_vs_field","horse_rank_pct_lag1_vs_field","class_hist_runs",
    "horse_top3_rate_vs_field","jockey_win_rate_vs_field","jockey_top3_rate",
    "horse_rank_pct_lag1","weight_burden_ratio","horse_agari_lag1_vs_field",
    "trainer_top3_rate_vs_field","horse_rank_pct_lag2","days_since_last",
    "horse_recent10_avg_rank","horse_front_style","jockey_venue_top3_rate",
    "field_avg_jockey_win_rate","horse_avg_agari","horse_win_rate_vs_field",
    "horse_agari_rank_pct_lag1","horse_recent5_avg_rank","race_back_pace",
    "jockey_win_rate","trainer_win_rate","field_avg_trainer_top3_rate",
    "horse_passing_first_rate_lag1","race_pace_diff","trainer_venue_win_rate",
    "field_avg_horse_top3_rate","trainer_venue_top3_rate","trainer_top3_rate",
    "jockey_course_top3_rate","horse_prev_back_pace1_vs_field","horse_passing_last_rate_lag1",
    "horse_recent_avg_rank3","horse_agari_diff_12","horse_dist_top3_rate_vs_field",
    "jockey_venue_win_rate","horse_passing_last_rate_lag2","jockey_runs",
    "field_avg_horse_rank","field_avg_front_pace","sire_dist_win_rate",
    "broodmare_sire_dist_win_rate","sire_hist_win_rate","horse_agari_vs_avg",
    "horse_agari_lag2","horse_prev_pace_diff2","sire_hist_top3_rate",
    "broodmare_sire_hist_win_rate","horse_prev_front_pace2","race_front_pace",
    "horse_prev_pace_diff3","age","horse_prev_front_pace1_vs_field",
    "horse_surface_top3_rate","horse_prev_front_pace1","horse_distance_diff_lag1",
    "field_avg_back_pace","field_avg_pace_diff","horse_prev_pace_diff1_vs_field",
    "horse_prev_front_pace3","jockey_recent20_top3_rate","horse_surface_win_rate",
    "broodmare_sire_hist_top3_rate","horse_prev_back_pace3","horse_prev_pace_diff1",
    "horse_prev_back_pace2","horse_prev_back_pace1","horse_bw_trend_3","jh_top3_rate",
    "horse_closing_style","jockey_recent50_win_rate","body_weight_diff",
    "trainer_recent20_top3_rate","closing_style_x_dist_diff","body_weight_diff_abs",
    "jockey_recent20_win_rate","trainer_recent20_win_rate","jockey_change",
    "horse_going_top3_rate_vs_field_diff","expected_pace_fit",
    "horse_going_win_rate","horse_going_top3_rate",
    "horse_dist_cat_top3_rate","horse_dist_cat_win_rate",
    "venue_frame_win_rate","venue_frame_top3_rate",
    "class_change",
    "sire_line_win_rate","sire_line_top3_rate",
    "sire_line_surface_top3_rate","sire_line_dist_top3_rate",
    "bms_line_win_rate","bms_line_top3_rate",
    "bms_line_surface_top3_rate","bms_line_dist_top3_rate",
    "bloodline_cross_win_rate","bloodline_cross_top3_rate",
    "bloodline_cross_surface_top3_rate","bloodline_cross_dist_top3_rate",
    # v11 新規 (9個)
    "sex_encoded",
    "going_encoded",
    "direction_encoded",
    "horse_venue_top3_rate_vs_field",
    "horse_venue_win_rate_vs_field",
    "horse_course_top3_rate_vs_field",
    "trainer_course_top3_rate_vs_field",
    "jockey_course_win_rate_vs_field",
    "horse_passing_gain_lag1",
    "horse_passing_gain_lag2",
]

LGB_PARAMS_TOP3 = dict(
    objective="binary", metric="auc", verbosity=-1,
    learning_rate=0.05, num_leaves=63, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
)
LGB_PARAMS_WIN = dict(
    objective="binary", metric="auc", verbosity=-1,
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
)


def main():
    print(">>> Loading processed data...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"   rows={len(df):,}, cols={df.shape[1]}", flush=True)

    steps = [
        ("sire/class",    add_sire_features),
        ("horse rolling", add_horse_rolling_features),
        ("jockey/trainer",add_jockey_trainer_rolling),
        ("field strength",add_field_strength),
        ("weight trend",  add_weight_trend),
        ("lap rolling",   add_lap_rolling_features),
        ("field pace",    add_field_pace_features),
        ("v6",            add_v6_features),
        ("v7",            add_v7_features),
        ("v10",           add_v10_features),
        ("v11",           add_v11_features),
    ]
    for label, fn in steps:
        print(f">>> Adding {label}...", flush=True)
        df = fn(df); gc.collect()

    print(">>> Adding vs_field features...", flush=True)
    df = add_model_features(df, race_col="race_id"); gc.collect()

    print(">>> Adding combo features...", flush=True)
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_w"] = (df["rank"] == 1).astype(float)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"]  = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]      = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_w", "_t"], inplace=True)

    avail = [c for c in FEATURE_COLS_V11 if c in df.columns]
    missing = [c for c in FEATURE_COLS_V11 if c not in df.columns]
    if missing:
        print(f"   [WARN] missing ({len(missing)}): {missing}", flush=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rank"])
    mask_train = df["date"].dt.year < 2025
    mask_val   = df["date"].dt.year == 2025
    y_top3 = (df["rank"] <= 3).astype(int)
    y_win  = (df["rank"] == 1).astype(int)
    X = df[avail]
    print(f"\nFeatures: {len(avail)}  Train: {mask_train.sum():,}  Val: {mask_val.sum():,}\n", flush=True)

    models_top3, models_win = [], []
    for i, (seed, leaves) in enumerate([(42, 63), (7, 95), (2024, 47)], 1):
        params = {**LGB_PARAMS_TOP3, "num_leaves": leaves, "seed": seed, "random_state": seed}
        print(f">>> [{i}/5] top3 s{seed} l{leaves} ...", flush=True)
        tr = lgb.Dataset(X[mask_train], y_top3[mask_train])
        va = lgb.Dataset(X[mask_val],   y_top3[mask_val], reference=tr)
        m = lgb.train(params, tr, 3000, valid_sets=[va],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)])
        print(f"   AUC={roc_auc_score(y_top3[mask_val], m.predict(X[mask_val])):.4f}", flush=True)
        models_top3.append(m)

    p_top3 = np.mean([m.predict(X[mask_val]) for m in models_top3], axis=0)
    print(f"   Ensemble TOP3 AUC={roc_auc_score(y_top3[mask_val], p_top3):.4f}\n", flush=True)

    for i, seed in enumerate([42, 7], 4):
        params = {**LGB_PARAMS_WIN, "seed": seed, "random_state": seed}
        print(f">>> [{i}/5] win s{seed} ...", flush=True)
        tr = lgb.Dataset(X[mask_train], y_win[mask_train])
        va = lgb.Dataset(X[mask_val],   y_win[mask_val], reference=tr)
        m = lgb.train(params, tr, 3000, valid_sets=[va],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)])
        auc = roc_auc_score(y_win[mask_val], m.predict(X[mask_val]))
        models_win.append(m)
        if i == 5:
            p_win = np.mean([m.predict(X[mask_val]) for m in models_win], axis=0)
            print(f"   AUC={auc:.4f}  Ensemble WIN={roc_auc_score(y_win[mask_val], p_win):.4f}\n", flush=True)
        else:
            print(f"   AUC={auc:.4f}", flush=True)

    p_win = np.mean([m.predict(X[mask_val]) for m in models_win], axis=0)
    val_df = df[mask_val].copy()
    val_df["score"] = 0.6 * p_win + 0.4 * p_top3

    def calc_hit_rates(vdf):
        win1 = fuku = pred3 = cnt3 = n = 0
        for _, g in vdf.groupby("race_id"):
            best = g["score"].idxmax()
            win1 += int(g.loc[best, "rank"] == 1)
            fuku += int(g.loc[best, "rank"] <= 3)
            t3   = g.nlargest(3, "score").index
            a3   = set(g[g["rank"] <= 3].index)
            pred3 += len(set(t3) & a3); cnt3 += 3; n += 1
        return win1/n, fuku/n, pred3/cnt3

    win1, fuku, top3p = calc_hit_rates(val_df)
    top3_auc = roc_auc_score(y_top3[mask_val], p_top3)
    win_auc  = roc_auc_score(y_win[mask_val], p_win)
    print("=" * 65)
    print(f"              v10({len(avail) - 10}個)  v11({len(avail)}個)")
    print(f"  1位的中率:    33.1%       {win1:.1%}")
    print(f"  複勝率:       62.9%       {fuku:.1%}")
    print(f"  予想3頭精度:   50.0%       {top3p:.1%}")
    print(f"  TOP3 AUC:    0.7929      {top3_auc:.4f}")
    print(f"  WIN AUC:     0.8246      {win_auc:.4f}")
    print("=" * 65)

    v11_new = {
        "sex_encoded","going_encoded","direction_encoded",
        "horse_venue_top3_rate_vs_field","horse_venue_win_rate_vs_field",
        "horse_course_top3_rate_vs_field","trainer_course_top3_rate_vs_field",
        "jockey_course_win_rate_vs_field",
        "horse_passing_gain_lag1","horse_passing_gain_lag2",
    }
    imp = {}
    for m in models_top3 + models_win:
        for feat, val in zip(m.feature_name(), m.feature_importance("gain")):
            imp[feat] = imp.get(feat, 0) + val
    total_imp = sum(imp.values())
    print("\n特徴量重要度:")
    for rank_i, (feat, val) in enumerate(sorted(imp.items(), key=lambda x: -x[1]), 1):
        tag = " [NEW]" if feat in v11_new else ""
        print(f" {rank_i:3d}. {feat:<52s} {val/total_imp:.2%}{tag}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, m in enumerate(models_top3): m.save_model(os.path.join(OUT_DIR, f"top3_{i}.lgb"))
    for i, m in enumerate(models_win):  m.save_model(os.path.join(OUT_DIR, f"win_{i}.lgb"))
    meta = {
        "version": "no_market_v11", "feature_engineering": "v11",
        "features": avail, "n_top3_models": len(models_top3), "n_win_models": len(models_win),
        "sire_line_categories": LINE_NAMES, "n_lines": N_LINES,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}", flush=True)

    print("\n>>> Regenerating lookup tables for v11...", flush=True)
    from no_market_v4_lookups import regenerate_v4_lookups
    lmeta = regenerate_v4_lookups(model_dir=OUT_DIR)
    for k, v in lmeta.get("lookup_counts", {}).items():
        print(f"   {k}: {v:,}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
