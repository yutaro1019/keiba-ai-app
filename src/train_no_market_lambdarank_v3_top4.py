"""
no_market_lambdarank_v2 正式モデル
====================================
グリッドサーチ (6特徴量数 × 8 label_gain = 48条件) の結果:
  最高的中率: 44.8% → top50特徴量 × label_gain=[0,1,40]

変更点 (vs lambdarank v1):
  - 特徴量: 107個 → 50個 (重要度上位50位)
  - label_gain: [0,1,3] → [0,1,40] (1位を40倍重視)
  - シード数: 3 (本番モデルとして安定化)
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
from train_no_market_v7 import add_v7_features
from train_no_market_v10 import add_v10_features, LINE_NAMES, N_LINES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL  = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")


def _gain_from_env():
    raw = os.environ.get("KEIBA_LABEL_GAIN")
    if not raw:
        return [0, 1, 5, 15, 40]
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 5 or vals[0] != 0:
        raise ValueError("KEIBA_LABEL_GAIN must be 5 comma-separated ints like 0,1,5,15,40")
    return vals


MODEL_VERSION = os.environ.get("KEIBA_MODEL_VERSION", "no_market_lambdarank_v3_top4")
OUT_DIR   = os.path.join(BASE_DIR, "models", MODEL_VERSION)
os.makedirs(OUT_DIR, exist_ok=True)

# グリッドサーチで決定した最適設定
LABEL_GAIN = _gain_from_env()
SEEDS = [42, 7, 2024]

# 重要度上位50特徴量
FEATURE_COLS = [
    "class_hist_win_rate","horse_avg_rank_vs_field","horse_recent_top3_rate3_vs_field",
    "class_hist_runs","horse_rank_pct_lag1","class_hist_top3_rate",
    "jockey_top3_rate_vs_field","horse_recent_avg_rank3_vs_field","horse_top3_rate_vs_field",
    "horse_rank_lag1","jockey_win_rate_vs_field","horse_rank_pct_lag1_vs_field",
    "horse_recent5_avg_rank","horse_recent_avg_rank3","horse_rank_pct_lag2",
    "horse_recent10_avg_rank","jockey_venue_top3_rate","weight_burden_ratio",
    "horse_agari_lag1_vs_field","jockey_top3_rate","trainer_top3_rate_vs_field",
    "days_since_last","horse_agari_rank_pct_lag1","horse_front_style",
    "age","horse_win_rate_vs_field","horse_avg_agari","trainer_top3_rate",
    "class_change","horse_surface_top3_rate","trainer_win_rate","jockey_win_rate",
    "trainer_venue_top3_rate","trainer_venue_win_rate","venue_frame_top3_rate",
    "horse_passing_first_rate_lag1","horse_passing_last_rate_lag1","jockey_venue_win_rate",
    "horse_prev_back_pace1_vs_field","sire_hist_top3_rate","sire_hist_win_rate",
    "horse_distance_diff_lag1","jockey_course_top3_rate","field_avg_front_pace",
    "jockey_recent20_top3_rate","bms_line_dist_top3_rate","jockey_recent50_win_rate",
    "venue_frame_win_rate","horse_going_top3_rate_vs_field_diff","horse_dist_top3_rate_vs_field",
]


def make_label(rank_series):
    return np.where(rank_series == 1, 4,
           np.where(rank_series == 2, 3,
           np.where(rank_series == 3, 2,
           np.where(rank_series == 4, 1, 0)))).astype(np.int32)


def calc_hit_rates(df_val, score_col="lr_score"):
    win1 = fuku = pred3 = cnt3 = n = 0
    for _, g in df_val.groupby("race_id"):
        best = g[score_col].idxmax()
        win1 += int(g.loc[best, "rank"] == 1)
        fuku += int(g.loc[best, "rank"] <= 3)
        t3 = g.nlargest(3, score_col).index
        a3 = set(g[g["rank"] <= 3].index)
        pred3 += len(set(t3) & a3)
        cnt3  += 3
        n     += 1
    return win1 / n, fuku / n, pred3 / cnt3


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
    ]
    for label, fn in steps:
        print(f">>> Adding {label}...", flush=True)
        df = fn(df); gc.collect()

    print(">>> Adding vs_field features...", flush=True)
    df = add_model_features(df, race_col="race_id"); gc.collect()

    print(">>> Adding combo features (jh_top3_rate)...", flush=True)
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"]  = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]      = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_t"], inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rank"])
    df = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)

    avail = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"   [WARN] missing: {missing}", flush=True)

    mask_train = df["date"].dt.year < 2025
    mask_val   = df["date"].dt.year == 2025

    y_label = make_label(df["rank"])
    y_win   = (df["rank"] == 1).astype(int)
    X = df[avail]

    def make_groups(mask):
        return df[mask].groupby("race_id", sort=False)["race_id"].count().tolist()
    groups_train = make_groups(mask_train)
    groups_val   = make_groups(mask_val)

    print(f"\nFeatures: {len(avail)}  label_gain={LABEL_GAIN}")
    print(f"Train: {mask_train.sum():,}行 / {len(groups_train):,}レース")
    print(f"Val  : {mask_val.sum():,}行 / {len(groups_val):,}レース\n")

    lgb_params = dict(
        objective       = "lambdarank",
        metric          = "ndcg",
        eval_at         = [1, 3],
        label_gain      = LABEL_GAIN,
        verbosity       = -1,
        learning_rate   = 0.05,
        num_leaves      = 63,
        min_child_samples = 20,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
    )

    models = []
    for i, seed in enumerate(SEEDS, 1):
        params = {**lgb_params, "seed": seed, "random_state": seed}
        print(f">>> [{i}/{len(SEEDS)}] seed={seed} ...", flush=True)
        tr_ds = lgb.Dataset(X[mask_train], label=y_label[mask_train], group=groups_train)
        va_ds = lgb.Dataset(X[mask_val],   label=y_label[mask_val],   group=groups_val, reference=tr_ds)
        m = lgb.train(
            params, tr_ds, num_boost_round=3000,
            valid_sets=[va_ds],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)],
        )
        raw = m.predict(X[mask_val])
        val_tmp = df[mask_val].copy()
        val_tmp["_raw"] = raw
        def _softmax(g):
            e = np.exp(g["_raw"] - g["_raw"].max())
            return e / e.sum()
        val_tmp["_prob"] = val_tmp.groupby("race_id", group_keys=False).apply(_softmax)
        auc = roc_auc_score(y_win[mask_val], val_tmp["_prob"])
        print(f"   AUC(win)={auc:.4f}", flush=True)
        models.append(m)

    # アンサンブル評価
    raw_scores = np.mean([m.predict(X[mask_val]) for m in models], axis=0)
    val_df = df[mask_val].copy()
    val_df["lr_score"] = raw_scores

    def softmax_in_race(g):
        e = np.exp(g["lr_score"] - g["lr_score"].max())
        return e / e.sum()
    val_df["lr_prob"] = val_df.groupby("race_id", group_keys=False).apply(softmax_in_race)

    win1, fuku, top3p = calc_hit_rates(val_df)
    auc_ens = roc_auc_score(y_win[mask_val], val_df["lr_prob"])

    print("\n" + "=" * 68)
    print(f"              v1(107feat/gain3)  v2(top50/gain40) [3seed]")
    print(f"  1位的中率:       39.7%              {win1:.1%}")
    print(f"  複勝率:          66.9%              {fuku:.1%}")
    print(f"  予想3頭精度:      50.9%              {top3p:.1%}")
    print(f"  WIN AUC:        0.8506             {auc_ens:.4f}")
    print("=" * 68)

    # 特徴量重要度
    imp = {}
    for m in models:
        for feat, val in zip(m.feature_name(), m.feature_importance("gain")):
            imp[feat] = imp.get(feat, 0) + val
    total_imp = sum(imp.values())
    print("\n特徴量重要度:")
    for rank_i, (feat, val) in enumerate(sorted(imp.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank_i:3d}. {feat:<50s} {val/total_imp:.2%}")

    # 保存
    for i, m in enumerate(models):
        m.save_model(os.path.join(OUT_DIR, f"rank_{i}.lgb"))

    meta = {
        "version": MODEL_VERSION,
        "feature_engineering": "v10",
        "model_type": "lambdarank",
        "label_design": "1st=4, 2nd=3, 3rd=2, 4th=1, 5th+=0",
        "label_gain": LABEL_GAIN,
        "eval_at": [1, 3],
        "features": avail,
        "n_rank_models": len(models),
        "sire_line_categories": LINE_NAMES,
        "n_lines": N_LINES,
        "hit_rate_win1": float(win1),
        "hit_rate_fuku": float(fuku),
        "hit_rate_top3pred": float(top3p),
        "win_auc_softmax": float(auc_ens),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}", flush=True)

    print("\n>>> Regenerating lookup tables...", flush=True)
    from no_market_v4_lookups import regenerate_v4_lookups
    lmeta = regenerate_v4_lookups(model_dir=OUT_DIR)
    for k, v in lmeta.get("lookup_counts", {}).items():
        print(f"   {k}: {v:,}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
