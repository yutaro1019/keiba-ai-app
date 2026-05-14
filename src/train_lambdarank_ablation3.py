"""
LambdaRank アブレーション第3弾
================================
label_gain=[0,1,30] (最適値) で特徴量数を変えてどうなるか。
第1弾は [0,1,3] 固定だったので、最適gainと特徴量数の交差検証。

テスト: top 10/20/30/50/70/107 × label_gain=[0,1,30]
各条件2シード (42, 7)。
"""
import os, sys, json, gzip, pickle, gc, warnings, time
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
from train_no_market_v10 import add_v10_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL  = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")

# 重要度順 107特徴量
FEATURES_BY_IMPORTANCE = [
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
    "broodmare_sire_hist_win_rate","sire_line_surface_top3_rate","jockey_runs",
    "horse_passing_last_rate_lag2","horse_prev_front_pace1","horse_agari_diff_12",
    "broodmare_sire_hist_top3_rate","horse_agari_vs_avg","field_avg_horse_top3_rate",
    "bms_line_win_rate","horse_prev_pace_diff2","horse_agari_lag2",
    "field_avg_jockey_win_rate","horse_surface_win_rate","horse_prev_pace_diff3",
    "horse_prev_front_pace2","sire_line_dist_top3_rate","broodmare_sire_dist_win_rate",
    "sire_dist_win_rate","jh_top3_rate",
    "horse_prev_front_pace1_vs_field","field_avg_trainer_top3_rate",
    "horse_prev_back_pace3","horse_prev_front_pace3",
    "bms_line_top3_rate","horse_prev_back_pace1",
    "horse_bw_trend_3","field_avg_horse_rank","field_avg_pace_diff","field_avg_back_pace",
    "horse_prev_pace_diff1","bms_line_surface_top3_rate","sire_line_win_rate","sire_line_top3_rate",
    "trainer_recent20_top3_rate","horse_prev_back_pace2",
    "horse_dist_cat_top3_rate","horse_closing_style","horse_prev_pace_diff1_vs_field",
    "horse_going_win_rate","body_weight_diff","horse_going_top3_rate",
    "expected_pace_fit","horse_dist_cat_win_rate","closing_style_x_dist_diff",
    "jockey_recent20_win_rate","body_weight_diff_abs",
    "bloodline_cross_surface_top3_rate","bloodline_cross_dist_top3_rate",
    "trainer_recent20_win_rate","bloodline_cross_win_rate",
    "jockey_change","bloodline_cross_top3_rate",
    "field_size","race_back_pace","race_pace_diff","race_front_pace",
]

SEEDS = [42, 7]
LABEL_GAIN = [0, 1, 30]


def make_label(rank_series):
    return np.where(rank_series == 1, 2,
           np.where(rank_series <= 3, 1, 0)).astype(np.int32)


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


def train_eval(X_tr, y_tr, g_tr, X_va, y_va, g_va, df_val, y_win_va):
    models = []
    params = dict(
        objective="lambdarank", metric="ndcg",
        eval_at=[1, 3], label_gain=LABEL_GAIN,
        verbosity=-1, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
    )
    for seed in SEEDS:
        p = {**params, "seed": seed, "random_state": seed}
        tr_ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
        va_ds = lgb.Dataset(X_va, label=y_va, group=g_va, reference=tr_ds)
        m = lgb.train(p, tr_ds, num_boost_round=3000,
                      valid_sets=[va_ds],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(99999)])
        models.append(m)

    raw = np.mean([m.predict(X_va) for m in models], axis=0)
    val_tmp = df_val.copy()
    val_tmp["lr_score"] = raw

    def softmax(g):
        e = np.exp(g["lr_score"] - g["lr_score"].max())
        return e / e.sum()
    val_tmp["lr_prob"] = val_tmp.groupby("race_id", group_keys=False).apply(softmax)

    win1, fuku, top3p = calc_hit_rates(val_tmp)
    auc = roc_auc_score(y_win_va, val_tmp["lr_prob"])
    return win1, fuku, top3p, auc


def main():
    print(">>> Loading & building features...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"   rows={len(df):,}", flush=True)

    for label, fn in [
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
        ("vs_field",      lambda d: add_model_features(d, race_col="race_id")),
    ]:
        print(f"   {label}...", flush=True)
        df = fn(df); gc.collect()

    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"] = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]     = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_t"], inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rank"])
    df = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)

    mask_tr = df["date"].dt.year < 2025
    mask_va = df["date"].dt.year == 2025
    y_label = make_label(df["rank"])
    y_win   = (df["rank"] == 1).astype(int)
    y_win_va = y_win[mask_va]

    def groups(mask):
        return df[mask].groupby("race_id", sort=False)["race_id"].count().tolist()
    g_tr = groups(mask_tr)
    g_va = groups(mask_va)

    avail_all = [f for f in FEATURES_BY_IMPORTANCE if f in df.columns]
    print(f"\nlabel_gain={LABEL_GAIN} で特徴量数を変えて検証\n", flush=True)

    print("=" * 62)
    print(f"特徴量数 × label_gain={LABEL_GAIN} (2シード)")
    print("=" * 62, flush=True)

    results = []
    for n_feat in [10, 20, 30, 50, 70, 107]:
        feats = avail_all[:n_feat]
        print(f"\n[top{n_feat:3d}個] ", end="", flush=True)
        t0 = time.time()

        df_val_sub = df[mask_va][feats + ["race_id", "rank"]].copy()
        win1, fuku, top3p, auc = train_eval(
            df[feats][mask_tr], y_label[mask_tr], g_tr,
            df[feats][mask_va], y_label[mask_va], g_va,
            df_val_sub, y_win_va,
        )
        elapsed = time.time() - t0
        print(f"1位={win1:.1%}  複勝={fuku:.1%}  AUC={auc:.4f}  ({elapsed:.0f}s)", flush=True)
        results.append({"n_feat": n_feat, "win1": win1, "fuku": fuku,
                        "top3p": top3p, "auc": auc})

    # ── 比較表 ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 62)
    print(f"結果まとめ (label_gain={LABEL_GAIN}、2シード)")
    print("=" * 62)
    print(f"{'特徴量数':>8}  {'1位的中率':>9}  {'複勝率':>7}  {'AUC':>8}")
    print("-" * 62)
    # 第1弾との比較 (label_gain=[0,1,3])
    ref = {10:0.385, 20:0.388, 30:0.396, 50:0.394, 70:0.397, 107:0.396}
    for r in results:
        diff = r["win1"] - ref.get(r["n_feat"], 0)
        diff_str = f"(+{diff:.1%})" if diff >= 0 else f"({diff:.1%})"
        best_mark = " ★" if r["win1"] == max(x["win1"] for x in results) else ""
        print(f"  top{r['n_feat']:3d}個  {r['win1']:>8.1%}  {diff_str:<8}  "
              f"{r['fuku']:>6.1%}  {r['auc']:>8.4f}{best_mark}")
    print("=" * 62)
    print("※ 括弧内は [0,1,3] との差分")

    best = max(results, key=lambda x: x["win1"])
    print(f"\n最高1位的中率: {best['win1']:.1%} (top{best['n_feat']}個、label_gain={LABEL_GAIN})")

    out_path = os.path.join(BASE_DIR, "src", "ablation3_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"結果を {out_path} に保存しました。")


if __name__ == "__main__":
    main()
