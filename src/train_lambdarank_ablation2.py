"""
LambdaRank アブレーション第2弾
================================
ベースライン変更: top70特徴量 (AUC 0.8520 で最良)
label_gain の上限探索: [0,1,10]/[0,1,15]/[0,1,20]/[0,1,30]/[0,1,50]/[0,0,1]

どこで頭打ちになるかを検証する。
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

# 重要度順 top70特徴量
TOP70_FEATURES = [
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
]

SEEDS = [42, 7]


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


def train_eval(X_tr, y_tr, g_tr, X_va, y_va, g_va, df_val, y_win_va, label_gain):
    models = []
    params = dict(
        objective="lambdarank", metric="ndcg",
        eval_at=[1, 3], label_gain=label_gain,
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

    def groups(mask):
        return df[mask].groupby("race_id", sort=False)["race_id"].count().tolist()
    g_tr = groups(mask_tr)
    g_va = groups(mask_va)

    feats = [f for f in TOP70_FEATURES if f in df.columns]
    print(f"\nTop70特徴量 実際に使用: {len(feats)}個\n", flush=True)

    X_tr = df[feats][mask_tr]
    X_va = df[feats][mask_va]
    y_tr = y_label[mask_tr]
    y_va = y_label[mask_va]
    df_val = df[mask_va][feats + ["race_id", "rank"]].copy()
    y_win_va = y_win[mask_va]

    # label_gain候補: [0,1,3]は既知なので[0,1,10]から上を探索
    # [0,0,1]は1位以外を完全無視する極端ケース
    gain_candidates = [
        [0, 1, 10],
        [0, 1, 15],
        [0, 1, 20],
        [0, 1, 30],
        [0, 1, 50],
        [0, 0, 1],   # 1位のみ学習する極端ケース
    ]

    print("=" * 65)
    print("label_gain 探索 (top70特徴量固定、2シード)")
    print("=" * 65, flush=True)

    results = []
    for lg in gain_candidates:
        lg_str = str(lg)
        print(f"\n[gain={lg_str}] ", end="", flush=True)
        t0 = time.time()
        win1, fuku, top3p, auc = train_eval(
            X_tr, y_tr, g_tr, X_va, y_va, g_va, df_val, y_win_va, lg
        )
        elapsed = time.time() - t0
        print(f"1位={win1:.1%}  複勝={fuku:.1%}  AUC={auc:.4f}  ({elapsed:.0f}s)", flush=True)
        results.append({"label_gain": lg_str, "win1": win1, "fuku": fuku,
                        "top3p": top3p, "auc": auc})

    # ── 比較表 ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("label_gain 探索結果 (top70特徴量、2シード)")
    print("=" * 65)
    print(f"{'label_gain':<14} {'1位的中率':>9} {'複勝率':>7} {'AUC':>8}  備考")
    print("-" * 65)
    # 既知の結果も含めて表示
    known = [
        ("[0,1,2]",  0.370, 0.665, 0.8424, "← 第1弾(107feat)"),
        ("[0,1,3]",  0.397, 0.669, 0.8506, "← ベースライン(107feat)"),
        ("[0,1,3]",  0.397, 0.670, 0.8520, "← top70 ablation"),
        ("[0,1,5]",  0.413, 0.670, 0.8595, "← 第1弾(107feat)"),
        ("[0,1,10]", 0.427, 0.670, 0.8677, "← 第1弾(107feat)"),
    ]
    for lg_str, w, f, a, note in known:
        print(f"{lg_str:<14} {w:>8.1%} {f:>6.1%} {a:>8.4f}  {note}")
    print("...(以下 top70特徴量での結果)...")
    for r in results:
        best = " ★" if r["win1"] == max(x["win1"] for x in results) else ""
        print(f"{r['label_gain']:<14} {r['win1']:>8.1%} {r['fuku']:>6.1%} {r['auc']:>8.4f}{best}")
    print("=" * 65)

    best = max(results, key=lambda x: x["win1"])
    print(f"\n最高1位的中率: {best['win1']:.1%} (label_gain={best['label_gain']})")

    out_path = os.path.join(BASE_DIR, "src", "ablation2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"結果を {out_path} に保存しました。")


if __name__ == "__main__":
    main()
