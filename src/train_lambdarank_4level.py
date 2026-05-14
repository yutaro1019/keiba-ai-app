"""
LambdaRank 4段階ラベル探索
============================
問題: 3段階ラベル(0/1/2)では2〜3位をまとめて同じラベルにするため、
     gainを上げると1位重視になり予想3頭精度が下がるトレードオフがある。

解決策: 4段階ラベル (4位以下=0, 3位=1, 2位=2, 1位=3)
     label_gain=[0, g_3rd, g_2nd, g_1st] で1位/2位/3位を独立して重み付け。
     → 全指標(1位的中率・複勝率・予想3頭精度)の同時改善を目指す。

比較対象として3段階ラベルの中間gain([0,1,15],[0,1,20])も含める。

特徴量: top50固定 (グリッドサーチ最適)
シード: 2 (探索フェーズ)
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

# top50特徴量
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

SEEDS = [42, 7]


def make_label_3(rank):
    """3段階: 1位=2, 2〜3位=1, 4位以下=0"""
    return np.where(rank == 1, 2, np.where(rank <= 3, 1, 0)).astype(np.int32)


def make_label_4(rank):
    """4段階: 1位=3, 2位=2, 3位=1, 4位以下=0"""
    return np.where(rank == 1, 3,
           np.where(rank == 2, 2,
           np.where(rank == 3, 1, 0))).astype(np.int32)


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
    y_win   = (df["rank"] == 1).astype(int)
    y_win_va = y_win[mask_va]

    def groups(mask):
        return df[mask].groupby("race_id", sort=False)["race_id"].count().tolist()
    g_tr = groups(mask_tr)
    g_va = groups(mask_va)

    feats = [f for f in FEATURE_COLS if f in df.columns]
    X_tr = df[feats][mask_tr]
    X_va = df[feats][mask_va]
    df_val = df[mask_va][feats + ["race_id", "rank"]].copy()

    # ── テスト条件 ──────────────────────────────────────────────────────
    # (説明ラベル, ラベル関数, label_gain)
    conditions = [
        # 3段階ラベル・中間gain（バランス探索）
        ("3lv [0,1,15]",  make_label_3, [0, 1, 15]),
        ("3lv [0,1,20]",  make_label_3, [0, 1, 20]),
        ("3lv [0,1,25]",  make_label_3, [0, 1, 25]),
        # 4段階ラベル: g_3rd=1固定、g_2nd×g_1stを変える
        ("4lv [0,1,2,10]",  make_label_4, [0, 1, 2, 10]),
        ("4lv [0,1,2,20]",  make_label_4, [0, 1, 2, 20]),
        ("4lv [0,1,2,30]",  make_label_4, [0, 1, 2, 30]),
        ("4lv [0,1,3,10]",  make_label_4, [0, 1, 3, 10]),
        ("4lv [0,1,3,20]",  make_label_4, [0, 1, 3, 20]),
        ("4lv [0,1,3,30]",  make_label_4, [0, 1, 3, 30]),
        ("4lv [0,1,5,15]",  make_label_4, [0, 1, 5, 15]),
        ("4lv [0,1,5,25]",  make_label_4, [0, 1, 5, 25]),
        ("4lv [0,1,5,40]",  make_label_4, [0, 1, 5, 40]),
        ("4lv [0,2,5,20]",  make_label_4, [0, 2, 5, 20]),
        ("4lv [0,2,5,30]",  make_label_4, [0, 2, 5, 30]),
    ]

    print(f"\n>>> {len(conditions)}条件 × {len(SEEDS)}シード\n", flush=True)
    print(f"{'条件':<20} {'1位':>7} {'複勝':>7} {'予想3頭':>8} {'AUC':>8}", flush=True)
    print("-" * 58, flush=True)

    # 既知ベースライン
    print(f"{'v1 (3lv gain=3)':<20} {'39.7%':>7} {'66.9%':>7} {'50.9%':>8} {'0.8506':>8}  ← binary→LR v1")
    print(f"{'v2 (3lv gain=40)':<20} {'44.5%':>7} {'66.9%':>7} {'48.4%':>8} {'0.8735':>8}  ← 現行v2")
    print("-" * 58, flush=True)

    results = []
    for name, label_fn, lg in conditions:
        y_tr = label_fn(df[mask_tr]["rank"])
        y_va = label_fn(df[mask_va]["rank"])
        t0 = time.time()
        win1, fuku, top3p, auc = train_eval(
            X_tr, y_tr, g_tr, X_va, y_va, g_va, df_val, y_win_va, lg
        )
        elapsed = time.time() - t0
        # 全指標がv1より良い場合にマーク
        all_better = win1 > 0.397 and fuku > 0.669 and top3p > 0.509
        mark = " ★全↑" if all_better else ""
        print(f"{name:<20} {win1:>7.1%} {fuku:>7.1%} {top3p:>8.1%} {auc:>8.4f}  ({elapsed:.0f}s){mark}",
              flush=True)
        results.append({"name": name, "label_gain": lg,
                        "win1": win1, "fuku": fuku, "top3p": top3p, "auc": auc})

    # ── 結果まとめ ──────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("全指標ランキング (win1+複勝+top3p の合計スコア)")
    print("=" * 58)
    for r in sorted(results, key=lambda x: x["win1"]+x["fuku"]+x["top3p"], reverse=True)[:5]:
        total = r["win1"] + r["fuku"] + r["top3p"]
        print(f"  {r['name']:<20} 合計={total:.3f}  "
              f"win={r['win1']:.1%} 複勝={r['fuku']:.1%} top3={r['top3p']:.1%}")

    out_path = os.path.join(BASE_DIR, "src", "4level_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果を {out_path} に保存しました。")


if __name__ == "__main__":
    main()
