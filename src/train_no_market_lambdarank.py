"""
no_market_lambdarank モデル学習スクリプト
===========================================
【なぜLambdaRankか】
  binary分類(win/top3)は全レース混合でのグローバル識別を最適化しており、
  「同一レース内での相対順位」を損失関数レベルで扱っていない。
  LambdaRankは「レース内の全ペアを正しく並べることでNDCGがどれだけ改善するか」を
  勾配に使うため、的中率(=レース内1位当て)と直接整合する。

【実装方針】
  - 特徴量はv10と同じ107個
  - ラベル: 1位→2, 2-3位→1, 4位以下→0  (label_gain=[0,1,3]で1位を3倍重視)
  - NDCG@1をメイン指標、NDCG@3もモニター
  - 3シードアンサンブル (スコアを平均)
  - 出力は確率ではなくスコア(レース内大小のみ有効)
  - 既存のbinaryモデルとはblendせず単独モードとして独立させる
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
OUT_DIR   = os.path.join(BASE_DIR, "models", "no_market_lambdarank")
os.makedirs(OUT_DIR, exist_ok=True)

# v10 と同じ特徴量リスト
FEATURE_COLS = [
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
]

# ラベル設計: 1位=2, 2-3位=1, 4位以下=0
# label_gain=[0,1,3] → NDCGで1位のゲインを3倍重視
def make_label(rank_series: pd.Series) -> np.ndarray:
    return np.where(rank_series == 1, 2,
           np.where(rank_series <= 3, 1, 0)).astype(np.int32)


def calc_hit_rates(df_val: pd.DataFrame, score_col: str = "lr_score"):
    """レース内スコア最大の馬が1位/3位以内かを集計"""
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

    # ── 特徴量構築 ──────────────────────────────────────────────────
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
    df["_w"] = (df["rank"] == 1).astype(float)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"]  = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]      = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_w", "_t"], inplace=True)

    # ── LambdaRank用データ準備 ──────────────────────────────────────
    # 重要: race_idでソート → 同一レースの全馬が連続している必要がある
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rank"])
    df = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)

    avail = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"   [WARN] missing ({len(missing)}): {missing}", flush=True)

    mask_train = df["date"].dt.year < 2025
    mask_val   = df["date"].dt.year == 2025

    y_label = make_label(df["rank"])  # 0/1/2 ラベル
    y_win   = (df["rank"] == 1).astype(int)  # AUC比較用
    X = df[avail]

    # グループ (= 各レースの頭数リスト、データ順に対応)
    def make_groups(mask):
        return df[mask].groupby("race_id", sort=False)["race_id"].count().tolist()

    groups_train = make_groups(mask_train)
    groups_val   = make_groups(mask_val)

    n_train_races = len(groups_train)
    n_val_races   = len(groups_val)
    print(f"\nFeatures: {len(avail)}")
    print(f"Train: {mask_train.sum():,}行 / {n_train_races:,}レース")
    print(f"Val  : {mask_val.sum():,}行 / {n_val_races:,}レース\n")

    lgb_params = dict(
        objective       = "lambdarank",
        metric          = "ndcg",
        eval_at         = [1, 3],      # NDCG@1, NDCG@3
        label_gain      = [0, 1, 3],   # label 0→0点, 1→1点, 2→3点 (1位を3倍重視)
        verbosity       = -1,
        learning_rate   = 0.05,
        num_leaves      = 63,
        min_child_samples = 20,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
    )

    # ── 3シードアンサンブル ─────────────────────────────────────────
    models = []
    seeds  = [42, 7, 2024]

    for i, seed in enumerate(seeds, 1):
        params = {**lgb_params, "seed": seed, "random_state": seed}
        print(f">>> [{i}/3] lambdarank seed={seed} ...", flush=True)

        tr_ds = lgb.Dataset(
            X[mask_train], label=y_label[mask_train], group=groups_train
        )
        va_ds = lgb.Dataset(
            X[mask_val], label=y_label[mask_val], group=groups_val,
            reference=tr_ds
        )
        m = lgb.train(
            params, tr_ds, num_boost_round=3000,
            valid_sets=[va_ds],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)],
        )
        # 参考: win AUCも表示（binaryとの比較用）
        raw = m.predict(X[mask_val])
        # レース内でsoftmax正規化して確率様スコアに変換
        val_tmp = df[mask_val].copy()
        val_tmp["_raw"] = raw
        def _softmax_norm(g):
            e = np.exp(g["_raw"] - g["_raw"].max())
            return e / e.sum()
        val_tmp["_prob"] = val_tmp.groupby("race_id", group_keys=False).apply(_softmax_norm)
        auc_win = roc_auc_score(y_win[mask_val], val_tmp["_prob"])
        print(f"   AUC(win, softmax)={auc_win:.4f}", flush=True)
        models.append(m)

    # ── アンサンブルスコアで評価 ─────────────────────────────────────
    raw_scores = np.mean([m.predict(X[mask_val]) for m in models], axis=0)

    val_df = df[mask_val].copy()
    val_df["lr_score"] = raw_scores

    # レース内softmax → 確率様スコア (AUC計算用)
    def softmax_in_race(g):
        e = np.exp(g["lr_score"] - g["lr_score"].max())
        return e / e.sum()
    val_df["lr_prob"] = val_df.groupby("race_id", group_keys=False).apply(softmax_in_race)

    win1, fuku, top3p = calc_hit_rates(val_df, score_col="lr_score")
    auc_win_ens = roc_auc_score(y_win[mask_val], val_df["lr_prob"])

    # ── 比較表示 ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"              v10-binary   LambdaRank")
    print(f"  1位的中率:    33.1%       {win1:.1%}")
    print(f"  複勝率:       62.9%       {fuku:.1%}")
    print(f"  予想3頭精度:   50.0%       {top3p:.1%}")
    print(f"  WIN AUC:     0.8246      {auc_win_ens:.4f}  (softmax後)")
    print("=" * 65)

    # ── 特徴量重要度 ────────────────────────────────────────────────
    imp = {}
    for m in models:
        for feat, val in zip(m.feature_name(), m.feature_importance("gain")):
            imp[feat] = imp.get(feat, 0) + val
    total_imp = sum(imp.values())
    print("\n特徴量重要度:")
    for rank_i, (feat, val) in enumerate(sorted(imp.items(), key=lambda x: -x[1]), 1):
        print(f" {rank_i:3d}. {feat:<52s} {val/total_imp:.2%}")

    # ── 保存 ────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(os.path.join(OUT_DIR, f"rank_{i}.lgb"))

    meta = {
        "version": "no_market_lambdarank",
        "feature_engineering": "v10",
        "model_type": "lambdarank",
        "label_design": "1st=2, 2nd-3rd=1, 4th+=0",
        "label_gain": [0, 1, 3],
        "eval_at": [1, 3],
        "features": avail,
        "n_rank_models": len(models),
        "sire_line_categories": LINE_NAMES,
        "n_lines": N_LINES,
        "hit_rate_win1": float(win1),
        "hit_rate_fuku": float(fuku),
        "hit_rate_top3pred": float(top3p),
        "win_auc_softmax": float(auc_win_ens),
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
