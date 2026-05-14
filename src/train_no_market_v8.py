"""
no_market_v8 モデル学習スクリプト
v7 からの追加:
  ② 枠番適性: venue × frame_group(内/中/外) の勝率・複勝率
  ③ 昇降級: 前走からのクラス変化方向
  ④ 血統系統: sire_line(SS/MP/Roberto/ND/other) × 距離カテゴリ・芝ダート別成績
     ※ time_index が全NaN のため着差特徴量は未実装
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL  = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
OUT_DIR   = os.path.join(BASE_DIR, "models", "no_market_v8")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 血統系統マッピング
# ─────────────────────────────────────────────────────────────────
# 値: 0=SS系, 1=MP系(キングマンボ/ミスタープロスペクター), 2=Roberto系, 3=ND系, 4=その他
SIRE_LINE_MAP = {
    # サンデーサイレンス系 (0)
    "ディープインパクト": 0, "ハーツクライ": 0, "ダイワメジャー": 0,
    "キズナ": 0, "ブラックタイド": 0, "ジャスタウェイ": 0,
    "ヴィクトワールピサ": 0, "オルフェーヴル": 0, "キンシャサノキセキ": 0,
    "ゴールドシップ": 0, "ゴールドアリュール": 0, "ステイゴールド": 0,
    "ディープブリランテ": 0, "メイショウボーラー": 0, "シルバーステート": 0,
    "ミッキーアイル": 0, "ネオユニヴァース": 0, "マンハッタンカフェ": 0,
    "キタサンブラック": 0, "イスラボニータ": 0, "リアルスティール": 0,
    "ゼンノロブロイ": 0, "カレンブラックヒル": 0, "リアルインパクト": 0,
    "スマートファルコン": 0, "エスポワールシチー": 0, "コパノリッキー": 0,
    "サトノアラジン": 0, "ワールドエース": 0, "サトノダイヤモンド": 0,
    "ダノンシャンティ": 0, "ダノンバラード": 0, "フジキセキ": 0,
    "スペシャルウィーク": 0, "アグネスタキオン": 0, "マーベラスサンデー": 0,
    "ダンスインザダーク": 0, "フェノーメノ": 0, "トーセンジョーダン": 0,
    "ビッグウィーク": 0, "ディープスカイ": 0, "リオンディーズ": 0,
    "サトノクラウン": 0, "ダノンキングリー": 0, "コントレイル": 0,
    "グランアレグリア": 0, "フィエールマン": 0, "サリオス": 0,
    # ミスタープロスペクター/キングマンボ系 (1)
    "ロードカナロア": 1, "ルーラーシップ": 1, "キングカメハメハ": 1,
    "ドゥラメンテ": 1, "エイシンフラッシュ": 1, "ホッコータルマエ": 1,
    "アドマイヤムーン": 1, "サウスヴィグラス": 1, "エンパイアメーカー": 1,
    "マクフィ": 1, "ワークフォース": 1, "キングズベスト": 1,
    "スウェプトオーヴァーボード": 1, "プリサイスエンド": 1,
    "アイルハヴアナザー": 1, "マインドユアビスケッツ": 1,
    "ラブリーデイ": 1, "レイデオロ": 1, "ダノンプレミアム": 1,
    "アドマイヤコジーン": 1, "フラムドグロワール": 1,
    # ロベルト系 (2)
    "スクリーンヒーロー": 2, "シンボリクリスエス": 2, "マツリダゴッホ": 2,
    "エピファネイア": 2, "モーリス": 2, "フリオーソ": 2,
    "グラスワンダー": 2, "ブライアンズタイム": 2, "タニノギムレット": 2,
    "ウインバリアシオン": 2,
    # ノーザンダンサー系 (3)
    "ハービンジャー": 3, "ヘニーヒューズ": 3, "クロフネ": 3,
    "ドレフォン": 3, "ダンカーク": 3, "ストロングリターン": 3,
    "アジアエクスプレス": 3, "ディスクリートキャット": 3,
    "メイショウサムソン": 3, "タートルボウル": 3,
    "デクラレーションオブウォー": 3, "ローエングリン": 3,
    "アメリカンペイトリオット": 3, "ブリックスアンドモルタル": 3,
    "ヨハネスブルグ": 3, "ノヴェリスト": 3,
    "ストームキャット": 3, "ジャイアンツコーズウェイ": 3,
    "ウォーフロント": 3, "カラヴァッジオ": 3,
}

def normalize_sire_name(name: str) -> str:
    """年号・英語サフィックスを除去して種牡馬名を正規化"""
    if pd.isna(name):
        return ""
    # 末尾の半角数字4桁(年)を除去
    name = re.sub(r"\d{4}$", "", str(name))
    # 末尾の英字サフィックス(スペース区切りなし)を除去
    name = re.sub(r"[A-Za-z''\s]+$", "", name)
    return name.strip()

def get_sire_line(name: str) -> int:
    key = normalize_sire_name(name)
    return SIRE_LINE_MAP.get(key, 4)  # 4 = その他

# レースクラスの順序エンコード
CLASS_LEVEL = {
    "新馬": 0, "未勝利": 0,
    "500万": 1, "1勝": 1,
    "1000万": 2, "2勝": 2,
    "1600万": 3, "3勝": 3,
    "オープン": 4, "リステッド": 4,
}

def encode_race_class(cls) -> float:
    if pd.isna(cls):
        return np.nan
    s = str(cls)
    if "G1" in s or "ＧⅠ" in s: return 7.0
    if "G2" in s or "ＧⅡ" in s: return 6.0
    if "G3" in s or "ＧⅢ" in s: return 5.0
    if "リステッド" in s:         return 4.5
    if "オープン" in s:           return 4.0
    if "3勝" in s or "1600万" in s: return 3.0
    if "2勝" in s or "1000万" in s: return 2.0
    if "1勝" in s or "500万" in s:  return 1.0
    if "新馬" in s or "未勝利" in s: return 0.0
    return 2.0  # fallback


def add_v8_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # ── ② 枠番グループ × 競馬場 ──────────────────────────────────────
    if "frame_no" in df.columns and "venue" in df.columns:
        fn = df["frame_no"].fillna(0)
        df["_frame_grp"] = np.where(fn <= 2, 0, np.where(fn <= 6, 1, 2)).astype("int8")

        df["_win"]  = (df["rank"] == 1).astype(float)
        df["_top3"] = (df["rank"] <= 3).astype(float)

        # venue × frame_group 全体累積勝率
        vf = df.groupby(["venue", "_frame_grp"], sort=False, observed=True)
        vf_runs  = vf.cumcount()
        vf_wins  = vf["_win"].cumsum()  - df["_win"]
        vf_top3s = vf["_top3"].cumsum() - df["_top3"]
        df["venue_frame_win_rate"]  = (vf_wins  / vf_runs.replace(0, np.nan)).astype("float32")
        df["venue_frame_top3_rate"] = (vf_top3s / vf_runs.replace(0, np.nan)).astype("float32")

        # 馬個人の枠グループ成績
        hf = df.groupby(["horse_id", "_frame_grp"], sort=False, observed=True)
        hf_runs  = hf.cumcount()
        hf_wins  = hf["_win"].cumsum()  - df["_win"]
        hf_top3s = hf["_top3"].cumsum() - df["_top3"]
        df["horse_frame_win_rate"]  = (hf_wins  / hf_runs.replace(0, np.nan)).astype("float32")
        df["horse_frame_top3_rate"] = (hf_top3s / hf_runs.replace(0, np.nan)).astype("float32")

        df.drop(columns=["_frame_grp", "_win", "_top3"], inplace=True)

    # ── ③ 昇降級 ─────────────────────────────────────────────────
    if "race_class" in df.columns and "horse_id" in df.columns:
        df["_cls_level"] = df["race_class"].map(encode_race_class).astype("float32")
        df["_cls_lag1"] = (
            df.sort_values(["horse_id", "date"])
              .groupby("horse_id", sort=False)["_cls_level"]
              .shift(1)
        )
        df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
        # class_change: + = 昇級, 0 = 同, - = 降級
        df["class_change"] = (df["_cls_level"] - df["_cls_lag1"]).astype("float32")
        df.drop(columns=["_cls_level", "_cls_lag1"], inplace=True)

    # ── ④ 血統系統 ────────────────────────────────────────────────
    for col_name, line_col in [("sire", "sire_line"), ("broodmare_sire", "bms_line")]:
        if col_name not in df.columns:
            continue

        df[line_col] = df[col_name].map(get_sire_line).astype("float32")
        df["_win2"]  = (df["rank"] == 1).astype(float)
        df["_top3_2"] = (df["rank"] <= 3).astype(float)

        # 系統全体累積成績
        grp = df.groupby(line_col, sort=False, observed=True)
        runs  = grp.cumcount()
        wins  = grp["_win2"].cumsum()  - df["_win2"]
        top3s = grp["_top3_2"].cumsum() - df["_top3_2"]
        df[f"{line_col}_win_rate"]  = (wins  / runs.replace(0, np.nan)).astype("float32")
        df[f"{line_col}_top3_rate"] = (top3s / runs.replace(0, np.nan)).astype("float32")

        # 芝/ダート別
        if "surface" in df.columns:
            grp2 = df.groupby([line_col, "surface"], sort=False, observed=True)
            r2 = grp2.cumcount()
            t2 = grp2["_top3_2"].cumsum() - df["_top3_2"]
            df[f"{line_col}_surface_top3_rate"] = (t2 / r2.replace(0, np.nan)).astype("float32")

        # 距離カテゴリ別
        if "_dist_cat_v8" not in df.columns:
            df["_dist_cat_v8"] = pd.cut(
                df["distance"], bins=DIST_BINS, labels=DIST_LABELS
            ).astype("float32")
        grp3 = df.groupby([line_col, "_dist_cat_v8"], sort=False, observed=True)
        r3 = grp3.cumcount()
        t3 = grp3["_top3_2"].cumsum() - df["_top3_2"]
        df[f"{line_col}_dist_top3_rate"] = (t3 / r3.replace(0, np.nan)).astype("float32")

        df.drop(columns=["_win2", "_top3_2"], inplace=True)

    if "_dist_cat_v8" in df.columns:
        df.drop(columns=["_dist_cat_v8"], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────
# モデル学習
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS_V8 = [
    # v6まで
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
    # v7
    "horse_going_top3_rate_vs_field_diff","expected_pace_fit",
    "horse_going_win_rate","horse_going_top3_rate",
    "horse_dist_cat_top3_rate","horse_dist_cat_win_rate",
    # v8 new
    "venue_frame_win_rate","venue_frame_top3_rate",
    "horse_frame_win_rate","horse_frame_top3_rate",
    "class_change",
    "sire_line","sire_line_win_rate","sire_line_top3_rate",
    "sire_line_surface_top3_rate","sire_line_dist_top3_rate",
    "bms_line","bms_line_win_rate","bms_line_top3_rate",
    "bms_line_surface_top3_rate","bms_line_dist_top3_rate",
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

    print(">>> Adding sire/class...", flush=True);  df = add_sire_features(df)
    print(">>> Adding horse rolling...", flush=True); df = add_horse_rolling_features(df)
    print(">>> Adding jockey/trainer...", flush=True); df = add_jockey_trainer_rolling(df)
    print(">>> Adding field strength...", flush=True); df = add_field_strength(df)
    print(">>> Adding weight trend...", flush=True); df = add_weight_trend(df)
    print(">>> Adding lap rolling...", flush=True); df = add_lap_rolling_features(df)
    print(">>> Adding field pace...", flush=True); df = add_field_pace_features(df)
    print(">>> Adding v6 features...", flush=True); df = add_v6_features(df)
    print(">>> Adding v7 features...", flush=True); df = add_v7_features(df)
    print(">>> Adding v8 features...", flush=True); df = add_v8_features(df)

    print(">>> Adding vs_field features...", flush=True)
    df = add_model_features(df, race_col="race_id"); gc.collect()

    print(">>> Adding combo features...", flush=True)
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_w"] = (df["rank"] == 1).astype(float)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"] = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]     = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_w", "_t"], inplace=True)

    # 使える特徴量のみ抽出
    avail = [c for c in FEATURE_COLS_V8 if c in df.columns]
    missing = [c for c in FEATURE_COLS_V8 if c not in df.columns]
    if missing:
        print(f"   [WARN] missing features ({len(missing)}): {missing[:10]}", flush=True)

    # train/val split — v6/v7と同じ: train=~2024末, val=2025のみ
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rank"])
    mask_train = df["date"].dt.year < 2025
    mask_val   = df["date"].dt.year == 2025

    y_top3 = (df["rank"] <= 3).astype(int)
    y_win  = (df["rank"] == 1).astype(int)

    X = df[avail]
    print(f"\nFeatures: {len(avail)}  Train: {mask_train.sum():,}  Val: {mask_val.sum():,}\n", flush=True)

    models_top3, models_win = [], []

    seeds_top3 = [(42, 63), (7, 95), (2024, 47)]
    for i, (seed, leaves) in enumerate(seeds_top3, 1):
        params = {**LGB_PARAMS_TOP3, "num_leaves": leaves, "seed": seed, "random_state": seed}
        print(f">>> [{i}/5] top3 s{seed} l{leaves} ...", flush=True)
        tr = lgb.Dataset(X[mask_train], y_top3[mask_train])
        va = lgb.Dataset(X[mask_val],   y_top3[mask_val], reference=tr)
        m = lgb.train(params, tr, 3000, valid_sets=[va],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)])
        p = m.predict(X[mask_val])
        print(f"   AUC={roc_auc_score(y_top3[mask_val], p):.4f}", flush=True)
        models_top3.append(m)

    p_top3 = np.mean([m.predict(X[mask_val]) for m in models_top3], axis=0)
    print(f"   Ensemble TOP3 AUC={roc_auc_score(y_top3[mask_val], p_top3):.4f}\n", flush=True)

    seeds_win = [42, 7]
    for i, seed in enumerate(seeds_win, 4):
        params = {**LGB_PARAMS_WIN, "seed": seed, "random_state": seed}
        print(f">>> [{i}/5] win s{seed} ...", flush=True)
        tr = lgb.Dataset(X[mask_train], y_win[mask_train])
        va = lgb.Dataset(X[mask_val],   y_win[mask_val], reference=tr)
        m = lgb.train(params, tr, 3000, valid_sets=[va],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)])
        p = m.predict(X[mask_val])
        auc = roc_auc_score(y_win[mask_val], p)
        models_win.append(m)
        if i == 5:
            p_win = np.mean([m.predict(X[mask_val]) for m in models_win], axis=0)
            print(f"   AUC={auc:.4f}  Ensemble WIN={roc_auc_score(y_win[mask_val], p_win):.4f}\n", flush=True)
        else:
            print(f"   AUC={auc:.4f}", flush=True)

    # 的中率計算 (v6/v7と同じ定義)
    val_df = df[mask_val].copy()
    score = 0.6 * np.mean([m.predict(X[mask_val]) for m in models_win], axis=0) + \
            0.4 * np.mean([m.predict(X[mask_val]) for m in models_top3], axis=0)
    val_df["score"] = score

    def calc_hit_rates(vdf):
        win1 = fuku = pred3 = pred3_cnt = n = 0
        for _, g in vdf.groupby("race_id"):
            best = g["score"].idxmax()
            win1  += int(g.loc[best, "rank"] == 1)
            fuku  += int(g.loc[best, "rank"] <= 3)
            top3_idx = g.nlargest(3, "score").index
            actual_top3 = set(g[g["rank"] <= 3].index)
            pred3 += len(set(top3_idx) & actual_top3)
            pred3_cnt += 3
            n += 1
        return win1/n, fuku/n, pred3/pred3_cnt

    win1, fuku, top3p = calc_hit_rates(val_df)

    print("=" * 65)
    print(f"              v7(92個)    v8({len(avail)}個)")
    print(f"  1位的中率:    33.2%       {win1:.1%}")
    print(f"  複勝率:       62.3%       {fuku:.1%}")
    print(f"  予想3頭精度:   50.1%       {top3p:.1%}")
    print(f"  TOP3 AUC:    0.7929      {roc_auc_score(y_top3[mask_val], p_top3):.4f}")
    print(f"  WIN AUC:     0.8251      {roc_auc_score(y_win[mask_val], p_win):.4f}")
    print("=" * 65)

    # 特徴量重要度
    imp = {}
    for m in models_top3 + models_win:
        for feat, val in zip(m.feature_name(), m.feature_importance("gain")):
            imp[feat] = imp.get(feat, 0) + val
    total_imp = sum(imp.values())
    ranked = sorted(imp.items(), key=lambda x: -x[1])

    v8_new = {
        "venue_frame_win_rate","venue_frame_top3_rate",
        "horse_frame_win_rate","horse_frame_top3_rate","class_change",
        "sire_line","sire_line_win_rate","sire_line_top3_rate",
        "sire_line_surface_top3_rate","sire_line_dist_top3_rate",
        "bms_line","bms_line_win_rate","bms_line_top3_rate",
        "bms_line_surface_top3_rate","bms_line_dist_top3_rate",
    }
    print("\n特徴量重要度:")
    for rank_i, (feat, val) in enumerate(ranked, 1):
        tag = " [NEW]" if feat in v8_new else ""
        print(f" {rank_i:2d}. {feat:<50s} {val/total_imp:.2%}{tag}")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, m in enumerate(models_top3):
        m.save_model(os.path.join(OUT_DIR, f"top3_{i}.lgb"))
    for i, m in enumerate(models_win):
        m.save_model(os.path.join(OUT_DIR, f"win_{i}.lgb"))

    meta = {
        "version": "no_market_v8",
        "feature_engineering": "v8",
        "features": avail,
        "n_top3_models": len(models_top3),
        "n_win_models": len(models_win),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUT_DIR}", flush=True)

    # ルックアップテーブル再生成
    print("\n>>> Regenerating lookup tables for v8...", flush=True)
    from no_market_v4_lookups import regenerate_v4_lookups
    lmeta = regenerate_v4_lookups(model_dir=OUT_DIR)
    for k, v in lmeta.items():
        print(f"   {k}: {v:,}" if isinstance(v, int) else f"   {k}: {v}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
