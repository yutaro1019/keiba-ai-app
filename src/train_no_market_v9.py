"""
no_market_v9 モデル学習スクリプト
v8 からの変更:
  ・SIRE_LINE_MAP 大幅拡張 (AP Indy系=4 を新設, その他=5)
  ・bloodline_cross: 父系統 × 母父系統 の交差特徴量 (win/top3/dist/surface)
  ・horse_frame_win/top3_rate 削除 (重要度低い・コース枠は venue_frame で代用)
  ・sire_line / bms_line の生カテゴリ値削除 (重要度ほぼ0%)
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
OUT_DIR   = os.path.join(BASE_DIR, "models", "no_market_v9")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 血統系統マッピング (v9: 6カテゴリ)
#   0=SS系, 1=MP系, 2=Roberto系, 3=ND系, 4=APIndy系, 5=その他
# ─────────────────────────────────────────────────────────────────
SIRE_LINE_MAP = {
    # ── サンデーサイレンス系 (0) ──────────────────────────────────
    # SS直系
    "ディープインパクト": 0, "ハーツクライ": 0, "ダイワメジャー": 0,
    "ステイゴールド": 0, "ネオユニヴァース": 0, "マンハッタンカフェ": 0,
    "スペシャルウィーク": 0, "フジキセキ": 0, "アグネスタキオン": 0,
    "ゴールドアリュール": 0, "ブラックタイド": 0, "ダンスインザダーク": 0,
    "ゼンノロブロイ": 0, "マーベラスサンデー": 0, "タイキシャトル": 0,
    "スズカマンボ": 0, "スズカフェニックス": 0,
    # SS第2世代
    "キズナ": 0, "ジャスタウェイ": 0, "ヴィクトワールピサ": 0,
    "オルフェーヴル": 0, "キンシャサノキセキ": 0, "ゴールドシップ": 0,
    "ディープブリランテ": 0, "メイショウボーラー": 0, "シルバーステート": 0,
    "ミッキーアイル": 0, "キタサンブラック": 0, "イスラボニータ": 0,
    "リアルスティール": 0, "カレンブラックヒル": 0, "リアルインパクト": 0,
    "スマートファルコン": 0, "エスポワールシチー": 0, "コパノリッキー": 0,
    "サトノアラジン": 0, "ワールドエース": 0, "サトノダイヤモンド": 0,
    "ダノンシャンティ": 0, "ダノンバラード": 0, "フェノーメノ": 0,
    "ディープスカイ": 0, "リオンディーズ": 0, "ダノンキングリー": 0,
    "コントレイル": 0, "グランアレグリア": 0, "フィエールマン": 0,
    "サリオス": 0, "スワーヴリチャード": 0, "ディーマジェスティ": 0,
    "スピルバーグ": 0, "トーセンラー": 0, "エイシンヒカリ": 0,
    "ローズキングダム": 0, "グランプリボス": 0, "ロゴタイプ": 0,
    "ヴァンセンヌ": 0, "シュヴァルグラン": 0, "アンライバルド": 0,
    "グレーターロンドン": 0, "ロジャーバローズ": 0, "アルアイン": 0,
    "アドマイヤマーズ": 0, "サートゥルナーリア": 0,  # ← 実際はRoberto経由だが近年扱い
    "ドリームジャーニー": 0, "ナカヤマフェスタ": 0,
    "リーチザクラウン": 0, "トーセンホマレボシ": 0,
    "キタサンブラック": 0,
    # SS孫以降で重要なもの
    "ビッグウィーク": 0,
    # ── キングマンボ/ミスタープロスペクター系 (1) ──────────────
    "ロードカナロア": 1, "ルーラーシップ": 1, "キングカメハメハ": 1,
    "ドゥラメンテ": 1, "エイシンフラッシュ": 1, "ホッコータルマエ": 1,
    "アドマイヤムーン": 1, "サウスヴィグラス": 1, "エンパイアメーカー": 1,
    "マクフィ": 1, "ワークフォース": 1, "キングズベスト": 1,
    "スウェプトオーヴァーボード": 1, "プリサイスエンド": 1,
    "アイルハヴアナザー": 1, "マインドユアビスケッツ": 1,
    "ラブリーデイ": 1, "レイデオロ": 1, "ダノンプレミアム": 1,
    "アドマイヤコジーン": 1, "ベルシャザール": 1, "トゥザグローリー": 1,
    "トゥザワールド": 1, "ヴァーミリアン": 1, "タイムパラドックス": 1,
    "ダノンレジェンド": 1, "ファインニードル": 1, "レッドファルクス": 1,
    "ビーチパトロール": 1, "モンテロッソ": 1, "アグネスデジタル": 1,
    "ケイムホーム": 1, "ニューイヤーズデイ": 1, "ストリートセンス": 1,
    "ストーミングホーム": 1, "タワーオブロンドン": 1,
    "フォーウィールドライブ": 1, "サマーバード": 1,
    # ── ロベルト系 (2) ────────────────────────────────────────
    "スクリーンヒーロー": 2, "シンボリクリスエス": 2, "マツリダゴッホ": 2,
    "エピファネイア": 2, "モーリス": 2, "フリオーソ": 2,
    "グラスワンダー": 2, "ブライアンズタイム": 2, "タニノギムレット": 2,
    "ウインバリアシオン": 2, "ナダル": 2,
    # ── ノーザンダンサー系 (3) ────────────────────────────────
    "ハービンジャー": 3, "ヘニーヒューズ": 3, "クロフネ": 3,
    "ドレフォン": 3, "ダンカーク": 3, "ストロングリターン": 3,
    "アジアエクスプレス": 3, "ディスクリートキャット": 3,
    "メイショウサムソン": 3, "タートルボウル": 3,
    "デクラレーションオブウォー": 3, "ローエングリン": 3,
    "アメリカンペイトリオット": 3, "ブリックスアンドモルタル": 3,
    "ヨハネスブルグ": 3, "ノヴェリスト": 3,
    "ジョーカプチーノ": 3, "バトルプラン": 3, "トランセンド": 3,
    "キングヘイロー": 3, "サトノクラウン": 3, "カネヒキリ": 3,
    "フレンチデピュティ": 3, "ロージズインメイ": 3, "シャンハイボビー": 3,
    "エスケンデレヤ": 3, "ザファクター": 3, "サンダースノー": 3,
    "ケープブランコ": 3, "トビーズコーナー": 3, "タリスマニック": 3,
    "モーニン": 3, "スズカコーズウェイ": 3, "モズアスコット": 3,
    "ストームキャット": 3, "ジャイアンツコーズウェイ": 3,
    "ウォーフロント": 3, "カラヴァッジオ": 3,
    # ── A.P.Indy系 (4) ── 米国ダート/スプリント血統 ──────────
    "シニスターミニスター": 4, "パイロ": 4, "マジェスティックウォリアー": 4,
    "カジノドライヴ": 4, "カリフォルニアクローム": 4, "クリエイター": 4,
    "ラニ": 4, "ルヴァンスレーヴ": 4,
}

LINE_NAMES = {0: "SS", 1: "MP", 2: "Roberto", 3: "ND", 4: "APIndy", 5: "other"}
N_LINES = 6  # 0-5

def normalize_sire_name(name) -> str:
    if pd.isna(name): return ""
    name = re.sub(r"\d{4}$", "", str(name))
    name = re.sub(r"[A-Za-z''\s]+$", "", name)
    return name.strip()

def get_sire_line(name) -> int:
    return SIRE_LINE_MAP.get(normalize_sire_name(name), 5)  # 5=その他


def encode_race_class(cls) -> float:
    if pd.isna(cls): return np.nan
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
    return 2.0


def add_v9_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # ── 枠番グループ × 競馬場 (コースレベルのみ、馬個人は削除) ───────
    if "frame_no" in df.columns and "venue" in df.columns:
        fn = df["frame_no"].fillna(0)
        df["_frame_grp"] = np.where(fn <= 2, 0, np.where(fn <= 6, 1, 2)).astype("int8")
        df["_win"]  = (df["rank"] == 1).astype(float)
        df["_top3"] = (df["rank"] <= 3).astype(float)
        vf = df.groupby(["venue", "_frame_grp"], sort=False, observed=True)
        vf_runs  = vf.cumcount()
        df["venue_frame_win_rate"]  = (
            (vf["_win"].cumsum()  - df["_win"])  / vf_runs.replace(0, np.nan)
        ).astype("float32")
        df["venue_frame_top3_rate"] = (
            (vf["_top3"].cumsum() - df["_top3"]) / vf_runs.replace(0, np.nan)
        ).astype("float32")
        df.drop(columns=["_frame_grp", "_win", "_top3"], inplace=True)

    # ── 昇降級 ──────────────────────────────────────────────────
    if "race_class" in df.columns and "horse_id" in df.columns:
        df["_cls_level"] = df["race_class"].map(encode_race_class).astype("float32")
        df["_cls_lag1"] = (
            df.sort_values(["horse_id", "date"])
              .groupby("horse_id", sort=False)["_cls_level"]
              .shift(1)
        )
        df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
        df["class_change"] = (df["_cls_level"] - df["_cls_lag1"]).astype("float32")
        df.drop(columns=["_cls_level", "_cls_lag1"], inplace=True)

    # ── 血統系統 + 交差特徴量 ────────────────────────────────────
    df["_win2"]   = (df["rank"] == 1).astype(float)
    df["_top3_2"] = (df["rank"] <= 3).astype(float)

    # 距離カテゴリ
    df["_dist_cat"] = pd.cut(
        df["distance"], bins=DIST_BINS, labels=DIST_LABELS
    ).astype("float32")

    for col_src, line_col in [("sire", "sire_line"), ("broodmare_sire", "bms_line")]:
        if col_src not in df.columns:
            continue
        df[line_col] = df[col_src].map(get_sire_line).astype("float32")

        # 系統全体累積成績
        grp = df.groupby(line_col, sort=False, observed=True)
        runs  = grp.cumcount()
        df[f"{line_col}_win_rate"]  = (
            (grp["_win2"].cumsum()   - df["_win2"])  / runs.replace(0, np.nan)
        ).astype("float32")
        df[f"{line_col}_top3_rate"] = (
            (grp["_top3_2"].cumsum() - df["_top3_2"]) / runs.replace(0, np.nan)
        ).astype("float32")

        # 芝ダート別
        if "surface" in df.columns:
            g2 = df.groupby([line_col, "surface"], sort=False, observed=True)
            r2 = g2.cumcount()
            df[f"{line_col}_surface_top3_rate"] = (
                (g2["_top3_2"].cumsum() - df["_top3_2"]) / r2.replace(0, np.nan)
            ).astype("float32")

        # 距離カテゴリ別
        g3 = df.groupby([line_col, "_dist_cat"], sort=False, observed=True)
        r3 = g3.cumcount()
        df[f"{line_col}_dist_top3_rate"] = (
            (g3["_top3_2"].cumsum() - df["_top3_2"]) / r3.replace(0, np.nan)
        ).astype("float32")

    # ── 父系統 × 母父系統 交差特徴量 ─────────────────────────────
    if "sire_line" in df.columns and "bms_line" in df.columns:
        # cross_id: 0-35 (6×6)
        df["_cross"] = (df["sire_line"] * N_LINES + df["bms_line"]).astype("float32")

        gc_cross = df.groupby("_cross", sort=False, observed=True)
        rc = gc_cross.cumcount()
        df["bloodline_cross_win_rate"]  = (
            (gc_cross["_win2"].cumsum()   - df["_win2"])  / rc.replace(0, np.nan)
        ).astype("float32")
        df["bloodline_cross_top3_rate"] = (
            (gc_cross["_top3_2"].cumsum() - df["_top3_2"]) / rc.replace(0, np.nan)
        ).astype("float32")

        # 芝ダート別
        if "surface" in df.columns:
            gc2 = df.groupby(["_cross", "surface"], sort=False, observed=True)
            rc2 = gc2.cumcount()
            df["bloodline_cross_surface_top3_rate"] = (
                (gc2["_top3_2"].cumsum() - df["_top3_2"]) / rc2.replace(0, np.nan)
            ).astype("float32")

        # 距離カテゴリ別
        gc3 = df.groupby(["_cross", "_dist_cat"], sort=False, observed=True)
        rc3 = gc3.cumcount()
        df["bloodline_cross_dist_top3_rate"] = (
            (gc3["_top3_2"].cumsum() - df["_top3_2"]) / rc3.replace(0, np.nan)
        ).astype("float32")

        df.drop(columns=["_cross"], inplace=True)

    df.drop(columns=["_win2", "_top3_2", "_dist_cat"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 特徴量リスト
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS_V9 = [
    # v6 コア
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
    # v9 (venue_frame=コースレベルのみ、horse_frameは削除)
    "venue_frame_win_rate","venue_frame_top3_rate",
    "class_change",
    # 系統別統計 (生カテゴリ値sire_line/bms_lineは削除)
    "sire_line_win_rate","sire_line_top3_rate",
    "sire_line_surface_top3_rate","sire_line_dist_top3_rate",
    "bms_line_win_rate","bms_line_top3_rate",
    "bms_line_surface_top3_rate","bms_line_dist_top3_rate",
    # 父系×母父系 交差特徴量
    "bloodline_cross_win_rate","bloodline_cross_top3_rate",
    "bloodline_cross_surface_top3_rate","bloodline_cross_dist_top3_rate",
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

    # カバー率を事前表示
    df["_sn"] = df["sire"].map(normalize_sire_name) if "sire" in df.columns else ""
    df["_sl"] = df["_sn"].map(lambda x: SIRE_LINE_MAP.get(x, 5))
    total = len(df[df["_sn"] != ""])
    print(">>> Sire line coverage:")
    for line, name in LINE_NAMES.items():
        cnt = (df["_sl"] == line).sum()
        print(f"   {name:8s}: {cnt:>7,}行 ({cnt/total:.1%})")
    df.drop(columns=["_sn","_sl"], inplace=True)

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
        ("v9",            add_v9_features),
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
        df["jh_top3_cum"] = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]     = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_w", "_t"], inplace=True)

    avail = [c for c in FEATURE_COLS_V9 if c in df.columns]
    missing = [c for c in FEATURE_COLS_V9 if c not in df.columns]
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

    for i, (seed, leaves) in enumerate([(42,63),(7,95),(2024,47)], 1):
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
    score  = 0.6 * p_win + 0.4 * p_top3
    val_df["score"] = score

    def calc_hit_rates(vdf):
        win1 = fuku = pred3 = pred3_cnt = n = 0
        for _, g in vdf.groupby("race_id"):
            best = g["score"].idxmax()
            win1 += int(g.loc[best, "rank"] == 1)
            fuku += int(g.loc[best, "rank"] <= 3)
            top3_idx = g.nlargest(3, "score").index
            actual_top3 = set(g[g["rank"] <= 3].index)
            pred3 += len(set(top3_idx) & actual_top3)
            pred3_cnt += 3; n += 1
        return win1/n, fuku/n, pred3/pred3_cnt

    win1, fuku, top3p = calc_hit_rates(val_df)

    print("=" * 65)
    print(f"              v8(107個)   v9({len(avail)}個)")
    print(f"  1位的中率:    33.3%       {win1:.1%}")
    print(f"  複勝率:       62.9%       {fuku:.1%}")
    print(f"  予想3頭精度:   50.0%       {top3p:.1%}")
    print(f"  TOP3 AUC:    0.7923      {roc_auc_score(y_top3[mask_val], p_top3):.4f}")
    print(f"  WIN AUC:     0.8234      {roc_auc_score(y_win[mask_val], p_win):.4f}")
    print("=" * 65)

    imp = {}
    for m in models_top3 + models_win:
        for feat, val in zip(m.feature_name(), m.feature_importance("gain")):
            imp[feat] = imp.get(feat, 0) + val
    total_imp = sum(imp.values())
    ranked = sorted(imp.items(), key=lambda x: -x[1])

    v9_new = {
        "venue_frame_win_rate","venue_frame_top3_rate","class_change",
        "sire_line_win_rate","sire_line_top3_rate","sire_line_surface_top3_rate","sire_line_dist_top3_rate",
        "bms_line_win_rate","bms_line_top3_rate","bms_line_surface_top3_rate","bms_line_dist_top3_rate",
        "bloodline_cross_win_rate","bloodline_cross_top3_rate",
        "bloodline_cross_surface_top3_rate","bloodline_cross_dist_top3_rate",
    }
    print("\n特徴量重要度:")
    for rank_i, (feat, val) in enumerate(ranked, 1):
        tag = " [NEW]" if feat in v9_new else ""
        print(f" {rank_i:3d}. {feat:<52s} {val/total_imp:.2%}{tag}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, m in enumerate(models_top3):
        m.save_model(os.path.join(OUT_DIR, f"top3_{i}.lgb"))
    for i, m in enumerate(models_win):
        m.save_model(os.path.join(OUT_DIR, f"win_{i}.lgb"))

    meta = {
        "version": "no_market_v9",
        "feature_engineering": "v9",
        "features": avail,
        "n_top3_models": len(models_top3),
        "n_win_models":  len(models_win),
        "sire_line_map_size": len(SIRE_LINE_MAP),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}", flush=True)

    print("\n>>> Regenerating lookup tables for v9...", flush=True)
    from no_market_v4_lookups import regenerate_v4_lookups
    lmeta = regenerate_v4_lookups(model_dir=OUT_DIR)
    for k, v in lmeta.get("lookup_counts", {}).items():
        print(f"   {k}: {v:,}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
