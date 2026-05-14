import sys, io, os, json, lightgbm as lgb, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "no_market_v5")
with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
feats = meta["features"]

m_win  = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_win_s42.txt"))
m_top3 = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_top3_s42.txt"))
fi_win  = pd.Series(m_win.feature_importance(importance_type="gain"), index=feats)
fi_top3 = pd.Series(m_top3.feature_importance(importance_type="gain"), index=feats)
fi = ((fi_win / fi_win.sum() + fi_top3 / fi_top3.sum()) / 2 * 100).round(2)
fi = fi.sort_values(ascending=False)

DESCRIPTIONS = {
    # 対フィールド比較
    "horse_avg_rank_vs_field":            "馬の通算平均着順 − 同レース他馬の平均着順（マイナスほど強い）",
    "horse_recent_avg_rank3_vs_field":    "馬の直近3走平均着順 − 同レース他馬の同指標",
    "jockey_top3_rate_vs_field":          "騎手の複勝率 − 同レース他騎手の平均複勝率",
    "horse_rank_pct_lag1_vs_field":       "前走着順パーセンタイル − 同レース他馬の同指標",
    "horse_recent_top3_rate3_vs_field":   "直近3走複勝率 − 同レース他馬の同指標",
    "horse_top3_rate_vs_field":           "通算複勝率 − 同レース他馬の平均複勝率",
    "jockey_win_rate_vs_field":           "騎手の勝率 − 同レース他騎手の平均勝率",
    "horse_agari_lag1_vs_field":          "前走上がり3F − 同レース他馬の前走平均上がり",
    "trainer_top3_rate_vs_field":         "調教師複勝率 − 同レース他調教師の平均複勝率",
    "horse_win_rate_vs_field":            "通算勝率 − 同レース他馬の平均勝率",
    "horse_dist_top3_rate_vs_field":      "この距離での複勝率 − 同レース他馬の同指標",

    # クラスTE
    "class_hist_win_rate":                "出走クラス（1勝・2勝等）の歴史的勝率（ターゲットエンコーディング）",
    "class_hist_top3_rate":               "出走クラスの歴史的複勝率",
    "class_hist_runs":                    "出走クラスの累計出走数（クラスの学習量）",

    # フィールド
    "field_size":                         "出走頭数（多いほど難しい）",

    # 馬統計（前走・成績）
    "horse_rank_lag1":                    "前走着順",
    "horse_rank_pct_lag1":                "前走着順パーセンタイル（頭数補正済み、0=1着, 1=最下位）",
    "horse_rank_pct_lag2":                "2走前着順パーセンタイル",
    "horse_rank_lag2":                    "2走前着順",
    "horse_avg_rank":                     "通算平均着順",
    "horse_avg_agari":                    "通算平均上がり3F",
    "horse_agari_diff_12":                "前走と2走前の上がり差（調子の変化）",
    "horse_agari_lag2":                   "2走前の上がり3F",
    "horse_agari_vs_avg":                 "前走上がり − 自馬の通算平均上がり",
    "horse_front_style":                  "脚質スコア（逃げ=低, 追込=高, 通過順位から計算）",
    "horse_passing_first_rate_lag1":      "前走の通過順位（1コーナー）/ 頭数（位置取り指標）",
    "horse_passing_last_rate_lag1":       "前走の最終通過順位 / 頭数",
    "horse_passing_last_rate_lag2":       "2走前の最終通過順位 / 頭数",
    "horse_distance_diff_lag1":           "今回距離 − 前走距離（距離変化）",
    "horse_surface_top3_rate":            "今回馬場（芝/ダ）での複勝率",
    "horse_surface_win_rate":             "今回馬場（芝/ダ）での勝率",
    "horse_avg_rank_vs_field":            "馬の通算平均着順 − 同レース他馬の平均着順",  # 再掲

    # ローリング統計
    "horse_recent5_avg_rank":             "直近5走の平均着順",
    "horse_recent10_avg_rank":            "直近10走の平均着順",
    "horse_recent_avg_rank3":             "直近3走の平均着順",
    "horse_recent5_top3_rate":            "直近5走の複勝率",
    "horse_recent10_top3_rate":           "直近10走の複勝率",
    "horse_recent5_win_rate":             "直近5走の勝率",
    "horse_bw_trend_3":                   "馬体重の3走前比較（増減トレンド）",

    # 騎手
    "jockey_top3_rate":                   "騎手の通算複勝率",
    "jockey_win_rate":                    "騎手の通算勝率",
    "jockey_venue_top3_rate":             "騎手のこの会場での複勝率",
    "jockey_venue_win_rate":              "騎手のこの会場での勝率",
    "jockey_course_top3_rate":            "騎手のこのコース（芝/ダ・距離帯）での複勝率",
    "jockey_runs":                        "騎手の通算出走数（経験値）",
    "jockey_recent20_win_rate":           "騎手の直近20走勝率",
    "jockey_recent20_top3_rate":          "騎手の直近20走複勝率",
    "jockey_recent50_win_rate":           "騎手の直近50走勝率",
    "field_avg_jockey_win_rate":          "同レース他騎手の平均勝率（対戦騎手の総合強度）",

    # 調教師
    "trainer_top3_rate":                  "調教師の通算複勝率",
    "trainer_win_rate":                   "調教師の通算勝率",
    "trainer_venue_top3_rate":            "調教師のこの会場での複勝率",
    "trainer_venue_win_rate":             "調教師のこの会場での勝率",
    "trainer_recent20_win_rate":          "調教師の直近20走勝率",
    "trainer_recent20_top3_rate":         "調教師の直近20走複勝率",
    "field_avg_trainer_top3_rate":        "同レース他調教師の平均複勝率（対戦調教師の強度）",

    # 種牡馬・母父TE
    "sire_hist_win_rate":                 "種牡馬（父）の歴史的勝率",
    "sire_hist_top3_rate":                "種牡馬（父）の歴史的複勝率",
    "sire_hist_runs":                     "種牡馬（父）の累計出走数",
    "sire_dist_win_rate":                 "種牡馬（父）のこの距離帯での勝率",
    "broodmare_sire_hist_win_rate":       "母父の歴史的勝率",
    "broodmare_sire_hist_top3_rate":      "母父の歴史的複勝率",
    "broodmare_sire_hist_runs":           "母父の累計出走数",
    "broodmare_sire_dist_win_rate":       "母父のこの距離帯での勝率",

    # フィールド強度
    "field_avg_horse_rank":               "同レース他馬の通算平均着順（対戦相手の実力平均）",
    "field_avg_horse_top3_rate":          "同レース他馬の通算複勝率（対戦相手の強度）",

    # その他
    "body_weight":                        "馬体重（kg）",
    "days_since_last":                    "前走からの休養日数",
    "weight_burden_ratio":                "斤量 / 馬体重（体重比負担）",
    "age":                                "馬齢",
    "round_no":                           "レース番号（1〜12R、時間帯の影響）",

    # コンボ
    "jh_top3_rate":                       "この騎手×この馬のコンビ複勝率",
    "th_runs":                            "この調教師×この馬のコンビ出走数",

    # LAP特徴量（v5新規）
    "race_front_pace":                    "今レースの前半3区間の平均ラップ（前半ペース）",
    "race_back_pace":                     "今レースの後半3区間の平均ラップ（後半ペース）",
    "race_pace_diff":                     "後半ペース − 前半ペース（負=後半加速=スローペース）",
    "horse_prev_front_pace1":             "馬の前走の前半3区間平均ラップ",
    "horse_prev_front_pace2":             "馬の2走前の前半3区間平均ラップ",
    "horse_prev_front_pace3":             "馬の3走前の前半3区間平均ラップ",
    "horse_prev_back_pace1":              "馬の前走の後半3区間平均ラップ",
    "horse_prev_back_pace2":              "馬の2走前の後半3区間平均ラップ",
    "horse_prev_back_pace3":              "馬の3走前の後半3区間平均ラップ",
    "horse_prev_pace_diff1":              "馬の前走のペース差（後半−前半）",
    "horse_prev_pace_diff2":              "馬の2走前のペース差",
    "horse_prev_pace_diff3":              "馬の3走前のペース差",
    "field_avg_front_pace":               "同レース他馬の前走前半ペース平均（今回のペース予測値）",
    "field_avg_back_pace":                "同レース他馬の前走後半ペース平均",
    "field_avg_pace_diff":                "同レース他馬の前走ペース差平均",
    "horse_prev_front_pace1_vs_field":    "馬の前走前半ペース − 同レース他馬の平均前走前半ペース",
    "horse_prev_back_pace1_vs_field":     "馬の前走後半ペース − 同レース他馬の平均前走後半ペース",
    "horse_prev_pace_diff1_vs_field":     "馬の前走ペース差 − 同レース他馬の平均前走ペース差",
}

LAP = {f for f in feats if "pace" in f}
LAP |= {"race_front_pace", "race_back_pace", "race_pace_diff",
        "field_avg_front_pace", "field_avg_back_pace", "field_avg_pace_diff"}

print("=" * 100)
print(f"{'順位':>3}  {'特徴量名':<46}  {'重要度':>6}  説明")
print("=" * 100)
for rank, (feat, val) in enumerate(fi.items(), 1):
    desc = DESCRIPTIONS.get(feat, "(説明なし)")
    lap  = " [LAP]" if feat in LAP else ""
    print(f"{rank:>3}. {feat:<46}  {val:>5.2f}%  {desc}{lap}")
print("=" * 100)
print(f"合計90特徴量  |  LAP特徴量(v5新規): {sum(1 for f in fi.index if f in LAP)}個  {sum(v for f,v in fi.items() if f in LAP):.2f}%寄与")
