"""
予想エンジン
- 学習済モデル(LGBアンサンブル)をロードし、レース単位で順位確率を計算
- レース内ソフトマックス正規化で「3着以内に入る相対確率」「1着になる相対確率」を返す
"""
import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PKL = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
MODEL_VARIANTS = {
    "market": {
        "label": "通常モデル（オッズ・人気あり）",
        "model_dir": MODEL_DIR,
        "description": "オッズと人気も予想順位の特徴量に使う高精度モデル",
    },
    "no_market": {
        "label": "オッズ・人気なしモデル",
        "model_dir": os.path.join(MODEL_DIR, "no_market_v4"),
        "description": "オッズ・人気なし。種牡馬統計・クラス強度・フィールド強度・長期ローリング特徴量を使う改良モデル",
        "feature_engineering": "v4",
    },
    "no_market_v5": {
        "label": "オッズ・人気なしモデル v5（ラップタイム特徴量）",
        "model_dir": os.path.join(MODEL_DIR, "no_market_v5"),
        "description": "v4にラップタイム特徴量（前半ペース・後半ペース・ペース差）を追加",
        "feature_engineering": "v5",
    },
    "no_market_v9": {
        "label": "オッズ・人気なしモデル v9（血統系統・交差特徴量）",
        "model_dir": os.path.join(MODEL_DIR, "no_market_v9"),
        "description": "枠番コース適性・昇降級・血統系統（SS/MP/Roberto/ND/APIndy）・父×母父交差特徴量を追加",
        "feature_engineering": "v9",
    },
    "no_market_v10": {
        "label": "オッズ・人気なしモデル v10（StormCat/SW系分離）",
        "model_dir": os.path.join(MODEL_DIR, "no_market_v10"),
        "description": "v9からND系をStormCat系・Sadler's Wells系にさらに分離した8カテゴリ血統分類",
        "feature_engineering": "v10",
    },
    "no_market_v11": {
        "label": "オッズ・人気なしモデル v11（馬場/性別/コース適性）",
        "model_dir": os.path.join(MODEL_DIR, "no_market_v11"),
        "description": "v10に牝馬フラグ・馬場状態・馬&調教師コース適性・通過順改善を追加",
        "feature_engineering": "v11",
    },
    "no_market_lambdarank": {
        "label": "LambdaRankモデル（直接順位最適化）",
        "model_dir": os.path.join(MODEL_DIR, "no_market_lambdarank"),
        "description": "v10特徴量 + LambdaRank目的関数によるNDCG@1/3最適化モデル",
        "feature_engineering": "v10",
    },
    "no_market_lambdarank_v2": {
        "label": "LambdaRankモデル v2（top50特徴量 / gain=40）",
        "model_dir": os.path.join(MODEL_DIR, "no_market_lambdarank_v2"),
        "description": "グリッドサーチ最適解: 重要度上位50特徴量 × label_gain=[0,1,40]で1位的中率44.8%",
        "feature_engineering": "v10",
    },
}
DEFAULT_MODEL_VARIANT = "market"


def normalize_model_variant(model_variant=None):
    key = str(model_variant or os.environ.get("KEIBA_MODEL_VARIANT") or DEFAULT_MODEL_VARIANT)
    return key if key in MODEL_VARIANTS else DEFAULT_MODEL_VARIANT


def model_dir_for_variant(model_variant=None):
    variant = normalize_model_variant(model_variant)
    return MODEL_VARIANTS[variant]["model_dir"]


class KeibaPredictor:
    def __init__(self, model_dir=None, model_variant=None):
        if model_dir is None:
            model_dir = model_dir_for_variant(model_variant)
        self.model_variant = normalize_model_variant(model_variant)
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feats = self.meta["features"]
        self.cat_cols = self.meta.get("categorical", [])
        self.cat_categories = self.meta.get("cat_categories", {})
        self.score_win_weight = float(self.meta.get("score_win_weight", self.meta.get("ranking_metrics", {}).get("score_win_weight", 0.6)))
        self.model_type = self.meta.get("model_type", "binary")

        # アンサンブルモデルをロード
        if self.model_type == "lambdarank":
            n_rank = self.meta.get("n_rank_models", 3)
            self.lr_models = [lgb.Booster(model_file=os.path.join(model_dir, f"rank_{i}.lgb")) for i in range(n_rank)]
            self.top3_models = []
            self.win_models = []
        elif "ensemble_top3" in self.meta:
            self.lr_models = []
            self.top3_models = [lgb.Booster(model_file=os.path.join(model_dir, m)) for m in self.meta["ensemble_top3"]]
            self.win_models  = [lgb.Booster(model_file=os.path.join(model_dir, m)) for m in self.meta["ensemble_win"]]
        else:
            self.lr_models = []
            n3 = self.meta.get("n_top3_models", 3)
            nw = self.meta.get("n_win_models", 2)
            self.top3_models = [lgb.Booster(model_file=os.path.join(model_dir, f"top3_{i}.lgb")) for i in range(n3)]
            self.win_models  = [lgb.Booster(model_file=os.path.join(model_dir, f"win_{i}.lgb"))  for i in range(nw)]
        time_path = os.path.join(model_dir, "lgb_time.txt")
        self.time_model = lgb.Booster(model_file=time_path) if os.path.exists(time_path) else None
        rank_name = self.meta.get("rank_model")
        rank_path = os.path.join(model_dir, rank_name) if rank_name else None
        self.rank_model = lgb.Booster(model_file=rank_path) if (rank_path and os.path.exists(rank_path)) else None
        self.rank_blend_weight = float(self.meta.get("rank_blend_weight", 0.0))

        # v4/v5 lookup tables（ある場合のみロード）
        self._v4_lookups = {}
        for name in ["sire", "broodmare_sire", "race_class", "jockey", "trainer", "horse", "jockey_horse", "trainer_horse", "horse_lap"]:
            lp = os.path.join(model_dir, f"{name}_lookup.json")
            if os.path.exists(lp):
                with open(lp, encoding="utf-8") as f2:
                    self._v4_lookups[name] = json.load(f2)

    def _align_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """推論データのカテゴリを学習時と同じcategoriesに揃える"""
        for c in self.cat_cols:
            if c in df.columns:
                cats = self.cat_categories.get(c, [])
                df[c] = pd.Categorical(df[c].astype(str), categories=cats)
        return df

    def _apply_v4_lookups(self, df: pd.DataFrame) -> pd.DataFrame:
        """v4 lookup テーブルから特徴量を付与する"""
        lu = self._v4_lookups

        def _get(table_name, key_col, stat_key, default=np.nan):
            table = lu.get(table_name, {})
            return df[key_col].astype(str).map(
                lambda k: table.get(k, {}).get(stat_key, default)
            ).astype("float32") if key_col in df.columns else pd.Series(default, index=df.index, dtype="float32")

        # 種牡馬
        for entity in ["sire", "broodmare_sire"]:
            df[f"{entity}_hist_runs"]      = _get(entity, entity, "hist_runs")
            df[f"{entity}_hist_win_rate"]  = _get(entity, entity, "hist_win_rate")
            df[f"{entity}_hist_top3_rate"] = _get(entity, entity, "hist_top3_rate")
            # 距離カテゴリ別
            if "distance" in df.columns and entity in lu:
                dist_bins = pd.cut(df["distance"], bins=[0, 1400, 1700, 2100, 9999], labels=False)
                table = lu[entity]
                df[f"{entity}_dist_win_rate"] = [
                    table.get(str(s), {}).get("dist_win_rates", {}).get(str(int(db)) if pd.notna(db) else "", np.nan)
                    for s, db in zip(df[entity].astype(str), dist_bins)
                ]
                df[f"{entity}_dist_win_rate"] = df[f"{entity}_dist_win_rate"].astype("float32")

        # レースクラス
        df["class_hist_runs"]      = _get("race_class", "race_class", "hist_runs")
        df["class_hist_win_rate"]  = _get("race_class", "race_class", "hist_win_rate")
        df["class_hist_top3_rate"] = _get("race_class", "race_class", "hist_top3_rate")

        # 騎手直近
        df["jockey_recent20_win_rate"]  = _get("jockey", "jockey_id", "recent20_win_rate")
        df["jockey_recent20_top3_rate"] = _get("jockey", "jockey_id", "recent20_top3_rate")
        df["jockey_recent50_win_rate"]  = _get("jockey", "jockey_id", "recent50_win_rate")

        # 調教師直近
        df["trainer_recent20_win_rate"]  = _get("trainer", "trainer_id", "recent20_win_rate")
        df["trainer_recent20_top3_rate"] = _get("trainer", "trainer_id", "recent20_top3_rate")

        # 馬直近
        df["horse_recent5_avg_rank"]   = _get("horse", "horse_id", "recent5_avg_rank")
        df["horse_recent10_avg_rank"]  = _get("horse", "horse_id", "recent10_avg_rank")
        df["horse_recent5_top3_rate"]  = _get("horse", "horse_id", "recent5_top3_rate")
        df["horse_recent10_top3_rate"] = _get("horse", "horse_id", "recent10_top3_rate")
        df["horse_recent5_win_rate"]   = _get("horse", "horse_id", "recent5_win_rate")

        # 騎手×馬 / 調教師×馬コンボ
        if "jockey_id" in df.columns and "horse_id" in df.columns and "jockey_horse" in lu:
            table = lu["jockey_horse"]
            keys = [f"{str(int(j)) if pd.notna(j) else ''}|{str(int(h)) if pd.notna(h) else ''}" for j, h in zip(df["jockey_id"], df["horse_id"])]
            df["jh_runs"] = [table.get(k, {}).get("jh_runs", np.nan) for k in keys]
            df["jh_win_rate"] = [table.get(k, {}).get("jh_win_rate", np.nan) for k in keys]
            df["jh_top3_rate"] = [table.get(k, {}).get("jh_top3_rate", np.nan) for k in keys]
            df[["jh_runs", "jh_win_rate", "jh_top3_rate"]] = df[["jh_runs", "jh_win_rate", "jh_top3_rate"]].astype("float32")

        if "trainer_id" in df.columns and "horse_id" in df.columns and "trainer_horse" in lu:
            table = lu["trainer_horse"]
            keys = [f"{str(int(t)) if pd.notna(t) else ''}|{str(int(h)) if pd.notna(h) else ''}" for t, h in zip(df["trainer_id"], df["horse_id"])]
            df["th_runs"] = [table.get(k, {}).get("th_runs", np.nan) for k in keys]
            df["th_win_rate"] = [table.get(k, {}).get("th_win_rate", np.nan) for k in keys]
            df["th_top3_rate"] = [table.get(k, {}).get("th_top3_rate", np.nan) for k in keys]
            df[["th_runs", "th_win_rate", "th_top3_rate"]] = df[["th_runs", "th_win_rate", "th_top3_rate"]].astype("float32")

        # フィールド強度（同レース内の対戦相手平均）
        race_col = "race_id"
        if race_col in df.columns:
            for col, new_col in [
                ("jockey_win_rate",   "field_avg_jockey_win_rate"),
                ("horse_avg_rank",    "field_avg_horse_rank"),
                ("horse_top3_rate",   "field_avg_horse_top3_rate"),
                ("trainer_top3_rate", "field_avg_trainer_top3_rate"),
            ]:
                if col in df.columns:
                    race_sum = df.groupby(race_col, sort=False)[col].transform("sum")
                    race_cnt = df.groupby(race_col, sort=False)[col].transform("count")
                    df[new_col] = ((race_sum - df[col]) / (race_cnt - 1).replace(0, np.nan)).astype("float32")

        # v5: 馬ごとの過去ラップペース（horse_lap_lookup から）
        if "horse_lap" in lu and "horse_id" in df.columns:
            table = lu["horse_lap"]
            for lag in [1, 2, 3]:
                for pace in ["front_pace", "back_pace", "pace_diff"]:
                    col_name = f"horse_prev_{pace}{lag}"
                    lookup_key = f"prev_{pace}{lag}"
                    df[col_name] = df["horse_id"].astype(str).map(
                        lambda k, t=table, lk=lookup_key: t.get(k, {}).get(lk, np.nan)
                    ).astype("float32")

            # フィールド期待ペース（同レース他馬の前走前半ペース平均）
            if race_col in df.columns:
                for src, fname in [
                    ("horse_prev_front_pace1", "field_avg_front_pace"),
                    ("horse_prev_back_pace1",  "field_avg_back_pace"),
                    ("horse_prev_pace_diff1",  "field_avg_pace_diff"),
                ]:
                    if src in df.columns:
                        race_sum = df.groupby(race_col, sort=False)[src].transform("sum")
                        race_cnt = df.groupby(race_col, sort=False)[src].transform("count")
                        df[fname] = ((race_sum - df[src]) / (race_cnt - 1).replace(0, np.nan)).astype("float32")
                        df[f"{src}_vs_field"] = (df[src] - df[fname]).astype("float32")

        return df

    def _apply_v9_lookups(self, df: pd.DataFrame) -> pd.DataFrame:
        """v9/v10 特有の特徴量をルックアップテーブルから付与する。
        ルックアップファイルが存在しない場合は NaN のまま（LightGBM はそのまま処理可能）。
        """
        lu = self._v4_lookups

        # ── sire_line / bms_line の統計 ──────────────────────────────────
        has_surface = "surface" in df.columns
        has_dist = "distance" in df.columns
        if has_dist:
            dist_bin = pd.cut(df["distance"], bins=[0, 1400, 1700, 2100, 9999], labels=False)

        for lookup_key, src_col, prefix in [
            ("sire_line_stats", "sire", "sire_line"),
            ("bms_line_stats", "broodmare_sire", "bms_line"),
        ]:
            table = lu.get(lookup_key)
            if table is None or src_col not in df.columns:
                continue

            n2l = table.get("name_to_line", {})
            lstats = table.get("line_stats", {})

            def _get_line_stat(name, stat, n2l=n2l, lstats=lstats):
                line = n2l.get(str(name), "7")
                return lstats.get(str(line), {}).get(stat, np.nan)

            df[f"{prefix}_win_rate"]  = df[src_col].astype(str).map(lambda x: _get_line_stat(x, "win_rate")).astype("float32")
            df[f"{prefix}_top3_rate"] = df[src_col].astype(str).map(lambda x: _get_line_stat(x, "top3_rate")).astype("float32")

            if has_surface:
                surf_key = df["surface"].map(lambda s: "turf" if s == "芝" else "dirt")
                df[f"{prefix}_surface_top3_rate"] = [
                    _get_line_stat(nm, f"surface_top3_rate_{sk}")
                    for nm, sk in zip(df[src_col].astype(str), surf_key)
                ]
                df[f"{prefix}_surface_top3_rate"] = df[f"{prefix}_surface_top3_rate"].astype("float32")
            else:
                df[f"{prefix}_surface_top3_rate"] = np.nan

            if has_dist:
                df[f"{prefix}_dist_top3_rate"] = [
                    _get_line_stat(nm, f"dist_top3_rate_{int(db) if pd.notna(db) else 1}")
                    for nm, db in zip(df[src_col].astype(str), dist_bin)
                ]
                df[f"{prefix}_dist_top3_rate"] = df[f"{prefix}_dist_top3_rate"].astype("float32")
            else:
                df[f"{prefix}_dist_top3_rate"] = np.nan

        # ── bloodline_cross の統計 ────────────────────────────────────────
        cx_table = lu.get("bloodline_cross_stats")
        if cx_table and "sire" in df.columns and "broodmare_sire" in df.columns:
            sl_map = lu.get("sire_line_stats", {}).get("name_to_line", {})
            bm_map = lu.get("bms_line_stats",  {}).get("name_to_line", {})
            n_lines = int(cx_table.get("n_lines", 8))
            cx_stats = cx_table.get("cross_stats", {})

            def _cx_key(sire, bms):
                sl = int(sl_map.get(str(sire), n_lines - 1))
                bl = int(bm_map.get(str(bms),  n_lines - 1))
                return str(sl * n_lines + bl)

            cx_keys = [_cx_key(s, b) for s, b in zip(df["sire"].astype(str), df["broodmare_sire"].astype(str))]
            df["bloodline_cross_win_rate"]  = [cx_stats.get(k, {}).get("win_rate", np.nan) for k in cx_keys]
            df["bloodline_cross_top3_rate"] = [cx_stats.get(k, {}).get("top3_rate", np.nan) for k in cx_keys]

            if has_surface:
                surf_keys = ["turf" if s == "芝" else "dirt" for s in df["surface"].astype(str)]
                df["bloodline_cross_surface_top3_rate"] = [
                    cx_stats.get(ck, {}).get(f"surface_top3_rate_{sk}", np.nan)
                    for ck, sk in zip(cx_keys, surf_keys)
                ]
            else:
                df["bloodline_cross_surface_top3_rate"] = np.nan

            if has_dist:
                df["bloodline_cross_dist_top3_rate"] = [
                    cx_stats.get(ck, {}).get(f"dist_top3_rate_{int(db) if pd.notna(db) else 1}", np.nan)
                    for ck, db in zip(cx_keys, dist_bin)
                ]
            else:
                df["bloodline_cross_dist_top3_rate"] = np.nan

            for c in ["bloodline_cross_win_rate","bloodline_cross_top3_rate",
                      "bloodline_cross_surface_top3_rate","bloodline_cross_dist_top3_rate"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

        # ── venue_frame の統計 ───────────────────────────────────────────
        vf_table = lu.get("venue_frame_stats")
        if vf_table and "venue" in df.columns and "frame_no" in df.columns:
            fn = df["frame_no"].fillna(0)
            fg = np.where(fn <= 2, 0, np.where(fn <= 6, 1, 2))
            vf_keys = [f"{v}_{int(f)}" for v, f in zip(df["venue"].astype(str), fg)]
            df["venue_frame_win_rate"]  = [vf_table.get(k, {}).get("win_rate",  np.nan) for k in vf_keys]
            df["venue_frame_top3_rate"] = [vf_table.get(k, {}).get("top3_rate", np.nan) for k in vf_keys]
            df["venue_frame_win_rate"]  = pd.to_numeric(df["venue_frame_win_rate"],  errors="coerce").astype("float32")
            df["venue_frame_top3_rate"] = pd.to_numeric(df["venue_frame_top3_rate"], errors="coerce").astype("float32")

        # ── class_change (昇降級) ────────────────────────────────────────
        horse_table = lu.get("horse")
        if horse_table and "race_class" in df.columns and "horse_id" in df.columns:
            def _encode_class(cls) -> float:
                if cls is None or (isinstance(cls, float) and np.isnan(cls)):
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
                return 2.0

            cur_level  = df["race_class"].map(_encode_class)
            prev_level = df["horse_id"].astype(str).map(
                lambda k: horse_table.get(k, {}).get("last_race_class_level", np.nan)
            ).astype("float32")
            df["class_change"] = (cur_level - prev_level).astype("float32")

        return df

    def _apply_v11_lookups(self, df: pd.DataFrame) -> pd.DataFrame:
        """v11 追加特徴量を推論時に付与する。
        sex/going/direction はカラムから直接エンコード。
        venue/course 系はルックアップから取得（なければ NaN）。
        """
        lu = self._v4_lookups

        # ── 性別エンコード ──────────────────────────────────────────
        if "sex" in df.columns:
            df["sex_encoded"] = (df["sex"] == "牝").astype("float32")

        # ── 馬場状態エンコード ──────────────────────────────────────
        if "going" in df.columns:
            going_map = {"良": 0.0, "稍重": 1.0, "稍": 1.0, "重": 2.0, "不良": 3.0, "不": 3.0}
            df["going_encoded"] = df["going"].map(going_map).astype("float32")

        # ── コース方向エンコード ────────────────────────────────────
        if "direction" in df.columns:
            dir_map = {"右": 0.0, "左": 1.0}
            df["direction_encoded"] = df["direction"].map(dir_map).astype("float32")

        # ── 馬の競馬場/コース別成績 → フィールド内パーセンタイル ────────
        for col in ["horse_venue_top3_rate", "horse_venue_win_rate", "horse_course_top3_rate"]:
            new_col = f"{col}_vs_field"
            if col in df.columns and new_col not in df.columns:
                n = df[col].notna().sum()
                if n > 1:
                    ranked = df[col].rank(method="average", ascending=False, na_option="keep")
                    df[new_col] = (1.0 - (ranked - 1) / max(n - 1, 1)).where(df[col].notna()).astype("float32")
                else:
                    df[new_col] = np.nan

        # ── 調教師/騎手のコース別成績 → フィールド内パーセンタイル ──────
        for col in ["trainer_course_top3_rate", "jockey_course_win_rate"]:
            new_col = f"{col}_vs_field"
            if col in df.columns and new_col not in df.columns:
                n = df[col].notna().sum()
                if n > 1:
                    ranked = df[col].rank(method="average", ascending=False, na_option="keep")
                    df[new_col] = (1.0 - (ranked - 1) / max(n - 1, 1)).where(df[col].notna()).astype("float32")
                else:
                    df[new_col] = np.nan

        # ── 通過順改善 ───────────────────────────────────────────────
        for col in ["horse_passing_gain_lag1", "horse_passing_gain_lag2"]:
            if col in df.columns:
                df[col] = df[col].astype("float32")

        return df

    def _build_features(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """学習時と同じ特徴量列を作る。欠けていればNaNで補う"""
        df = race_df.copy()
        fe = self.meta.get("feature_engineering", "")
        if fe in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"):
            from feature_engineering import add_model_features
            df = add_model_features(df, race_col="race_id")
        if fe in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"):
            df = self._apply_v4_lookups(df)
        if fe in ("v9", "v10", "v11"):
            df = self._apply_v9_lookups(df)
        if fe in ("v11",):
            df = self._apply_v11_lookups(df)
        for f in self.feats:
            if f not in df.columns:
                df[f] = np.nan
        for f in self.feats:
            if f not in self.cat_cols:
                df[f] = pd.to_numeric(df[f], errors="coerce")
        df = self._align_categories(df)
        return df[self.feats]

    def predict_frame(self, race_df: pd.DataFrame, normalize=True) -> pd.DataFrame:
        """
        race_df: 1レースまたは複数レースの出走馬データ。
        Returns: 同じ行に確率列を追加したDataFrame。
        """
        X = self._build_features(race_df)

        # LambdaRankモデル: 生スコアをレース内softmaxで正規化
        if self.model_type == "lambdarank":
            raw_scores = np.mean(
                [m.predict(X, num_iteration=m.best_iteration) for m in self.lr_models], axis=0
            )
            p_win = np.zeros_like(raw_scores)
            if "race_id" in race_df.columns and race_df["race_id"].nunique(dropna=False) > 1:
                for grp_idx in race_df.groupby("race_id").groups.values():
                    idx = list(grp_idx)
                    chunk = raw_scores[idx]
                    chunk = chunk - chunk.max()
                    exp_c = np.exp(chunk)
                    p_win[idx] = exp_c / exp_c.sum()
            else:
                shifted = raw_scores - raw_scores.max()
                exp_s = np.exp(shifted)
                p_win = exp_s / exp_s.sum()
            p_top3 = np.clip(p_win * 3.0, 0.0, 0.999)
            out = race_df.copy()
            out["p_top3_raw"] = p_top3
            out["p_win_raw"] = p_win
            out["pred_time_idx"] = np.nan
            out["p_top3"] = np.clip(p_top3, 1e-6, 0.999)
            out["p_win"] = np.clip(p_win, 1e-6, 0.999)
            out["score"] = out["p_win"]
            if "race_id" in out.columns and out["race_id"].nunique(dropna=False) > 1:
                out["pred_rank"] = out.groupby("race_id")["score"].rank(ascending=False, method="min").astype(int)
                return out.sort_values(["race_id", "pred_rank"]).reset_index(drop=True)
            out["pred_rank"] = out["score"].rank(ascending=False, method="min").astype(int)
            return out.sort_values("pred_rank").reset_index(drop=True)

        # バイナリアンサンブル平均
        p_top3_list = [m.predict(X, num_iteration=m.best_iteration) for m in self.top3_models]
        p_top3 = np.mean(p_top3_list, axis=0)

        p_win_list = [m.predict(X, num_iteration=m.best_iteration) for m in self.win_models]
        p_win = np.mean(p_win_list, axis=0)

        # LambdaRankスコアをブレンド(モデルがある場合のみ)
        if self.rank_model is not None and self.rank_blend_weight > 0:
            rank_raw = self.rank_model.predict(X, num_iteration=self.rank_model.best_iteration)
            bw = self.rank_blend_weight
            # レースごとにsoftmax正規化（複数レースのバッチ対応）
            if "race_id" in race_df.columns and race_df["race_id"].nunique(dropna=False) > 1:
                rank_prob = np.zeros_like(rank_raw)
                p_win_norm = np.zeros_like(p_win)
                for rid, grp_idx in race_df.groupby("race_id").groups.items():
                    idx = list(grp_idx)
                    chunk_r = rank_raw[idx]
                    chunk_r = chunk_r - chunk_r.max()
                    exp_r = np.exp(chunk_r)
                    rank_prob[idx] = exp_r / exp_r.sum()
                    chunk_w = p_win[idx]
                    sw = chunk_w.sum()
                    p_win_norm[idx] = chunk_w / sw if sw > 0 else chunk_w
            else:
                rank_shifted = rank_raw - rank_raw.max()
                rank_prob = np.exp(rank_shifted) / np.exp(rank_shifted).sum()
                p_win_s = p_win.sum()
                p_win_norm = p_win / p_win_s if p_win_s > 0 else p_win
            p_win = (1 - bw) * p_win_norm + bw * rank_prob

        if self.time_model is not None:
            p_time = self.time_model.predict(X, num_iteration=self.time_model.best_iteration)
        else:
            p_time = np.full(len(X), np.nan)

        out = race_df.copy()
        out["p_top3_raw"] = p_top3
        out["p_win_raw"] = p_win
        out["pred_time_idx"] = p_time

        if normalize and "race_id" in out.columns and out["race_id"].nunique(dropna=False) > 1:
            s3 = out.groupby("race_id")["p_top3_raw"].transform("sum")
            sw = out.groupby("race_id")["p_win_raw"].transform("sum")
            out["p_top3"] = np.where(s3 > 0, out["p_top3_raw"] * (3.0 / s3), out["p_top3_raw"])
            out["p_win"] = np.where(sw > 0, out["p_win_raw"] * (1.0 / sw), out["p_win_raw"])
            out["p_top3"] = np.clip(out["p_top3"], 1e-6, 0.999)
            out["p_win"] = np.clip(out["p_win"], 1e-6, 0.999)
        elif normalize and len(out) > 1:
            # レース内で確率の合計を「3着が出る期待数=3」「1着が出る=1」になるよう正規化
            # ソフトマックスではなく、生確率の和でスケーリング
            s3 = p_top3.sum()
            sw = p_win.sum()
            out["p_top3"] = (p_top3 * (3.0 / s3)) if s3 > 0 else p_top3
            out["p_win"] = (p_win * (1.0 / sw)) if sw > 0 else p_win
            # 上限1.0
            out["p_top3"] = np.clip(out["p_top3"], 1e-6, 0.999)
            out["p_win"] = np.clip(out["p_win"], 1e-6, 0.999)
        else:
            out["p_top3"] = np.clip(p_top3, 1e-6, 0.999)
            out["p_win"] = np.clip(p_win, 1e-6, 0.999)

        # 期待順位推定: モデルごとの検証で決めた重みを使う
        out["score"] = self.score_win_weight * out["p_win"] + (1.0 - self.score_win_weight) * out["p_top3"]
        if "race_id" in out.columns and out["race_id"].nunique(dropna=False) > 1:
            out["pred_rank"] = out.groupby("race_id")["score"].rank(ascending=False, method="min").astype(int)
            return out.sort_values(["race_id", "pred_rank"]).reset_index(drop=True)
        out["pred_rank"] = out["score"].rank(ascending=False, method="min").astype(int)
        return out.sort_values("pred_rank").reset_index(drop=True)

    def predict_race(self, race_df: pd.DataFrame, normalize=True) -> pd.DataFrame:
        """
        race_df: 1レースの出走馬データ(複数行=複数頭)
        Returns: 同じ並び順で確率列が追加されたDataFrame
        """
        return self.predict_frame(race_df, normalize=normalize)

    def recommend_bets(
        self,
        race_df: pd.DataFrame,
        odds_data: dict,
        bankroll: float = 100_000,
        kelly_factor: float = 0.25,
        min_ev: float = 0.1,
        top_k: int = 4,
        ticket_types=None,
    ) -> list:
        """
        LambdaRank v2 スコアとマーケットオッズからKelly基準で馬券推薦を返す。

        バックテストで実証された最良設定:
          ticket_types = ["umaren", "umatan"]  ROI +43.9% (2025年検証)
          min_ev = 0.1  (マーケット対比10%以上の期待値のみ)
          top_k  = 4    (上位4頭の組合せ対象)
          kelly  = 0.25 (クォーターKelly)

        Parameters
        ----------
        race_df      : 1レースの出走馬データ (horse_no 列必須)
        odds_data    : {ticket_kind: [{horses, odds}, ...]}
                       data/odds/<race_id>.json または data/payouts/<race_id>.json の内容
        bankroll     : 手元資金 (円)
        kelly_factor : Kelly係数の倍率 (0.25=クォーターKelly)
        min_ev       : 最低期待値 (0.1=市場比10%以上の優位性がある馬券のみ)
        top_k        : 組合せ馬券で対象にする上位頭数
        ticket_types : 対象券種 (Noneで["umaren","umatan"])

        Returns
        -------
        list of dicts: [{ticket_kind, horses, model_prob, market_odds, ev, kelly, bet_amount}, ...]
        """
        if self.model_type != "lambdarank":
            return []

        from kelly_betting import KellyBettingAI, softmax_probs, market_probs_from_odds
        from kelly_betting import enumerate_race_bets, size_bets

        if ticket_types is None:
            ticket_types = ["umaren", "umatan"]

        X = self._build_features(race_df)
        raw_scores = np.mean(
            [m.predict(X, num_iteration=m.best_iteration) for m in self.lr_models], axis=0
        )
        model_probs = softmax_probs(raw_scores)
        horse_nos = race_df["horse_no"].tolist()

        # 単勝オッズからマーケット確率を逆算
        tansho_entries = odds_data.get("tansho", [])
        h2odds = {e["horses"][0]: e["odds"] for e in tansho_entries if e.get("horses")}
        tansho_odds_arr = np.array([h2odds.get(h, 100.0) for h in horse_nos], dtype=float)
        market_probs = market_probs_from_odds(tansho_odds_arr)

        bets_raw = enumerate_race_bets(
            model_probs, market_probs, horse_nos,
            tansho_odds_arr, ticket_types, top_k=top_k,
        )
        bets_raw = [b for b in bets_raw if b["ev"] >= min_ev]
        return size_bets(bets_raw, bankroll, kelly_factor,
                         max_bet_frac=0.10, max_total_frac=0.20)


# ===========================================================
# 過去データからレースを取り出して予想
# ===========================================================
def load_race_from_data(race_id: int, pkl_path=DATA_PKL) -> pd.DataFrame:
    """保存済データから race_id のレースを取り出す"""
    with gzip.open(pkl_path, "rb") as f:
        df = pickle.load(f)
    sub = df[df["race_id"] == int(race_id)].copy()
    return sub.reset_index(drop=True)


def list_recent_races(year: int = 2025, limit: int = 20, pkl_path=DATA_PKL) -> pd.DataFrame:
    with gzip.open(pkl_path, "rb") as f:
        df = pickle.load(f)
    sub = df[df["date"].dt.year == year]
    races = sub.groupby("race_id").agg(
        date=("date", "first"),
        venue=("venue", "first"),
        round_no=("round_no", "first"),
        race_class=("race_class", "first"),
        surface=("surface", "first"),
        distance=("distance", "first"),
        n_horses=("horse_no", "count"),
    ).reset_index()
    races = races.sort_values("date", ascending=False).head(limit)
    return races


if __name__ == "__main__":
    p = KeibaPredictor()
    print("Loaded models:")
    print(f"  TOP3 ensemble: {len(p.top3_models)} models")
    print(f"  WIN ensemble: {len(p.win_models)} models")
    print(f"  Time model: {'yes' if p.time_model else 'no'}")
    print(f"  Features: {len(p.feats)}")
    print()
    races = list_recent_races(2025, 5)
    print("Recent races:\n", races)
    rid = int(races.iloc[0]["race_id"])
    df = load_race_from_data(rid)
    pred = p.predict_race(df)
    print(f"\n=== 予想結果 race_id={rid} ===")
    cols = ["pred_rank", "horse_no", "frame_no", "p_win", "p_top3", "odds", "popularity", "rank"]
    cols = [c for c in cols if c in pred.columns]
    print(pred[cols].to_string(index=False))
