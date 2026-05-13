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


class KeibaPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feats = self.meta["features"]
        self.cat_cols = self.meta["categorical"]
        self.cat_categories = self.meta["cat_categories"]

        # アンサンブルモデルをロード
        self.top3_models = [
            lgb.Booster(model_file=os.path.join(model_dir, m))
            for m in self.meta["ensemble_top3"]
        ]
        self.win_models = [
            lgb.Booster(model_file=os.path.join(model_dir, m))
            for m in self.meta["ensemble_win"]
        ]
        time_path = os.path.join(model_dir, "lgb_time.txt")
        self.time_model = lgb.Booster(model_file=time_path) if os.path.exists(time_path) else None

    def _align_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """推論データのカテゴリを学習時と同じcategoriesに揃える"""
        for c in self.cat_cols:
            if c in df.columns:
                cats = self.cat_categories.get(c, [])
                df[c] = pd.Categorical(df[c].astype(str), categories=cats)
        return df

    def _build_features(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """学習時と同じ特徴量列を作る。欠けていればNaNで補う"""
        df = race_df.copy()
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

        # アンサンブル平均
        p_top3_list = [m.predict(X, num_iteration=m.best_iteration) for m in self.top3_models]
        p_top3 = np.mean(p_top3_list, axis=0)

        p_win_list = [m.predict(X, num_iteration=m.best_iteration) for m in self.win_models]
        p_win = np.mean(p_win_list, axis=0)

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

        # 期待順位推定: 1=p_win, 2-3=p_top3-p_win
        # スコア(高いほど強い)= 0.6*p_win + 0.4*p_top3
        out["score"] = 0.6 * out["p_win"] + 0.4 * out["p_top3"]
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
