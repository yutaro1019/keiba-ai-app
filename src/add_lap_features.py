"""
ラップタイム特徴量をpickleに追加するスクリプト。

CSVから race_front_pace / race_back_pace / race_pace_diff を取り出して
既存のprocessed_all.pkl.gz にマージして上書き保存する。

Usage:
    python add_lap_features.py
"""
import sys, io, os, re, gzip, pickle, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from predictor import DATA_PKL

# CSVが置かれているディレクトリ（preprocess.pyのDATA_DIR相当）
CSV_DIR = r"C:\Users\yukim\OneDrive\Desktop\バックアップ\datalist"
YEARS = list(range(2016, 2026))


def parse_lap_times(s):
    """'12.9 - 12.0 - 12.5 - ...' -> [12.9, 12.0, 12.5, ...]  (None if invalid)"""
    if pd.isna(s) or str(s).strip() in ("**", ""):
        return None
    parts = re.findall(r"\d+\.\d+", str(s))
    return [float(x) for x in parts] if len(parts) >= 3 else None


def lap_to_pace_features(laps):
    """ラップリストから前半・後半ペースと差を計算"""
    if laps is None:
        return np.nan, np.nan, np.nan
    front = float(np.mean(laps[:3]))
    back  = float(np.mean(laps[-3:]))
    diff  = back - front  # 負 = 後半加速（ハイペース）
    return front, back, diff


def load_lap_data_from_csvs():
    """全年のCSVからrace_id+ラップタイムを1行/レースで抽出してDataFrameにまとめる"""
    frames = []
    lap_col = "ラップタイム"
    for year in YEARS:
        path = os.path.join(CSV_DIR, f"raw_keiba_data_{year}_full_final.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path, encoding="utf-8", low_memory=False,
                         usecols=lambda c: c in ("race_id", lap_col))
        df.columns = [c.replace("﻿", "") for c in df.columns]
        if lap_col not in df.columns:
            print(f"  [skip] {year}: ラップタイム column not found")
            continue
        df["race_id"] = pd.to_numeric(df["race_id"], errors="coerce").astype("int64")
        # 1レース=全馬同一値なので先頭行を取る
        race_df = (
            df.dropna(subset=["race_id"])
            .groupby("race_id", sort=False)[lap_col]
            .first()
            .reset_index()
        )
        laps_parsed = race_df[lap_col].apply(parse_lap_times)
        pace = laps_parsed.apply(lap_to_pace_features)
        race_df["race_front_pace"] = [x[0] for x in pace]
        race_df["race_back_pace"]  = [x[1] for x in pace]
        race_df["race_pace_diff"]  = [x[2] for x in pace]
        race_df = race_df.drop(columns=[lap_col])

        valid = race_df["race_front_pace"].notna().sum()
        total = len(race_df)
        print(f"  {year}: {total} races, {valid} with valid lap times ({valid/total*100:.1f}%)")
        frames.append(race_df)
        gc.collect()

    if not frames:
        raise RuntimeError("No CSV data loaded — check CSV_DIR path")
    return pd.concat(frames, ignore_index=True)


def main():
    print(">>> Loading pickle...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        hist = pickle.load(f)
    print(f"   rows={len(hist):,}  cols={len(hist.columns)}")

    # 既存lap列があれば一度削除（再マージ）
    for col in ("race_front_pace", "race_back_pace", "race_pace_diff"):
        if col in hist.columns:
            hist.drop(columns=[col], inplace=True)

    print(">>> Loading lap times from CSVs...", flush=True)
    lap_df = load_lap_data_from_csvs()
    print(f"   total lap rows: {len(lap_df):,}")

    print(">>> Merging...", flush=True)
    hist["race_id"] = pd.to_numeric(hist["race_id"], errors="coerce").astype("int64")
    merged = hist.merge(lap_df, on="race_id", how="left")

    for col in ("race_front_pace", "race_back_pace", "race_pace_diff"):
        merged[col] = merged[col].astype("float32")

    matched = merged["race_front_pace"].notna().sum()
    print(f"   matched rows: {matched:,} / {len(merged):,} ({matched/len(merged)*100:.1f}%)")

    print(">>> Saving updated pickle...", flush=True)
    with gzip.open(DATA_PKL, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Saved to {DATA_PKL}")
    print("Done.")


if __name__ == "__main__":
    main()
