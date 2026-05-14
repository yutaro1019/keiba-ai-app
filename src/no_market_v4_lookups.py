"""
Refresh lookup tables used by the no_market_v4 prediction model.

The model files are trained separately, but production predictions need fresh
history snapshots such as recent horse form and jockey/trainer rolling form.
These lookup files are regenerated from data/processed_all.pkl.gz after the
dataset update mode appends new finished races.
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PKL = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
NO_MARKET_V4_DIR = os.path.join(BASE_DIR, "models", "no_market_v4")

LOOKUP_NAMES = (
    "sire",
    "broodmare_sire",
    "race_class",
    "jockey",
    "trainer",
    "horse",
    "jockey_horse",
    "trainer_horse",
    "horse_lap",
)


def _emit(progress: Callable[[Dict[str, Any]], None] | None, **payload):
    if progress is not None:
        progress(payload)


def _key(value) -> str:
    if pd.isna(value):
        return ""
    try:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
    except Exception:
        pass
    return str(value)


def _combo_key(a, b) -> str:
    ka = _key(a)
    kb = _key(b)
    return f"{ka}|{kb}" if ka and kb else ""


def _finite(value, default: float | int = 0.0):
    try:
        f = float(value)
    except Exception:
        return default
    if not np.isfinite(f):
        return default
    return f


def _rate(values: pd.Series, default=np.nan) -> float:
    if len(values) == 0:
        return _finite(default, default)
    return _finite(values.mean(), default)


def _load_processed(data_path: str = DATA_PKL) -> pd.DataFrame:
    with gzip.open(data_path, "rb") as f:
        df = pickle.load(f)
    df = df.dropna(subset=["rank"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    sort_cols = ["date", "race_id"] + (["horse_no"] if "horse_no" in df.columns else [])
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df["_win"] = (df["rank"] == 1).astype("float32")
    df["_top3"] = (df["rank"] <= 3).astype("float32")
    return df


def _build_blood_lookup(df: pd.DataFrame, column: str) -> Dict[str, Dict[str, Any]]:
    if column not in df.columns:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    dist_bin = pd.cut(df["distance"], bins=[0, 1400, 1700, 2100, 9999], labels=False) if "distance" in df.columns else None
    work = df[[column, "_win", "_top3"]].copy()
    if dist_bin is not None:
        work["_dist_bin"] = dist_bin

    for key, group in work.dropna(subset=[column]).groupby(column, sort=False, observed=True):
        item = {
            "hist_runs": int(len(group)),
            "hist_win_rate": _rate(group["_win"]),
            "hist_top3_rate": _rate(group["_top3"]),
            "dist_win_rates": {},
        }
        if "_dist_bin" in group.columns:
            for db, db_group in group.dropna(subset=["_dist_bin"]).groupby("_dist_bin", sort=False, observed=True):
                item["dist_win_rates"][str(int(db))] = _rate(db_group["_win"])
        out[_key(key)] = item
    return out


def _build_class_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if "race_class" not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, group in df.dropna(subset=["race_class"]).groupby("race_class", sort=False, observed=True):
        out[_key(key)] = {
            "hist_runs": int(len(group)),
            "hist_win_rate": _rate(group["_win"]),
            "hist_top3_rate": _rate(group["_top3"]),
        }
    return out


def _recent_rate(group: pd.DataFrame, col: str, window: int, min_rows: int = 1) -> float:
    tail = group.tail(window)
    if len(tail) < min_rows:
        return np.nan
    return _rate(tail[col])


def _build_jockey_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if "jockey_id" not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, group in df.dropna(subset=["jockey_id"]).groupby("jockey_id", sort=False):
        out[_key(key)] = {
            "recent20_win_rate": _recent_rate(group, "_win", 20, 3),
            "recent20_top3_rate": _recent_rate(group, "_top3", 20, 3),
            "recent50_win_rate": _recent_rate(group, "_win", 50, 3),
            "recent50_top3_rate": _recent_rate(group, "_top3", 50, 3),
            "hist_runs": int(len(group)),
        }
    return out


def _build_trainer_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if "trainer_id" not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, group in df.dropna(subset=["trainer_id"]).groupby("trainer_id", sort=False):
        out[_key(key)] = {
            "recent20_win_rate": _recent_rate(group, "_win", 20, 3),
            "recent20_top3_rate": _recent_rate(group, "_top3", 20, 3),
            "recent50_win_rate": _recent_rate(group, "_win", 50, 3),
            "recent50_top3_rate": _recent_rate(group, "_top3", 50, 3),
            "hist_runs": int(len(group)),
        }
    return out


def _build_horse_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if "horse_id" not in df.columns:
        return {}
    out: Dict[str, Dict[str, Any]] = {}

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

    has_class = "race_class" in df.columns
    for key, group in df.dropna(subset=["horse_id"]).groupby("horse_id", sort=False):
        recent5  = group.tail(5)
        recent10 = group.tail(10)
        entry = {
            "recent5_avg_rank":   _rate(recent5["rank"]),
            "recent10_avg_rank":  _rate(recent10["rank"]),
            "recent5_top3_rate":  _rate(recent5["_top3"]),
            "recent10_top3_rate": _rate(recent10["_top3"]),
            "recent5_win_rate":   _rate(recent5["_win"]),
            "recent10_win_rate":  _rate(recent10["_win"]),
            "hist_runs":          int(len(group)),
        }
        if has_class:
            last_cls = group["race_class"].dropna().iloc[-1] if len(group["race_class"].dropna()) > 0 else None
            entry["last_race_class_level"] = _finite(_encode_class(last_cls), np.nan)
        out[_key(key)] = entry
    return out


def _build_horse_lap_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """馬ごとの直近3走ラップペース（horse_prev_*1/2/3 の推論時ルックアップ）"""
    lap_cols = ["race_front_pace", "race_back_pace", "race_pace_diff"]
    present = [c for c in lap_cols if c in df.columns]
    if "horse_id" not in df.columns or not present:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, group in df.dropna(subset=["horse_id"]).groupby("horse_id", sort=False):
        item: Dict[str, Any] = {}
        for lag in [1, 2, 3]:
            for src in present:
                tag = src.replace("race_", "")
                col_name = f"prev_{tag}{lag}"
                g_valid = group[src].dropna()
                if len(g_valid) >= lag:
                    item[col_name] = _finite(g_valid.iloc[-lag], np.nan)
                else:
                    item[col_name] = np.nan
        if item:
            out[_key(key)] = item
    return out


def _build_jockey_horse_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if "jockey_id" not in df.columns or "horse_id" not in df.columns:
        return {}
    work = df.dropna(subset=["jockey_id", "horse_id"]).copy()
    work["_combo_key"] = [_combo_key(j, h) for j, h in zip(work["jockey_id"], work["horse_id"])]
    out: Dict[str, Dict[str, Any]] = {}
    for key, group in work[work["_combo_key"] != ""].groupby("_combo_key", sort=False):
        out[key] = {
            "jh_runs": int(len(group)),
            "jh_win_rate": _rate(group["_win"]),
            "jh_top3_rate": _rate(group["_top3"]),
        }
    return out


def _build_trainer_horse_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if "trainer_id" not in df.columns or "horse_id" not in df.columns:
        return {}
    work = df.dropna(subset=["trainer_id", "horse_id"]).copy()
    work["_combo_key"] = [_combo_key(t, h) for t, h in zip(work["trainer_id"], work["horse_id"])]
    out: Dict[str, Dict[str, Any]] = {}
    for key, group in work[work["_combo_key"] != ""].groupby("_combo_key", sort=False):
        out[key] = {
            "th_runs": int(len(group)),
            "th_win_rate": _rate(group["_win"]),
            "th_top3_rate": _rate(group["_top3"]),
        }
    return out


def _build_sire_line_lookup(df: pd.DataFrame, col: str, get_sire_line_fn) -> Dict[str, Any]:
    """sire/broodmare_sire ごとに系統IDを割り当て、系統別統計を集計する。
    Returns: {name_to_line: {sire_name: line_id_str}, line_stats: {line_id_str: {win_rate,...}}}
    """
    if col not in df.columns:
        return {}

    name_to_line: Dict[str, str] = {}
    for name in df[col].dropna().unique():
        name_to_line[str(name)] = str(get_sire_line_fn(name))

    work = df[[col, "_win", "_top3"]].copy()
    work["_line"] = work[col].map(lambda x: get_sire_line_fn(x) if pd.notna(x) else -1)
    if "surface" in df.columns:
        work["_surface"] = df["surface"]
    if "distance" in df.columns:
        work["_dist_bin"] = pd.cut(df["distance"], bins=[0, 1400, 1700, 2100, 9999], labels=False)

    line_stats: Dict[str, Any] = {}
    for line, grp in work[work["_line"] >= 0].groupby("_line", sort=False, observed=True):
        stat: Dict[str, Any] = {
            "win_rate":  _rate(grp["_win"]),
            "top3_rate": _rate(grp["_top3"]),
        }
        if "_surface" in grp.columns:
            for surf_val, surf_name in [("芝", "turf"), ("ダート", "dirt")]:
                sg = grp[grp["_surface"] == surf_val]
                stat[f"surface_top3_rate_{surf_name}"] = _rate(sg["_top3"]) if len(sg) > 0 else np.nan
        if "_dist_bin" in grp.columns:
            for db in [0, 1, 2, 3]:
                dg = grp[grp["_dist_bin"] == db]
                stat[f"dist_top3_rate_{db}"] = _rate(dg["_top3"]) if len(dg) > 0 else np.nan
        line_stats[str(int(line))] = {k: _finite(v, np.nan) for k, v in stat.items()}

    return {"name_to_line": name_to_line, "line_stats": line_stats}


def _build_bloodline_cross_lookup(df: pd.DataFrame, get_sire_line_fn, n_lines: int = 8) -> Dict[str, Any]:
    """父系統 × 母父系統 (n_lines×n_lines) の組み合わせ別統計を集計する。"""
    if "sire" not in df.columns or "broodmare_sire" not in df.columns:
        return {}

    work = df[["sire", "broodmare_sire", "_win", "_top3"]].copy()
    work["_sl"] = work["sire"].map(lambda x: get_sire_line_fn(x) if pd.notna(x) else n_lines - 1)
    work["_bl"] = work["broodmare_sire"].map(lambda x: get_sire_line_fn(x) if pd.notna(x) else n_lines - 1)
    work["_cx"] = work["_sl"] * n_lines + work["_bl"]
    if "surface" in df.columns:
        work["_surface"] = df["surface"]
    if "distance" in df.columns:
        work["_dist_bin"] = pd.cut(df["distance"], bins=[0, 1400, 1700, 2100, 9999], labels=False)

    cross_stats: Dict[str, Any] = {}
    for cx, grp in work.groupby("_cx", sort=False, observed=True):
        stat: Dict[str, Any] = {
            "win_rate":  _rate(grp["_win"]),
            "top3_rate": _rate(grp["_top3"]),
        }
        if "_surface" in grp.columns:
            for surf_val, surf_name in [("芝", "turf"), ("ダート", "dirt")]:
                sg = grp[grp["_surface"] == surf_val]
                stat[f"surface_top3_rate_{surf_name}"] = _rate(sg["_top3"]) if len(sg) > 0 else np.nan
        if "_dist_bin" in grp.columns:
            for db in [0, 1, 2, 3]:
                dg = grp[grp["_dist_bin"] == db]
                stat[f"dist_top3_rate_{db}"] = _rate(dg["_top3"]) if len(dg) > 0 else np.nan
        cross_stats[str(int(cx))] = {k: _finite(v, np.nan) for k, v in stat.items()}

    return {"n_lines": n_lines, "cross_stats": cross_stats}


def _build_venue_frame_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """競馬場 × 枠番グループ (内/中/外) の成績を集計する。"""
    if "venue" not in df.columns or "frame_no" not in df.columns:
        return {}

    work = df[["venue", "frame_no", "_win", "_top3"]].copy()
    fn = work["frame_no"].fillna(0)
    work["_fg"] = np.where(fn <= 2, 0, np.where(fn <= 6, 1, 2))

    out: Dict[str, Dict[str, Any]] = {}
    for (venue, fg), grp in work.dropna(subset=["venue"]).groupby(["venue", "_fg"], sort=False, observed=True):
        key = f"{venue}_{int(fg)}"
        out[key] = {
            "win_rate":  _rate(grp["_win"]),
            "top3_rate": _rate(grp["_top3"]),
        }
    return out


def build_lookup_tables(df: pd.DataFrame, progress: Callable[[Dict[str, Any]], None] | None = None) -> Dict[str, Dict[str, Any]]:
    builders = [
        ("sire", lambda: _build_blood_lookup(df, "sire")),
        ("broodmare_sire", lambda: _build_blood_lookup(df, "broodmare_sire")),
        ("race_class", lambda: _build_class_lookup(df)),
        ("jockey", lambda: _build_jockey_lookup(df)),
        ("trainer", lambda: _build_trainer_lookup(df)),
        ("horse", lambda: _build_horse_lookup(df)),
        ("jockey_horse", lambda: _build_jockey_horse_lookup(df)),
        ("trainer_horse", lambda: _build_trainer_horse_lookup(df)),
        ("horse_lap", lambda: _build_horse_lap_lookup(df)),
    ]
    tables: Dict[str, Dict[str, Any]] = {}
    total = len(builders)
    for idx, (name, builder) in enumerate(builders, start=1):
        _emit(progress, phase="lookup", current=idx - 1, total=total, message=f"{name} lookup を作成中")
        tables[name] = builder()
        _emit(progress, phase="lookup", current=idx, total=total, message=f"{name} lookup 作成完了: {len(tables[name]):,}件")
    return tables


def write_lookup_tables(tables: Dict[str, Dict[str, Any]], model_dir: str = NO_MARKET_V4_DIR) -> Dict[str, Any]:
    os.makedirs(model_dir, exist_ok=True)
    written = {}
    for name, table in tables.items():
        path = os.path.join(model_dir, f"{name}_lookup.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, separators=(",", ":"))
        written[name] = {"path": path, "entries": len(table)}

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lookups": written,
    }
    meta_path = os.path.join(model_dir, "lookup_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    meta["meta_path"] = meta_path
    return meta


def regenerate_v4_lookups(
    data_path: str = DATA_PKL,
    model_dir: str = NO_MARKET_V4_DIR,
    progress: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    _emit(progress, phase="lookup", current=0, total=10, message="processed_all から v4 lookup を読み込み準備中")
    df = _load_processed(data_path)
    _emit(progress, phase="lookup", current=1, total=10, message=f"lookup 元データ: {len(df):,}行")
    tables = build_lookup_tables(df, progress=progress)

    # v9/v10 モデル向け追加ルックアップ（meta.json が存在すれば検出）
    meta_path = os.path.join(model_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as _f:
            _model_meta = json.load(_f)
        fe = _model_meta.get("feature_engineering", "")
        if fe in ("v9", "v10", "v11"):
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import importlib
                # v11 は sire_line 系統を v10 から継承
                mod_name = "train_no_market_v10" if fe == "v11" else f"train_no_market_{fe}"
                mod = importlib.import_module(mod_name)
                get_sire_line_fn = mod.get_sire_line
                n_lines = mod.N_LINES
                _emit(progress, phase="lookup", current=8, total=10, message=f"{fe} 血統系統ルックアップを構築中")
                tables["sire_line_stats"] = _build_sire_line_lookup(df, "sire", get_sire_line_fn)
                tables["bms_line_stats"]  = _build_sire_line_lookup(df, "broodmare_sire", get_sire_line_fn)
                tables["bloodline_cross_stats"] = _build_bloodline_cross_lookup(df, get_sire_line_fn, n_lines)
                tables["venue_frame_stats"] = _build_venue_frame_lookup(df)
            except Exception as _e:
                _emit(progress, phase="lookup", current=8, total=10, message=f"[WARN] {fe} 追加ルックアップ構築失敗: {_e}")

    _emit(progress, phase="lookup", current=9, total=10, message="lookup JSON を保存中")
    meta = write_lookup_tables(tables, model_dir=model_dir)
    _emit(progress, phase="lookup", current=10, total=10, message="v4 lookup 更新完了")
    return {
        "updated": True,
        "model_dir": model_dir,
        "data_path": data_path,
        "rows": int(len(df)),
        "lookup_counts": {name: len(table) for name, table in tables.items()},
        "meta_path": meta.get("meta_path", ""),
        "generated_at": meta.get("generated_at", ""),
    }


if __name__ == "__main__":
    result = regenerate_v4_lookups()
    print(json.dumps(result, ensure_ascii=False, indent=2))
