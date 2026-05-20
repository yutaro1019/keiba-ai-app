from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import pickle
import re
import sys
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))

from betting import STYLE_CONFIG, format_suggestion, rank_predictions, suggest
from keiba_ai import display_probability_percent, race_confidence, settle_bet
from predictor import DATA_PKL, DEFAULT_MODEL_VARIANT, MODEL_VARIANTS, KeibaPredictor, normalize_model_variant
from race_scraper import (
    auto_update_missing,
    candidate_race_dates,
    discover_race_ids_for_date,
    list_cached_races,
    load_payout_cache,
    load_or_scrape,
    parse_date,
    VENUE_NAMES,
)


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.json.ensure_ascii = False
_predictors: Dict[str, KeibaPredictor] = {}
VISIBLE_STYLE_KEYS = ("roi_focus", "hit_focus", "kelly_ai")
VISIBLE_STYLE_CONFIG = {key: STYLE_CONFIG[key] for key in VISIBLE_STYLE_KEYS if key in STYLE_CONFIG}
VISIBLE_STYLE_CONFIG["kelly_ai"] = {
    "name": "Kelly AI（馬連・馬単）",
    "desc": "LambdaRank v2 + Kelly基準。EV+10%以上の馬連・馬単のみ推薦。no_market_lambdarank_v2専用",
}
DEFAULT_STYLE = "roi_focus"
PRIMARY_MODEL_KEY = "no_market_lambdarank_v3_g40_8_3_1"
KELLY_MODEL_KEY = "no_market_lambdarank_v2"
VISIBLE_MODEL_KEYS = (PRIMARY_MODEL_KEY, KELLY_MODEL_KEY, "market")
VISIBLE_MODEL_VARIANTS = {k: MODEL_VARIANTS[k] for k in VISIBLE_MODEL_KEYS}
VENUE_OPTIONS = [(code, name) for code, name in sorted(VENUE_NAMES.items(), key=lambda item: int(item[0]))]
UPDATE_JOBS: Dict[str, Dict] = {}
UPDATE_LOCK = threading.Lock()


class JobCancelled(Exception):
    pass


@app.after_request
def force_utf8_response(response):
    if response.mimetype == "text/html":
        response.headers["Content-Type"] = "text/html; charset=utf-8"
    elif response.mimetype == "application/json":
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


KELLY_DEFAULT_BANKROLL = 100_000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_CACHE_DIR      = os.path.join(BASE_DIR, "data", "sim_cache")
PIPELINE_CACHE_DIR = os.path.join(BASE_DIR, "data", "pipeline_cache")
SCORE_CACHE_DIR    = os.path.join(BASE_DIR, "data", "score_cache")
os.makedirs(SIM_CACHE_DIR,      exist_ok=True)
os.makedirs(PIPELINE_CACHE_DIR, exist_ok=True)
os.makedirs(SCORE_CACHE_DIR,    exist_ok=True)

# スコアキャッシュに保存する列（特徴量は除き、シミュレーションに必要な列のみ）
_SCORE_CACHE_COLS = [
    "race_id", "date", "horse_no", "odds", "rank",
    "score", "p_win", "p_top3", "pred_rank",
    "horse_dist_top3_rate", "horse_top3_rate",
]


def _score_cache_path(model_variant: str) -> str:
    return os.path.join(SCORE_CACHE_DIR, f"{model_variant}.pkl.gz")


def _load_score_cache(model_variant: str):
    """モデルスコアキャッシュを読み込む。データ更新後は無効。"""
    path = _score_cache_path(model_variant)
    if not os.path.exists(path):
        return None
    try:
        if os.path.getmtime(path) >= os.path.getmtime(DATA_PKL):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def _save_score_cache(model_variant: str, df: "pd.DataFrame") -> None:
    """予測スコア済みDataFrameを列を絞って保存する。"""
    path = _score_cache_path(model_variant)
    try:
        cols = [c for c in _SCORE_CACHE_COLS if c in df.columns]
        with gzip.open(path, "wb") as f:
            pickle.dump(df[cols].reset_index(drop=True), f)
    except Exception:
        pass


def _sim_cache_key(start_date, end_date, style, model_variant, budget, kelly_bankroll, min_confidence) -> str:
    try:
        data_mtime = int(os.path.getmtime(DATA_PKL))
    except OSError:
        data_mtime = 0
    raw = f"{start_date}_{end_date}_{style}_{model_variant}_{budget}_{kelly_bankroll}_{min_confidence:.2f}_{data_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_sim_cache(key: str):
    path = os.path.join(SIM_CACHE_DIR, f"{key}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_sim_cache(key: str, result: dict) -> None:
    path = os.path.join(SIM_CACHE_DIR, f"{key}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass


_SIM_CACHE_INDEX = os.path.join(SIM_CACHE_DIR, "index.json")
_sim_index_lock = threading.Lock()


def _save_sim_cache_meta(key: str, start_date: str, end_date: str, style: str,
                          model_variant: str, budget: int, kelly_bankroll: int,
                          min_confidence: float) -> None:
    try:
        data_mtime = int(os.path.getmtime(DATA_PKL))
    except OSError:
        data_mtime = 0
    entry = {
        "start_date": start_date, "end_date": end_date,
        "style": style, "model_variant": model_variant,
        "budget": budget, "kelly_bankroll": kelly_bankroll,
        "min_confidence": round(min_confidence, 4), "data_mtime": data_mtime,
    }
    with _sim_index_lock:
        try:
            if os.path.exists(_SIM_CACHE_INDEX):
                with open(_SIM_CACHE_INDEX, "r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = {}
            index[key] = entry
            with open(_SIM_CACHE_INDEX, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False)
        except Exception:
            pass


def _find_superset_cache(start_date: str, end_date: str, style: str, model_variant: str,
                          budget: int, kelly_bankroll: int, min_confidence: float):
    """同じ条件でより広い日付範囲のキャッシュを探す。見つかれば (key, meta) を返す。"""
    if not os.path.exists(_SIM_CACHE_INDEX):
        return None, None
    try:
        data_mtime = int(os.path.getmtime(DATA_PKL))
    except OSError:
        data_mtime = 0
    req_start = parse_date(start_date)
    req_end = parse_date(end_date)
    if req_start is None or req_end is None:
        return None, None
    try:
        with open(_SIM_CACHE_INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return None, None
    for key, meta in index.items():
        if (meta.get("style") == style and
                meta.get("model_variant") == model_variant and
                meta.get("budget") == budget and
                meta.get("kelly_bankroll") == kelly_bankroll and
                abs(meta.get("min_confidence", 0) - min_confidence) < 0.01 and
                meta.get("data_mtime") == data_mtime):
            cached_start = parse_date(meta.get("start_date", ""))
            cached_end = parse_date(meta.get("end_date", ""))
            if (cached_start is not None and cached_end is not None and
                    cached_start <= req_start and cached_end >= req_end and
                    (cached_start != req_start or cached_end != req_end)):
                if os.path.exists(os.path.join(SIM_CACHE_DIR, f"{key}.pkl")):
                    return key, meta
    return None, None


def _slice_sim_result(superset: dict, req_start, req_end) -> "Optional[dict]":
    """スーパーセットのキャッシュから指定期間のサブセット結果を再集計する。"""
    from datetime import date as _date
    raw_rows = superset.get("_raw_rows")
    raw_tested = superset.get("_raw_tested")
    if raw_rows is None or raw_tested is None:
        return None  # 古いキャッシュ形式

    def _in_range(d_str):
        try:
            d = _date.fromisoformat(str(d_str)[:10])
            return req_start <= d <= req_end
        except Exception:
            return False

    rows = [r for r in raw_rows if _in_range(r["date"])]
    tested_rows = [r for r in raw_tested if _in_range(r["date"])]

    tested = len(tested_rows)
    top1_win = sum(1 for r in tested_rows if r["top1_win"])
    top1_top3 = sum(1 for r in tested_rows if r["top1_top3"])
    top3_hit_sum = sum(r["top3_hits"] for r in tested_rows)
    skipped_conf = sum(1 for r in tested_rows if r.get("skipped_conf"))
    no_bet = sum(1 for r in tested_rows if r.get("no_bet"))

    bought_races = len(rows)
    hit_races = sum(1 for r in rows if r["payout"] > 0)
    total_bet = sum(r["bet"] for r in rows)
    total_payout = sum(r["payout"] for r in rows)
    ticket_count = sum(r["tickets"] for r in rows)
    ticket_hits = sum(r.get("ticket_hits", 0) for r in rows)
    actual_payout_races = sum(1 for r in rows if r.get("payout_source") == "実払戻")

    roi = total_payout / total_bet * 100 if total_bet else 0.0
    profit = total_payout - total_bet
    hit_rate = hit_races / bought_races * 100 if bought_races else 0.0
    ticket_hit_rate = ticket_hits / ticket_count * 100 if ticket_count else 0.0
    actual_payout_rate = actual_payout_races / bought_races * 100 if bought_races else 0.0

    rows_by_payout = sorted(rows, key=lambda r: r["payout"], reverse=True)[:20]
    rows_recent = sorted(rows, key=lambda r: (r["date"], r["race_id"]), reverse=True)[:80]

    by_conf = []
    if rows:
        detail = pd.DataFrame(rows)
        grouped = detail.groupby("conf_label").agg(
            races=("race_id", "count"),
            bet=("bet", "sum"),
            payout=("payout", "sum"),
            hits=("payout", lambda s: int((s > 0).sum())),
        ).reset_index()
        grouped["roi"] = grouped["payout"] / grouped["bet"] * 100
        grouped["hit_rate"] = grouped["hits"] / grouped["races"] * 100
        by_conf = grouped.sort_values("roi", ascending=False).to_dict(orient="records")

    return {
        "start_date": req_start.isoformat(),
        "end_date": req_end.isoformat(),
        "style": superset["style"],
        "style_name": superset["style_name"],
        "model_variant": superset["model_variant"],
        "model_variant_label": superset["model_variant_label"],
        "budget": superset["budget"],
        "min_confidence": superset["min_confidence"],
        "tested": tested,
        "bought": bought_races,
        "skipped_confidence": skipped_conf,
        "no_bet": no_bet,
        "hit_races": hit_races,
        "hit_rate": hit_rate,
        "ticket_count": ticket_count,
        "ticket_hits": ticket_hits,
        "ticket_hit_rate": ticket_hit_rate,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "roi": roi,
        "avg_tickets": ticket_count / bought_races if bought_races else 0.0,
        "top1_win_rate": top1_win / tested * 100 if tested else 0.0,
        "top1_top3_rate": top1_top3 / tested * 100 if tested else 0.0,
        "top3_hits_avg": top3_hit_sum / tested if tested else 0.0,
        "top3_hit_rate": top3_hit_sum / (tested * 3) * 100 if tested else 0.0,
        "top3_hit_fraction": f"{int(top3_hit_sum)}/{int(tested * 3)}" if tested else "0/0",
        "actual_payout_races": actual_payout_races,
        "actual_payout_rate": actual_payout_rate,
        "payout_mode": superset.get("payout_mode", "実払戻優先"),
        "top_payout_rows": rows_by_payout,
        "recent_rows": rows_recent,
        "by_conf": by_conf,
        "_raw_rows": rows,
        "_raw_tested": tested_rows,
    }


def _build_pipeline_df(model_variant: str, df_all: "pd.DataFrame", emit) -> "pd.DataFrame":
    """
    フルパイプラインで特徴量計算済みのDataFrameを返す。
    data/pipeline_cache/<model_variant>.pkl.gz にキャッシュし、
    processed_all.pkl.gz が更新されていなければキャッシュを再利用する。
    """
    cache_path = os.path.join(PIPELINE_CACHE_DIR, f"{model_variant}.pkl.gz")
    if os.path.exists(cache_path):
        try:
            if os.path.getmtime(cache_path) >= os.path.getmtime(DATA_PKL):
                emit({"phase": "model", "current": 0, "total": 1,
                      "message": "パイプラインキャッシュをロード中..."})
                with gzip.open(cache_path, "rb") as f:
                    df = pickle.load(f)
                emit({"phase": "model", "current": 1, "total": 1,
                      "message": "パイプラインキャッシュ読み込み完了"})
                return df
        except Exception:
            pass

    meta_path = os.path.join(MODEL_VARIANTS[model_variant]["model_dir"], "meta.json")
    with open(meta_path, encoding="utf-8") as _f:
        fe = json.load(_f).get("feature_engineering", "")

    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    steps_v5 = []
    if fe in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"):
        from train_no_market_v5 import (
            add_sire_features, add_horse_rolling_features, add_jockey_trainer_rolling,
            add_field_strength, add_weight_trend, add_lap_rolling_features, add_field_pace_features,
        )
        steps_v5 = [
            ("sire/class",    add_sire_features),
            ("horse rolling", add_horse_rolling_features),
            ("jockey/trainer",add_jockey_trainer_rolling),
            ("field strength",add_field_strength),
            ("weight trend",  add_weight_trend),
            ("lap rolling",   add_lap_rolling_features),
            ("field pace",    add_field_pace_features),
        ]
    for label, fn in steps_v5:
        emit({"phase": "model", "current": 0, "total": 1, "message": f"特徴量構築中: {label}..."})
        df = fn(df); gc.collect()

    if fe in ("v6", "v7", "v8", "v9", "v10", "v11"):
        from train_no_market_v6 import add_v6_features
        emit({"phase": "model", "current": 0, "total": 1, "message": "特徴量構築中: v6..."})
        df = add_v6_features(df); gc.collect()
    if fe in ("v7", "v8", "v9", "v10", "v11"):
        from train_no_market_v7 import add_v7_features
        emit({"phase": "model", "current": 0, "total": 1, "message": "特徴量構築中: v7..."})
        df = add_v7_features(df); gc.collect()
    if fe in ("v8", "v9", "v10", "v11"):
        from train_no_market_v8 import add_v8_features
        emit({"phase": "model", "current": 0, "total": 1, "message": "特徴量構築中: v8..."})
        df = add_v8_features(df); gc.collect()
    if fe in ("v9", "v10", "v11"):
        from train_no_market_v10 import add_v10_features
        emit({"phase": "model", "current": 0, "total": 1, "message": "特徴量構築中: v10..."})
        df = add_v10_features(df); gc.collect()
    if fe in ("v11",):
        from train_no_market_v11 import add_v11_features
        emit({"phase": "model", "current": 0, "total": 1, "message": "特徴量構築中: v11..."})
        df = add_v11_features(df); gc.collect()

    from feature_engineering import add_model_features
    emit({"phase": "model", "current": 0, "total": 1, "message": "特徴量構築中: vs_field..."})
    df = add_model_features(df, race_col="race_id"); gc.collect()

    # jockey-horse コンボ（lambdarank と同一ロジック）
    emit({"phase": "model", "current": 0, "total": 1, "message": "特徴量構築中: jockey×horse combo..."})
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"] = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]     = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_t"], inplace=True)
    gc.collect()

    emit({"phase": "model", "current": 0, "total": 1, "message": "パイプラインキャッシュを保存中..."})
    try:
        with gzip.open(cache_path, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass
    emit({"phase": "model", "current": 1, "total": 1, "message": "フルパイプライン構築完了（キャッシュ保存済み）"})
    return df


def _parse_kelly_bankroll(req=None) -> int:
    src = req or request
    try:
        return max(1000, int(str(src.values.get("kelly_bankroll", KELLY_DEFAULT_BANKROLL)).replace(",", "")))
    except (ValueError, TypeError):
        return KELLY_DEFAULT_BANKROLL


@app.context_processor
def inject_form_options():
    return {
        "venue_options": VENUE_OPTIONS,
        "model_variants": VISIBLE_MODEL_VARIANTS,
        "selected_model_variant": normalize_visible_model(request.values.get("model_variant")),
        "kelly_bankroll": _parse_kelly_bankroll(),
    }


DEFAULT_MODEL_KEY = PRIMARY_MODEL_KEY

def normalize_style(style: str | None) -> str:
    return style if style in VISIBLE_STYLE_CONFIG else DEFAULT_STYLE


def normalize_visible_model(model_variant: str | None) -> str:
    key = str(model_variant or DEFAULT_MODEL_KEY)
    return key if key in VISIBLE_MODEL_VARIANTS else DEFAULT_MODEL_KEY


def effective_model_for_style(style: str | None, model_variant: str | None) -> str:
    if normalize_style(style) == "kelly_ai":
        return KELLY_MODEL_KEY
    return normalize_visible_model(model_variant)


def predictor(model_variant: str | None = None) -> KeibaPredictor:
    variant = normalize_model_variant(model_variant)
    if variant not in _predictors:
        try:
            _predictors[variant] = KeibaPredictor(model_variant=variant)
        except Exception as exc:
            if variant != DEFAULT_MODEL_VARIANT:
                # モデルファイルが見つからない場合はデフォルトへフォールバック
                import logging
                logging.warning(f"モデル'{variant}'の読込に失敗。デフォルトモデルを使用します: {exc}")
                if DEFAULT_MODEL_VARIANT not in _predictors:
                    _predictors[DEFAULT_MODEL_VARIANT] = KeibaPredictor(model_variant=DEFAULT_MODEL_VARIANT)
                _predictors[variant] = _predictors[DEFAULT_MODEL_VARIANT]
            else:
                raise
    return _predictors[variant]


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not np.isfinite(seconds):
        return "計算中"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}秒"
    minutes = seconds // 60
    rest = seconds % 60
    if minutes < 60:
        return f"{minutes}分{rest:02d}秒"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}時間{minutes:02d}分"


def set_job_state(job_id: str, **updates):
    with UPDATE_LOCK:
        job = UPDATE_JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()


def get_job_state(job_id: str) -> Dict:
    with UPDATE_LOCK:
        return dict(UPDATE_JOBS.get(job_id, {}))


def request_job_cancel(job_id: str) -> bool:
    with UPDATE_LOCK:
        job = UPDATE_JOBS.get(job_id)
        if not job:
            return False
        if job.get("status") in {"completed", "failed", "cancelled"}:
            return True
        job["cancel_requested"] = True
        job["status"] = "cancelling"
        job["eta_seconds"] = 0
        job["eta_label"] = "停止中"
        job["message"] = "停止要求を受け付けました。処理中のレースが終わり次第停止します。"
        job["updated_at"] = time.time()
        return True


def job_cancel_requested(job_id: str) -> bool:
    return bool(get_job_state(job_id).get("cancel_requested"))


def raise_if_cancelled(job_id: str):
    if job_cancel_requested(job_id):
        raise JobCancelled("ユーザー操作で停止しました。")


def mark_job_cancelled(job_id: str):
    set_job_state(
        job_id,
        status="cancelled",
        phase="cancelled",
        eta_seconds=0,
        eta_label="停止",
        message="停止しました",
        finished_at=time.time(),
    )


def progress_to_job(job_id: str, payload: Dict):
    now = time.time()
    current = int(payload.get("current") or 0)
    total = max(1, int(payload.get("total") or 1))
    percent = min(100.0, max(0.0, current / total * 100.0))
    state = get_job_state(job_id)
    phase = payload.get("phase", state.get("phase", "running"))
    phase_started_at = float(state.get("phase_started_at") or now)
    last_progress_at = float(state.get("last_progress_at") or now)
    last_current = int(state.get("last_current") or 0)
    item_seconds_ema = state.get("item_seconds_ema")

    if phase != state.get("phase"):
        phase_started_at = now
        last_progress_at = now
        last_current = current
        item_seconds_ema = None

    if current > last_current:
        delta_items = current - last_current
        sample_seconds = max(0.001, (now - last_progress_at) / delta_items)
        if np.isfinite(sample_seconds):
            if item_seconds_ema is None:
                item_seconds_ema = sample_seconds
            else:
                item_seconds_ema = item_seconds_ema * 0.65 + sample_seconds * 0.35

    elapsed = max(0.001, now - phase_started_at)
    remaining_items = max(0, total - current)
    eta_seconds = None
    if current >= total:
        eta_seconds = 0
    elif remaining_items > 0:
        avg_item_seconds = elapsed / current if current > 0 else None
        if item_seconds_ema is not None and np.isfinite(item_seconds_ema):
            recent = float(item_seconds_ema)
            if avg_item_seconds is None:
                eta_item_seconds = recent
            elif current < 10:
                eta_item_seconds = max(recent, avg_item_seconds)
            else:
                eta_item_seconds = recent * 0.55 + avg_item_seconds * 0.45
            eta_seconds = eta_item_seconds * remaining_items
        elif avg_item_seconds is not None:
            eta_seconds = avg_item_seconds * remaining_items

    if state.get("status") == "cancelled":
        return
    next_status = "completed" if phase == "done" else "running"
    if state.get("cancel_requested") and phase != "done":
        next_status = "cancelling"

    set_job_state(
        job_id,
        status=next_status,
        phase=phase,
        phase_started_at=phase_started_at,
        phase_elapsed_seconds=elapsed,
        last_progress_at=now,
        last_current=current,
        item_seconds_ema=item_seconds_ema,
        current=current,
        total=total,
        percent=percent,
        eta_seconds=eta_seconds,
        eta_label=format_eta(eta_seconds),
        message=payload.get("message", ""),
    )


def run_update_job(job_id: str, start_date: str | None, end_date: str | None, limit: int):
    set_job_state(
        job_id,
        status="running",
        phase="prepare",
        current=0,
        total=1,
        percent=0.0,
        eta_seconds=None,
        eta_label="計算中",
        message="更新準備中",
        started_at=time.time(),
        phase_started_at=time.time(),
    )
    try:
        result = auto_update_missing(
            start_date=start_date or None,
            end_date=end_date or None,
            limit=limit,
            skip_cached=True,
            progress=lambda payload: progress_to_job(job_id, payload),
        )
        if result.get("lookup_updated"):
            _predictors.pop("no_market", None)
        set_job_state(
            job_id,
            status="completed",
            phase="done",
            current=1,
            total=1,
            percent=100.0,
            eta_seconds=0,
            eta_label="0秒",
            message="更新完了",
            result=result,
            finished_at=time.time(),
        )
    except Exception as exc:
        set_job_state(
            job_id,
            status="failed",
            phase="error",
            percent=100.0,
            eta_seconds=0,
            eta_label="停止",
            message=str(exc),
            error=str(exc),
            finished_at=time.time(),
        )


def run_predict_day_job(
    job_id: str,
    kaisai_date: str,
    budget: int,
    style: str,
    model_variant: str,
    sort_mode: str,
    force_update: bool,
    kelly_bankroll: int = KELLY_DEFAULT_BANKROLL,
):
    set_job_state(
        job_id,
        id=job_id,
        kind="predict_day",
        status="running",
        phase="prepare",
        current=0,
        total=1,
        percent=0.0,
        eta_seconds=None,
        eta_label="計算中",
        message="未来レース予想を準備中",
        started_at=time.time(),
        phase_started_at=time.time(),
    )
    try:
        result = predict_day_payload(
            kaisai_date=kaisai_date,
            budget=budget,
            style=style,
            model_variant=model_variant,
            sort_mode=sort_mode,
            force_update=force_update,
            progress=lambda payload: progress_to_job(job_id, payload),
            should_cancel=lambda: job_cancel_requested(job_id),
            kelly_bankroll=kelly_bankroll,
        )
        raise_if_cancelled(job_id)
        set_job_state(
            job_id,
            status="completed",
            phase="done",
            current=1,
            total=1,
            percent=100.0,
            eta_seconds=0,
            eta_label="0秒",
            message="未来レース予想完了",
            result=result,
            finished_at=time.time(),
        )
    except JobCancelled as exc:
        mark_job_cancelled(job_id)
    except Exception as exc:
        set_job_state(
            job_id,
            status="failed",
            phase="error",
            percent=100.0,
            eta_seconds=0,
            eta_label="停止",
            message=str(exc),
            error=str(exc),
            finished_at=time.time(),
        )


def run_sim_job(
    job_id: str,
    start_date: str,
    end_date: str,
    budget: int,
    style: str,
    model_variant: str,
    min_confidence: float,
):
    set_job_state(
        job_id,
        id=job_id,
        kind="simulate",
        status="running",
        phase="prepare",
        current=0,
        total=1,
        percent=0.0,
        eta_seconds=None,
        eta_label="計算中",
        message="シミュレーション準備中",
        started_at=time.time(),
        phase_started_at=time.time(),
    )
    try:
        result = simulate_period(
            start_date=start_date,
            end_date=end_date,
            budget=budget,
            style=style,
            model_variant=model_variant,
            min_confidence=min_confidence,
            progress=lambda payload: progress_to_job(job_id, payload),
            should_cancel=lambda: job_cancel_requested(job_id),
        )
        raise_if_cancelled(job_id)
        set_job_state(
            job_id,
            status="completed",
            phase="done",
            current=1,
            total=1,
            percent=100.0,
            eta_seconds=0,
            eta_label="0秒",
            message="シミュレーション完了",
            result=result,
            finished_at=time.time(),
        )
    except JobCancelled as exc:
        mark_job_cancelled(job_id)
    except Exception as exc:
        set_job_state(
            job_id,
            status="failed",
            phase="error",
            percent=100.0,
            eta_seconds=0,
            eta_label="停止",
            message=str(exc),
            error=str(exc),
            finished_at=time.time(),
        )


def rank_for_style(pred: pd.DataFrame, style: str) -> pd.DataFrame:
    return rank_predictions(pred, style)


def _hit_points_for_bet(bet: Dict, pred: pd.DataFrame) -> int:
    ranks = {
        int(row["horse_no"]): int(row["rank"])
        for _, row in pred.iterrows()
        if pd.notna(row.get("horse_no")) and pd.notna(row.get("rank"))
    }

    def hit(kind: str, horses: List[int]) -> bool:
        if any(h not in ranks for h in horses):
            return False
        if kind == "tansho":
            return ranks[horses[0]] == 1
        if kind == "fukusho":
            return ranks[horses[0]] <= 3
        if kind == "wide":
            return all(ranks[h] <= 3 for h in horses)
        if kind == "umaren":
            return set(ranks[h] for h in horses) == {1, 2}
        if kind == "umatan":
            return ranks[horses[0]] == 1 and ranks[horses[1]] == 2
        if kind == "sanrenpuku":
            return set(ranks[h] for h in horses) == {1, 2, 3}
        if kind == "sanrentan":
            return ranks[horses[0]] == 1 and ranks[horses[1]] == 2 and ranks[horses[2]] == 3
        return False

    if str(bet.get("ticket_kind", "")).endswith("_box"):
        return sum(
            1
            for combo in bet.get("combos", [])
            if hit(str(combo.get("ticket_kind", "")), [int(h) for h in combo.get("horses", [])])
        )
    return int(hit(str(bet.get("ticket_kind", "")), [int(h) for h in bet.get("horses", [])]))


def _settle_bets_and_hit_points(bets: List[Dict], pred: pd.DataFrame, payout_cache: Optional[Dict] = None) -> tuple[int, int]:
    total_payout = sum(settle_bet(bet, pred, payout_cache=payout_cache) for bet in bets)
    hit_points = sum(_hit_points_for_bet(bet, pred) for bet in bets)
    return int(total_payout), int(hit_points)


def allocation_label(style: str) -> str:
    if style == "kelly_ai":
        return "LambdaRank v2スコア × マーケットオッズでKelly係数を計算。EV+10%以上の馬連・馬単のみ。クォーターKelly(×0.25)。"
    if STYLE_CONFIG.get(style, {}).get("_is_smart"):
        return "自信度45点以上のレースのみ自動選択。HIGH(≥75)→ROI重視、MID→バランス。LOW(<45)はデフォルトでスキップ。"
    cfg = STYLE_CONFIG.get(style, {})
    mode = cfg.get("stake_mode", "kelly")
    min_ev = cfg.get("min_ev", 0.85)
    if mode == "kind_weight":
        weights = cfg.get("kind_weights", {})
        labels = {
            "tansho": "単勝", "fukusho": "複勝", "wide": "ワイド", "umaren": "馬連",
            "umatan": "馬単", "sanrenpuku": "三連複", "sanrentan": "三連単",
            "wide_box": "ワイドBOX", "umaren_box": "馬連BOX", "umatan_box": "馬単BOX",
            "sanrenpuku_box": "三連複BOX", "sanrentan_box": "三連単BOX",
        }
        parts = [f"{labels.get(k, k)}{weights[k]:.1f}" for k in cfg.get("tickets", []) if k in weights]
        weight_label = "、".join(parts) if parts else "全券種1.0"
        if len(parts) > 6:
            weight_label = "全券種+BOXを比較、単勝・複勝は高EV時に厚め、高配当券は100円単位"
        return (
            f"各買い目に最低100円を確保し、残りをEVと券種ウェイトで配分。"
            f"基本EV下限={min_ev:.2f}、ウェイト={weight_label}。"
        )
    if mode == "base_best":
        base_bet = int(cfg.get("base_bet", 100))
        return (
            f"候補すべてに最低{base_bet}円を置き、残り資金を最高EVの買い目へ集中。"
            f"EV下限={min_ev:.2f}。"
        )
    if mode == "top1_hit":
        return f"買い目の推定的中率が最も高い1点へ全額集中。EV下限={min_ev:.2f}。"
    if mode == "rank_plan":
        labels = {
            "tansho": "単勝", "fukusho": "複勝", "wide": "ワイド", "umaren": "馬連",
            "umatan": "馬単", "sanrenpuku": "三連複", "sanrentan": "三連単",
        }
        parts = []
        for plan in cfg.get("rank_plan", []):
            kind = str(plan.get("ticket_kind", ""))
            ranks = "-".join(str(r) for r in plan.get("ranks", []))
            weight = float(plan.get("weight", 1.0))
            parts.append(f"{labels.get(kind, kind)}予想{ranks}位 x{weight:g}")
        return f"順位固定配分: {'、'.join(parts)}。100円単位で予算に合わせて配分。"
    if mode == "fixed_kind_amounts":
        labels = {
            "tansho": "単勝", "fukusho": "複勝", "wide": "ワイド", "umaren": "馬連",
            "umatan": "馬単", "sanrenpuku": "三連複", "sanrentan": "三連単",
        }
        fixed = cfg.get("fixed_amounts", {})
        parts = [f"{labels.get(k, k)}{int(v):,}円" for k, v in fixed.items()]
        return f"固定配分: {'、'.join(parts)}。回収率用と的中率用の買い目を同時に買う。"
    if mode == "ev_prop":
        return f"EV下限={min_ev:.2f}を超えた買い目に、EV差に比例して配分。"
    if mode == "equal":
        return f"EV下限={min_ev:.2f}を超えた買い目へ均等配分。"
    if mode == "top1":
        return f"EV下限={min_ev:.2f}を超えた最上位1点へ集中。"
    return f"EV下限={min_ev:.2f}を超えた買い目に、分数ケリー基準で配分。"


def prediction_payload_from_race(race, budget: int, style: str, model_variant: str = DEFAULT_MODEL_VARIANT, kelly_bankroll: int = KELLY_DEFAULT_BANKROLL) -> Dict:
    model_variant = effective_model_for_style(style, model_variant)
    raw_pred = predictor(model_variant).predict_race(race.rows)
    rank_style = "smart" if style == "kelly_ai" else style
    pred = rank_for_style(raw_pred, rank_style)
    conf = race_confidence(pred)
    result = suggest(pred, budget=budget, style=rank_style)

    prediction_rows: List[Dict] = []
    horse_no_labels: Dict[int, str] = {}
    has_temporary_no = False
    field_size = max(1, len(pred))
    for _, row in pred.iterrows():
        horse_no_value = int(row["horse_no"]) if pd.notna(row.get("horse_no")) else None
        confirmed = bool(row.get("horse_no_confirmed", True))
        if not confirmed:
            has_temporary_no = True
        horse_no_label = "" if horse_no_value is None else str(horse_no_value if confirmed else f"仮{horse_no_value}")
        if horse_no_value is not None:
            horse_no_labels[horse_no_value] = horse_no_label
        odds_source_value = row.get("odds_source", "")
        odds_source = str(odds_source_value) if pd.notna(odds_source_value) and odds_source_value else ""
        prediction_rows.append({
            "rank": int(row["pred_rank"]),
            "horse_no": horse_no_label,
            "horse_no_value": horse_no_value,
            "horse_no_confirmed": confirmed,
            "frame_no": int(row["frame_no"]) if pd.notna(row.get("frame_no")) else "",
            "name": str(row.get("馬名", "")),
            "p_win": display_probability_percent(row["p_win"], field_size, 1),
            "p_top3": display_probability_percent(row["p_top3"], field_size, 3),
            "odds": float(row["odds"]) if "odds" in row and pd.notna(row.get("odds")) else None,
            "odds_source": odds_source,
            "popularity": int(row["popularity"]) if "popularity" in row and pd.notna(row.get("popularity")) else None,
            "horse_runs": int(row["horse_runs"]) if "horse_runs" in row and pd.notna(row.get("horse_runs")) else None,
        })

    bets = []
    for b in result["bets"]:
        bet_name = b["name"]
        if has_temporary_no:
            bet_name = re.sub(
                r"(?<!\d)(\d+)(?!\d)",
                lambda m: horse_no_labels.get(int(m.group(1)), m.group(1)),
                bet_name,
            )
        bets.append({
            "type": b["type"],
            "name": bet_name,
            "bet": int(b["bet"]),
            "p_hit": min(95.0, float(b["p_hit"]) * 100),
            "odds_est": float(b["odds_est"]),
            "ev": float(b["ev"]),
            "payout_est": int(b["payout_est"]),
            "kind": b["ticket_kind"],
            "unit_count": int(b.get("unit_count", 1)),
        })

    kelly_bets = []
    if model_variant == "no_market_lambdarank_v2":
        try:
            tansho_entries = [
                {
                    "horses": [int(row["horse_no"])],
                    "odds": float(row["odds"]) if pd.notna(row.get("odds")) else 100.0,
                }
                for _, row in pred.iterrows()
                if pd.notna(row.get("horse_no"))
            ]
            kelly_bets = predictor(model_variant).recommend_bets(
                race.rows,
                {"tansho": tansho_entries},
                bankroll=kelly_bankroll,
                kelly_factor=0.25,
                min_ev=0.10,
                top_k=4,
                ticket_types=["umaren", "umatan"],
                min_bet=100,
            )
            # 予算上限を超える場合は比例縮小
            total_kelly = sum(b["bet_amount"] for b in kelly_bets)
            if total_kelly > budget:
                scale = budget / total_kelly
                kelly_bets = [
                    {**b, "bet_amount": int(b["bet_amount"] * scale / 100) * 100}
                    for b in kelly_bets
                    if int(b["bet_amount"] * scale / 100) * 100 >= 100
                ]
        except Exception:
            kelly_bets = []

    chosen_style = result.get("chosen_style") or style
    return {
        "race": race,
        "meta": race.meta,
        "confidence": conf,
        "style": style,
        "style_name": result.get("style", VISIBLE_STYLE_CONFIG[style]["name"]),
        "style_desc": result.get("style_desc", VISIBLE_STYLE_CONFIG[style].get("desc", "")),
        "chosen_style": chosen_style,
        "chosen_confidence": result.get("confidence"),
        "model_variant": model_variant,
        "model_variant_label": MODEL_VARIANTS[model_variant]["label"],
        "allocation": allocation_label(chosen_style),
        "warning": (
            "馬番・枠番が未確定のため、買い目の番号は登録順の仮番号です。"
            "出馬確定後に情報更新してから購入判断してください。"
            if has_temporary_no else ""
        ),
        "prediction_rows": prediction_rows,
        "bets": bets,
        "kelly_bets": kelly_bets,
        "result": result,
        "plain_text": format_suggestion(result),
    }


def race_result_row_from_payload(race, payload: Dict) -> Dict:
    top = payload["prediction_rows"][0] if payload["prediction_rows"] else {}
    return {
        "race_id": race.race_id,
        "title": payload["meta"].get("race_name") or f"{int(payload['meta'].get('round_no') or 0)}R",
        "venue": payload["meta"].get("venue_name") or payload["meta"].get("venue") or "",
        "round_no": float(payload["meta"].get("round_no") or 0),
        "course": course_label(payload["meta"]),
        "confidence": payload["confidence"]["score"],
        "conf_label": payload["confidence"]["label"],
        "top_name": top.get("name", ""),
        "top_horse_no": top.get("horse_no", ""),
        "top3": top.get("p_top3", 0.0),
        "total_bet": payload["result"]["total_bet"],
        "tickets": payload["result"]["n_tickets"],
        "expected_roi": payload["result"]["expected_roi"] * 100,
        "payload": payload,
    }


def predict_selected_race_payload(
    kaisai_date: str,
    venue_code: str,
    round_no: int,
    budget: int,
    style: str,
    model_variant: str,
    force_update: bool,
    kelly_bankroll: int = KELLY_DEFAULT_BANKROLL,
) -> Dict:
    target_date = parse_date(kaisai_date)
    if target_date is None:
        raise ValueError("日付を選択してください。")

    venue_code = str(venue_code or "").strip().zfill(2)
    if venue_code not in VENUE_NAMES:
        raise ValueError("競馬場を選択してください。")

    try:
        round_no_int = int(round_no)
    except Exception:
        raise ValueError("R番号を入力してください。")
    if round_no_int < 1 or round_no_int > 12:
        raise ValueError("R番号は1〜12で入力してください。")

    race_ids = discover_race_ids_for_date(target_date)
    matched = [
        int(rid)
        for rid in race_ids
        if str(int(rid))[4:6] == venue_code and int(str(int(rid))[-2:]) == round_no_int
    ]
    if not matched:
        available = []
        for rid in race_ids:
            text = str(int(rid))
            venue_name = VENUE_NAMES.get(text[4:6], text[4:6])
            available.append(f"{venue_name}{int(text[-2:])}R")
        hint = " / ".join(available[:24])
        raise ValueError(
            f"{target_date.isoformat()} {VENUE_NAMES[venue_code]}{round_no_int}R が見つかりません。"
            + (f" 候補: {hint}" if hint else "")
        )

    race = load_or_scrape(str(matched[0]), force=force_update)
    payload = prediction_payload_from_race(race, budget, style, model_variant, kelly_bankroll=kelly_bankroll)
    notices = [payload["warning"]] if payload.get("warning") else []
    return {
        "date": target_date.isoformat(),
        "race_ids": matched,
        "races": [race_result_row_from_payload(race, payload)],
        "errors": [],
        "notices": notices,
        "sort_mode": "race",
        "style": style,
        "model_variant": normalize_model_variant(model_variant),
        "budget": budget,
        "mode_title": f"{target_date.isoformat()} {VENUE_NAMES[venue_code]} {round_no_int}R 予想",
        "selected_race": True,
    }


def predict_day_payload(
    kaisai_date: str,
    budget: int,
    style: str,
    model_variant: str,
    sort_mode: str,
    force_update: bool,
    progress: Optional[Callable[[Dict], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    kelly_bankroll: int = KELLY_DEFAULT_BANKROLL,
) -> Dict:
    def _check_cancel():
        if should_cancel is not None and should_cancel():
            raise JobCancelled("ユーザー操作で停止しました。")

    target_date = parse_date(kaisai_date)
    if target_date is None:
        raise ValueError("日付を選択してください。")
    _emit_progress = progress or (lambda payload: None)
    _check_cancel()
    _emit_progress({
        "phase": "discover",
        "current": 0,
        "total": 1,
        "message": f"{target_date.isoformat()} の開催レースを検索中",
    })
    race_ids = discover_race_ids_for_date(target_date)
    _check_cancel()
    _emit_progress({
        "phase": "discover",
        "current": 1,
        "total": 1,
        "message": f"{target_date.isoformat()} の開催レース {len(race_ids)}R を発見",
    })
    races = []
    errors = []
    notices = []
    total_races = max(1, len(race_ids))
    _emit_progress({
        "phase": "predict",
        "current": 0,
        "total": total_races,
        "message": f"{len(race_ids)}Rを予想します",
    })
    for index, race_id in enumerate(race_ids, start=1):
        _check_cancel()
        _emit_progress({
            "phase": "predict",
            "current": index - 1,
            "total": total_races,
            "message": f"race_id={int(race_id)} を取得・予想中 ({index}/{len(race_ids)})",
        })
        try:
            race = load_or_scrape(str(race_id), force=force_update)
            _check_cancel()
            payload = prediction_payload_from_race(race, budget, style, model_variant, kelly_bankroll=kelly_bankroll)
            _check_cancel()
            if payload.get("warning") and payload["warning"] not in notices:
                notices.append(payload["warning"])
            races.append(race_result_row_from_payload(race, payload))
            _emit_progress({
                "phase": "predict",
                "current": index,
                "total": total_races,
                "message": f"{payload['meta'].get('venue_name') or ''} {payload['meta'].get('round_no') or ''}R 予想完了 ({index}/{len(race_ids)})",
            })
        except JobCancelled:
            raise
        except Exception as exc:
            errors.append({"race_id": race_id, "message": str(exc)})
            _emit_progress({
                "phase": "predict",
                "current": index,
                "total": total_races,
                "message": f"race_id={int(race_id)} は失敗: {exc} ({index}/{len(race_ids)})",
            })

    if sort_mode == "confidence":
        races.sort(key=lambda r: (r["confidence"], r["expected_roi"]), reverse=True)
    elif sort_mode == "expected_roi":
        races.sort(key=lambda r: (r["expected_roi"], r["confidence"]), reverse=True)
    else:
        races.sort(key=lambda r: (str(r["venue"]), r["round_no"], r["race_id"]))

    return {
        "date": target_date.isoformat(),
        "race_ids": race_ids,
        "races": races,
        "errors": errors,
        "notices": notices,
        "sort_mode": sort_mode,
        "style": style,
        "model_variant": normalize_model_variant(model_variant),
        "budget": budget,
    }


def course_label(meta: Dict) -> str:
    surface = {"turf": "芝", "dirt": "ダ", "jump": "障"}.get(str(meta.get("surface")), str(meta.get("surface") or ""))
    distance = meta.get("distance")
    if distance is None or pd.isna(distance):
        return surface
    return f"{surface}{int(float(distance))}m"


def simulate_period(
    start_date: str,
    end_date: str,
    budget: int,
    style: str,
    model_variant: str,
    min_confidence: float,
    progress: Optional[Callable[[Dict], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    kelly_bankroll: int = KELLY_DEFAULT_BANKROLL,
) -> Dict:
    def _check_cancel():
        if should_cancel is not None and should_cancel():
            raise JobCancelled("ユーザー操作で停止しました。")

    _check_cancel()
    start = parse_date(start_date)
    end = parse_date(end_date)
    if start is None or end is None:
        raise ValueError("開始日と終了日を入力してください。")
    if start > end:
        start, end = end, start

    _emit_sim_progress = progress or (lambda payload: None)

    # ── シミュレーション結果キャッシュ ──────────────────────────────────
    model_variant = effective_model_for_style(style, model_variant)
    cache_key = _sim_cache_key(start_date, end_date, style, model_variant, budget, kelly_bankroll, min_confidence)
    cached = _load_sim_cache(cache_key)
    if cached is not None:
        _emit_sim_progress({"phase": "simulate", "current": 1, "total": 1,
                            "message": "キャッシュから結果を読み込みました（即座に表示）"})
        return cached

    # スーパーセットキャッシュ（より広い期間）から部分取得
    sup_key, _ = _find_superset_cache(start_date, end_date, style, model_variant, budget, kelly_bankroll, min_confidence)
    if sup_key is not None:
        sup_cached = _load_sim_cache(sup_key)
        if sup_cached is not None:
            sliced = _slice_sim_result(sup_cached, start, end)
            if sliced is not None:
                _save_sim_cache(cache_key, sliced)
                _save_sim_cache_meta(cache_key, start.isoformat(), end.isoformat(),
                                     style, model_variant, budget, kelly_bankroll, min_confidence)
                _emit_sim_progress({"phase": "simulate", "current": 1, "total": 1,
                                    "message": "既存キャッシュから期間を絞り込んで表示（即座に表示）"})
                return sliced

    # ── データロード ────────────────────────────────────────────────────
    _emit_sim_progress({"phase": "model", "current": 0, "total": 1, "message": "データをロード中..."})
    with gzip.open(DATA_PKL, "rb") as f:
        df_all = pickle.load(f)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

    pred_model = predictor(model_variant)

    if pred_model.model_type == "lambdarank":
        # ── スコアキャッシュ確認（最優先）─────────────────────────────────
        scored_all = _load_score_cache(model_variant)
        if scored_all is not None:
            _emit_sim_progress({"phase": "model", "current": 1, "total": 1,
                                "message": "スコアキャッシュをロード中..."})
            scored_all["date"] = pd.to_datetime(scored_all["date"], errors="coerce")
        else:
            # ── フルパイプライン（訓練時と同じ特徴量、キャッシュ利用）──────────
            df_pipeline = _build_pipeline_df(model_variant, df_all, _emit_sim_progress)
            _check_cancel()

            df_all_pipe = df_pipeline.copy()
            df_all_pipe = df_all_pipe[df_all_pipe["rank"].notna()].copy() if "rank" in df_all_pipe.columns else df_all_pipe
            df_all_pipe = df_all_pipe.sort_values(["date", "race_id"]).reset_index(drop=True)

            _emit_sim_progress({"phase": "model", "current": 0, "total": 1,
                                "message": f"全データ {df_all_pipe['race_id'].nunique()}R を一括AI予測します..."})
            feats = pred_model.feats
            X_all = df_all_pipe.reindex(columns=feats)
            raw_all = np.mean(
                [m.predict(X_all, num_iteration=m.best_iteration) for m in pred_model.lr_models], axis=0
            )
            df_all_pipe = df_all_pipe.copy()
            df_all_pipe["_raw"] = raw_all

            def _sm(g):
                e = np.exp(g["_raw"] - g["_raw"].max()); return e / e.sum()
            df_all_pipe["score"]    = df_all_pipe.groupby("race_id", group_keys=False).apply(_sm)
            df_all_pipe["p_win"]    = np.clip(df_all_pipe["score"], 1e-6, 0.999)
            df_all_pipe["p_top3"]   = np.clip(df_all_pipe["score"] * 3.0, 1e-6, 0.999)
            df_all_pipe["pred_rank"] = df_all_pipe.groupby("race_id")["score"].rank(
                ascending=False, method="min").astype(int)
            df_all_pipe.drop(columns=["_raw"], inplace=True, errors="ignore")

            _emit_sim_progress({"phase": "model", "current": 0, "total": 1,
                                "message": "スコアキャッシュを保存中..."})
            _save_score_cache(model_variant, df_all_pipe)
            scored_all = _load_score_cache(model_variant)
            if scored_all is None:
                scored_all = df_all_pipe[[c for c in _SCORE_CACHE_COLS if c in df_all_pipe.columns]].copy()
            scored_all["date"] = pd.to_datetime(scored_all["date"], errors="coerce")
            gc.collect()

        _check_cancel()
        mask = (scored_all["date"].dt.date >= start) & (scored_all["date"].dt.date <= end)
        sim_df = scored_all[mask].copy()
        if sim_df.empty:
            raise ValueError("指定期間のデータがありません。")
        sim_df = sim_df.sort_values(["date", "race_id"]).reset_index(drop=True)
        predicted_all = sim_df
    else:
        # ── 従来の predict_frame (lookup テーブル) ──────────────────────
        # スコアキャッシュ確認
        scored_all = _load_score_cache(model_variant)
        if scored_all is not None:
            _emit_sim_progress({"phase": "model", "current": 1, "total": 1,
                                "message": "スコアキャッシュをロード中..."})
            scored_all["date"] = pd.to_datetime(scored_all["date"], errors="coerce")
            mask = (scored_all["date"].dt.date >= start) & (scored_all["date"].dt.date <= end)
            sim_df = scored_all[mask].copy()
            if sim_df.empty:
                raise ValueError("指定期間のデータがありません。")
            predicted_all = sim_df.sort_values(["date", "race_id"]).reset_index(drop=True)
        else:
            dates = df_all["date"].dt.date
            sim_df = df_all[(dates >= start) & (dates <= end)].copy()
            if sim_df.empty:
                raise ValueError("指定期間のデータがありません。")
            if "rank" in sim_df.columns:
                sim_df = sim_df[sim_df["rank"].notna()].copy()
            if sim_df.empty:
                raise ValueError("有効な結果付きデータがありません。")
            sim_df = sim_df.sort_values(["date", "race_id"]).reset_index(drop=True)
            _emit_sim_progress({"phase": "model", "current": 0, "total": 1,
                                "message": f"{sim_df['race_id'].nunique()}R分をまとめてAI予測します"})
            all_df = df_all[df_all["rank"].notna()].copy() if "rank" in df_all.columns else df_all.copy()
            all_df = all_df.sort_values(["date", "race_id"]).reset_index(drop=True)
            predicted_full = predictor(model_variant).predict_frame(all_df)
            _save_score_cache(model_variant, predicted_full)
            dates2 = pd.to_datetime(predicted_full["date"], errors="coerce").dt.date
            predicted_all = predicted_full[(dates2 >= start) & (dates2 <= end)].copy()

    _check_cancel()
    _emit_sim_progress({"phase": "model", "current": 1, "total": 1, "message": "AI予測を一括完了しました"})
    groups = [(race_id, race_df.reset_index(drop=True)) for race_id, race_df in predicted_all.groupby("race_id", sort=False)]
    _emit_sim_progress({
        "phase": "simulate",
        "current": 0,
        "total": max(1, len(groups)),
        "message": f"{len(groups)}Rを検証します",
    })

    total_bet = 0
    total_payout = 0
    bought_races = 0
    hit_races = 0
    skipped_conf = 0
    no_bet = 0
    ticket_count = 0
    ticket_hits = 0
    top1_win = 0
    top1_top3 = 0
    top3_hit_sum = 0
    tested = 0
    actual_payout_races = 0
    rows = []
    tested_rows_data = []

    total_groups = max(1, len(groups))
    for index, (race_id, race_df) in enumerate(groups, start=1):
        _check_cancel()
        date_text_for_progress = str(race_df["date"].iloc[0])[:10]
        _emit_sim_progress({
            "phase": "simulate",
            "current": index - 1,
            "total": total_groups,
            "message": f"{date_text_for_progress} race_id={int(race_id)} を検証中 ({index}/{len(groups)})",
        })
        if race_df["rank"].isna().all():
            _emit_sim_progress({
                "phase": "simulate",
                "current": index,
                "total": total_groups,
                "message": f"{date_text_for_progress} race_id={int(race_id)} は結果なしでスキップ ({index}/{len(groups)})",
            })
            continue
        race_df = race_df[race_df["rank"].notna()].reset_index(drop=True)
        if len(race_df) < 3:
            continue
        tested += 1
        is_kelly = style == "kelly_ai"
        rank_style = "smart" if is_kelly else style
        pred = rank_for_style(race_df, rank_style)
        conf = race_confidence(pred)

        top = pred.iloc[0]
        top1_win += int(top["rank"] == 1)
        top1_top3 += int(top["rank"] <= 3)
        pred_top3 = set(pred.head(3)["horse_no"].astype(int))
        actual_top3 = set(pred[pred["rank"] <= 3]["horse_no"].astype(int))
        top3_hits = len(pred_top3 & actual_top3)
        top3_hit_sum += top3_hits
        _t = {"race_id": int(race_id), "date": date_text_for_progress,
              "top1_win": int(top["rank"]) == 1, "top1_top3": int(top["rank"]) <= 3,
              "top3_hits": top3_hits, "skipped_conf": False, "no_bet": False}

        if not is_kelly and conf["score"] < min_confidence:
            skipped_conf += 1
            _t["skipped_conf"] = True
            tested_rows_data.append(_t)
            _emit_sim_progress({
                "phase": "simulate",
                "current": index,
                "total": total_groups,
                "message": f"{date_text_for_progress} race_id={int(race_id)} は自信度不足でスキップ ({index}/{len(groups)})",
            })
            continue

        if is_kelly:
            if pred_model.model_type == "lambdarank":
                # フルパイプラインのスコアを直接使用（lookup テーブルなし）
                from kelly_betting import enumerate_race_bets, size_bets, market_probs_from_odds
                h_rows = [
                    (int(row["horse_no"]),
                     float(row["odds"]) if pd.notna(row.get("odds")) else 100.0,
                     float(row.get("score", 0.0)))
                    for _, row in race_df.iterrows() if pd.notna(row.get("horse_no"))
                ]
                horse_nos      = [h for h, _, _ in h_rows]
                tansho_odds_arr = np.array([o for _, o, _ in h_rows])
                model_probs    = np.array([s for _, _, s in h_rows])
                market_probs   = market_probs_from_odds(tansho_odds_arr)
                bets_raw = enumerate_race_bets(
                    model_probs, market_probs, horse_nos, tansho_odds_arr,
                    ticket_types=["umaren", "umatan"], top_k=4,
                )
                bets_raw   = [b for b in bets_raw if b["ev"] >= 0.10]
                kelly_bets = size_bets(bets_raw, kelly_bankroll, kelly_factor=0.25, min_bet=100)
            else:
                tansho_entries = [
                    {"horses": [int(row["horse_no"])], "odds": float(row["odds"]) if pd.notna(row.get("odds")) else 100.0}
                    for _, row in race_df.iterrows()
                    if pd.notna(row.get("horse_no"))
                ]
                kelly_bets = predictor(model_variant).recommend_bets(
                    race_df, {"tansho": tansho_entries},
                    bankroll=kelly_bankroll, kelly_factor=0.25, min_ev=0.10, top_k=4,
                    ticket_types=["umaren", "umatan"], min_bet=100,
                )
            # 予算上限を超える場合は比例縮小
            total_kelly = sum(b["bet_amount"] for b in kelly_bets)
            if total_kelly > budget:
                scale = budget / total_kelly
                kelly_bets = [
                    {**b, "bet_amount": int(b["bet_amount"] * scale / 100) * 100}
                    for b in kelly_bets
                    if int(b["bet_amount"] * scale / 100) * 100 >= 100
                ]
            _check_cancel()
            if not kelly_bets:
                no_bet += 1
                _t["no_bet"] = True
                tested_rows_data.append(_t)
                _emit_sim_progress({
                    "phase": "simulate",
                    "current": index,
                    "total": total_groups,
                    "message": f"{date_text_for_progress} race_id={int(race_id)} はKelly対象買い目なしでスキップ ({index}/{len(groups)})",
                })
                continue
            settle_bets = [
                {"ticket_kind": b["ticket_kind"], "horses": b["horses"],
                 "bet": b["bet_amount"], "odds_est": b["market_odds"]}
                for b in kelly_bets
            ]
            total_b = sum(b["bet_amount"] for b in kelly_bets)
            avg_ev = sum(b["ev"] for b in kelly_bets) / len(kelly_bets)
            result = {
                "bets": settle_bets,
                "total_bet": total_b,
                "n_tickets": len(kelly_bets),
                "expected_roi": avg_ev,
            }
        else:
            result = suggest(pred, budget=budget, style=style)
        _check_cancel()
        race_bet = int(result["total_bet"])
        if race_bet <= 0:
            no_bet += 1
            _t["no_bet"] = True
            tested_rows_data.append(_t)
            _emit_sim_progress({
                "phase": "simulate",
                "current": index,
                "total": total_groups,
                "message": f"{date_text_for_progress} race_id={int(race_id)} は買い目なしでスキップ ({index}/{len(groups)})",
            })
            continue

        payout_cache = load_payout_cache(int(race_id))
        payout_source = "実払戻" if payout_cache else "推定"
        if payout_cache:
            actual_payout_races += 1
        race_payout, race_ticket_hits = _settle_bets_and_hit_points(result["bets"], pred, payout_cache=payout_cache)
        total_bet += race_bet
        total_payout += race_payout
        bought_races += 1
        hit_races += int(race_payout > 0)
        ticket_count += int(result["n_tickets"])
        ticket_hits += race_ticket_hits

        date_text = str(race_df["date"].iloc[0])[:10]
        tested_rows_data.append(_t)
        rows.append({
            "race_id": int(race_id),
            "date": date_text,
            "confidence": float(conf["score"]),
            "conf_label": conf["label"],
            "bet": race_bet,
            "payout": race_payout,
            "profit": race_payout - race_bet,
            "roi": race_payout / race_bet * 100 if race_bet else 0.0,
            "tickets": int(result["n_tickets"]),
            "ticket_hits": race_ticket_hits,
            "expected_roi": float(result["expected_roi"]) * 100,
            "top_horse_no": int(top["horse_no"]) if pd.notna(top.get("horse_no")) else "",
            "top_win": float(top["p_win"]) * 100,
            "top_top3": float(top["p_top3"]) * 100,
            "top3_hits": int(top3_hits),
            "top3_fraction": f"{int(top3_hits)}/3",
            "top3_hit_rate": top3_hits / 3 * 100,
            "payout_source": payout_source,
        })
        _emit_sim_progress({
            "phase": "simulate",
            "current": index,
            "total": total_groups,
            "message": f"{date_text_for_progress} race_id={int(race_id)} 完了 ROI {rows[-1]['roi']:.1f}% / {payout_source} ({index}/{len(groups)})",
        })

    roi = total_payout / total_bet * 100 if total_bet else 0.0
    profit = total_payout - total_bet
    hit_rate = hit_races / bought_races * 100 if bought_races else 0.0
    ticket_hit_rate = ticket_hits / ticket_count * 100 if ticket_count else 0.0
    actual_payout_rate = actual_payout_races / bought_races * 100 if bought_races else 0.0

    rows_by_payout = sorted(rows, key=lambda r: r["payout"], reverse=True)[:20]
    rows_recent = sorted(rows, key=lambda r: (r["date"], r["race_id"]), reverse=True)[:80]

    by_conf = []
    if rows:
        detail = pd.DataFrame(rows)
        grouped = detail.groupby("conf_label").agg(
            races=("race_id", "count"),
            bet=("bet", "sum"),
            payout=("payout", "sum"),
            hits=("payout", lambda s: int((s > 0).sum())),
        ).reset_index()
        grouped["roi"] = grouped["payout"] / grouped["bet"] * 100
        grouped["hit_rate"] = grouped["hits"] / grouped["races"] * 100
        by_conf = grouped.sort_values("roi", ascending=False).to_dict(orient="records")

    result = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "style": style,
        "style_name": (VISIBLE_STYLE_CONFIG.get(style) or STYLE_CONFIG.get(style, {})).get("name", style),
        "model_variant": model_variant,
        "model_variant_label": MODEL_VARIANTS[model_variant]["label"],
        "budget": budget,
        "min_confidence": min_confidence,
        "tested": tested,
        "bought": bought_races,
        "skipped_confidence": skipped_conf,
        "no_bet": no_bet,
        "hit_races": hit_races,
        "hit_rate": hit_rate,
        "ticket_count": ticket_count,
        "ticket_hits": ticket_hits,
        "ticket_hit_rate": ticket_hit_rate,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "roi": roi,
        "avg_tickets": ticket_count / bought_races if bought_races else 0.0,
        "top1_win_rate": top1_win / tested * 100 if tested else 0.0,
        "top1_top3_rate": top1_top3 / tested * 100 if tested else 0.0,
        "top3_hits_avg": top3_hit_sum / tested if tested else 0.0,
        "top3_hit_rate": top3_hit_sum / (tested * 3) * 100 if tested else 0.0,
        "top3_hit_fraction": f"{int(top3_hit_sum)}/{int(tested * 3)}" if tested else "0/0",
        "actual_payout_races": actual_payout_races,
        "actual_payout_rate": actual_payout_rate,
        "payout_mode": "実払戻優先",
        "top_payout_rows": rows_by_payout,
        "recent_rows": rows_recent,
        "by_conf": by_conf,
        "_raw_rows": rows,
        "_raw_tested": tested_rows_data,
    }
    _save_sim_cache(cache_key, result)
    _save_sim_cache_meta(cache_key, start.isoformat(), end.isoformat(),
                         style, model_variant, budget, kelly_bankroll, min_confidence)
    return result


@app.get("/")
def index():
    return render_template(
        "index.html",
        styles=VISIBLE_STYLE_CONFIG,
        cached=list_cached_races(),
        date_candidates=candidate_race_dates(),
        selected_style=DEFAULT_STYLE,
        budget=3000,
        result=None,
        day_result=None,
        sim_result=None,
        updates=None,
        error=None,
    )


@app.post("/predict-day")
def predict_day_route():
    style = normalize_style(request.form.get("style", DEFAULT_STYLE))
    model_variant = effective_model_for_style(style, request.form.get("model_variant"))
    sort_mode = request.form.get("sort_mode", "race")
    try:
        budget = int(str(request.form.get("budget", "3000")).replace(",", ""))
    except ValueError:
        budget = 3000
    force_update = request.form.get("force_update") == "on"
    kaisai_date = request.form.get("kaisai_date", "")
    kelly_bankroll = _parse_kelly_bankroll()
    try:
        day_result = predict_day_payload(kaisai_date, budget, style, model_variant, sort_mode, force_update,
                                         kelly_bankroll=kelly_bankroll)
        error = None
    except Exception as exc:
        day_result = None
        error = str(exc)
    return render_template(
        "index.html",
        styles=VISIBLE_STYLE_CONFIG,
        cached=list_cached_races(),
        date_candidates=candidate_race_dates(),
        selected_style=style,
        selected_model_variant=model_variant,
        budget=budget,
        result=None,
        day_result=day_result,
        sim_result=None,
        updates=None,
        error=error,
    )


@app.post("/predict-race")
def predict_race_select_route():
    style = normalize_style(request.form.get("style", DEFAULT_STYLE))
    model_variant = effective_model_for_style(style, request.form.get("model_variant"))
    try:
        budget = int(str(request.form.get("budget", "3000")).replace(",", ""))
    except ValueError:
        budget = 3000
    force_update = request.form.get("force_update") == "on"
    kaisai_date = request.form.get("kaisai_date", "")
    venue_code = request.form.get("venue_code", "")
    round_no = request.form.get("round_no", "")
    kelly_bankroll = _parse_kelly_bankroll()
    try:
        day_result = predict_selected_race_payload(
            kaisai_date=kaisai_date,
            venue_code=venue_code,
            round_no=int(round_no),
            budget=max(100, budget),
            style=style,
            model_variant=model_variant,
            force_update=force_update,
            kelly_bankroll=kelly_bankroll,
        )
        error = None
    except Exception as exc:
        day_result = None
        error = str(exc)
    return render_template(
        "index.html",
        styles=VISIBLE_STYLE_CONFIG,
        cached=list_cached_races(),
        date_candidates=candidate_race_dates(),
        selected_style=style,
        selected_model_variant=model_variant,
        budget=budget,
        result=None,
        day_result=day_result,
        sim_result=None,
        updates=None,
        error=error,
    )


@app.post("/api/predict-day/start")
def api_predict_day_start():
    data = request.get_json(silent=True) or request.form
    style = normalize_style(str(data.get("style", DEFAULT_STYLE)))
    model_variant = effective_model_for_style(style, str(data.get("model_variant", DEFAULT_MODEL_KEY)))
    sort_mode = str(data.get("sort_mode", "race") or "race")
    try:
        budget = int(str(data.get("budget", "3000")).replace(",", ""))
    except ValueError:
        budget = 3000
    force_update = data.get("force_update") == "on" or str(data.get("force_update", "")).lower() == "true"
    kaisai_date = str(data.get("kaisai_date", ""))
    try:
        kelly_bankroll = max(1000, int(str(data.get("kelly_bankroll", KELLY_DEFAULT_BANKROLL)).replace(",", "")))
    except (ValueError, TypeError):
        kelly_bankroll = KELLY_DEFAULT_BANKROLL

    job_id = uuid.uuid4().hex
    set_job_state(
        job_id,
        id=job_id,
        kind="predict_day",
        status="queued",
        phase="queued",
        current=0,
        total=1,
        percent=0.0,
        eta_label="計算中",
        message="未来予想ジョブを開始します",
        started_at=time.time(),
    )
    thread = threading.Thread(
        target=run_predict_day_job,
        args=(job_id, kaisai_date, max(100, budget), style, model_variant, sort_mode, force_update, kelly_bankroll),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/api/predict-day/status/<job_id>")
def api_predict_day_status(job_id: str):
    state = get_job_state(job_id)
    if not state:
        return jsonify({"status": "missing", "message": "ジョブが見つかりません"}), 404
    result = state.pop("result", None)
    if result:
        races = result.get("races", [])
        state["summary"] = {
            "date": result.get("date"),
            "races": len(races),
            "errors": len(result.get("errors", [])),
            "top_confidence": max((r.get("confidence", 0) for r in races), default=0),
        }
    return jsonify(state)


@app.post("/api/predict-day/cancel/<job_id>")
def api_predict_day_cancel(job_id: str):
    if not request_job_cancel(job_id):
        return jsonify({"status": "missing", "message": "ジョブが見つかりません"}), 404
    return jsonify(get_job_state(job_id))


@app.get("/predict-day/result/<job_id>")
def predict_day_result_route(job_id: str):
    state = get_job_state(job_id)
    day_result = state.get("result") if state else None
    error = None
    if not state:
        error = "未来予想結果が見つかりません。もう一度実行してください。"
    elif state.get("status") == "failed":
        error = state.get("error") or state.get("message") or "未来予想に失敗しました。"
    elif not day_result:
        error = "未来予想がまだ完了していません。"
    return render_template(
        "index.html",
        styles=VISIBLE_STYLE_CONFIG,
        cached=list_cached_races(),
        date_candidates=candidate_race_dates(),
        selected_style=normalize_style(day_result.get("style") if day_result else DEFAULT_STYLE),
        selected_model_variant=normalize_visible_model(day_result.get("model_variant") if day_result else DEFAULT_MODEL_KEY),
        budget=int(day_result.get("budget", 3000)) if day_result else 3000,
        result=None,
        day_result=day_result,
        sim_result=None,
        updates=None,
        error=error,
    )


@app.post("/simulate")
def simulate_route():
    style = normalize_style(request.form.get("style", DEFAULT_STYLE))
    model_variant = effective_model_for_style(style, request.form.get("model_variant"))
    try:
        budget = int(str(request.form.get("budget", "3000")).replace(",", ""))
    except ValueError:
        budget = 3000
    try:
        min_conf = float(str(request.form.get("min_confidence", "")).strip() or STYLE_CONFIG.get(style, {}).get("default_min_confidence", 0.0))
    except ValueError:
        min_conf = float(STYLE_CONFIG.get(style, {}).get("default_min_confidence", 0.0))

    kelly_bankroll = _parse_kelly_bankroll()
    try:
        sim_result = simulate_period(
            start_date=request.form.get("sim_start", ""),
            end_date=request.form.get("sim_end", ""),
            budget=budget,
            style=style,
            model_variant=model_variant,
            min_confidence=min_conf,
            kelly_bankroll=kelly_bankroll,
        )
        error = None
    except Exception as exc:
        sim_result = None
        error = str(exc)

    return render_template(
        "index.html",
        styles=VISIBLE_STYLE_CONFIG,
        cached=list_cached_races(),
        date_candidates=candidate_race_dates(),
        selected_style=style,
        selected_model_variant=model_variant,
        budget=budget,
        result=None,
        day_result=None,
        sim_result=sim_result,
        updates=None,
        error=error,
    )


@app.post("/api/simulate/start")
def api_simulate_start():
    data = request.get_json(silent=True) or request.form
    style = normalize_style(str(data.get("style", DEFAULT_STYLE)))
    model_variant = effective_model_for_style(style, str(data.get("model_variant", DEFAULT_MODEL_KEY)))
    try:
        budget = int(str(data.get("budget", "3000")).replace(",", ""))
    except ValueError:
        budget = 3000
    try:
        min_conf = float(str(data.get("min_confidence", "")).strip() or STYLE_CONFIG.get(style, {}).get("default_min_confidence", 0.0))
    except ValueError:
        min_conf = float(STYLE_CONFIG.get(style, {}).get("default_min_confidence", 0.0))

    job_id = uuid.uuid4().hex
    set_job_state(
        job_id,
        id=job_id,
        kind="simulate",
        status="queued",
        phase="queued",
        current=0,
        total=1,
        percent=0.0,
        eta_label="計算中",
        message="シミュレーションジョブを開始します",
        started_at=time.time(),
    )
    thread = threading.Thread(
        target=run_sim_job,
        args=(
            job_id,
            str(data.get("sim_start", "")),
            str(data.get("sim_end", "")),
            max(100, budget),
            style,
            model_variant,
            min_conf,
        ),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/api/simulate/status/<job_id>")
def api_simulate_status(job_id: str):
    state = get_job_state(job_id)
    if not state:
        return jsonify({"status": "missing", "message": "ジョブが見つかりません"}), 404
    result = state.get("result")
    if result:
        state["summary"] = {
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "tested": result.get("tested"),
            "bought": result.get("bought"),
            "roi": result.get("roi"),
            "profit": result.get("profit"),
            "actual_payout_races": result.get("actual_payout_races"),
            "actual_payout_rate": result.get("actual_payout_rate"),
        }
    return jsonify(state)


@app.post("/api/simulate/cancel/<job_id>")
def api_simulate_cancel(job_id: str):
    if not request_job_cancel(job_id):
        return jsonify({"status": "missing", "message": "ジョブが見つかりません"}), 404
    return jsonify(get_job_state(job_id))


@app.get("/simulate/result/<job_id>")
def simulate_result_route(job_id: str):
    state = get_job_state(job_id)
    sim_result = state.get("result") if state else None
    error = None
    if not state:
        error = "シミュレーション結果が見つかりません。もう一度実行してください。"
    elif state.get("status") == "failed":
        error = state.get("error") or state.get("message") or "シミュレーションに失敗しました。"
    elif not sim_result:
        error = "シミュレーションがまだ完了していません。"
    return render_template(
        "index.html",
        styles=VISIBLE_STYLE_CONFIG,
        cached=list_cached_races(),
        date_candidates=candidate_race_dates(),
        selected_style=normalize_style(sim_result.get("style") if sim_result else DEFAULT_STYLE),
        selected_model_variant=normalize_visible_model(sim_result.get("model_variant") if sim_result else DEFAULT_MODEL_KEY),
        budget=int(sim_result.get("budget", 3000)) if sim_result else 3000,
        result=None,
        day_result=None,
        sim_result=sim_result,
        updates=None,
        error=error,
    )


@app.post("/update")
def update_route():
    updates = None
    error = None
    limit_raw = str(request.form.get("limit", "0")).strip()
    try:
        limit = int(limit_raw) if limit_raw else 0
    except ValueError:
        limit = 0
    try:
        updates = auto_update_missing(
            start_date=request.form.get("start_date") or None,
            end_date=request.form.get("end_date") or None,
            limit=limit,
            skip_cached=True,
        )
        if updates.get("lookup_updated"):
            _predictors.pop("no_market", None)
    except Exception as exc:
        error = str(exc)
    return render_template(
        "index.html",
        styles=VISIBLE_STYLE_CONFIG,
        cached=list_cached_races(),
        date_candidates=candidate_race_dates(),
        selected_style=normalize_style(request.form.get("style", DEFAULT_STYLE)),
        budget=int(request.form.get("budget", "3000") or 3000),
        result=None,
        day_result=None,
        sim_result=None,
        updates=updates,
        error=error,
    )


@app.post("/api/update/start")
def api_update_start():
    data = request.get_json(silent=True) or request.form
    limit_raw = str(data.get("limit", "0")).strip()
    try:
        limit = int(limit_raw) if limit_raw else 0
    except ValueError:
        limit = 0
    job_id = uuid.uuid4().hex
    set_job_state(
        job_id,
        id=job_id,
        status="queued",
        phase="queued",
        current=0,
        total=1,
        percent=0.0,
        eta_label="計算中",
        message="更新ジョブを開始します",
        started_at=time.time(),
    )
    thread = threading.Thread(
        target=run_update_job,
        args=(job_id, data.get("start_date") or None, data.get("end_date") or None, limit),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/api/update/status/<job_id>")
def api_update_status(job_id: str):
    state = get_job_state(job_id)
    if not state:
        return jsonify({"status": "missing", "message": "ジョブが見つかりません"}), 404
    result = state.get("result")
    if result:
        state["summary"] = {
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "scanned_dates": result.get("scanned_dates"),
            "discovered_races": result.get("discovered_races"),
            "skipped_existing": result.get("skipped_existing"),
            "missing_races": result.get("missing_races"),
            "updated": result.get("updated"),
            "failed": result.get("failed"),
            "races_added": result.get("races_added"),
            "rows_added": result.get("rows_added"),
            "latest_date": result.get("latest_date"),
            "backup_path": result.get("backup_path"),
            "lookup_updated": result.get("lookup_updated"),
            "stopped_by_limit": result.get("stopped_by_limit"),
        }
        state["updates"] = result.get("updates", [])[:80]
    return jsonify(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    predictor()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
