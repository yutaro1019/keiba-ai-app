import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import CATEGORICAL_FEATS, DATA_PKL, MARKET_FEATS, NUMERIC_FEATS  # noqa: E402


def load_data():
    with gzip.open(DATA_PKL, "rb") as f:
        df = pd.read_pickle(f)

    keep_cols = ["race_id", "horse_no", "rank", "date"] + NUMERIC_FEATS + MARKET_FEATS + CATEGORICAL_FEATS
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in df.columns))
    df = df[keep_cols].dropna(subset=["rank"]).copy()
    df["target_top3"] = (df["rank"] <= 3).astype("int8")
    df["target_win"] = (df["rank"] == 1).astype("int8")
    df["__year"] = df["date"].dt.year

    tr_mask = df["__year"] < 2025
    cat_meta = {}
    for c in CATEGORICAL_FEATS:
        if c in df.columns:
            df[c] = df[c].astype("category")
            tr_cats = df.loc[tr_mask, c].cat.categories
            cat_meta[c] = list(tr_cats.astype(str))
            df[c] = pd.Categorical(df[c], categories=tr_cats)

    return df, cat_meta


def current_importance(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    feats = list(meta["features"])
    gain = np.zeros(len(feats), dtype=float)
    for fn in meta.get("ensemble_top3", []) + meta.get("ensemble_win", []):
        path = model_dir / fn
        if not path.exists():
            continue
        booster = lgb.Booster(model_file=str(path))
        gain += booster.feature_importance(importance_type="gain")
    order = np.argsort(-gain)
    return [feats[i] for i in order], {feats[i]: float(gain[i]) for i in range(len(feats))}


def train_binary(X_tr, y_tr, X_va, y_va, cat_feats, seed, leaves, rounds):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": leaves,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 2,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feats)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_feats, reference=dtr)
    return lgb.train(
        params,
        dtr,
        num_boost_round=rounds,
        valid_sets=[dva],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
    )


def ranking_metrics(valid_meta, p_top3_raw, p_win_raw, win_weight=0.40):
    pred = valid_meta.copy()
    pred["p_top3_raw"] = p_top3_raw
    pred["p_win_raw"] = p_win_raw
    s3 = pred.groupby("race_id")["p_top3_raw"].transform("sum")
    sw = pred.groupby("race_id")["p_win_raw"].transform("sum")
    pred["p_top3"] = np.where(s3 > 0, pred["p_top3_raw"] * (3.0 / s3), pred["p_top3_raw"])
    pred["p_win"] = np.where(sw > 0, pred["p_win_raw"] * (1.0 / sw), pred["p_win_raw"])
    pred["p_top3"] = pred["p_top3"].clip(1e-6, 0.999)
    pred["p_win"] = pred["p_win"].clip(1e-6, 0.999)
    pred["_score"] = win_weight * pred["p_win"] + (1.0 - win_weight) * pred["p_top3"]
    pred = pred.sort_values(["race_id", "_score", "horse_no"], ascending=[True, False, True])

    races = pred["race_id"].nunique()
    top = pred.groupby("race_id", sort=False).head(1)
    top1_win = float((top["rank"] == 1).mean() * 100)
    top1_top3 = float((top["rank"] <= 3).mean() * 100)

    pred_top3 = pred.groupby("race_id", sort=False).head(3)
    pred_sets = pred_top3.groupby("race_id")["horse_no"].apply(lambda s: set(s.astype(int)))
    actual_sets = pred[pred["rank"] <= 3].groupby("race_id")["horse_no"].apply(lambda s: set(s.astype(int)))
    hits = int(sum(len(pred_sets[rid] & actual_sets.get(rid, set())) for rid in pred_sets.index))
    top3_rate = float(hits / (races * 3) * 100)
    avg_hits = float(hits / races)
    return {
        "races": int(races),
        "top1_win_rate": top1_win,
        "top1_top3_rate": top1_top3,
        "top3_hit_rate": top3_rate,
        "top3_hits": hits,
        "top3_total": int(races * 3),
        "top3_avg": avg_hits,
    }


def evaluate_subset(df, features, rounds, save_dir=None, cat_meta=None, win_weights=None):
    tr = df[df["__year"] < 2025].reset_index(drop=True)
    va = df[df["__year"] == 2025].reset_index(drop=True)
    cat_feats = [c for c in CATEGORICAL_FEATS if c in features]

    X_tr = tr[features]
    X_va = va[features]
    y_tr_top3 = tr["target_top3"].to_numpy()
    y_va_top3 = va["target_top3"].to_numpy()
    y_tr_win = tr["target_win"].to_numpy()
    y_va_win = va["target_win"].to_numpy()

    top3_specs = [(42, 63, "lgb_top3_s42.txt"), (7, 95, "lgb_top3_s7.txt"), (2024, 47, "lgb_top3_s2024.txt")]
    win_specs = [(42, 63, "lgb_win_s42.txt"), (7, 95, "lgb_win_s7.txt")]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    top3_preds = []
    top3_aucs = []
    for seed, leaves, filename in top3_specs:
        model = train_binary(X_tr, y_tr_top3, X_va, y_va_top3, cat_feats, seed, leaves, rounds)
        pred = model.predict(X_va, num_iteration=model.best_iteration)
        top3_preds.append(pred)
        top3_aucs.append(float(roc_auc_score(y_va_top3, pred)))
        if save_dir is not None:
            model.save_model(str(save_dir / filename))

    win_preds = []
    win_aucs = []
    for seed, leaves, filename in win_specs:
        model = train_binary(X_tr, y_tr_win, X_va, y_va_win, cat_feats, seed, leaves, rounds)
        pred = model.predict(X_va, num_iteration=model.best_iteration)
        win_preds.append(pred)
        win_aucs.append(float(roc_auc_score(y_va_win, pred)))
        if save_dir is not None:
            model.save_model(str(save_dir / filename))

    p_top3 = np.mean(top3_preds, axis=0)
    p_win = np.mean(win_preds, axis=0)
    valid_meta = va[["race_id", "horse_no", "rank"]].copy()
    win_weights = win_weights or [0.6]
    rank_candidates = [
        {"score_win_weight": float(w), **ranking_metrics(valid_meta, p_top3, p_win, win_weight=float(w))}
        for w in win_weights
    ]
    rank = max(
        rank_candidates,
        key=lambda r: (
            r["top1_win_rate"] >= 30.0,
            r["top1_top3_rate"] >= 60.0,
            r["top1_win_rate"],
            r["top1_top3_rate"],
            r["top3_avg"],
        ),
    )
    result = {
        "features": len(features),
        "ens_top3_auc": float(roc_auc_score(y_va_top3, p_top3)),
        "ens_win_auc": float(roc_auc_score(y_va_win, p_win)),
        "top3_seed_aucs": top3_aucs,
        "win_seed_aucs": win_aucs,
        **rank,
    }
    if save_dir is not None:
        metrics = {
            "lgb_top3_seed42": top3_aucs[0],
            "lgb_top3_seed7": top3_aucs[1],
            "lgb_top3_seed2024": top3_aucs[2],
            "ens_top3_auc": result["ens_top3_auc"],
            "lgb_win_seed42": win_aucs[0],
            "lgb_win_seed7": win_aucs[1],
            "ens_win_auc": result["ens_win_auc"],
        }
        meta = {
            "features": features,
            "numeric": [f for f in features if f in NUMERIC_FEATS],
            "excluded_market_features": MARKET_FEATS,
            "categorical": cat_feats,
            "cat_categories": cat_meta or {},
            "metrics": metrics,
            "score_win_weight": rank["score_win_weight"],
            "ranking_metrics": rank,
            "ranking_metrics_by_score_weight": rank_candidates,
            "ensemble_top3": ["lgb_top3_s42.txt", "lgb_top3_s7.txt", "lgb_top3_s2024.txt"],
            "ensemble_win": ["lgb_win_s42.txt", "lgb_win_s7.txt"],
        }
        (save_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    parser.add_argument("--sizes", default="73,60,50,40,30")
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--out", default=str(ROOT / "feature_subset_results.json"))
    parser.add_argument("--exclude-market", action="store_true", help="Do not use odds/popularity as model features")
    parser.add_argument("--save-size", type=int, default=None, help="Save this feature count as a model directory")
    parser.add_argument("--save-dir", default=None, help="Directory for --save-size model")
    parser.add_argument("--win-weights", default="0.3,0.4,0.5,0.6,0.7", help="Score weights to test for p_win")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    ranked_features, gains = current_importance(model_dir)
    if args.exclude_market:
        ranked_features = [f for f in ranked_features if f not in set(MARKET_FEATS)]
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    sizes = [min(s, len(ranked_features)) for s in sizes]
    win_weights = [float(x.strip()) for x in args.win_weights.split(",") if x.strip()]

    print(f"model_dir={model_dir}")
    print(f"exclude_market={args.exclude_market}")
    print(f"sizes={sizes}")
    print("loading data...")
    df, cat_meta = load_data()
    results = []
    for size in sizes:
        features = ranked_features[:size]
        print(f"\n=== feature subset: top {size} ===", flush=True)
        save_dir = args.save_dir if args.save_size == size else None
        result = evaluate_subset(df, features, args.rounds, save_dir=save_dir, cat_meta=cat_meta, win_weights=win_weights)
        result["feature_names"] = features
        results.append(result)
        print(
            f"top3_auc={result['ens_top3_auc']:.4f} win_auc={result['ens_win_auc']:.4f} "
            f"score_w={result['score_win_weight']:.2f} "
            f"top1_win={result['top1_win_rate']:.1f}% top1_top3={result['top1_top3_rate']:.1f}% "
            f"top3_rate={result['top3_hit_rate']:.1f}% avg={result['top3_avg']:.2f}"
        )

    out = Path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    print("\nSummary")
    for r in sorted(results, key=lambda x: (x["top1_win_rate"], x["top1_top3_rate"], x["top3_avg"]), reverse=True):
        print(
            f"{r['features']:>3} features | TOP3 AUC {r['ens_top3_auc']:.4f} | WIN AUC {r['ens_win_auc']:.4f} | "
            f"score_w {r['score_win_weight']:.2f} | "
            f"1着 {r['top1_win_rate']:.1f}% | 3着内 {r['top1_top3_rate']:.1f}% | "
            f"上位3頭 {r['top3_hit_rate']:.1f}% | 平均 {r['top3_avg']:.2f}"
        )


if __name__ == "__main__":
    main()
