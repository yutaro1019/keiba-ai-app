import argparse
import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import predictor as predictor_mod
import web_app
from betting import STYLE_CONFIG


MODELS = {
    "no_market_lambdarank_v2": {
        "label": "v2 gain 1着40/2-3着1",
        "model_dir": os.path.join(BASE_DIR, "models", "no_market_lambdarank_v2"),
        "feature_engineering": "v10",
    },
    "no_market_lambdarank_v3_top4": {
        "label": "v3 gain 40/15/5/1",
        "model_dir": os.path.join(BASE_DIR, "models", "no_market_lambdarank_v3_top4"),
        "feature_engineering": "v10",
    },
    "no_market_lambdarank_v3_g40_8_3_1": {
        "label": "v3 gain 40/8/3/1",
        "model_dir": os.path.join(BASE_DIR, "models", "no_market_lambdarank_v3_g40_8_3_1"),
        "feature_engineering": "v10",
    },
    "no_market_lambdarank_v3_g40_5_2_1": {
        "label": "v3 gain 40/5/2/1",
        "model_dir": os.path.join(BASE_DIR, "models", "no_market_lambdarank_v3_g40_5_2_1"),
        "feature_engineering": "v10",
    },
    "no_market_lambdarank_v3_g40_4_2_1": {
        "label": "v3 gain 40/4/2/1",
        "model_dir": os.path.join(BASE_DIR, "models", "no_market_lambdarank_v3_g40_4_2_1"),
        "feature_engineering": "v10",
    },
    "no_market_lambdarank_v3_g50_10_4_1": {
        "label": "v3 gain 50/10/4/1",
        "model_dir": os.path.join(BASE_DIR, "models", "no_market_lambdarank_v3_g50_10_4_1"),
        "feature_engineering": "v10",
    },
}


def register_models():
    for key, cfg in MODELS.items():
        predictor_mod.MODEL_VARIANTS[key] = cfg
        web_app.MODEL_VARIANTS[key] = cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--bankroll", type=int, default=100000)
    parser.add_argument(
        "--styles",
        default="roi_focus,hit_focus,kelly_ai",
        help="Comma-separated styles.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODELS.keys()),
        help="Comma-separated model keys.",
    )
    args = parser.parse_args()

    register_models()
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []

    for model_key in model_keys:
        for style in styles:
            min_conf = float(STYLE_CONFIG.get(style, {}).get("default_min_confidence", 0.0))
            print(f">>> simulate model={model_key} style={style}", flush=True)
            result = web_app.simulate_period(
                args.start,
                args.end,
                args.budget,
                style,
                model_key,
                min_conf,
                kelly_bankroll=args.bankroll,
            )
            rows.append({
                "model": model_key,
                "label": MODELS[model_key]["label"],
                "style": style,
                "roi": result["roi"],
                "hit_rate": result["hit_rate"],
                "ticket_hit_rate": result["ticket_hit_rate"],
                "bought": result["bought"],
                "tested": result["tested"],
                "no_bet": result["no_bet"],
                "skipped_confidence": result["skipped_confidence"],
                "total_bet": result["total_bet"],
                "total_payout": result["total_payout"],
                "profit": result["profit"],
                "top1_win_rate": result["top1_win_rate"],
                "top1_top3_rate": result["top1_top3_rate"],
                "top3_hit_rate": result["top3_hit_rate"],
            })

    out_path = os.path.join(BASE_DIR, "data", "roi_compare_lambdarank_models.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nmodel,style,roi,hit_rate,ticket_hit_rate,bought,total_bet,total_payout,profit")
    for r in sorted(rows, key=lambda x: (x["style"], -x["roi"])):
        print(
            f"{r['model']},{r['style']},{r['roi']:.2f},{r['hit_rate']:.2f},"
            f"{r['ticket_hit_rate']:.2f},{r['bought']},{r['total_bet']},"
            f"{r['total_payout']},{r['profit']}"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
