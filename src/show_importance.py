import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "no_market_v5")

with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
feats = meta["features"]

m_win  = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_win_s42.txt"))
m_top3 = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_top3_s42.txt"))

fi_win  = pd.Series(m_win.feature_importance(importance_type="gain"),  index=feats)
fi_top3 = pd.Series(m_top3.feature_importance(importance_type="gain"), index=feats)
fi = ((fi_win / fi_win.sum() + fi_top3 / fi_top3.sum()) / 2 * 100).round(2)
fi = fi.sort_values(ascending=False)

LAP_FEATS = {f for f in feats if "pace" in f or ("front" in f and "pace" in f) or ("back" in f and "pace" in f)}
LAP_FEATS |= {"race_front_pace", "race_back_pace", "race_pace_diff",
              "field_avg_front_pace", "field_avg_back_pace", "field_avg_pace_diff"}

def get_cat(f):
    if f in LAP_FEATS:            return "LAP[v5新規]"
    if "vs_field" in f:           return "対フィールド比較"
    if "class_hist" in f:         return "クラスTE"
    if "sire" in f or "broodmare" in f: return "種牡馬/母父TE"
    if "jockey" in f:             return "騎手"
    if "trainer" in f:            return "調教師"
    if "recent" in f:             return "ローリング統計"
    if "bw_trend" in f:           return "体重トレンド"
    if "field_avg" in f:          return "フィールド強度"
    if "horse" in f:              return "馬統計"
    if "field_size" in f:         return "フィールド"
    return "その他"

print("=" * 70)
print("特徴量重要度  (win+top3モデル平均, gain基準)")
print("=" * 70)
print(f"{'順位':>4}  {'特徴量':<46}  {'重要度':>6}  カテゴリ")
print("-" * 70)
lap_total = 0.0
for rank, (feat, val) in enumerate(fi.items(), 1):
    tag = " [LAP]" if feat in LAP_FEATS else ""
    if feat in LAP_FEATS:
        lap_total += val
    print(f"{rank:>4}. {feat:<46}  {val:>5.2f}%  {get_cat(feat)}{tag}")

print("-" * 70)
print(f"\nLAPフィーチャー合計寄与: {lap_total:.2f}%  ({sum(1 for f in feats if f in LAP_FEATS)}個)")
print(f"総特徴量数: {len(feats)}")
