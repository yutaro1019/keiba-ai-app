"""
Kelly基準 バックテスト (修正版)
================================
【正しい設計】
  1. モデル確率 = LambdaRank v2 スコアの softmax
  2. マーケット確率 = 単勝オッズ(processed_all.pkl.gzの odds列) から逆算
  3. Kelly = (モデル確率 × マーケットオッズ - 1) / (マーケットオッズ - 1)
  4. 組合せ馬券のオッズ → 単勝オッズからPlackett-Luceで推定
  5. 的中・払戻 → 払戻JSONと実着順データを使用

【なぜ2025年のみ評価するか】
  LambdaRank v2 は 2024年以前のデータで学習済み。
  2023-2024に走らせるとモデルが訓練データに過学習しており、ROIが過大評価される。
"""
import os, sys, json, gzip, pickle, gc, warnings, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from feature_engineering import add_model_features
from train_no_market_v5 import (
    add_sire_features, add_horse_rolling_features, add_jockey_trainer_rolling,
    add_field_strength, add_weight_trend, add_lap_rolling_features, add_field_pace_features,
)
from train_no_market_v6 import add_v6_features
from train_no_market_v7 import add_v7_features
from train_no_market_v10 import add_v10_features
from kelly_betting import (
    softmax_probs, kelly_fraction,
    pl_win, pl_place, pl_pair, pl_order2, pl_wide, pl_trio, pl_order3,
    market_probs_from_odds, enumerate_race_bets, size_bets,
)

DATA_PKL    = os.path.join(BASE_DIR, "data", "processed_all.pkl.gz")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "no_market_lambdarank_v2")
PAYOUT_DIR  = os.path.join(BASE_DIR, "data", "payouts")

# top50特徴量（v2モデルと同じ）
FEATURE_COLS = [
    "class_hist_win_rate","horse_avg_rank_vs_field","horse_recent_top3_rate3_vs_field",
    "class_hist_runs","horse_rank_pct_lag1","class_hist_top3_rate",
    "jockey_top3_rate_vs_field","horse_recent_avg_rank3_vs_field","horse_top3_rate_vs_field",
    "horse_rank_lag1","jockey_win_rate_vs_field","horse_rank_pct_lag1_vs_field",
    "horse_recent5_avg_rank","horse_recent_avg_rank3","horse_rank_pct_lag2",
    "horse_recent10_avg_rank","jockey_venue_top3_rate","weight_burden_ratio",
    "horse_agari_lag1_vs_field","jockey_top3_rate","trainer_top3_rate_vs_field",
    "days_since_last","horse_agari_rank_pct_lag1","horse_front_style",
    "age","horse_win_rate_vs_field","horse_avg_agari","trainer_top3_rate",
    "class_change","horse_surface_top3_rate","trainer_win_rate","jockey_win_rate",
    "trainer_venue_top3_rate","trainer_venue_win_rate","venue_frame_top3_rate",
    "horse_passing_first_rate_lag1","horse_passing_last_rate_lag1","jockey_venue_win_rate",
    "horse_prev_back_pace1_vs_field","sire_hist_top3_rate","sire_hist_win_rate",
    "horse_distance_diff_lag1","jockey_course_top3_rate","field_avg_front_pace",
    "jockey_recent20_top3_rate","bms_line_dist_top3_rate","jockey_recent50_win_rate",
    "venue_frame_win_rate","horse_going_top3_rate_vs_field_diff","horse_dist_top3_rate_vs_field",
]

TICKET_TYPES = ["tansho", "fukusho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]


def build_features(df):
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
        ("v10",           add_v10_features),
    ]
    for label, fn in steps:
        print(f"   {label}...", flush=True)
        df = fn(df); gc.collect()

    print("   vs_field...", flush=True)
    df = add_model_features(df, race_col="race_id"); gc.collect()

    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_t"] = (df["rank"] <= 3).astype(float)
    if "jockey_id" in df.columns and "horse_id" in df.columns:
        grp = df.groupby(["jockey_id", "horse_id"], sort=False)
        df["jh_top3_cum"]  = grp["_t"].cumsum() - df["_t"]
        df["jh_runs"]      = grp.cumcount().astype("float32")
        df["jh_top3_rate"] = (df["jh_top3_cum"] / df["jh_runs"].replace(0, np.nan)).astype("float32")
        df.drop(columns=["jh_top3_cum"], inplace=True)
    df.drop(columns=["_t"], inplace=True)
    return df


def load_models():
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n = meta.get("n_rank_models", 3)
    models = [lgb.Booster(model_file=os.path.join(MODEL_DIR, f"rank_{i}.lgb")) for i in range(n)]
    return models, meta


def load_payout(race_id):
    path = os.path.join(PAYOUT_DIR, f"{race_id}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    payouts = data.get("payouts", {})
    # key化: kind -> {frozenset(horses): odds}
    result = {}
    for kind, entries in payouts.items():
        result[kind] = {}
        for e in entries:
            hs = e.get("horses", [])
            odds = e.get("odds", 0.0)
            # tanshoとfukushoは horse_no単体、馬単/三連単は順序あり、他は順不同
            if kind in ("umatan", "sanrentan"):
                key = tuple(hs)
            else:
                key = frozenset(hs)
            result[kind][key] = odds
    return result


def check_hit_and_payout(kind, horses, rank1, rank2, rank3, top3_set, top2_set, payout_map):
    """
    的中判定と実際の払戻オッズを返す。
    hit: bool, actual_odds: float (ヒット時のみ有効)
    """
    kind_map = payout_map.get(kind, {})
    hs = horses

    if kind == "tansho":
        hit = (hs[0] == rank1)
        key = frozenset([hs[0]])
    elif kind == "fukusho":
        hit = (hs[0] in top3_set)
        key = frozenset([hs[0]])
    elif kind == "umaren":
        hit = (frozenset(hs) == top2_set)
        key = frozenset(hs)
    elif kind == "umatan":
        hit = (hs[0] == rank1 and hs[1] == rank2)
        key = tuple(hs)
    elif kind == "wide":
        hit = (set(hs).issubset(top3_set))
        key = frozenset(hs)
    elif kind == "sanrenpuku":
        hit = (frozenset(hs) == top3_set)
        key = frozenset(hs)
    elif kind == "sanrentan":
        hit = (hs[0] == rank1 and hs[1] == rank2 and hs[2] == rank3)
        key = tuple(hs)
    else:
        return False, 0.0

    actual_odds = kind_map.get(key, 0.0) if hit else 0.0
    return hit, actual_odds


def run_backtest(df, models, feats,
                 years, kelly_factor, min_ev, initial_bankroll,
                 ticket_types, top_k, verbose, flat_bet=0):

    df["date"] = pd.to_datetime(df["date"])
    mask = df["date"].dt.year.isin(years) & df["rank"].notna()
    df_eval = df[mask].copy()

    bankroll = float(initial_bankroll)
    stats = defaultdict(lambda: {"bet": 0, "ret": 0, "n": 0, "hit": 0})
    year_stats = defaultdict(lambda: {"bet": 0, "ret": 0, "n": 0, "hit": 0})
    bankroll_history = [bankroll]
    n_races = n_races_bet = 0
    detail_rows = []

    all_race_ids = sorted(df_eval["race_id"].unique())
    total = len(all_race_ids)
    print(f"\n対象レース数: {total:,}  対象期間: {years}", flush=True)

    for idx, race_id in enumerate(all_race_ids):
        if bankroll <= 0 and flat_bet == 0:
            print("バンクロール枯渇 → 終了")
            break
        if idx % 500 == 0:
            print(f"  [{idx:5d}/{total}] bankroll={bankroll:,.0f}円", flush=True)

        g = df_eval[df_eval["race_id"] == race_id].copy()
        if len(g) < 2:
            continue
        if "odds" not in g.columns or g["odds"].isna().all():
            continue

        # ── モデルスコア計算 ──────────────────────────────────────
        X = g[feats].values
        raw = np.mean([m.predict(X) for m in models], axis=0)
        model_probs = softmax_probs(raw)

        # ── マーケット確率 (単勝オッズから逆算) ──────────────────
        tansho_odds_arr = g["odds"].fillna(100.0).values.astype(float)
        market_probs = market_probs_from_odds(tansho_odds_arr)
        horse_nos = g["horse_no"].tolist()

        # ── 馬券候補列挙 ─────────────────────────────────────────
        bets_raw = enumerate_race_bets(
            model_probs, market_probs, horse_nos,
            tansho_odds_arr, ticket_types, top_k=top_k,
        )
        # EV フィルタ
        bets_raw = [b for b in bets_raw if b["ev"] >= min_ev]

        if not bets_raw:
            n_races += 1
            bankroll_history.append(bankroll)
            continue

        # ── サイズ決定 ───────────────────────────────────────────
        if flat_bet > 0:
            bets = [{**b, "bet_amount": flat_bet} for b in bets_raw]
        else:
            bets = size_bets(bets_raw, bankroll, kelly_factor,
                             max_bet_frac=0.10, max_total_frac=0.20)
        if not bets:
            n_races += 1
            bankroll_history.append(bankroll)
            continue

        # ── 着順情報 ─────────────────────────────────────────────
        ranks = dict(zip(g["horse_no"], g["rank"].astype(int)))
        rank1  = next((h for h, r in ranks.items() if r == 1), None)
        rank2  = next((h for h, r in ranks.items() if r == 2), None)
        rank3  = next((h for h, r in ranks.items() if r == 3), None)
        top3   = {h for h, r in ranks.items() if r <= 3}
        top2   = {h for h, r in ranks.items() if r <= 2}
        payout_map = load_payout(race_id)

        year = pd.to_datetime(str(race_id)[:8], format="%Y%m%d").year \
               if len(str(race_id)) >= 8 else 0

        race_bet = race_ret = 0
        for b in bets:
            amount = b["bet_amount"]
            kind   = b["ticket_kind"]

            hit, actual_odds = check_hit_and_payout(
                kind, b["horses"], rank1, rank2, rank3, top3, top2, payout_map
            )

            # 的中時の払戻: 実際のオッズを使用、なければマーケット推定値で代替
            if hit:
                use_odds = actual_odds if actual_odds > 1.0 else b["market_odds"]
                payout = int(amount * use_odds)
            else:
                payout = 0

            race_bet += amount
            race_ret += payout

            stats[kind]["bet"] += amount
            stats[kind]["ret"] += payout
            stats[kind]["n"]   += 1
            stats[kind]["hit"] += int(hit)

            year_stats[year]["bet"] += amount
            year_stats[year]["ret"] += payout
            year_stats[year]["n"]   += 1
            year_stats[year]["hit"] += int(hit)

            detail_rows.append({
                "race_id":    race_id,
                "ticket_kind": kind,
                "horses":     b["horses"],
                "model_prob": b["model_prob"],
                "market_odds": b["market_odds"],
                "ev":         b["ev"],
                "kelly":      b["kelly"],
                "bet_amount": amount,
                "hit":        hit,
                "actual_odds": actual_odds,
                "payout":     payout,
            })

        n_races += 1
        n_races_bet += 1
        bankroll += (race_ret - race_bet)
        bankroll_history.append(bankroll)

        if verbose and (race_ret != race_bet):
            profit = race_ret - race_bet
            print(f"  race {race_id}: bet={race_bet:,} ret={race_ret:,} "
                  f"profit={profit:+,} bank={bankroll:,.0f}")

    return stats, year_stats, bankroll, bankroll_history, n_races, n_races_bet, detail_rows


def print_results(stats, year_stats, bankroll, initial_bankroll, n_races, n_races_bet):
    total_bet = sum(v["bet"] for v in stats.values())
    total_ret = sum(v["ret"] for v in stats.values())
    total_n   = sum(v["n"]   for v in stats.values())
    total_hit = sum(v["hit"] for v in stats.values())

    print("\n" + "=" * 80)
    print("【券種別 ROI】")
    print(f"{'券種':<12} {'ベット数':>8} {'的中数':>7} {'的中率':>8} "
          f"{'投資額':>12} {'回収額':>12} {'ROI':>8}")
    print("-" * 80)
    for kind in TICKET_TYPES:
        v = stats.get(kind, {"bet":0,"ret":0,"n":0,"hit":0})
        if v["n"] == 0:
            continue
        roi = v["ret"] / v["bet"] - 1.0 if v["bet"] > 0 else 0.0
        hr  = v["hit"] / v["n"]
        print(f"  {kind:<10} {v['n']:>8,} {v['hit']:>7,} {hr:>8.1%} "
              f"{v['bet']:>12,} {v['ret']:>12,} {roi:>+8.1%}")
    print("-" * 80)
    if total_bet > 0:
        print(f"  {'合計':<10} {total_n:>8,} {total_hit:>7,} {total_hit/total_n:>8.1%} "
              f"{total_bet:>12,} {total_ret:>12,} {(total_ret/total_bet-1)*100:>+7.1f}%")

    if year_stats:
        print("\n【年別 ROI】")
        print(f"{'年':<6} {'ベット数':>8} {'投資額':>12} {'回収額':>12} {'ROI':>8}")
        print("-" * 50)
        for yr in sorted(year_stats.keys()):
            v = year_stats[yr]
            if v["bet"] == 0:
                continue
            roi = v["ret"] / v["bet"] - 1.0
            print(f"  {yr:<4} {v['n']:>8,} {v['bet']:>12,} {v['ret']:>12,} {roi:>+8.1%}")

    print("\n【サマリー】")
    print(f"  対象レース数   : {n_races:,}")
    if n_races > 0:
        print(f"  ベットしたレース: {n_races_bet:,} ({n_races_bet/n_races*100:.1f}%)")
    print(f"  初期バンクロール: {initial_bankroll:,.0f} 円")
    print(f"  最終バンクロール: {bankroll:,.0f} 円")
    pnl = bankroll - initial_bankroll
    print(f"  損益           : {pnl:+,.0f} 円")
    if total_bet > 0:
        print(f"  総合ROI        : {(total_ret/total_bet-1)*100:+.2f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Kelly基準バックテスト (2025年ホールドアウト)")
    parser.add_argument("--years",    nargs="+", type=int, default=[2025],
                        help="評価年 (デフォルト: 2025のみ。2023/2024は学習データなので除外推奨)")
    parser.add_argument("--kelly",    type=float, default=0.25,
                        help="Kelly係数の倍率 (0.25=クォーターKelly)")
    parser.add_argument("--min-ev",   type=float, default=0.0,
                        help="最低期待値フィルタ (0.0=EV>0のみ)")
    parser.add_argument("--bankroll", type=float, default=100_000,
                        help="初期資金 (円)")
    parser.add_argument("--top-k",   type=int, default=5,
                        help="組合せ馬券の対象上位K頭 (デフォルト5)")
    parser.add_argument("--tickets",  nargs="+", default=None,
                        choices=TICKET_TYPES, help="対象券種 (省略=全券種)")
    parser.add_argument("--flat-bet", type=int, default=0,
                        help="フラットベット額 (円)。指定時はKellyを無視して固定額でベット")
    parser.add_argument("--verbose",  action="store_true",
                        help="各レースの詳細を表示")
    args = parser.parse_args()

    ticket_types = args.tickets or TICKET_TYPES
    flat_bet = args.flat_bet
    print(f">>> 設定:")
    print(f"    years={args.years}  kelly={args.kelly}  min_ev={args.min_ev}")
    print(f"    bankroll={args.bankroll:,.0f}  top_k={args.top_k}  tickets={ticket_types}")
    if flat_bet:
        print(f"    ★ フラットベットモード: 1ベット固定 {flat_bet:,}円", flush=True)
    else:
        print(f"    compound Kelly (bankroll更新あり)", flush=True)

    # ── データ読み込み & 特徴量構築 ────────────────────────────────────────
    print(">>> Loading data...", flush=True)
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"   rows={len(df):,}", flush=True)

    print(">>> Building features...", flush=True)
    df = build_features(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rank"])
    df = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)

    feats = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"   [WARN] missing features: {missing}", flush=True)
    print(f"   特徴量数: {len(feats)}", flush=True)

    # ── モデルロード ────────────────────────────────────────────────────────
    print(">>> Loading models...", flush=True)
    models, meta = load_models()
    print(f"   {len(models)}モデルアンサンブル  label_gain={meta.get('label_gain')}", flush=True)

    # ── バックテスト実行 ────────────────────────────────────────────────────
    print(">>> Running backtest...", flush=True)
    stats, year_stats, final_bankroll, hist, n_races, n_races_bet, details = run_backtest(
        df, models, feats,
        years            = args.years,
        kelly_factor     = args.kelly,
        min_ev           = args.min_ev,
        initial_bankroll = args.bankroll,
        ticket_types     = ticket_types,
        top_k            = args.top_k,
        verbose          = args.verbose,
        flat_bet         = flat_bet,
    )

    # ── 結果出力 ────────────────────────────────────────────────────────────
    print_results(stats, year_stats, final_bankroll, args.bankroll, n_races, n_races_bet)

    # バンクロール推移CSV
    out_csv = os.path.join(BASE_DIR, "src", "backtest_bankroll.csv")
    pd.DataFrame({"bankroll": hist}).to_csv(out_csv, index=False)
    print(f"\n(バンクロール推移 → {out_csv})")

    # 詳細CSV
    if details:
        out_detail = os.path.join(BASE_DIR, "src", "backtest_details.csv")
        pd.DataFrame(details).to_csv(out_detail, index=False)
        print(f"(ベット詳細 → {out_detail})")

    # サマリーJSON
    total_bet = sum(v["bet"] for v in stats.values())
    total_ret = sum(v["ret"] for v in stats.values())
    summary = {
        "config": {
            "years": args.years, "kelly_factor": args.kelly,
            "min_ev": args.min_ev, "bankroll": args.bankroll, "top_k": args.top_k,
        },
        "final_bankroll": final_bankroll,
        "total_bet":   total_bet,
        "total_return": total_ret,
        "roi": (total_ret/total_bet-1.0) if total_bet > 0 else 0.0,
        "n_races": n_races, "n_races_bet": n_races_bet,
        "by_ticket": {k: {
            "n": v["n"], "hit": v["hit"],
            "bet": v["bet"], "ret": v["ret"],
            "roi": (v["ret"]/v["bet"]-1.0) if v["bet"] > 0 else 0.0,
        } for k, v in stats.items()},
        "by_year": {str(yr): {
            "n": v["n"], "bet": v["bet"], "ret": v["ret"],
            "roi": (v["ret"]/v["bet"]-1.0) if v["bet"] > 0 else 0.0,
        } for yr, v in year_stats.items()},
    }
    out_json = os.path.join(BASE_DIR, "src", "backtest_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"(サマリー → {out_json})")


if __name__ == "__main__":
    main()
