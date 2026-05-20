"""
🏇 競馬予想AI - メインCLI
- モデル: LightGBMアンサンブル(3シードTOP3 + 2シードWIN) + 血統特徴
- 入力: 過去レースID指定 / 未来レース手入力 両対応
- 賭け方: 4スタイル(堅実/バランス/一発/AIおまかせ)から毎回選択

Usage:
    python keiba_ai.py             # 対話モード
    python keiba_ai.py --race-id 202506050811 --budget 3000
    python keiba_ai.py --manual    # 未来レース手入力
"""
import os
import sys
import argparse
import gzip
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import KeibaPredictor, load_race_from_data, list_recent_races, DATA_PKL, MODEL_VARIANTS, DEFAULT_MODEL_VARIANT
from betting import suggest, format_suggestion, rank_predictions, STYLE_CONFIG


PUBLIC_STYLE_KEYS = ("smart", "hybrid", "roi_focus", "hit_focus")
DEFAULT_STYLE = "smart"


VENUE_NAMES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

BANNER = r"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║      🏇  KEIBA AI  —  AI競馬予想 & 賭け方エンジン  🏇        ║
║                                                           ║
║      LightGBM アンサンブル × 血統特徴量 × ケリー基準         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


# ==================================================================
# 入力ヘルパ
# ==================================================================
def ask(prompt, default=None, choices=None, type_=str):
    while True:
        msg = prompt
        if default is not None:
            msg += f" [{default}]"
        msg += ": "
        try:
            ans = input(msg).strip()
        except EOFError:
            return default
        if not ans and default is not None:
            ans = str(default)
        if choices and ans not in [str(c) for c in choices]:
            print(f"   ⚠️  選択肢から選んでください: {choices}")
            continue
        if type_ != str:
            try:
                ans = type_(ans)
            except Exception:
                print(f"   ⚠️  形式が違います ({type_.__name__})")
                continue
        return ans


def show_race_info(df: pd.DataFrame):
    if len(df) == 0:
        print("⚠️  該当レースが見つかりません")
        return False
    rid = int(df["race_id"].iloc[0])
    venue_code = str(df["venue"].iloc[0]) if "venue" in df.columns else "??"
    venue_name = VENUE_NAMES.get(venue_code, venue_code)
    date = df["date"].iloc[0] if "date" in df.columns else "?"
    rno = df["round_no"].iloc[0] if "round_no" in df.columns else "?"
    surf = df["surface"].iloc[0] if "surface" in df.columns else "?"
    surf_jp = {"turf": "芝", "dirt": "ダート", "jump": "障害"}.get(str(surf), str(surf))
    dist = df["distance"].iloc[0] if "distance" in df.columns else "?"
    rclass = df["race_class"].iloc[0] if "race_class" in df.columns else "?"
    n = len(df)
    print(f"\n📍 レースID: {rid}  ({date}, {venue_name} {int(rno) if pd.notna(rno) else '?'}R)")
    print(f"   {rclass}  {surf_jp}{int(dist) if pd.notna(dist) else '?'}m  /  出走頭数: {n}頭")
    return True


def race_confidence(pred: pd.DataFrame) -> dict:
    if len(pred) == 0:
        return {"score": 0.0, "label": "NONE", "top_win": 0.0, "top_top3": 0.0, "margin": 0.0}

    ordered = pred.sort_values("score", ascending=False).reset_index(drop=True)
    top = ordered.iloc[0]
    second_score = float(ordered.iloc[1]["score"]) if len(ordered) > 1 else 0.0
    top_score = float(top["score"])
    top_win = float(top["p_win"])
    top_top3 = float(top["p_top3"])
    margin = max(0.0, top_score - second_score)
    field_size = int(top.get("field_size", len(pred))) if pd.notna(top.get("field_size", len(pred))) else len(pred)
    field_size = max(1, field_size)

    def scale(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def damp_probability(value: float, places: int, strength: float) -> float:
        base = min(0.95, max(0.0, float(places) / field_size))
        return base + (max(0.0, min(0.999, value)) - base) * strength

    # 以前の式は p_top3=42% や margin=0.035 で上限に当たり、
    # 100点が多発していた。ここでは2025年の分布を基準に、強い
    # レースだけ90点台になるよう緩やかにスケーリングする。
    confidence_win = damp_probability(top_win, 1, 0.90)
    confidence_top3 = damp_probability(top_top3, 3, 0.78)
    win_part = scale(confidence_win, 0.08, 0.50) * 30.0
    top3_part = scale(confidence_top3, 0.25, 0.90) * 40.0
    margin_part = scale(margin, 0.00, 0.35) * 20.0
    score = max(0.0, min(98.0, 10.0 + win_part + top3_part + margin_part))

    if score >= 75:
        label = "HIGH"
    elif score >= 55:
        label = "MID"
    else:
        label = "LOW"

    return {
        "score": score,
        "label": label,
        "top_win": top_win,
        "top_top3": top_top3,
        "margin": margin,
    }


def display_probability_percent(probability: float, field_size: int, places: int = 1) -> float:
    """画面表示用の補正済み確率。100%張り付きに見える過信表示を避ける。"""
    try:
        p = float(probability)
    except Exception:
        return 0.0
    if not np.isfinite(p):
        return 0.0
    p = max(0.0, min(0.999, p))
    n = max(int(field_size or 0), int(places), 1)
    base = min(0.95, max(0.0, float(places) / n))
    strength = 0.90 if places <= 1 else 0.78
    adjusted = base + (p - base) * strength
    return max(0.0, min(95.0, adjusted * 100.0))


def show_prediction(pred: pd.DataFrame, top_n=10, show_actual=False):
    conf = race_confidence(pred)
    field_size_for_conf = max(1, len(pred))
    print(
        f"\nAI confidence: {conf['score']:.1f}/100 ({conf['label']})"
        f"  top_win={display_probability_percent(conf['top_win'], field_size_for_conf, 1):.1f}%"
        f"  top3={display_probability_percent(conf['top_top3'], field_size_for_conf, 3):.1f}%"
        f"  margin={conf['margin']:.4f}"
    )
    print("\n🏆 AI 予想順位:")
    print("─" * 90)
    print(f"  {'予想':>4}  {'馬番':>4}  {'枠':>3}  {'馬名':<14}  {'1着率':>7}  {'3着内率':>8}  {'単勝オッズ':>9}  {'人気':>5}", end="")
    if show_actual and "rank" in pred.columns:
        print(f"  {'実着順':>5}", end="")
    print()
    print("─" * 90)
    for i, row in pred.head(top_n).iterrows():
        name = str(row.get("馬名", ""))[:14] if "馬名" in row else ""
        odds = row.get("odds", np.nan)
        pop = row.get("popularity", np.nan)
        field_size = int(row.get("field_size", len(pred))) if pd.notna(row.get("field_size", len(pred))) else len(pred)
        p_win_display = display_probability_percent(row["p_win"], field_size, 1)
        p_top3_display = display_probability_percent(row["p_top3"], field_size, 3)
        line = (
            f"  {int(row['pred_rank']):>4}  "
            f"{int(row['horse_no']):>4}  {int(row['frame_no']) if pd.notna(row['frame_no']) else '?':>3}  "
            f"{name:<14}  "
            f"{p_win_display:>6.1f}%  "
            f"{p_top3_display:>7.1f}%  "
        )
        if pd.notna(odds):
            line += f"{odds:>8.1f}倍  "
        else:
            line += f"{'?':>9}  "
        line += f"{int(pop) if pd.notna(pop) else '?':>5}"
        if show_actual and "rank" in pred.columns and pd.notna(row.get("rank")):
            line += f"  {int(row['rank']):>5}"
        print(line)
    print()


# ==================================================================
# 過去データから予想
# ==================================================================
def predict_past_race(race_id: int, predictor: KeibaPredictor):
    df = load_race_from_data(race_id)
    if not show_race_info(df):
        return None
    pred = predictor.predict_race(df)
    show_prediction(pred, top_n=min(12, len(pred)), show_actual=True)
    return pred


def list_and_pick_race(predictor: KeibaPredictor, year: int = 2025) -> int:
    races = list_recent_races(year=year, limit=15)
    print(f"\n📋 直近の{year}年レース(最新15件):")
    print("─" * 80)
    print(f"  {'No':>3}  {'race_id':<14}  {'日付':<12}  {'場':<6}  {'クラス':<14}  {'コース':<14}")
    print("─" * 80)
    for i, row in races.reset_index(drop=True).iterrows():
        venue = VENUE_NAMES.get(str(row["venue"]), str(row["venue"]))
        surf = {"turf": "芝", "dirt": "ダ", "jump": "障"}.get(str(row["surface"]), "?")
        dist = int(row["distance"]) if pd.notna(row["distance"]) else "?"
        cls = str(row["race_class"])[:12] if pd.notna(row["race_class"]) else ""
        date = str(row["date"])[:10]
        print(f"  {i+1:>3}  {row['race_id']:<14}  {date:<12}  {venue:<6}  {cls:<14}  {surf}{dist}m")
    print()
    no = ask("番号を入力(または race_idを直接)", default="1")
    try:
        idx = int(no) - 1
        if 0 <= idx < len(races):
            return int(races.iloc[idx]["race_id"])
    except ValueError:
        pass
    # race_id直接入力扱い
    try:
        return int(no)
    except ValueError:
        return int(races.iloc[0]["race_id"])


# ==================================================================
# 未来レース手入力
# ==================================================================
def manual_input_race(predictor: KeibaPredictor) -> pd.DataFrame:
    """未来レースの基本情報と出走馬を対話的に入力"""
    print("\n" + "=" * 60)
    print("✍️  未来レース手入力モード")
    print("=" * 60)

    # レース基本情報
    surface = ask("コース(turf=芝 / dirt=ダ)", default="turf", choices=["turf", "dirt"])
    direction = ask("方向(右/左/直)", default="右")
    distance = ask("距離(m)", default="1600", type_=int)
    venue_code = ask("競馬場コード(05=東京/06=中山/08=京都/09=阪神 等)", default="05")
    weather = ask("天候(晴/曇/小雨/雨/小雪/雪)", default="晴")
    going = ask("馬場(良/稍重/重/不良)", default="良")
    race_class = ask("クラス(新馬/未勝利/1勝クラス/2勝クラス/3勝クラス/オープン/G3/G2/G1)", default="3歳以上1勝クラス")
    n_horses = ask("出走頭数", default="12", type_=int)

    print("\n各馬の情報を入力(最低限: 馬番・性齢・斤量・推定単勝オッズ)")
    print("(過去成績がわかる場合は『horse_id』を入れると履歴を反映します。空欄でOK)")
    print()

    rows = []
    for i in range(n_horses):
        print(f"--- {i+1}/{n_horses}頭目 ---")
        horse_no = i + 1
        frame_no = ((horse_no - 1) // max(1, n_horses // 8)) + 1
        sex_age = ask(f"  馬{horse_no} 性齢(例: 牡4 / 牝3 / セ5)", default="牡4")
        weight_carry = ask(f"  馬{horse_no} 斤量(kg)", default="56", type_=float)
        odds = ask(f"  馬{horse_no} 推定単勝オッズ(分からなければ空)", default="")
        body_weight = ask(f"  馬{horse_no} 馬体重(kg) 不明=空", default="")
        body_diff = ask(f"  馬{horse_no} 馬体重前差(+5/-3 等)不明=空", default="0")
        horse_id = ask(f"  馬{horse_no} horse_id(過去成績連携用、不明=空)", default="")

        # 性齢パース
        sx, ag = None, np.nan
        if len(sex_age) >= 2 and sex_age[0] in "牡牝セ騙":
            sx = sex_age[0]
            try:
                ag = int(sex_age[1:])
            except Exception:
                ag = np.nan

        row = {
            "race_id": 999900000000 + i,  # ダミー
            "horse_no": float(horse_no),
            "frame_no": float(frame_no),
            "sex": sx, "age": ag,
            "weight_carry": float(weight_carry),
            "surface": surface, "direction": direction,
            "distance": float(distance), "venue": venue_code,
            "weather": weather, "going": going,
            "race_class": race_class,
            "round_no": np.nan,
            "odds": float(odds) if odds else np.nan,
            "popularity": np.nan,
            "body_weight": float(body_weight) if body_weight else np.nan,
            "body_weight_diff": float(body_diff) if body_diff else np.nan,
            "agari": np.nan,
        }
        if horse_id:
            try:
                row["horse_id"] = int(horse_id)
            except Exception:
                pass
        rows.append(row)
        print()

    df = pd.DataFrame(rows)

    # 過去成績データから履歴特徴量をマージ(horse_idがあれば)
    if "horse_id" in df.columns and df["horse_id"].notna().any():
        print("📚 過去データから履歴特徴量をマージ中...")
        with gzip.open(DATA_PKL, "rb") as f:
            hist = pickle.load(f)
        # 各馬の最新履歴をマージ
        hist = hist[hist["horse_id"].isin(df["horse_id"].dropna().astype("Int32"))]
        if len(hist) > 0:
            hist = hist.sort_values("date").groupby("horse_id").tail(1)
            hist_cols = [c for c in [
                "horse_runs", "horse_avg_rank", "horse_win_rate", "horse_top3_rate",
                "horse_avg_time_idx", "horse_best_time_idx", "horse_avg_agari",
                "horse_dist_runs", "horse_surface_runs",
                "sire", "broodmare_sire",
                "jockey_runs", "jockey_win_rate", "jockey_top3_rate",
                "trainer_win_rate", "trainer_top3_rate",
            ] if c in hist.columns]
            df = df.merge(hist[["horse_id"] + hist_cols], on="horse_id", how="left")
            print(f"   {df['horse_runs'].notna().sum()}/{len(df)}頭の履歴をマージしました")

    # オッズから人気を推定
    if df["odds"].notna().any():
        df["popularity"] = df["odds"].rank(method="min")

    return df


# ==================================================================
# 賭け方提案ループ
# ==================================================================
def betting_loop(pred: pd.DataFrame):
    while True:
        print("\n" + "─" * 60)
        print("賭け方提案")
        print("─" * 60)
        print("  1) 自動選択   —  自信度45点以上のレースのみ自動選択(HIGH→ROI重視/MID→バランス)")
        print("  2) ハイブリッド —  単勝・複勝・ワイドをEV重視で組み合わせ")
        print("  3) 期待値重視  —  三連単系を除外し、高EV買い目を優先")
        print("  4) 的中率重視  —  3着内率を軸にEVも見て動的に選択")
        print("  0) 終了")
        choice = ask("\n選択", default="1", choices=["0", "1", "2", "3", "4"])
        if choice == "0":
            break
        budget_str = ask("予算(円)", default="3000")
        try:
            budget = int(budget_str.replace(",", "").replace("円", ""))
        except ValueError:
            print("⚠️  予算は数字で入力してください")
            continue
        if budget < 100:
            print("⚠️  予算は100円以上にしてください")
            continue

        style_map = {"1": "smart", "2": "hybrid", "3": "roi_focus", "4": "hit_focus"}
        style = style_map[choice]
        result = suggest(pred, budget=budget, style=style)
        if result.get("chosen_style"):
            conf = result.get("confidence", 0.0)
            print(f"\n  → 自信度 {conf:.1f}点 → {result['style']}")
        print(format_suggestion(result))

        again = ask("別のスタイル/予算で再提案しますか? (y/n)", default="n")
        if again.lower() not in ("y", "yes"):
            break


# ==================================================================
# メイン
# ==================================================================
def _rank_by_horse(pred: pd.DataFrame) -> dict:
    return {
        int(row["horse_no"]): int(row["rank"])
        for _, row in pred.iterrows()
        if pd.notna(row.get("horse_no")) and pd.notna(row.get("rank"))
    }


def _win_odds_by_horse(pred: pd.DataFrame) -> dict:
    return {
        int(row["horse_no"]): float(row["odds"])
        for _, row in pred.iterrows()
        if pd.notna(row.get("horse_no")) and pd.notna(row.get("odds"))
    }


def _ticket_hit(kind: str, horses: list[int], ranks: dict) -> bool:
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


def settle_bet(bet: dict, pred: pd.DataFrame, payout_cache: dict | None = None) -> int:
    ranks = _rank_by_horse(pred)
    horses = [int(h) for h in bet["horses"]]
    if any(h not in ranks for h in horses):
        return 0

    kind = bet["ticket_kind"]
    odds = float(bet["odds_est"])
    try:
        from race_scraper import actual_payout_multiplier
    except Exception:
        actual_payout_multiplier = None

    if kind.endswith("_box") and bet.get("combos"):
        combos = bet.get("combos") or []
        unit_count = max(1, int(bet.get("unit_count") or len(combos)))
        unit_bet = int(bet["bet"] // unit_count)
        payout = 0
        win_odds = _win_odds_by_horse(pred)
        for combo in combos:
            combo_kind = str(combo.get("ticket_kind", ""))
            combo_horses = [int(h) for h in combo.get("horses", [])]
            if not _ticket_hit(combo_kind, combo_horses, ranks):
                continue
            actual_odds = actual_payout_multiplier(payout_cache, combo_kind, combo_horses) if actual_payout_multiplier else None
            combo_odds = float(actual_odds) if actual_odds is not None else float(combo.get("odds_est", odds))
            if actual_odds is None and combo_kind == "tansho" and combo_horses:
                combo_odds = win_odds.get(combo_horses[0], combo_odds)
            payout += int(unit_bet * combo_odds)
        return payout

    actual_odds = actual_payout_multiplier(payout_cache, kind, horses) if actual_payout_multiplier else None
    if actual_odds is not None:
        odds = float(actual_odds)
    elif kind == "tansho":
        odds = _win_odds_by_horse(pred).get(horses[0], odds)
    return int(bet["bet"] * odds) if _ticket_hit(kind, horses, ranks) else 0


def show_buy_decision(pred: pd.DataFrame, result: dict, min_confidence: float = 75.0, min_expected_roi: float = 0.0):
    conf = race_confidence(pred)
    expected_roi = float(result["expected_roi"]) * 100
    ok_conf = conf["score"] >= min_confidence
    ok_roi = expected_roi >= min_expected_roi
    ok_bet = result.get("total_bet", 0) > 0
    decision = "BUY" if ok_conf and ok_roi and ok_bet else "SKIP"
    print(
        f"\nDecision: {decision}"
        f"  confidence={conf['score']:.1f} >= {min_confidence:.1f}: {ok_conf}"
        f"  expected_roi={expected_roi:+.1f}% >= {min_expected_roi:+.1f}%: {ok_roi}"
        f"  has_bets={ok_bet}"
    )


def backtest_year(
    predictor: KeibaPredictor,
    year: int,
    budget: int,
    style: str,
    limit: int = None,
    min_confidence: float = 0.0,
    min_expected_roi: float = None,
    show_races: bool = False,
):
    with gzip.open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    year_df = df[df["date"].dt.year == year].copy()
    if len(year_df) == 0:
        print(f"No races found for year={year}")
        return

    year_df = year_df.sort_values(["date", "race_id"]).reset_index(drop=True)
    race_groups = [(race_id, race_df.reset_index(drop=True)) for race_id, race_df in year_df.groupby("race_id", sort=False)]
    if limit:
        race_groups = race_groups[:limit]

    valid_groups = []
    for race_id, race_df in race_groups:
        if race_df["rank"].isna().all():
            continue
        race_df = race_df[race_df["rank"].notna()].reset_index(drop=True)
        if len(race_df) >= 3:
            valid_groups.append((race_id, race_df))

    if not valid_groups:
        print(f"No valid finished races found for year={year}")
        return

    predict_source = pd.concat([race_df for _, race_df in valid_groups], ignore_index=True)
    predicted_all = predictor.predict_frame(predict_source)
    race_groups = [
        (race_id, race_df.reset_index(drop=True))
        for race_id, race_df in predicted_all.groupby("race_id", sort=False)
    ]

    total_bet = 0
    total_payout = 0
    hit_races = 0
    bought_races = 0
    skipped_races = 0
    actual_payout_races = 0
    rows = []

    print(
        f"\nBacktest start: year={year}, races={len(race_groups)}, budget={budget:,}, "
        f"style={style}, min_confidence={min_confidence:.1f}, "
        f"min_expected_roi={min_expected_roi if min_expected_roi is not None else 'none'}"
    )
    print("Note: cached actual payouts are used when data/payouts has the race; otherwise estimated odds are used.")
    if show_races:
        print("\nrace_id       date        conf label  bet     payout  roi")
        print("-" * 62)

    for idx, (race_id, race_df) in enumerate(race_groups, start=1):
        pred = rank_predictions(race_df, style)
        conf = race_confidence(pred)
        if conf["score"] < min_confidence:
            skipped_races += 1
            if show_races:
                date = str(race_df["date"].iloc[0])[:10]
                print(f"{int(race_id)}  {date}  {conf['score']:5.1f} {conf['label']:<5}  SKIP")
            continue

        result = suggest(pred, budget=budget, style=style)
        expected_roi_pct = float(result["expected_roi"]) * 100
        if min_expected_roi is not None and expected_roi_pct < min_expected_roi:
            skipped_races += 1
            if show_races:
                date = str(race_df["date"].iloc[0])[:10]
                print(
                    f"{int(race_id)}  {date}  {conf['score']:5.1f} {conf['label']:<5}  "
                    f"SKIP expected_roi={expected_roi_pct:.1f}%"
                )
            continue

        race_bet = int(result["total_bet"])
        if race_bet <= 0:
            continue

        payout_cache = None
        try:
            from race_scraper import load_payout_cache
            payout_cache = load_payout_cache(int(race_id))
        except Exception:
            payout_cache = None
        if payout_cache:
            actual_payout_races += 1
        race_payout = sum(settle_bet(b, pred, payout_cache=payout_cache) for b in result["bets"])
        total_bet += race_bet
        total_payout += race_payout
        bought_races += 1
        if race_payout > 0:
            hit_races += 1

        rows.append({
            "race_id": int(race_id),
            "date": race_df["date"].iloc[0],
            "bet": race_bet,
            "payout": race_payout,
            "roi": race_payout / race_bet if race_bet else 0.0,
            "tickets": int(result["n_tickets"]),
            "confidence": conf["score"],
            "conf_label": conf["label"],
            "expected_roi": expected_roi_pct,
        })

        if show_races:
            date = str(race_df["date"].iloc[0])[:10]
            race_roi = race_payout / race_bet * 100 if race_bet else 0.0
            print(
                f"{int(race_id)}  {date}  {conf['score']:5.1f} {conf['label']:<5}  "
                f"{race_bet:7,} {race_payout:8,} {race_roi:5.1f}%"
            )

        if idx % 100 == 0 or idx == len(race_groups):
            roi = total_payout / total_bet * 100 if total_bet else 0.0
            print(f"  {idx:>4}/{len(race_groups)} races  bet={total_bet:,} payout={total_payout:,} roi={roi:.1f}%")

    roi = total_payout / total_bet * 100 if total_bet else 0.0
    profit = total_payout - total_bet
    hit_rate = hit_races / bought_races * 100 if bought_races else 0.0

    print("\n" + "=" * 72)
    print(f"Backtest result: {year}")
    print("=" * 72)
    print(f"Races tested : {len(race_groups):,}")
    print(f"Races bought : {bought_races:,}")
    print(f"Actual payout: {actual_payout_races:,} races")
    print(f"Races skipped: {skipped_races:,}")
    print(f"Hit races    : {hit_races:,} ({hit_rate:.1f}%)")
    print(f"Total bet    : Yen {total_bet:,}")
    print(f"Total payout : Yen {total_payout:,}")
    print(f"Profit       : Yen {profit:+,}")
    print(f"ROI          : {roi:.1f}%")

    if rows:
        detail = pd.DataFrame(rows).sort_values("payout", ascending=False).head(10)
        print("\nTop payout races:")
        print(detail[["race_id", "date", "confidence", "conf_label", "expected_roi", "bet", "payout", "roi", "tickets"]].to_string(index=False))

        by_conf = pd.DataFrame(rows).groupby("conf_label").agg(
            races=("race_id", "count"),
            bet=("bet", "sum"),
            payout=("payout", "sum"),
        ).reset_index()
        by_conf["roi"] = by_conf["payout"] / by_conf["bet"] * 100
        print("\nBy confidence:")
        print(by_conf.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-id", type=int, help="過去レースID")
    parser.add_argument("--budget", type=int, default=3000, help="予算(円)")
    parser.add_argument("--style", default=DEFAULT_STYLE, choices=PUBLIC_STYLE_KEYS)
    parser.add_argument("--manual", action="store_true", help="未来レース手入力モード")
    parser.add_argument("--year", type=int, default=2025, help="一覧表示する年")
    parser.add_argument("--backtest", action="store_true", help="Run one-year historical ROI backtest")
    parser.add_argument("--limit", type=int, default=None, help="Limit backtest races for quick testing")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Skip backtest races below this AI confidence")
    parser.add_argument("--min-expected-roi", type=float, default=None, help="Skip races below this expected ROI percent")
    parser.add_argument("--show-races", action="store_true", help="Print each race line during backtest")
    parser.add_argument("--fetch-payouts", action="store_true", help="Fetch actual payout tables for races in --year")
    parser.add_argument("--fetch-odds", action="store_true", help="Fetch all available JRA odds tables for races in --year")
    parser.add_argument("--force", action="store_true", help="Overwrite cached payout/odds files when fetching market data")
    parser.add_argument("--sleep-seconds", type=float, default=0.25, help="Delay between market-data requests")
    parser.add_argument("--model-variant", default=DEFAULT_MODEL_VARIANT, choices=MODEL_VARIANTS.keys(), help="Prediction model variant")
    parser.add_argument("--no-banner", action="store_true")
    args = parser.parse_args()

    if not args.no_banner:
        print(BANNER)

    if args.fetch_payouts or args.fetch_odds:
        from race_scraper import fetch_market_data_for_year

        def show_progress(payload):
            print(
                f"[{payload.get('phase', '-')}] "
                f"{payload.get('current', 0)}/{payload.get('total', 0)} "
                f"{payload.get('message', '')}",
                flush=True,
            )

        result = fetch_market_data_for_year(
            year=args.year,
            fetch_payouts=args.fetch_payouts,
            fetch_odds=args.fetch_odds,
            force=args.force,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            progress=show_progress,
        )
        print("\nMarket data fetch result")
        print(f"  year        : {result['year']}")
        print(f"  target races: {result['target_races']:,} / found {result['total_found']:,}")
        print(f"  payouts     : got {result['payout_ok']:,}, cached {result['payout_skip']:,}, failed {result['payout_fail']:,}")
        print(f"  odds        : got {result['odds_ok']:,}, cached {result['odds_skip']:,}, failed {result['odds_fail']:,}")
        print(f"  payout dir  : {result['payout_dir']}")
        print(f"  odds dir    : {result['odds_dir']}")
        return

    print("⚙️  モデル読込中...", flush=True)
    predictor = KeibaPredictor(model_variant=args.model_variant)
    metrics = predictor.meta.get("metrics", {})
    variant_label = MODEL_VARIANTS.get(args.model_variant, {}).get("label", args.model_variant)
    print(f"   ✓ モデル種別: {variant_label}")
    print(f"   ✓ TOP3アンサンブル AUC: {metrics.get('ens_top3_auc', 0):.4f}")
    print(f"   ✓ WIN  アンサンブル AUC: {metrics.get('ens_win_auc', 0):.4f}")
    print(f"   ✓ 特徴量: {len(predictor.feats)}個 (血統・履歴・コース適性 含む)")

    # ==== コマンドラインモード ====
    if args.backtest:
        if args.min_confidence == 0.0:
            args.min_confidence = STYLE_CONFIG.get(args.style, {}).get("default_min_confidence", args.min_confidence)
        backtest_year(
            predictor,
            year=args.year,
            budget=args.budget,
            style=args.style,
            limit=args.limit,
            min_confidence=args.min_confidence,
            min_expected_roi=args.min_expected_roi,
            show_races=args.show_races,
        )
        return

    if args.race_id:
        pred = predict_past_race(args.race_id, predictor)
        if pred is None:
            return
        result = suggest(pred, budget=args.budget, style=args.style)
        print(format_suggestion(result))
        decision_conf = STYLE_CONFIG.get(args.style, {}).get("default_min_confidence", 75.0)
        show_buy_decision(pred, result, min_confidence=decision_conf, min_expected_roi=0.0)
        return

    if args.manual:
        df = manual_input_race(predictor)
        pred = predictor.predict_race(df)
        # 馬名がないので horse_no で
        pred["馬名"] = pred["horse_no"].apply(lambda x: f"#{int(x)}番")
        show_prediction(pred, top_n=len(pred))
        betting_loop(pred)
        return

    # ==== 対話モード ====
    print("\n📋 モード選択:")
    print("  1) 過去レースを予想して検証(2025年データ)")
    print("  2) 未来レースを手入力して予想")
    print("  3) race_idを直接指定して予想")
    print("  0) 終了")
    mode = ask("\n選択", default="1", choices=["0", "1", "2", "3"])

    if mode == "0":
        return

    if mode == "1":
        rid = list_and_pick_race(predictor, year=args.year)
        pred = predict_past_race(rid, predictor)
    elif mode == "2":
        df = manual_input_race(predictor)
        pred = predictor.predict_race(df)
        pred["馬名"] = pred["horse_no"].apply(lambda x: f"#{int(x)}番")
        show_prediction(pred, top_n=len(pred))
    elif mode == "3":
        rid = ask("race_id (例: 202506050811)", type_=int)
        pred = predict_past_race(rid, predictor)
    else:
        return

    if pred is None or len(pred) == 0:
        print("⚠️  予想結果が空です")
        return

    betting_loop(pred)
    print("\n👋 ご利用ありがとうございました。GOOD LUCK! 🍀\n")


if __name__ == "__main__":
    main()
