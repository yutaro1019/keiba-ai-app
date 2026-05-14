"""
Kelly基準 資金分配AI
======================
LambdaRankモデルのスコアから全券種の的中確率を推定し、
Kelly基準で期待値プラスの馬券だけを選択して資金配分する。

【対応券種】
  tansho    単勝   1頭
  fukusho   複勝   1頭 (3着以内)
  umaren    馬連   2頭 (順不同, 1-2着)
  umatan    馬単   2頭 (順あり, 1-2着)
  wide      ワイド 2頭 (順不同, 3着以内に両方)
  sanrenpuku 三連複 3頭 (順不同, 1-3着)
  sanrentan  三連単 3頭 (順あり, 1-3着)

【確率推定: Plackett-Luce モデル】
  P(i 1着) = softmax(score_i)
  P(i 2着 | j 1着) = softmax(score_i, i≠j上で再正規化)
  → 全順列確率を解析的に計算

【Kelly基準】
  f* = (p × odds - 1) / (odds - 1)
  f* > 0 のとき期待値プラス → ベット対象
  実際の掛け金 = f* × kelly_factor × バンクロール
"""
import os, json, glob, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ── Plackett-Luce 確率計算 ──────────────────────────────────────────────

def softmax_probs(scores: np.ndarray) -> np.ndarray:
    """スコア → 勝利確率 (Plackett-Luce の1着確率)"""
    e = np.exp(scores - scores.max())
    return e / e.sum()


def pl_win(probs: np.ndarray, i: int) -> float:
    """P(馬i が1着)"""
    return float(probs[i])


def pl_place(probs: np.ndarray, i: int, n: int = 3) -> float:
    """P(馬i が n着以内) — Plackett-Luce 解析計算"""
    p = probs.copy()
    pi = p[i]

    # P(i が1着)
    result = pi

    if n >= 2:
        # P(i が2着) = pi * Σ_{j≠i} p_j/(1-p_j)
        s2 = sum(p[j] / (1 - p[j]) for j in range(len(p)) if j != i and (1 - p[j]) > 1e-9)
        result += pi * s2

    if n >= 3:
        # P(i が3着) = pi * Σ_{j≠i} Σ_{k≠i,k≠j} p_j/(1-p_j) * p_k/(1-p_j-p_k)
        s3 = 0.0
        for j in range(len(p)):
            if j == i: continue
            pj = p[j]
            rem_j = 1.0 - pj
            if rem_j < 1e-9: continue
            for k in range(len(p)):
                if k == i or k == j: continue
                pk = p[k]
                rem_jk = rem_j - pk
                if rem_jk < 1e-9: continue
                s3 += (pj / rem_j) * (pk / rem_jk)
        result += pi * s3

    return min(float(result), 1.0)


def pl_order2(probs: np.ndarray, i: int, j: int) -> float:
    """P(馬i が1着 かつ 馬j が2着) — 馬単の確率"""
    if i == j: return 0.0
    rem = 1.0 - probs[i]
    if rem < 1e-9: return 0.0
    return float(probs[i] * probs[j] / rem)


def pl_pair(probs: np.ndarray, i: int, j: int) -> float:
    """P(馬i, 馬j が共に1-2着, 順不同) — 馬連の確率"""
    return pl_order2(probs, i, j) + pl_order2(probs, j, i)


def pl_wide(probs: np.ndarray, i: int, j: int) -> float:
    """P(馬i, 馬j が共に3着以内, 順不同) — ワイドの確率"""
    if i == j: return 0.0
    p = probs
    pi, pj = p[i], p[j]
    result = 0.0
    n = len(p)
    # 6通りの (i,j の1着, 2着, 3着への配置) を合計
    for pos_i, pos_j in itertools.permutations([0, 1, 2], 2):
        # i が pos_i 着, j が pos_j 着
        rem = 1.0
        prob = 1.0
        chosen = []
        valid = True
        for pos in range(3):
            if pos == pos_i:
                prob *= pi / rem
                rem -= pi
                chosen.append(i)
            elif pos == pos_j:
                prob *= pj / rem
                rem -= pj
                chosen.append(j)
            else:
                # 3番目の馬は i でも j でもない誰か → 確率の和 = rem - (i,j の残り)
                other_sum = rem - sum(p[h] for h in [i, j] if h not in chosen)
                if other_sum < 1e-9:
                    valid = False
                    break
                prob *= other_sum / rem
                rem = rem - other_sum
                chosen.append(-1)
            if rem < -1e-9:
                valid = False
                break
        if valid:
            result += prob
    return min(float(result), 1.0)


def pl_trio(probs: np.ndarray, i: int, j: int, k: int) -> float:
    """P(馬i, j, k が共に3着以内, 順不同) — 三連複の確率"""
    if len({i, j, k}) < 3: return 0.0
    total = 0.0
    for perm in itertools.permutations([i, j, k]):
        total += pl_order3(probs, *perm)
    return min(float(total), 1.0)


def pl_order3(probs: np.ndarray, i: int, j: int, k: int) -> float:
    """P(馬i 1着, 馬j 2着, 馬k 3着) — 三連単の確率"""
    if len({i, j, k}) < 3: return 0.0
    pi = probs[i]
    rem_i = 1.0 - pi
    if rem_i < 1e-9: return 0.0
    pj_given_i = probs[j] / rem_i
    rem_ij = rem_i - probs[j]
    if rem_ij < 1e-9: return 0.0
    pk_given_ij = probs[k] / rem_ij
    return float(pi * pj_given_i * pk_given_ij)


# ── Kelly 計算 ──────────────────────────────────────────────────────────

def kelly_fraction(p: float, odds: float) -> float:
    """
    Kelly係数を計算する
    p    : 的中確率
    odds : 払い戻し倍率 (100円 → odds×100円)
    戻り値: バンクロールに対する最適賭け率 (負ならベット不要)
    """
    if odds <= 1.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    # f* = (p*odds - 1) / (odds - 1)
    return (p * odds - 1.0) / (odds - 1.0)


# ── 全馬券の確率 × オッズ → ベット候補リスト ────────────────────────────

def enumerate_bets(
    probs: np.ndarray,
    horse_nos: List[int],
    odds_data: Dict,
    ticket_types: Optional[List[str]] = None,
) -> List[Dict]:
    """
    全馬券の期待値とKelly係数を計算して返す。

    Parameters
    ----------
    probs      : 各馬の勝利確率 (horse_nos の順)
    horse_nos  : 馬番リスト
    odds_data  : {ticket_kind: [{horses:[馬番...], odds:float}, ...]}
    ticket_types: 対象券種リスト (Noneで全券種)

    Returns
    -------
    List of dicts: {ticket_kind, horses, model_prob, odds, ev, kelly, ...}
    """
    if ticket_types is None:
        ticket_types = ["tansho", "fukusho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]

    # 馬番→インデックス変換
    h2i = {h: i for i, h in enumerate(horse_nos)}

    bets = []

    for kind in ticket_types:
        entries = odds_data.get(kind, [])
        for entry in entries:
            hs = entry["horses"]
            odds = entry.get("odds", 0.0)
            if odds <= 1.0:
                continue

            # 的中確率を計算
            idx = [h2i[h] for h in hs if h in h2i]
            if len(idx) != len(hs):
                continue  # 馬番がデータにない

            if kind == "tansho":
                p = pl_win(probs, idx[0])
            elif kind == "fukusho":
                p = pl_place(probs, idx[0], n=3)
            elif kind == "umaren":
                p = pl_pair(probs, idx[0], idx[1])
            elif kind == "umatan":
                p = pl_order2(probs, idx[0], idx[1])
            elif kind == "wide":
                p = pl_wide(probs, idx[0], idx[1])
            elif kind == "sanrenpuku":
                p = pl_trio(probs, idx[0], idx[1], idx[2])
            elif kind == "sanrentan":
                p = pl_order3(probs, idx[0], idx[1], idx[2])
            else:
                continue

            ev = p * odds - 1.0  # 期待値 (0以上なら有利)
            kelly = kelly_fraction(p, odds)

            bets.append({
                "ticket_kind": kind,
                "horses":      hs,
                "model_prob":  round(p, 5),
                "odds":        odds,
                "ev":          round(ev, 4),
                "kelly":       round(kelly, 5),
            })

    return bets


def select_and_size(
    bets: List[Dict],
    bankroll: float,
    kelly_factor: float = 0.25,
    min_ev: float = 0.0,
    max_bet_fraction: float = 0.10,
    max_total_fraction: float = 0.20,
) -> List[Dict]:
    """
    期待値プラスの馬券を選択し掛け金を決定する。

    Parameters
    ----------
    bets            : enumerate_bets() の出力
    bankroll        : 現在のバンクロール (円)
    kelly_factor    : フルKellyに対する掛け率 (0.25 = クォーターKelly)
    min_ev          : 最低期待値フィルタ (0.0=期待値プラスのみ)
    max_bet_fraction: 1馬券あたりの上限 (バンクロール比)
    max_total_fraction: 1レース合計の上限 (バンクロール比)

    Returns
    -------
    List of selected bets with bet_amount added
    """
    # 期待値フィルタ
    candidates = [b for b in bets if b["ev"] > min_ev and b["kelly"] > 0]

    if not candidates:
        return []

    # Kelly額を計算
    for b in candidates:
        raw_fraction = b["kelly"] * kelly_factor
        capped = min(raw_fraction, max_bet_fraction)
        # 100円単位に切り捨て
        amount = int(bankroll * capped / 100) * 100
        b = b.copy()
        b["kelly_fraction"] = round(raw_fraction, 5)
        b["bet_fraction"]   = round(capped, 5)
        b["bet_amount"]     = amount
        candidates[candidates.index(b) if b in candidates else -1] = b

    # 再構築 (copyしたので)
    sized = []
    for b in bets:
        if b["ev"] > min_ev and b["kelly"] > 0:
            raw_fraction = b["kelly"] * kelly_factor
            capped = min(raw_fraction, max_bet_fraction)
            amount = int(bankroll * capped / 100) * 100
            if amount >= 100:
                sized.append({**b,
                    "kelly_fraction": round(raw_fraction, 5),
                    "bet_fraction":   round(capped, 5),
                    "bet_amount":     amount,
                })

    if not sized:
        return []

    # 合計上限チェック
    total_amount = sum(b["bet_amount"] for b in sized)
    max_total = bankroll * max_total_fraction
    if total_amount > max_total:
        scale = max_total / total_amount
        for b in sized:
            b["bet_amount"] = int(b["bet_amount"] * scale / 100) * 100
        sized = [b for b in sized if b["bet_amount"] >= 100]

    # EVの高い順にソート
    sized.sort(key=lambda x: -x["ev"])
    return sized


# ── オッズ取得 ────────────────────────────────────────────────────────────

def load_odds_from_payout(payout_path: str) -> Dict:
    """払い戻しJSONからオッズを読み込む (シミュレーション用)"""
    with open(payout_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("payouts", {})


def load_odds_live(race_id: int, data_dir: str) -> Dict:
    """
    本番用: 確定オッズを優先、なければ推定オッズを使う
    data_dir: keiba_ai/data/
    """
    # 確定オッズ (payoutsディレクトリ)
    payout_path = os.path.join(data_dir, "payouts", f"{race_id}.json")
    if os.path.exists(payout_path):
        return load_odds_from_payout(payout_path)

    # 推定オッズ (oddsディレクトリ)
    odds_path = os.path.join(data_dir, "odds", f"{race_id}.json")
    if os.path.exists(odds_path):
        with open(odds_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("odds", {})

    return {}


# ── メイン API ──────────────────────────────────────────────────────────

class KellyBettingAI:
    """
    使い方:
        ai = KellyBettingAI(kelly_factor=0.25, min_ev=0.0)
        bets = ai.recommend(scores, horse_nos, odds_data, bankroll=100000)
        # または
        bets = ai.recommend_for_race(race_id, scores, horse_nos, data_dir, bankroll)
    """
    def __init__(
        self,
        kelly_factor: float = 0.25,
        min_ev: float = 0.0,
        max_bet_fraction: float = 0.10,
        max_total_fraction: float = 0.20,
        ticket_types: Optional[List[str]] = None,
    ):
        self.kelly_factor       = kelly_factor
        self.min_ev             = min_ev
        self.max_bet_fraction   = max_bet_fraction
        self.max_total_fraction = max_total_fraction
        self.ticket_types       = ticket_types  # None = 全券種

    def recommend(
        self,
        scores: np.ndarray,
        horse_nos: List[int],
        odds_data: Dict,
        bankroll: float = 100_000,
    ) -> List[Dict]:
        """
        スコアとオッズデータから馬券推薦を返す。

        Parameters
        ----------
        scores    : LambdaRankの生スコア (馬番順)
        horse_nos : 馬番リスト
        odds_data : {ticket_kind: [{horses, odds}, ...]}
        bankroll  : 手元資金 (円)
        """
        probs = softmax_probs(np.array(scores, dtype=float))
        all_bets = enumerate_bets(probs, horse_nos, odds_data, self.ticket_types)
        return select_and_size(
            all_bets, bankroll,
            kelly_factor       = self.kelly_factor,
            min_ev             = self.min_ev,
            max_bet_fraction   = self.max_bet_fraction,
            max_total_fraction = self.max_total_fraction,
        )

    def recommend_for_race(
        self,
        race_id: int,
        scores: np.ndarray,
        horse_nos: List[int],
        data_dir: str,
        bankroll: float = 100_000,
    ) -> List[Dict]:
        """本番用: data_dir からオッズを自動取得して推薦"""
        odds_data = load_odds_live(race_id, data_dir)
        if not odds_data:
            return []
        return self.recommend(scores, horse_nos, odds_data, bankroll)

    def recommend_from_payout(
        self,
        payout_path: str,
        scores: np.ndarray,
        horse_nos: List[int],
        bankroll: float = 100_000,
    ) -> List[Dict]:
        """シミュレーション用: 払い戻しJSONから直接推薦"""
        odds_data = load_odds_from_payout(payout_path)
        return self.recommend(scores, horse_nos, odds_data, bankroll)


# ── バックテスト ────────────────────────────────────────────────────────

def backtest(
    race_results: List[Dict],
    kelly_factor: float = 0.25,
    min_ev: float = 0.0,
    initial_bankroll: float = 100_000,
    ticket_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict:
    """
    過去レース結果を使ってバックテストを実行する。

    Parameters
    ----------
    race_results: [{"race_id", "scores", "horse_nos", "ranks", "payout_path"}, ...]
    kelly_factor: Kelly係数の倍率
    min_ev      : 最低期待値
    initial_bankroll: 初期資金

    Returns
    -------
    dict: {total_bet, total_return, roi, n_bets, n_hit, hit_rate, bankroll_history}
    """
    ai = KellyBettingAI(kelly_factor=kelly_factor, min_ev=min_ev,
                        ticket_types=ticket_types)
    bankroll = initial_bankroll
    total_bet = total_return = n_bets = n_hit = 0
    bankroll_history = [bankroll]
    detail_rows = []

    for r in race_results:
        if bankroll <= 0:
            break

        payout_path = r.get("payout_path", "")
        if not payout_path or not os.path.exists(payout_path):
            continue

        odds_data = load_odds_from_payout(payout_path)
        bets = ai.recommend(
            np.array(r["scores"]), r["horse_nos"], odds_data, bankroll
        )

        if not bets:
            bankroll_history.append(bankroll)
            continue

        # 実際の着順から的中判定
        ranks = {h: rk for h, rk in zip(r["horse_nos"], r["ranks"])}
        win_horses  = [h for h, rk in ranks.items() if rk == 1]
        top3_horses = [h for h, rk in ranks.items() if rk <= 3]
        top3_set    = set(top3_horses)
        top2_set    = set(h for h, rk in ranks.items() if rk <= 2)
        # 実着順の上位3頭 (馬単/三連単用)
        rank1 = win_horses[0] if win_horses else None
        rank2 = next((h for h, rk in ranks.items() if rk == 2), None)
        rank3 = next((h for h, rk in ranks.items() if rk == 3), None)

        race_bet = race_return = 0
        for b in bets:
            amount = b["bet_amount"]
            hs = b["horses"]
            kind = b["ticket_kind"]
            hit = False

            if kind == "tansho":
                hit = (hs[0] == rank1)
            elif kind == "fukusho":
                hit = (hs[0] in top3_set)
            elif kind == "umaren":
                hit = (set(hs) == top2_set)
            elif kind == "umatan":
                hit = (hs[0] == rank1 and hs[1] == rank2)
            elif kind == "wide":
                hit = (set(hs).issubset(top3_set))
            elif kind == "sanrenpuku":
                hit = (set(hs) == top3_set)
            elif kind == "sanrentan":
                hit = (hs[0] == rank1 and hs[1] == rank2 and hs[2] == rank3)

            payout = int(amount * b["odds"]) if hit else 0
            race_bet    += amount
            race_return += payout
            n_bets      += 1
            if hit:
                n_hit += 1

            detail_rows.append({**b, "race_id": r["race_id"],
                                 "hit": hit, "bet_amount": amount, "payout": payout})

        total_bet    += race_bet
        total_return += race_return
        bankroll     += (race_return - race_bet)
        bankroll_history.append(bankroll)

        if verbose:
            print(f"  race {r['race_id']}: bet={race_bet:,} return={race_return:,} "
                  f"bank={bankroll:,}")

    roi = (total_return / total_bet - 1.0) if total_bet > 0 else 0.0
    hit_rate = n_hit / n_bets if n_bets > 0 else 0.0

    return {
        "total_bet":        total_bet,
        "total_return":     total_return,
        "roi":              roi,
        "n_bets":           n_bets,
        "n_hit":            n_hit,
        "hit_rate":         hit_rate,
        "final_bankroll":   bankroll,
        "bankroll_history": bankroll_history,
        "details":          detail_rows,
    }


# ── バックテスト向けユーティリティ (predictor.py からも使う) ──────────────────

import itertools as _itertools


def market_probs_from_odds(tansho_odds_arr: np.ndarray) -> np.ndarray:
    """単勝オッズ配列 → 正規化マーケット確率 (Plackett-Luce の前提)"""
    inv = 1.0 / np.maximum(np.asarray(tansho_odds_arr, dtype=float), 1.01)
    return inv / inv.sum()


# 日本競馬の払戻率
_PAYOUT_RATES = {
    "tansho": 0.80, "fukusho": 0.75,
    "umaren": 0.75, "umatan": 0.75, "wide": 0.75,
    "sanrenpuku": 0.75, "sanrentan": 0.725,
}


def _estimate_market_odds(market_probs: np.ndarray, kind: str, horse_idxs: list) -> float:
    """マーケット確率から券種・馬の組合せオッズを推定する"""
    rate = _PAYOUT_RATES.get(kind, 0.75)
    if kind == "tansho":
        p = pl_win(market_probs, horse_idxs[0])
    elif kind == "fukusho":
        p = pl_place(market_probs, horse_idxs[0], n=3)
    elif kind == "umaren":
        p = pl_pair(market_probs, horse_idxs[0], horse_idxs[1])
    elif kind == "umatan":
        p = pl_order2(market_probs, horse_idxs[0], horse_idxs[1])
    elif kind == "wide":
        p = pl_wide(market_probs, horse_idxs[0], horse_idxs[1])
    elif kind == "sanrenpuku":
        p = pl_trio(market_probs, horse_idxs[0], horse_idxs[1], horse_idxs[2])
    elif kind == "sanrentan":
        p = pl_order3(market_probs, horse_idxs[0], horse_idxs[1], horse_idxs[2])
    else:
        return 1.0
    return (rate / p) if p > 1e-9 else 9999.0


def enumerate_race_bets(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    horse_nos: List[int],
    tansho_odds_arr: np.ndarray,
    ticket_types: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict]:
    """
    1レース分の全馬券候補を列挙し Kelly 係数を計算する。

    Parameters
    ----------
    model_probs    : LambdaRank の勝利確率 (softmax済)
    market_probs   : 単勝オッズから逆算したマーケット確率
    horse_nos      : 馬番リスト
    tansho_odds_arr: 各馬の単勝オッズ
    ticket_types   : 対象券種 (Noneで全券種)
    top_k          : 組合せ馬券の対象上位頭数

    Returns
    -------
    List of dicts: {ticket_kind, horses, model_prob, market_odds, ev, kelly}
    """
    if ticket_types is None:
        ticket_types = list(_PAYOUT_RATES.keys())

    top_idx = np.argsort(model_probs)[::-1][:top_k]
    bets = []

    for kind in ticket_types:
        if kind in ("tansho", "fukusho"):
            candidates = [(i,) for i in range(len(horse_nos))]
        elif kind in ("umaren", "wide"):
            candidates = list(_itertools.combinations(top_idx, 2))
        elif kind == "umatan":
            candidates = list(_itertools.permutations(top_idx, 2))
        elif kind == "sanrenpuku":
            candidates = list(_itertools.combinations(top_idx, 3))
        elif kind == "sanrentan":
            candidates = list(_itertools.permutations(top_idx, 3))
        else:
            continue

        for idx_tuple in candidates:
            if kind == "tansho":
                model_p = pl_win(model_probs, idx_tuple[0])
                market_odds = float(tansho_odds_arr[idx_tuple[0]])
            elif kind == "fukusho":
                model_p = pl_place(model_probs, idx_tuple[0], n=3)
                market_odds = _estimate_market_odds(market_probs, kind, list(idx_tuple))
            elif kind == "umaren":
                model_p = pl_pair(model_probs, idx_tuple[0], idx_tuple[1])
                market_odds = _estimate_market_odds(market_probs, kind, list(idx_tuple))
            elif kind == "umatan":
                model_p = pl_order2(market_probs, idx_tuple[0], idx_tuple[1])
                model_p = pl_order2(model_probs, idx_tuple[0], idx_tuple[1])
                market_odds = _estimate_market_odds(market_probs, kind, list(idx_tuple))
            elif kind == "wide":
                model_p = pl_wide(model_probs, idx_tuple[0], idx_tuple[1])
                market_odds = _estimate_market_odds(market_probs, kind, list(idx_tuple))
            elif kind == "sanrenpuku":
                model_p = pl_trio(model_probs, idx_tuple[0], idx_tuple[1], idx_tuple[2])
                market_odds = _estimate_market_odds(market_probs, kind, list(idx_tuple))
            elif kind == "sanrentan":
                model_p = pl_order3(model_probs, idx_tuple[0], idx_tuple[1], idx_tuple[2])
                market_odds = _estimate_market_odds(market_probs, kind, list(idx_tuple))
            else:
                continue

            if market_odds <= 1.0 or model_p <= 0:
                continue

            ev = model_p * market_odds - 1.0
            kelly = kelly_fraction(model_p, market_odds)

            if kelly > 0:
                bets.append({
                    "ticket_kind":  kind,
                    "horses":       [int(horse_nos[i]) for i in idx_tuple],
                    "model_prob":   round(model_p, 5),
                    "market_odds":  round(market_odds, 2),
                    "ev":           round(ev, 4),
                    "kelly":        round(kelly, 5),
                })
    return bets


def size_bets(
    bets: List[Dict],
    bankroll: float,
    kelly_factor: float = 0.25,
    max_bet_frac: float = 0.10,
    max_total_frac: float = 0.20,
) -> List[Dict]:
    """Kelly係数から掛け金を決定し bet_amount を付与して返す。"""
    sized = []
    for b in bets:
        raw = b["kelly"] * kelly_factor
        capped = min(raw, max_bet_frac)
        amount = int(bankroll * capped / 100) * 100
        if amount >= 100:
            sized.append({**b, "bet_amount": amount})

    if not sized:
        return []

    total = sum(b["bet_amount"] for b in sized)
    max_total = bankroll * max_total_frac
    if total > max_total:
        scale = max_total / total
        sized_scaled = []
        for b in bets:
            raw = b["kelly"] * kelly_factor
            capped = min(raw, max_bet_frac)
            amount = int(bankroll * capped * scale / 100) * 100
            if amount >= 100:
                sized_scaled.append({**b, "bet_amount": amount})
        sized = sized_scaled

    sized.sort(key=lambda x: -x["ev"])
    return sized


if __name__ == "__main__":
    # 動作確認: 仮想レース
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== KellyBettingAI 動作確認 ===")
    scores = np.array([2.1, 1.8, 1.2, 0.9, 0.5, 0.3, 0.1, -0.2])
    horse_nos = list(range(1, 9))
    probs = softmax_probs(scores)
    print(f"勝利確率: {[f'{p:.1%}' for p in probs]}")

    odds_data = {
        "tansho":     [{"horses": [1], "odds": 3.5}],
        "fukusho":    [{"horses": [1], "odds": 1.4}, {"horses": [2], "odds": 1.6}],
        "umaren":     [{"horses": [1, 2], "odds": 8.2}],
        "sanrenpuku": [{"horses": [1, 2, 3], "odds": 25.0}],
    }
    ai = KellyBettingAI(kelly_factor=0.25, min_ev=0.0)
    bets = ai.recommend(scores, horse_nos, odds_data, bankroll=100_000)
    print(f"\n推薦馬券 ({len(bets)}件):")
    for b in bets:
        print(f"  {b['ticket_kind']:12s} {b['horses']}  "
              f"確率={b['model_prob']:.1%}  オッズ={b['odds']:.1f}  "
              f"EV={b['ev']:+.1%}  Kelly={b['kelly']:.3f}  "
              f"→ {b['bet_amount']:,}円")
