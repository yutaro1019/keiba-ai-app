"""
賭け方提案エンジン
- AIが予測した確率と単勝オッズから期待値(EV)を計算
- 4スタイル: 堅実 / バランス / 一発狙い / AIおまかせ
- 券種: 単勝/複勝/ワイド/馬連/馬単/三連複/三連単
- 予算配分は「ケリー基準の分数」+ スタイル係数
"""
from itertools import combinations, permutations
from typing import List, Dict
import numpy as np
import pandas as pd

# JRA控除率(おおよそ)
TAKEOUT = {
    "tansho": 0.20,
    "fukusho": 0.20,
    "wide": 0.225,
    "umaren": 0.225,
    "umatan": 0.25,
    "sanrenpuku": 0.25,
    "sanrentan": 0.275,
}

STYLE_CONFIG = {
    "kenjitsu": {  # 堅実
        "name": "🛡️ 堅実型",
        "desc": "複勝・ワイド中心。的中率重視で資金を守る",
        "tickets": ["fukusho", "wide"],
        "max_combos": {"fukusho": 3, "wide": 4},
        "min_p": {"fukusho": 0.18, "wide": 0.08},  # 推定的中確率の下限
        "kelly_frac": 0.50,
    },
    "balance": {  # バランス
        "name": "⚖️ バランス型",
        "desc": "単勝・複勝・馬連・三連複を組み合わせ",
        "tickets": ["tansho", "fukusho", "umaren", "sanrenpuku"],
        "max_combos": {"tansho": 2, "fukusho": 2, "umaren": 4, "sanrenpuku": 6},
        "min_p": {"tansho": 0.10, "fukusho": 0.25, "umaren": 0.04, "sanrenpuku": 0.012},
        "kelly_frac": 0.30,
    },
    "ippatsu": {  # 一発狙い
        "name": "💥 一発狙い型",
        "desc": "三連単・馬単中心。少額で高配当を狙う",
        "tickets": ["tansho", "umatan", "sanrenpuku", "sanrentan"],
        "max_combos": {"tansho": 1, "umatan": 4, "sanrenpuku": 4, "sanrentan": 8},
        "min_p": {"tansho": 0.08, "umatan": 0.025, "sanrenpuku": 0.010, "sanrentan": 0.002},
        "kelly_frac": 0.40,
    },
    "ai": {  # AIおまかせ
        "name": "🤖 AIおまかせ",
        "desc": "EV(期待値)が最大の組み合わせをAIが自動選択",
        "tickets": ["fukusho", "wide", "umaren", "sanrenpuku", "tansho", "umatan", "sanrentan"],
        "max_combos": {"fukusho": 2, "wide": 3, "umaren": 4, "sanrenpuku": 6,
                       "tansho": 2, "umatan": 3, "sanrentan": 5},
        "min_p": {"fukusho": 0.20, "wide": 0.10, "umaren": 0.03, "sanrenpuku": 0.008,
                  "tansho": 0.06, "umatan": 0.015, "sanrentan": 0.0015},
        "kelly_frac": 0.30,
    },
    "roi150": {
        "name": "ROI150探索型",
        "desc": "複勝特化・高自信度・高EV条件で買う探索型",
        "tickets": ["fukusho"],
        "max_combos": {"fukusho": 2},
        "min_p": {"fukusho": 0.20},
        "kelly_frac": 0.30,
        "min_ev": 1.35,
        "rank_win_weight": 0.0,
        "stake_mode": "ev_prop",
        "default_min_confidence": 92.0,
        "strict_ev": True,
    },
    "diverse": {
        "name": "分散ROI型",
        "desc": "4点以上・2券種以上で買い、資金は3着内率の高い複勝へ厚く配分",
        "tickets": ["tansho", "fukusho", "wide", "umaren", "sanrenpuku"],
        "max_combos": {"tansho": 2, "fukusho": 3, "wide": 4, "umaren": 4, "sanrenpuku": 4},
        "min_p": {"tansho": 0.06, "fukusho": 0.20, "wide": 0.10, "umaren": 0.03, "sanrenpuku": 0.008},
        "kelly_frac": 0.30,
        "min_ev": 1.10,
        "rank_win_weight": 0.0,
        "stake_mode": "kind_weight",
        "kind_weights": {"fukusho": 120.0, "tansho": 1.0, "wide": 2.0, "umaren": 1.0, "sanrenpuku": 1.0},
        "default_min_confidence": 75.0,
        "strict_ev": True,
        "min_tickets": 4,
        "min_kinds": 2,
        "max_total_tickets": 8,
    },
    "diverse_more": {
        "name": "分散多め型",
        "desc": "分散ROI型より買うレース数を増やし、複勝厚め配分で回収を狙う",
        "tickets": ["tansho", "fukusho", "wide", "umaren", "sanrenpuku"],
        "max_combos": {"tansho": 2, "fukusho": 3, "wide": 4, "umaren": 4, "sanrenpuku": 4},
        "min_p": {"tansho": 0.06, "fukusho": 0.20, "wide": 0.10, "umaren": 0.03, "sanrenpuku": 0.008},
        "kelly_frac": 0.30,
        "min_ev": 1.05,
        "rank_win_weight": 0.0,
        "stake_mode": "kind_weight",
        "kind_weights": {"fukusho": 120.0, "tansho": 1.0, "wide": 2.0, "umaren": 1.0, "sanrenpuku": 1.0},
        "default_min_confidence": 75.0,
        "strict_ev": True,
        "min_tickets": 4,
        "min_kinds": 2,
        "max_total_tickets": 8,
    },
    "maxroi": {
        "name": "🎯 厳選・回収率重視",
        "desc": "全券種を比較し、単勝以外は市場よりかなり強い時だけ買って回収率を最優先",
        "tickets": ["tansho", "fukusho"],
        "max_combos": {
            "tansho": 1, "fukusho": 1,
        },
        "min_p": {
            "tansho": 0.05, "fukusho": 0.15,
        },
        "kelly_frac": 0.30,
        "min_ev": 2.00,
        "min_ev_by_kind": {
            "tansho": 2.00, "fukusho": 2.00,
        },
        "max_pred_rank_by_kind": {"tansho": 1, "fukusho": 3},
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "stake_mode": "equal",
        "default_min_confidence": 0.0,
        "strict_ev": True,
        "seed_each_kind": True,
        "min_tickets": 1,
        "min_kinds": 0,
        "max_total_tickets": 2,
        "max_total_points": 2,
    },
    "coverage120": {
        "name": "🧩 旧・広く買う比較用",
        "desc": "推定オッズ時代のワイド・三連複・BOX中心モード。実払戻では87%前後なので比較用",
        "tickets": [
            "wide", "sanrenpuku", "wide_box", "sanrenpuku_box",
        ],
        "max_combos": {
            "wide": 4, "sanrenpuku": 4, "wide_box": 1, "sanrenpuku_box": 1,
        },
        "min_p": {
            "wide": 0.07, "sanrenpuku": 0.003,
            "wide_box": 0.14, "sanrenpuku_box": 0.010,
        },
        "kelly_frac": 0.25,
        "min_ev": 1.00,
        "min_ev_by_kind": {
            "wide": 1.00, "sanrenpuku": 1.05,
            "wide_box": 1.00, "sanrenpuku_box": 1.05,
        },
        "odds_multiplier": {
            "wide": 2.70, "sanrenpuku": 3.80,
        },
        "max_pred_rank_by_kind": {
            "wide": 4, "sanrenpuku": 5,
            "wide_box": 4, "sanrenpuku_box": 4,
        },
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "stake_mode": "kind_weight",
        "kind_weights": {
            "wide": 2.8, "sanrenpuku": 2.0,
            "wide_box": 2.4, "sanrenpuku_box": 1.8,
        },
        "default_min_confidence": 0.0,
        "strict_ev": True,
        "seed_each_kind": True,
        "min_tickets": 2,
        "min_kinds": 0,
        "max_total_tickets": 7,
        "max_total_points": 16,
    },
    "actual120": {
        "name": "🔥 実払戻・三連単120狙い",
        "desc": "実払戻検証で強かった三連単1点型。2/3以上のレースで買い、期待値1.2以上の最上位1点に集中",
        "tickets": ["sanrentan"],
        "max_combos": {"sanrentan": 99},
        "min_p": {"sanrentan": 0.0},
        "kelly_frac": 0.30,
        "min_ev": 1.20,
        "min_ev_by_kind": {"sanrentan": 1.20},
        "max_pred_rank_by_kind": {"sanrentan": 5},
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "stake_mode": "top1",
        "default_min_confidence": 0.0,
        "strict_ev": True,
        "min_tickets": 1,
        "min_kinds": 0,
        "max_total_tickets": 1,
        "max_total_points": 1,
    },
    "roi_focus": {
        "name": "🎯 期待値重視",
        "desc": "単勝・複勝・ワイドだけでEVを計算。三連系や馬連の高配当外れ値は除外",
        "tickets": ["tansho", "fukusho", "wide"],
        "max_combos": {"tansho": 2, "fukusho": 2, "wide": 4},
        "min_p": {"tansho": 0.07, "fukusho": 0.24, "wide": 0.12},
        "kelly_frac": 0.30,
        "min_ev": 1.00,
        "min_ev_by_kind": {"tansho": 1.00, "fukusho": 0.95, "wide": 0.98},
        "max_odds_by_kind": {"tansho": 18.0, "fukusho": 7.0, "wide": 18.0},
        "max_pred_rank_by_kind": {"tansho": 4, "fukusho": 6, "wide": 5},
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "stake_mode": "kind_weight",
        "kind_weights": {"tansho": 1.1, "fukusho": 1.8, "wide": 2.8},
        "default_min_confidence": 0.0,
        "strict_ev": True,
        "min_tickets": 1,
        "min_kinds": 0,
        "max_total_tickets": 5,
        "max_total_points": 8,
    },
    "hit_focus": {
        "name": "🟢 的中率重視",
        "desc": "3着内率を最優先しつつEVも確認。複勝・ワイドから当たりやすい買い目を動的に選択",
        "tickets": ["fukusho", "wide"],
        "max_combos": {"fukusho": 3, "wide": 3},
        "min_p": {"fukusho": 0.22, "wide": 0.10},
        "kelly_frac": 0.30,
        "min_ev": 0.70,
        "min_ev_by_kind": {"fukusho": 0.70, "wide": 0.80},
        "max_odds_by_kind": {"fukusho": 7.0, "wide": 18.0},
        "max_pred_rank_by_kind": {"fukusho": 6, "wide": 5},
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "selection_key": "p_hit",
        "stake_mode": "kind_weight",
        "kind_weights": {"fukusho": 4.0, "wide": 1.4},
        "default_min_confidence": 0.0,
        "strict_ev": True,
        "min_tickets": 1,
        "min_kinds": 0,
        "max_total_tickets": 5,
        "max_total_points": 7,
    },
    "hybrid": {
        "name": "⚖️ ハイブリッド",
        "desc": "ワイドBOXで的中を拾い、EVの高い三連複へ厚く配分。三連単BOXも強い時だけ候補に入れる",
        "tickets": ["wide", "sanrenpuku", "wide_box", "sanrentan_box", "sanrenpuku_box"],
        "max_combos": {"wide": 4, "sanrenpuku": 4, "wide_box": 1, "sanrentan_box": 1, "sanrenpuku_box": 1},
        "min_p": {"wide": 0.07, "sanrenpuku": 0.003, "wide_box": 0.14, "sanrentan_box": 0.006, "sanrenpuku_box": 0.010},
        "kelly_frac": 0.30,
        "min_ev": 1.00,
        "min_ev_by_kind": {
            "wide": 1.00,
            "sanrenpuku": 1.05,
            "wide_box": 1.00,
            "sanrentan_box": 1.30,
            "sanrenpuku_box": 1.05,
        },
        "max_pred_rank_by_kind": {
            "wide": 4,
            "sanrenpuku": 5,
            "wide_box": 4,
            "sanrentan_box": 3,
            "sanrenpuku_box": 4,
        },
        "odds_multiplier": {"wide": 2.70, "sanrenpuku": 3.80, "sanrentan_box": 1.00},
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "selection_key": "ev",
        "stake_mode": "kind_weight",
        "kind_weights": {
            "wide": 0.6,
            "sanrenpuku": 8.0,
            "wide_box": 0.6,
            "sanrentan_box": 0.2,
            "sanrenpuku_box": 0.5,
        },
        "default_min_confidence": 0.0,
        "strict_ev": True,
        "seed_each_kind": True,
        "min_tickets": 2,
        "min_kinds": 0,
        "max_total_tickets": 4,
        "max_total_points": 10,
    },
    "hybrid_hit": {
        "name": "⚖️ 的中ハイブリッド",
        "desc": "単勝・複勝も候補に入れ、複勝とワイドを厚めにしながらEVのある三連複も拾う",
        "tickets": ["tansho", "fukusho", "wide", "sanrenpuku", "wide_box", "sanrenpuku_box"],
        "max_combos": {
            "tansho": 2,
            "fukusho": 3,
            "wide": 4,
            "sanrenpuku": 3,
            "wide_box": 1,
            "sanrenpuku_box": 1,
        },
        "min_p": {
            "tansho": 0.09,
            "fukusho": 0.26,
            "wide": 0.10,
            "sanrenpuku": 0.006,
            "wide_box": 0.16,
            "sanrenpuku_box": 0.012,
        },
        "kelly_frac": 0.30,
        "min_ev": 0.90,
        "min_ev_by_kind": {
            "tansho": 0.95,
            "fukusho": 0.80,
            "wide": 0.88,
            "sanrenpuku": 1.05,
            "wide_box": 0.90,
            "sanrenpuku_box": 1.05,
        },
        "max_odds_by_kind": {
            "tansho": 18.0,
            "fukusho": 8.0,
            "wide": 20.0,
            "sanrenpuku": 120.0,
            "wide_box": 20.0,
            "sanrenpuku_box": 120.0,
        },
        "max_pred_rank_by_kind": {
            "tansho": 3,
            "fukusho": 5,
            "wide": 5,
            "sanrenpuku": 5,
            "wide_box": 4,
            "sanrenpuku_box": 4,
        },
        "odds_multiplier": {"wide": 2.30, "sanrenpuku": 3.20},
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "selection_key": "balanced",
        "stake_mode": "kind_weight",
        "kind_weights": {
            "tansho": 1.1,
            "fukusho": 4.0,
            "wide": 2.5,
            "sanrenpuku": 1.2,
            "wide_box": 2.0,
            "sanrenpuku_box": 0.9,
        },
        "default_min_confidence": 0.0,
        "strict_ev": True,
        "seed_each_kind": True,
        "min_tickets": 3,
        "min_kinds": 0,
        "max_total_tickets": 6,
        "max_total_points": 14,
    },
    "fukusho_roi": {
        "name": "🟢 複勝特化・前の高回収",
        "desc": "前の複勝中心モード。複勝だけを買い、回収率を優先して資金を集中",
        "tickets": ["fukusho"],
        "max_combos": {"fukusho": 4},
        "min_p": {"fukusho": 0.20},
        "kelly_frac": 0.30,
        "min_ev": 1.10,
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "stake_mode": "base_best",
        "base_bet": 100,
        "default_min_confidence": 50.0,
        "strict_ev": True,
        "min_tickets": 2,
        "min_kinds": 0,
        "max_total_tickets": 4,
    },
    "profitmax": {
        "name": "💹 多めに買う・利益重視",
        "desc": "全券種を比較し、単勝・複勝を土台に高配当券は強い時だけ混ぜる",
        "tickets": [
            "tansho", "fukusho", "wide", "umaren", "umatan",
            "sanrenpuku", "sanrentan", "wide_box", "umaren_box",
            "umatan_box", "sanrenpuku_box", "sanrentan_box",
        ],
        "max_combos": {
            "tansho": 3, "fukusho": 3, "wide": 4, "umaren": 4, "umatan": 4,
            "sanrenpuku": 5, "sanrentan": 5, "wide_box": 1, "umaren_box": 1,
            "umatan_box": 1, "sanrenpuku_box": 1, "sanrentan_box": 1,
        },
        "min_p": {
            "tansho": 0.045, "fukusho": 0.18, "wide": 0.07, "umaren": 0.020,
            "umatan": 0.008, "sanrenpuku": 0.003, "sanrentan": 0.0008,
            "wide_box": 0.16, "umaren_box": 0.050, "umatan_box": 0.030,
            "sanrenpuku_box": 0.012, "sanrentan_box": 0.005,
        },
        "kelly_frac": 0.30,
        "min_ev": 1.00,
        "min_ev_by_kind": {
            "tansho": 1.00, "fukusho": 1.00, "wide": 50.00, "umaren": 50.00,
            "umatan": 50.00, "sanrenpuku": 50.00, "sanrentan": 50.00,
            "wide_box": 50.00, "umaren_box": 50.00, "umatan_box": 50.00,
            "sanrenpuku_box": 50.00, "sanrentan_box": 50.00,
        },
        "rank_win_weight": 0.60,
        "rank_mode": "top1_win_fill_top3",
        "rank_top_extra": {"horse_dist_top3_rate": 0.04},
        "rank_rest_extra": {"horse_top3_rate": 0.02},
        "stake_mode": "kind_weight",
        "kind_weights": {
            "tansho": 3.0, "fukusho": 3.8, "wide": 2.4, "umaren": 1.8,
            "umatan": 1.4, "sanrenpuku": 1.5, "sanrentan": 0.9,
            "wide_box": 2.0, "umaren_box": 1.7, "umatan_box": 1.3,
            "sanrenpuku_box": 1.4, "sanrentan_box": 0.9,
        },
        "default_min_confidence": 40.0,
        "strict_ev": True,
        "min_tickets": 3,
        "min_kinds": 0,
        "seed_each_kind": True,
        "max_total_tickets": 12,
        "max_total_points": 22,
    },
}


# =====================================================
# オッズ推定(p_winとマーケット平均から、各券種のオッズを推定)
# =====================================================
def estimate_odds(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    pred_dfに 'odds'(単勝マーケット) があれば優先。なければp_winから推定。
    マーケットの暗黙確率(p_market)も計算し、AIエッジの判定に使う。
    """
    df = pred_df.copy()
    if "odds" not in df.columns or df["odds"].isna().all():
        # マーケットオッズが無い場合、p_winから単勝オッズを推定 (控除20%)
        p = df["p_win"].clip(1e-4, 0.95)
        df["odds_est"] = (1 - TAKEOUT["tansho"]) / p
        df["p_market"] = df["p_win"]
        df["p_market_top3"] = df["p_top3"]
    else:
        df["odds_est"] = df["odds"].fillna((1 - TAKEOUT["tansho"]) / df["p_win"].clip(1e-4, 0.95))
        # マーケット暗黙確率
        market_inv = (1 - TAKEOUT["tansho"]) / df["odds_est"].clip(1.01, 1e6)
        # レース内で1に正規化
        s = market_inv.sum()
        df["p_market"] = market_inv / s if s > 0 else df["p_win"]
        top3_seed = np.sqrt(df["p_market"].clip(1e-6, 1.0))
        top3_sum = top3_seed.sum()
        if top3_sum > 0:
            df["p_market_top3"] = (top3_seed * (3.0 / top3_sum)).clip(0.01, 0.97)
        else:
            df["p_market_top3"] = df["p_top3"]
    return df


def fukusho_odds_est(p_top3: float, win_odds: float = None) -> float:
    """複勝オッズ推定
    単勝オッズ(マーケット)がある場合はそれからダービー。なければモデル確率から推定。
    経験則: 複勝オッズ ≈ 1.0 + (単勝オッズ - 1) * 0.25
    """
    if win_odds is not None and not np.isnan(win_odds) and win_odds > 1.0:
        # 単勝オッズから複勝オッズを推定(人気馬は低い、人気させ马は高い)
        if win_odds < 5:
            return max(1.1, 1.0 + (win_odds - 1.0) * 0.30)
        elif win_odds < 20:
            return max(1.5, 1.0 + (win_odds - 1.0) * 0.22)
        else:
            return max(2.0, 1.0 + (win_odds - 1.0) * 0.15)
    # オッズ不明ならモデル確率より逆算(控除2割、ただ3頭中1頭にスケール)
    return max(1.1, (1 - TAKEOUT["fukusho"]) / max(p_top3 * 1.5, 1e-4))


def umaren_prob(p_a_win, p_b_win, p_a_top3, p_b_top3) -> float:
    """A,Bのいずれかが1着+他方が2着の確率を簡易推定"""
    # P(A=1,B=2) ≈ p_a_win * (p_b_top3/0.6)
    # P(B=1,A=2) ≈ p_b_win * (p_a_top3/0.6)
    pa = p_a_win * min(1.0, p_b_top3 / 0.6) * 0.5
    pb = p_b_win * min(1.0, p_a_top3 / 0.6) * 0.5
    return min(0.95, pa + pb)


def umatan_prob(p_a_win, p_b_top3) -> float:
    """A→Bの順(Aが1着・Bが2着)の確率"""
    return min(0.95, p_a_win * min(1.0, p_b_top3 / 0.6) * 0.5)


def trio_prob(p_top3_a, p_top3_b, p_top3_c) -> float:
    """A,B,Cの3頭が3着以内に入る確率(三連複)"""
    return min(0.95, p_top3_a * p_top3_b * p_top3_c * 1.0)


def trifecta_prob(p_a_win, p_b_top3, p_c_top3) -> float:
    """A→B→Cの順の確率(三連単)"""
    return min(0.95, p_a_win * (p_b_top3 / 0.6) * (p_c_top3 / 0.6) * 0.18)


def odds_est_for(ticket_type: str, ps: List[float]) -> float:
    """券種ごとの推定オッズ。market_pがあれば市場確率から、なければAI確率から逆算する。"""
    if isinstance(ps, list):
        market_p = ps[1] if len(ps) > 1 and ps[1] is not None else ps[0]
    else:
        market_p = ps
    p = max(float(market_p), 1e-5)
    return max(1.1, (1 - TAKEOUT[ticket_type]) / p)


def style_odds(ticket_kind: str, odds: float, cfg: Dict) -> float:
    multipliers = cfg.get("odds_multiplier", {})
    base_kind = BOX_BASE_KIND.get(ticket_kind, ticket_kind)
    multiplier = float(multipliers.get(ticket_kind, multipliers.get(base_kind, 1.0)))
    return max(1.1, float(odds) * multiplier)


def _load_actual_odds_for_pred(pred: pd.DataFrame):
    if "race_id" not in pred.columns or pred["race_id"].nunique(dropna=True) != 1:
        return None
    try:
        from race_scraper import load_odds_cache
    except Exception:
        return None
    race_id = pred["race_id"].dropna().iloc[0]
    try:
        return load_odds_cache(int(race_id))
    except Exception:
        return None


def candidate_odds(ticket_kind: str, horses: List[int], estimated_odds: float, cfg: Dict, odds_cache) -> float:
    if odds_cache:
        try:
            from race_scraper import actual_odds_multiplier
            actual = actual_odds_multiplier(odds_cache, ticket_kind, horses)
        except Exception:
            actual = None
        if actual is not None and actual > 0:
            return max(1.01, float(actual))
    return style_odds(ticket_kind, estimated_odds, cfg)


BOX_BASE_KIND = {
    "wide_box": "wide",
    "umaren_box": "umaren",
    "umatan_box": "umatan",
    "sanrenpuku_box": "sanrenpuku",
    "sanrentan_box": "sanrentan",
}

BOX_LABEL = {
    "wide_box": "ワイドBOX",
    "umaren_box": "馬連BOX",
    "umatan_box": "馬単BOX",
    "sanrenpuku_box": "三連複BOX",
    "sanrentan_box": "三連単BOX",
}

TICKET_LABEL = {
    "tansho": "単勝",
    "fukusho": "複勝",
    "wide": "ワイド",
    "umaren": "馬連",
    "umatan": "馬単",
    "sanrenpuku": "三連複",
    "sanrentan": "三連単",
}


def point_count(candidate: Dict) -> int:
    return max(1, int(candidate.get("unit_count", 1)))


def base_cost(candidate: Dict) -> int:
    cost = int(candidate.get("base_cost", 100))
    return max(100, (cost // 100) * 100)


def add_box_candidate(
    cands: List[Dict],
    box_kind: str,
    horse_nums: List[int],
    combos: List[Dict],
    min_p: float,
    p_factor: float = 0.75,
):
    """BOXを複数点の候補として追加する。betはBOX合計額、combosは100円単位の内訳。"""
    if not combos:
        return
    unit_count = len(combos)
    box_cost = unit_count * 100
    expected_return_base = sum(100 * float(c["p_hit"]) * float(c["odds_est"]) for c in combos)
    ev = expected_return_base / box_cost if box_cost > 0 else 0.0
    p_any = min(0.95, sum(float(c["p_hit"]) for c in combos) * p_factor)
    if p_any < min_p or p_any <= 0:
        return
    effective_odds = expected_return_base / (box_cost * p_any) if p_any > 0 else 1.1
    label = BOX_LABEL.get(box_kind, "BOX")
    nums = "-".join(str(n) for n in sorted(horse_nums))
    cands.append({
        "type": label,
        "name": f"{label} {nums} ({unit_count}点)",
        "horses": sorted(horse_nums),
        "p_hit": float(p_any),
        "odds_est": float(max(0.01, effective_odds)),
        "ev": float(ev),
        "ticket_kind": box_kind,
        "box_kind": BOX_BASE_KIND.get(box_kind, box_kind),
        "combos": combos,
        "unit_count": unit_count,
        "base_cost": box_cost,
        "max_pred_rank": int(len(horse_nums)),
    })


# =====================================================
# ケリー基準で配分
# =====================================================
def kelly_fraction(p: float, odds: float) -> float:
    """ケリー比率 f* = (b*p - q) / b, b=odds-1, q=1-p"""
    b = max(0.01, odds - 1.0)
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def _rank_plan_candidate(pred: pd.DataFrame, plan: Dict, cfg: Dict, odds_cache) -> Dict | None:
    """予想順位をそのまま使う固定買い目。荒い三連単ではなく、現実的な券種に向ける。"""
    kind = str(plan.get("ticket_kind", ""))
    ranks = [int(r) for r in plan.get("ranks", [])]
    if not kind or not ranks:
        return None
    if max(ranks) > len(pred) or min(ranks) < 1:
        return None

    rows = [pred.iloc[r - 1] for r in ranks]
    horse_nums = [int(row["horse_no"]) for row in rows]
    if kind in {"wide", "umaren", "sanrenpuku"}:
        horse_nums = sorted(horse_nums)

    if kind == "tansho":
        p = float(rows[0]["p_win"])
        odds = candidate_odds(kind, horse_nums, float(rows[0]["odds_est"]), cfg, odds_cache)
    elif kind == "fukusho":
        p = float(rows[0]["p_top3"])
        win_odds = float(rows[0]["odds_est"]) if pd.notna(rows[0].get("odds_est")) else None
        odds = candidate_odds(kind, horse_nums, fukusho_odds_est(p, win_odds), cfg, odds_cache)
    elif kind == "wide":
        p = min(0.95, float(rows[0]["p_top3"]) * float(rows[1]["p_top3"]) * 1.5)
        market_p = min(0.95, float(rows[0]["p_market_top3"]) * float(rows[1]["p_market_top3"]) * 1.5)
        odds = candidate_odds(kind, horse_nums, odds_est_for(kind, [p, market_p]), cfg, odds_cache)
    elif kind == "umaren":
        p = umaren_prob(rows[0]["p_win"], rows[1]["p_win"], rows[0]["p_top3"], rows[1]["p_top3"])
        market_p = umaren_prob(rows[0]["p_market"], rows[1]["p_market"], rows[0]["p_market_top3"], rows[1]["p_market_top3"])
        odds = candidate_odds(kind, horse_nums, odds_est_for(kind, [p, market_p]), cfg, odds_cache)
    elif kind == "umatan":
        p = umatan_prob(rows[0]["p_win"], rows[1]["p_top3"])
        market_p = umatan_prob(rows[0]["p_market"], rows[1]["p_market_top3"])
        odds = candidate_odds(kind, horse_nums, odds_est_for(kind, [p, market_p]), cfg, odds_cache)
    elif kind == "sanrenpuku":
        p = trio_prob(rows[0]["p_top3"], rows[1]["p_top3"], rows[2]["p_top3"])
        market_p = trio_prob(rows[0]["p_market_top3"], rows[1]["p_market_top3"], rows[2]["p_market_top3"])
        odds = candidate_odds(kind, horse_nums, odds_est_for(kind, [p, market_p]), cfg, odds_cache)
    else:
        return None

    label = TICKET_LABEL.get(kind, kind)
    joiner = "→" if kind in {"umatan", "sanrentan"} else "-"
    name = f"{label} {joiner.join(str(n) for n in horse_nums)}"
    return {
        "type": label,
        "name": name,
        "horses": horse_nums,
        "p_hit": float(p),
        "odds_est": float(odds),
        "ev": float(p * odds),
        "ticket_kind": kind,
        "max_pred_rank": int(max(ranks)),
        "plan_weight": float(plan.get("weight", 1.0)),
    }


def build_rank_plan_bets(pred: pd.DataFrame, budget: int, cfg: Dict) -> List[Dict]:
    """順位予想を素直に使い、固定ウェイトで100円単位に配分する。"""
    pred = estimate_odds(pred).reset_index(drop=True)
    odds_cache = _load_actual_odds_for_pred(pred)
    candidates = []
    for plan in cfg.get("rank_plan", []):
        c = _rank_plan_candidate(pred, plan, cfg, odds_cache)
        if c is not None:
            candidates.append(c)
    if not candidates:
        return []

    weights = np.array([max(0.01, float(c.get("plan_weight", 1.0))) for c in candidates], dtype=float)
    weights = weights / weights.sum()
    costs = np.array([base_cost(c) for c in candidates], dtype=float)
    amounts = np.floor((weights * int(budget)) / 100.0) * 100.0

    leftover = int(budget) - int(amounts.sum())
    order = sorted(
        range(len(candidates)),
        key=lambda i: (candidates[i]["plan_weight"], candidates[i]["p_hit"], candidates[i]["ev"]),
        reverse=True,
    )
    while leftover >= 100 and order:
        placed = False
        for i in order:
            if leftover >= costs[i]:
                amounts[i] += costs[i]
                leftover -= int(costs[i])
                placed = True
                break
        if not placed:
            break

    out = []
    for c, amount in zip(candidates, amounts):
        if amount < base_cost(c):
            continue
        c2 = dict(c)
        c2["bet"] = int(amount)
        c2["payout_est"] = int(amount * c["odds_est"])
        out.append(c2)
    return out


# =====================================================
# 候補生成
# =====================================================
def generate_candidates(pred: pd.DataFrame, style: str) -> List[Dict]:
    """
    pred: predict_race の結果(pred_rank昇順)
    style: 'kenjitsu' / 'balance' / 'ippatsu' / 'ai'
    Returns: 各候補の dict のリスト
    """
    cfg = STYLE_CONFIG[style]
    pred = estimate_odds(pred)
    odds_cache = _load_actual_odds_for_pred(pred)

    # pandas の iloc を組み合わせループ内で何万回も呼ぶとシミュレーションが重くなるため、
    # 候補生成では最初に軽い dict リストへ変換する。
    horses = pred.head(8).reset_index(drop=True).to_dict("records")  # 候補は上位8頭まで

    cands = []

    # ---- 単勝 ----
    if "tansho" in cfg["tickets"]:
        for row in horses:
            p = row["p_win"]
            if p < cfg["min_p"]["tansho"]:
                continue
            horse_nums = [int(row["horse_no"])]
            odds = candidate_odds("tansho", horse_nums, float(row["odds_est"]), cfg, odds_cache)
            ev = p * odds
            cands.append({
                "type": "単勝",
                "name": f"単勝 {int(row['horse_no'])}番",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": odds,
                "ev": float(ev),
                "ticket_kind": "tansho",
                "pred_rank": int(row.get("pred_rank", 99)),
                "max_pred_rank": int(row.get("pred_rank", 99)),
            })

    # ---- 複勝 ----
    if "fukusho" in cfg["tickets"]:
        for row in horses:
            p = row["p_top3"]
            if p < cfg["min_p"]["fukusho"]:
                continue
            win_odds = row.get("odds_est", None)
            horse_nums = [int(row["horse_no"])]
            odds = candidate_odds("fukusho", horse_nums, fukusho_odds_est(p, win_odds), cfg, odds_cache)
            ev = p * odds
            cands.append({
                "type": "複勝",
                "name": f"複勝 {int(row['horse_no'])}番",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": odds,
                "ev": float(ev),
                "ticket_kind": "fukusho",
                "pred_rank": int(row.get("pred_rank", 99)),
                "max_pred_rank": int(row.get("pred_rank", 99)),
            })

    # ---- ワイド ----
    if "wide" in cfg["tickets"]:
        for i, j in combinations(range(min(6, len(horses))), 2):
            r1 = horses[i]; r2 = horses[j]
            # ワイド: 2頭ともtop3に入る確率(独立近似)
            p = min(0.95, r1["p_top3"] * r2["p_top3"] * 1.5)
            if p < cfg["min_p"]["wide"]:
                continue
            market_p = min(0.95, r1["p_market_top3"] * r2["p_market_top3"] * 1.5)
            horse_nums = sorted([int(r1["horse_no"]), int(r2["horse_no"])])
            odds = candidate_odds("wide", horse_nums, odds_est_for("wide", [p, market_p]), cfg, odds_cache)
            ev = p * odds
            cands.append({
                "type": "ワイド",
                "name": f"ワイド {int(r1['horse_no'])}-{int(r2['horse_no'])}",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": odds,
                "ev": float(ev),
                "ticket_kind": "wide",
                "max_pred_rank": int(max(i, j) + 1),
            })

    # ---- 馬連 ----
    if "umaren" in cfg["tickets"]:
        for i, j in combinations(range(min(6, len(horses))), 2):
            r1 = horses[i]; r2 = horses[j]
            p = umaren_prob(r1["p_win"], r2["p_win"], r1["p_top3"], r2["p_top3"])
            if p < cfg["min_p"]["umaren"]:
                continue
            market_p = umaren_prob(r1["p_market"], r2["p_market"], r1["p_market_top3"], r2["p_market_top3"])
            horse_nums = sorted([int(r1["horse_no"]), int(r2["horse_no"])])
            odds = candidate_odds("umaren", horse_nums, odds_est_for("umaren", [p, market_p]), cfg, odds_cache)
            ev = p * odds
            cands.append({
                "type": "馬連",
                "name": f"馬連 {int(r1['horse_no'])}-{int(r2['horse_no'])}",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": odds,
                "ev": float(ev),
                "ticket_kind": "umaren",
                "max_pred_rank": int(max(i, j) + 1),
            })

    # ---- 馬単 ----
    if "umatan" in cfg["tickets"]:
        for i, j in permutations(range(min(5, len(horses))), 2):
            r1 = horses[i]; r2 = horses[j]
            p = umatan_prob(r1["p_win"], r2["p_top3"])
            if p < cfg["min_p"]["umatan"]:
                continue
            market_p = umatan_prob(r1["p_market"], r2["p_market_top3"])
            horse_nums = [int(r1["horse_no"]), int(r2["horse_no"])]
            odds = candidate_odds("umatan", horse_nums, odds_est_for("umatan", [p, market_p]), cfg, odds_cache)
            ev = p * odds
            cands.append({
                "type": "馬単",
                "name": f"馬単 {int(r1['horse_no'])}→{int(r2['horse_no'])}",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": odds,
                "ev": float(ev),
                "ticket_kind": "umatan",
                "max_pred_rank": int(max(i, j) + 1),
            })

    # ---- 三連複 ----
    if "sanrenpuku" in cfg["tickets"]:
        n = min(6, len(horses))
        for i, j, k in combinations(range(n), 3):
            r1 = horses[i]; r2 = horses[j]; r3 = horses[k]
            p = trio_prob(r1["p_top3"], r2["p_top3"], r3["p_top3"])
            if p < cfg["min_p"]["sanrenpuku"]:
                continue
            market_p = trio_prob(r1["p_market_top3"], r2["p_market_top3"], r3["p_market_top3"])
            nums = sorted([int(r1['horse_no']), int(r2['horse_no']), int(r3['horse_no'])])
            odds = candidate_odds("sanrenpuku", nums, odds_est_for("sanrenpuku", [p, market_p]), cfg, odds_cache)
            ev = p * odds
            cands.append({
                "type": "三連複",
                "name": f"三連複 {nums[0]}-{nums[1]}-{nums[2]}",
                "horses": nums,
                "p_hit": float(p),
                "odds_est": odds,
                "ev": float(ev),
                "ticket_kind": "sanrenpuku",
                "max_pred_rank": int(max(i, j, k) + 1),
            })

    # ---- 三連単 ----
    if "sanrentan" in cfg["tickets"]:
        n = min(5, len(horses))
        for i, j, k in permutations(range(n), 3):
            r1 = horses[i]; r2 = horses[j]; r3 = horses[k]
            p = trifecta_prob(r1["p_win"], r2["p_top3"], r3["p_top3"])
            if p < cfg["min_p"]["sanrentan"]:
                continue
            market_p = trifecta_prob(r1["p_market"], r2["p_market_top3"], r3["p_market_top3"])
            horse_nums = [int(r1["horse_no"]), int(r2["horse_no"]), int(r3["horse_no"])]
            odds = candidate_odds("sanrentan", horse_nums, odds_est_for("sanrentan", [p, market_p]), cfg, odds_cache)
            ev = p * odds
            cands.append({
                "type": "三連単",
                "name": f"三連単 {int(r1['horse_no'])}→{int(r2['horse_no'])}→{int(r3['horse_no'])}",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": odds,
                "ev": float(ev),
                "ticket_kind": "sanrentan",
                "max_pred_rank": int(max(i, j, k) + 1),
                "order_tiebreak": [float(r2["p_top3"]), -float(r3["p_top3"])],
            })

    # ---- BOX系 ----
    if "wide_box" in cfg["tickets"] and len(horses) >= 3:
        n = min(4, len(horses))
        box_rows = horses[:n]
        combo_specs = []
        for i, j in combinations(range(n), 2):
            r1 = box_rows[i]; r2 = box_rows[j]
            p = min(0.95, r1["p_top3"] * r2["p_top3"] * 1.5)
            market_p = min(0.95, r1["p_market_top3"] * r2["p_market_top3"] * 1.5)
            horse_nums = sorted([int(r1["horse_no"]), int(r2["horse_no"])])
            odds = candidate_odds("wide", horse_nums, odds_est_for("wide", [p, market_p]), cfg, odds_cache)
            combo_specs.append({
                "ticket_kind": "wide",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": float(odds),
            })
        add_box_candidate(
            cands,
            "wide_box",
            [int(r["horse_no"]) for r in box_rows],
            combo_specs,
            cfg["min_p"].get("wide_box", 0.0),
            p_factor=0.70,
        )

    if "umaren_box" in cfg["tickets"] and len(horses) >= 3:
        n = min(4, len(horses))
        box_rows = horses[:n]
        combo_specs = []
        for i, j in combinations(range(n), 2):
            r1 = box_rows[i]; r2 = box_rows[j]
            p = umaren_prob(r1["p_win"], r2["p_win"], r1["p_top3"], r2["p_top3"])
            market_p = umaren_prob(r1["p_market"], r2["p_market"], r1["p_market_top3"], r2["p_market_top3"])
            horse_nums = sorted([int(r1["horse_no"]), int(r2["horse_no"])])
            odds = candidate_odds("umaren", horse_nums, odds_est_for("umaren", [p, market_p]), cfg, odds_cache)
            combo_specs.append({
                "ticket_kind": "umaren",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": float(odds),
            })
        add_box_candidate(
            cands,
            "umaren_box",
            [int(r["horse_no"]) for r in box_rows],
            combo_specs,
            cfg["min_p"].get("umaren_box", 0.0),
            p_factor=0.78,
        )

    if "umatan_box" in cfg["tickets"] and len(horses) >= 3:
        n = min(3, len(horses))
        box_rows = horses[:n]
        combo_specs = []
        for i, j in permutations(range(n), 2):
            r1 = box_rows[i]; r2 = box_rows[j]
            p = umatan_prob(r1["p_win"], r2["p_top3"])
            market_p = umatan_prob(r1["p_market"], r2["p_market_top3"])
            horse_nums = [int(r1["horse_no"]), int(r2["horse_no"])]
            odds = candidate_odds("umatan", horse_nums, odds_est_for("umatan", [p, market_p]), cfg, odds_cache)
            combo_specs.append({
                "ticket_kind": "umatan",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": float(odds),
            })
        add_box_candidate(
            cands,
            "umatan_box",
            [int(r["horse_no"]) for r in box_rows],
            combo_specs,
            cfg["min_p"].get("umatan_box", 0.0),
            p_factor=0.62,
        )

    if "sanrenpuku_box" in cfg["tickets"] and len(horses) >= 4:
        n = min(4, len(horses))
        box_rows = horses[:n]
        combo_specs = []
        for i, j, k in combinations(range(n), 3):
            r1 = box_rows[i]; r2 = box_rows[j]; r3 = box_rows[k]
            p = trio_prob(r1["p_top3"], r2["p_top3"], r3["p_top3"])
            market_p = trio_prob(r1["p_market_top3"], r2["p_market_top3"], r3["p_market_top3"])
            horse_nums = sorted([int(r1["horse_no"]), int(r2["horse_no"]), int(r3["horse_no"])])
            odds = candidate_odds("sanrenpuku", horse_nums, odds_est_for("sanrenpuku", [p, market_p]), cfg, odds_cache)
            combo_specs.append({
                "ticket_kind": "sanrenpuku",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": float(odds),
            })
        add_box_candidate(
            cands,
            "sanrenpuku_box",
            [int(r["horse_no"]) for r in box_rows],
            combo_specs,
            cfg["min_p"].get("sanrenpuku_box", 0.0),
            p_factor=0.60,
        )

    if "sanrentan_box" in cfg["tickets"] and len(horses) >= 3:
        n = min(3, len(horses))
        box_rows = horses[:n]
        combo_specs = []
        for i, j, k in permutations(range(n), 3):
            r1 = box_rows[i]; r2 = box_rows[j]; r3 = box_rows[k]
            p = trifecta_prob(r1["p_win"], r2["p_top3"], r3["p_top3"])
            market_p = trifecta_prob(r1["p_market"], r2["p_market_top3"], r3["p_market_top3"])
            horse_nums = [int(r1["horse_no"]), int(r2["horse_no"]), int(r3["horse_no"])]
            odds = candidate_odds("sanrentan", horse_nums, odds_est_for("sanrentan", [p, market_p]), cfg, odds_cache)
            combo_specs.append({
                "ticket_kind": "sanrentan",
                "horses": horse_nums,
                "p_hit": float(p),
                "odds_est": float(odds),
                "order_tiebreak": [float(r2["p_top3"]), -float(r3["p_top3"])],
            })
        add_box_candidate(
            cands,
            "sanrentan_box",
            [int(r["horse_no"]) for r in box_rows],
            combo_specs,
            cfg["min_p"].get("sanrentan_box", 0.0),
            p_factor=0.50,
        )

    return cands


# =====================================================
# 予算配分
# =====================================================
def min_ev_for_candidate(candidate: Dict, cfg: Dict, default_min_ev: float) -> float:
    """券種別のEV下限。三連複など推定オッズ券種だけ別枠で調整する。"""
    by_kind = cfg.get("min_ev_by_kind", {})
    return float(by_kind.get(candidate.get("ticket_kind"), default_min_ev))


def rank_allowed_for_candidate(candidate: Dict, cfg: Dict) -> bool:
    by_kind = cfg.get("max_pred_rank_by_kind", {})
    limit = by_kind.get(candidate.get("ticket_kind"))
    if limit is None:
        return True
    rank_value = candidate.get("max_pred_rank", candidate.get("pred_rank", 999))
    try:
        return int(rank_value) <= int(limit)
    except Exception:
        return True


def odds_allowed_for_candidate(candidate: Dict, cfg: Dict) -> bool:
    """極端な推定オッズを外して、3連単的な外れ値に引っ張られないようにする。"""
    by_kind = cfg.get("max_odds_by_kind", {})
    limit = by_kind.get(candidate.get("ticket_kind"))
    if limit is None:
        return True
    try:
        return float(candidate.get("odds_est", 0.0)) <= float(limit)
    except Exception:
        return True


def selection_sort_tuple(candidate: Dict, cfg: Dict):
    kind = candidate.get("ticket_kind")
    key = cfg.get("selection_key_by_kind", {}).get(kind, cfg.get("selection_key", "ev"))
    # 三連単の2着/3着など、モデルが同じ式で評価する買い目は極小の浮動小数差で
    # 順番が入れ替わりやすい。丸めた指標と馬番の昇順でタイブレークして再現性を保つ。
    order_tiebreak = tuple(float(x) for x in candidate.get("order_tiebreak", []))
    horses = tuple(-int(h) for h in candidate.get("horses", []) if pd.notna(h))
    if key == "p_hit":
        return (
            round(float(candidate.get("p_hit", 0.0)), 10),
            round(float(candidate.get("ev", 0.0)), 10),
            *order_tiebreak,
            *horses,
        )
    if key == "balanced":
        return (
            round(float(candidate.get("ev", 0.0)) * float(candidate.get("p_hit", 0.0)), 10),
            round(float(candidate.get("ev", 0.0)), 10),
            *order_tiebreak,
            *horses,
        )
    return (
        round(float(candidate.get("ev", 0.0)), 10),
        round(float(candidate.get("p_hit", 0.0)), 10),
        *order_tiebreak,
        *horses,
    )


def trim_to_budget_and_points(selected: List[Dict], budget: int, cfg: Dict) -> List[Dict]:
    """BOXのような複数点候補を含め、最低購入額と最大点数に収まる候補だけ残す。"""
    max_points = cfg.get("max_total_points")
    out = []
    used_budget = 0
    used_points = 0
    for c in selected:
        cost = base_cost(c)
        points = point_count(c)
        if used_budget + cost > budget:
            continue
        if max_points is not None and used_points + points > int(max_points):
            continue
        out.append(c)
        used_budget += cost
        used_points += points
    return out


def allocate_budget(cands: List[Dict], budget: int, style: str) -> List[Dict]:
    """
    候補をEVでフィルタし、スタイルごとの分散条件と賭け金配分を適用する。
    """
    cfg = STYLE_CONFIG[style]
    min_ev = cfg.get("min_ev", 0.85)
    pos_ev = [
        c for c in cands
        if c["ev"] >= min_ev_for_candidate(c, cfg, min_ev)
        and rank_allowed_for_candidate(c, cfg)
        and odds_allowed_for_candidate(c, cfg)
    ]
    if len(pos_ev) == 0:
        if cfg.get("strict_ev"):
            return []
        pos_ev = sorted(cands, key=lambda c: c["ev"], reverse=True)[:5]

    by_kind = {}
    for c in sorted(pos_ev, key=lambda x: selection_sort_tuple(x, cfg), reverse=True):
        k = c["ticket_kind"]
        by_kind.setdefault(k, [])
        if len(by_kind[k]) < cfg["max_combos"].get(k, 99):
            by_kind[k].append(c)

    min_kinds = int(cfg.get("min_kinds", 0))
    max_total = cfg.get("max_total_tickets")

    if cfg.get("seed_each_kind"):
        selected = []
        seen = set()
        for kind in cfg.get("tickets", []):
            items = by_kind.get(kind)
            if not items:
                continue
            selected.append(items[0])
            seen.add(id(items[0]))
        rest = [
            c
            for items in by_kind.values()
            for c in items
            if id(c) not in seen
        ]
        rest.sort(key=lambda c: selection_sort_tuple(c, cfg), reverse=True)
        selected.extend(rest)
        if max_total is not None:
            selected = selected[:max_total]
    elif min_kinds > 0:
        if len(by_kind) < min_kinds:
            return []

        selected = []
        for _, items in sorted(by_kind.items(), key=lambda kv: selection_sort_tuple(kv[1][0], cfg), reverse=True):
            if len(selected) >= min_kinds:
                break
            selected.append(items[0])

        used = {id(c) for c in selected}
        rest = [
            c
            for items in by_kind.values()
            for c in items
            if id(c) not in used
        ]
        rest.sort(key=lambda c: selection_sort_tuple(c, cfg), reverse=True)
        for c in rest:
            if max_total is not None and len(selected) >= max_total:
                break
            selected.append(c)
    else:
        selected = [c for v in by_kind.values() for c in v]
        selected.sort(key=lambda c: selection_sort_tuple(c, cfg), reverse=True)
        if max_total is not None:
            selected = selected[:max_total]

    selected = trim_to_budget_and_points(selected, budget, cfg)

    if not selected:
        return []
    if cfg.get("stake_mode") == "top1":
        selected = sorted(selected, key=lambda c: c["ev"], reverse=True)[:1]
    if cfg.get("stake_mode") == "top1_hit":
        selected = sorted(selected, key=lambda c: (c["p_hit"], c["ev"]), reverse=True)[:1]

    if sum(point_count(c) for c in selected) < int(cfg.get("min_tickets", 1)):
        return []

    stake_mode = cfg.get("stake_mode", "kelly")
    if stake_mode == "equal":
        weights = np.ones(len(selected), dtype=float) / len(selected)
    elif stake_mode == "fixed_kind_amounts":
        fixed = cfg.get("fixed_amounts", {})
        out = []
        spent = 0
        used_kinds = set()
        for c in selected:
            kind = c["ticket_kind"]
            if kind in used_kinds:
                continue
            amount = int(fixed.get(kind, 0))
            amount = (amount // 100) * 100
            if amount < base_cost(c):
                continue
            if spent + amount > budget:
                continue
            c2 = dict(c)
            c2["bet"] = amount
            c2["payout_est"] = int(amount * c["odds_est"])
            out.append(c2)
            spent += amount
            used_kinds.add(kind)
        return out
    elif stake_mode == "ev_prop":
        weights = np.array(
            [max(0.01, c["ev"] - min_ev_for_candidate(c, cfg, min_ev) + 0.05) for c in selected],
            dtype=float,
        )
        weights = weights / weights.sum()
    elif stake_mode == "kind_weight":
        base = np.array([base_cost(c) for c in selected], dtype=float)
        if base.sum() > budget:
            return []
        remaining = budget - base.sum()
        kind_weights = cfg.get("kind_weights", {})
        weights = np.array(
            [
                float(kind_weights.get(c["ticket_kind"], 1.0))
                * max(0.01, c["ev"] - min_ev_for_candidate(c, cfg, min_ev) + 0.05)
                for c in selected
            ],
            dtype=float,
        )
        if weights.sum() <= 0:
            weights = np.ones(len(selected), dtype=float)
        weights = weights / weights.sum()
        extra = np.floor((weights * remaining) / base) * base
        amounts = base + extra
        leftover = budget - amounts.sum()
        while leftover >= 100:
            eligible = [i for i, cost in enumerate(base) if cost <= leftover]
            if not eligible:
                break
            idx = max(eligible, key=lambda i: weights[i])
            amounts[idx] += base[idx]
            leftover -= base[idx]
        out = []
        for c, amt in zip(selected, amounts):
            if amt < base_cost(c):
                continue
            c2 = dict(c)
            c2["bet"] = int(amt)
            c2["payout_est"] = int(amt * c["odds_est"])
            out.append(c2)
        out.sort(key=lambda c: c["bet"], reverse=True)
        return out
    elif stake_mode == "base_best":
        base_bet = int(cfg.get("base_bet", 100))
        base_bet = max(100, (base_bet // 100) * 100)
        base = np.array([max(base_bet, base_cost(c)) for c in selected], dtype=float)
        if base.sum() > budget:
            return []
        amounts = base.copy()
        leftover = budget - amounts.sum()
        if leftover >= 100:
            target_kind = cfg.get("bonus_target")
            target_indexes = [
                i for i, c in enumerate(selected)
                if target_kind is None or c["ticket_kind"] == target_kind
            ]
            if not target_indexes:
                target_indexes = list(range(len(selected)))
            idx = max(target_indexes, key=lambda i: (selected[i]["ev"], selected[i]["p_hit"]))
            cost = base_cost(selected[idx])
            amounts[idx] += int(leftover // cost) * cost
        out = []
        for c, amt in zip(selected, amounts):
            if amt < base_cost(c):
                continue
            c2 = dict(c)
            c2["bet"] = int(amt)
            c2["payout_est"] = int(amt * c["odds_est"])
            out.append(c2)
        out.sort(key=lambda c: c["bet"], reverse=True)
        return out
    else:
        weights = []
        for c in selected:
            kf = kelly_fraction(c["p_hit"], c["odds_est"]) * cfg["kelly_frac"]
            weights.append(kf)
        weights = np.array(weights)
        if weights.sum() <= 0:
            evs = np.array([max(0.01, c["ev"] - 1.0) for c in selected])
            weights = evs / evs.sum()
        else:
            weights = weights / weights.sum()

    costs = np.array([base_cost(c) for c in selected], dtype=float)
    raw_amounts = weights * budget
    amounts = np.floor(raw_amounts / costs) * costs
    leftover = budget - amounts.sum()
    while leftover >= 100 and len(amounts) > 0:
        eligible = [i for i, cost in enumerate(costs) if cost <= leftover]
        if not eligible:
            break
        idx = max(eligible, key=lambda i: selected[i]["ev"])
        amounts[idx] += costs[idx]
        leftover -= costs[idx]

    out = []
    for c, amt in zip(selected, amounts):
        if amt < base_cost(c):
            continue
        c2 = dict(c)
        c2["bet"] = int(amt)
        c2["payout_est"] = int(amt * c["odds_est"])
        out.append(c2)

    out.sort(key=lambda c: c["bet"], reverse=True)
    return out


# =====================================================
# 公開API
# =====================================================
def rank_predictions(pred_df: pd.DataFrame, style: str) -> pd.DataFrame:
    """スタイルごとの予想順位。1位は勝ち切り力、2〜3位は3着内率を重視できる。"""
    cfg = STYLE_CONFIG.get(style, {})
    w = cfg.get("rank_win_weight")
    if w is None:
        return pred_df.sort_values("pred_rank").reset_index(drop=True)

    def _zscore(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        med = s.median()
        if pd.isna(med):
            med = 0.0
        s = s.fillna(med)
        std = s.std(ddof=0)
        if pd.isna(std) or std <= 1e-9:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    def _add_extra_score(out: pd.DataFrame, base_score: pd.Series, extras: Dict[str, float]) -> pd.Series:
        score = base_score.copy()
        for col, coef in extras.items():
            if col in out.columns:
                score = score + float(coef) * _zscore(out[col])
        return score

    def _rank_one(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        top_base = float(w) * out["p_win"] + (1.0 - float(w)) * out["p_top3"]
        rest_w = float(cfg.get("rank_rest_win_weight", 0.0))
        rest_base = rest_w * out["p_win"] + (1.0 - rest_w) * out["p_top3"]
        out["_rank_top_score"] = _add_extra_score(out, top_base, cfg.get("rank_top_extra", {}))
        out["_rank_rest_score"] = _add_extra_score(out, rest_base, cfg.get("rank_rest_extra", {}))
        out["_tie_horse_no"] = pd.to_numeric(
            out["horse_no"] if "horse_no" in out.columns else pd.Series(np.arange(len(out)), index=out.index),
            errors="coerce",
        ).fillna(999)
        out["score"] = out["_rank_top_score"]
        if cfg.get("rank_mode") == "top1_win_fill_top3" and len(out) > 1:
            top_idx = out.sort_values(
                ["_rank_top_score", "_tie_horse_no"],
                ascending=[False, True],
                kind="mergesort",
            ).index[0]
            top = out.loc[[top_idx]]
            rest = out.drop(index=top_idx).sort_values(
                ["_rank_rest_score", "_tie_horse_no"],
                ascending=[False, True],
                kind="mergesort",
            )
            out = pd.concat([top, rest], axis=0)
        else:
            out = out.sort_values(
                ["_rank_top_score", "_tie_horse_no"],
                ascending=[False, True],
                kind="mergesort",
            )
        out = out.reset_index(drop=True)
        out["pred_rank"] = np.arange(1, len(out) + 1)
        out = out.drop(columns=["_rank_top_score", "_rank_rest_score", "_tie_horse_no"], errors="ignore")
        return out

    if "race_id" in pred_df.columns and pred_df["race_id"].nunique(dropna=False) > 1:
        ranked = [_rank_one(group) for _, group in pred_df.groupby("race_id", sort=False)]
        return pd.concat(ranked, ignore_index=True) if ranked else pred_df.copy()
    return _rank_one(pred_df)


def suggest(pred_df: pd.DataFrame, budget: int, style: str = "ai") -> Dict:
    """
    pred_df: KeibaPredictor.predict_race の出力
    budget: 予算(円)
    style: 'kenjitsu' / 'balance' / 'ippatsu' / 'ai'
    """
    if style not in STYLE_CONFIG:
        style = "ai"
    pred_df = rank_predictions(pred_df, style)
    cfg = STYLE_CONFIG[style]
    if cfg.get("stake_mode") == "rank_plan":
        bets = build_rank_plan_bets(pred_df, budget, cfg)
    else:
        cands = generate_candidates(pred_df, style)
        bets = allocate_budget(cands, budget, style)

    total_bet = sum(b["bet"] for b in bets)
    expected_return = sum(b["bet"] * b["odds_est"] * b["p_hit"] for b in bets)
    expected_profit = expected_return - total_bet
    n_tickets = sum(point_count(b) for b in bets)
    # 1点でも当たる確率(独立近似)
    p_any_hit = 1.0 - np.prod([1 - b["p_hit"] for b in bets]) if bets else 0.0

    return {
        "style": STYLE_CONFIG[style]["name"],
        "style_desc": STYLE_CONFIG[style]["desc"],
        "budget": int(budget),
        "total_bet": int(total_bet),
        "n_tickets": int(n_tickets),
        "expected_return": float(expected_return),
        "expected_profit": float(expected_profit),
        "expected_roi": float(expected_profit / total_bet) if total_bet > 0 else 0.0,
        "p_any_hit": float(p_any_hit),
        "bets": bets,
    }


def format_suggestion(result: Dict) -> str:
    """人が読める形式に整形"""
    lines = []
    lines.append(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"  {result['style']}  —  {result['style_desc']}")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"  予算: ¥{result['budget']:,}  /  購入合計: ¥{result['total_bet']:,}  /  点数: {result['n_tickets']}点")
    lines.append(f"  少なくとも1点当たる確率(推定): {result['p_any_hit']*100:.1f}%")
    lines.append(f"  期待払戻: ¥{result['expected_return']:,.0f}  (期待収支: ¥{result['expected_profit']:+,.0f}, 期待ROI: {result['expected_roi']*100:+.1f}%)")
    lines.append("")
    lines.append(f"  {'券種':<6}{'買い目':<28}{'購入額':>9}{'推定勝率':>10}{'推定オッズ':>10}{'期待払戻':>12}")
    lines.append("  " + "─" * 76)
    for b in result["bets"]:
        lines.append(
            f"  {b['type']:<6}{b['name']:<26}"
            f"¥{b['bet']:>7,}  "
            f"{b['p_hit']*100:>6.1f}%   "
            f"{b['odds_est']:>6.1f}倍   "
            f"¥{b['payout_est']:>9,}"
        )
    lines.append("")
    lines.append("  ※ オッズはAI推定値(レース直前の実オッズで微調整推奨)。")
    lines.append("  ※ 推定勝率はモデル予測の独立近似で参考値。実勝率と乖離する場合あり。")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from predictor import KeibaPredictor, load_race_from_data, list_recent_races

    p = KeibaPredictor()
    races = list_recent_races(2025, 3)
    rid = int(races.iloc[0]["race_id"])
    df = load_race_from_data(rid)
    pred = p.predict_race(df)
    print(f"\n=== レース {rid} の予想 ===")
    cols = [c for c in ["pred_rank", "horse_no", "frame_no", "p_win", "p_top3", "odds", "popularity", "rank"] if c in pred.columns]
    print(pred[cols].head(8).to_string(index=False))
    print()

    for style in ["kenjitsu", "balance", "ippatsu", "ai"]:
        result = suggest(pred, budget=3000, style=style)
        print(format_suggestion(result))
