import math
import os
import sys
import types
import unittest

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import betting


def sample_predictions():
    return pd.DataFrame(
        [
            {"horse_no": 1, "p_win": 0.24, "p_top3": 0.72, "odds": 3.2, "pred_rank": 1},
            {"horse_no": 2, "p_win": 0.18, "p_top3": 0.64, "odds": 5.0, "pred_rank": 2},
            {"horse_no": 3, "p_win": 0.15, "p_top3": 0.58, "odds": 6.5, "pred_rank": 3},
            {"horse_no": 4, "p_win": 0.12, "p_top3": 0.48, "odds": 9.0, "pred_rank": 4},
            {"horse_no": 5, "p_win": 0.08, "p_top3": 0.36, "odds": 14.0, "pred_rank": 5},
            {"horse_no": 6, "p_win": 0.05, "p_top3": 0.28, "odds": 22.0, "pred_rank": 6},
        ]
    )


class BettingMathTests(unittest.TestCase):
    def test_estimate_odds_without_market_odds_uses_probabilities(self):
        df = sample_predictions().drop(columns=["odds"])
        out = betting.estimate_odds(df)

        self.assertIn("odds_est", out.columns)
        self.assertAlmostEqual(out.loc[0, "odds_est"], 0.8 / 0.24, places=6)
        self.assertTrue(out["p_market"].equals(out["p_win"]))
        self.assertTrue(out["p_market_top3"].equals(out["p_top3"]))

    def test_estimate_odds_with_market_odds_normalizes_market_probability(self):
        out = betting.estimate_odds(sample_predictions())

        self.assertAlmostEqual(float(out["p_market"].sum()), 1.0, places=6)
        self.assertGreater(out.loc[0, "p_market_top3"], 0.0)
        self.assertLessEqual(out["p_market_top3"].max(), 0.97)

    def test_probability_helpers_are_bounded_and_ordered(self):
        self.assertGreater(betting.fukusho_odds_est(0.5, 20.0), betting.fukusho_odds_est(0.5, 3.0))
        self.assertAlmostEqual(betting.umatan_prob(0.2, 0.6), 0.1)
        self.assertLessEqual(betting.umaren_prob(0.9, 0.9, 0.9, 0.9), 0.95)
        self.assertLessEqual(betting.trio_prob(1.0, 1.0, 1.0), 0.95)
        self.assertLessEqual(betting.trifecta_prob(1.0, 1.0, 1.0), 0.95)
        self.assertGreater(betting.odds_est_for("wide", [0.2, 0.1]), 1.1)

    def test_budget_and_candidate_helpers(self):
        box = []
        betting.add_box_candidate(
            box,
            "wide_box",
            [1, 2, 3],
            [
                {"p_hit": 0.3, "odds_est": 4.0, "ticket_kind": "wide", "horses": [1, 2]},
                {"p_hit": 0.2, "odds_est": 6.0, "ticket_kind": "wide", "horses": [1, 3]},
            ],
            min_p=0.1,
        )

        self.assertEqual(betting.point_count(box[0]), 2)
        self.assertEqual(betting.base_cost(box[0]), 200)
        self.assertGreater(box[0]["ev"], 0.0)
        self.assertEqual(betting.kelly_fraction(0.1, 2.0), 0.0)
        self.assertGreater(betting.kelly_fraction(0.6, 3.0), 0.0)

    def test_candidate_odds_uses_actual_cache_when_available(self):
        original = sys.modules.get("race_scraper")
        sys.modules["race_scraper"] = types.SimpleNamespace(
            actual_odds_multiplier=lambda payload, kind, horses: 12.3
        )
        try:
            odds = betting.candidate_odds("wide", [1, 2], 4.5, {}, {"ok": True})
        finally:
            if original is None:
                sys.modules.pop("race_scraper", None)
            else:
                sys.modules["race_scraper"] = original

        self.assertEqual(odds, 12.3)

    def test_actual_odds_loader_handles_missing_and_single_race(self):
        self.assertIsNone(betting._load_actual_odds_for_pred(pd.DataFrame({"horse_no": [1]})))
        self.assertIsNone(
            betting._load_actual_odds_for_pred(pd.DataFrame({"race_id": [1, 2], "horse_no": [1, 2]}))
        )

        original = sys.modules.get("race_scraper")
        sys.modules["race_scraper"] = types.SimpleNamespace(load_odds_cache=lambda race_id: {"race_id": race_id})
        try:
            payload = betting._load_actual_odds_for_pred(pd.DataFrame({"race_id": [123456789012]}))
        finally:
            if original is None:
                sys.modules.pop("race_scraper", None)
            else:
                sys.modules["race_scraper"] = original

        self.assertEqual(payload, {"race_id": 123456789012})


class BettingFlowTests(unittest.TestCase):
    def tearDown(self):
        for name in [
            "_test_rank_plan",
            "_test_equal",
            "_test_fixed",
            "_test_ev_prop",
            "_test_base_best",
            "_test_kelly",
            "_test_min_kinds",
            "_test_top1_hit",
            "_test_select_p_hit",
            "_test_select_balanced",
            "_test_no_result",
        ]:
            betting.STYLE_CONFIG.pop(name, None)

    def cands(self):
        return [
            {
                "type": "A",
                "name": "A 1",
                "ticket_kind": "tansho",
                "horses": [1],
                "p_hit": 0.45,
                "odds_est": 3.0,
                "ev": 1.35,
                "max_pred_rank": 1,
            },
            {
                "type": "B",
                "name": "B 2",
                "ticket_kind": "fukusho",
                "horses": [2],
                "p_hit": 0.7,
                "odds_est": 1.8,
                "ev": 1.26,
                "max_pred_rank": 2,
            },
            {
                "type": "C",
                "name": "C 1-2",
                "ticket_kind": "wide",
                "horses": [1, 2],
                "p_hit": 0.35,
                "odds_est": 4.0,
                "ev": 1.4,
                "max_pred_rank": 2,
                "unit_count": 2,
                "base_cost": 200,
            },
        ]

    def test_generate_candidates_covers_ticket_types_and_boxes(self):
        cands = betting.generate_candidates(sample_predictions(), "profitmax")
        kinds = {c["ticket_kind"] for c in cands}

        self.assertIn("tansho", kinds)
        self.assertIn("fukusho", kinds)
        self.assertIn("wide", kinds)
        self.assertIn("umaren", kinds)
        self.assertIn("umatan", kinds)
        self.assertIn("sanrenpuku", kinds)
        self.assertIn("sanrentan", kinds)
        self.assertIn("wide_box", kinds)
        self.assertIn("sanrenpuku_box", kinds)

    def test_allocate_budget_filters_and_rounds_to_100_yen_units(self):
        bets = betting.suggest(sample_predictions(), budget=3000, style="hybrid")["bets"]

        self.assertTrue(bets)
        self.assertLessEqual(sum(b["bet"] for b in bets), 3000)
        self.assertTrue(all(b["bet"] % 100 == 0 for b in bets))
        self.assertTrue(all(b["ev"] >= 1.0 for b in bets))

    def test_allocate_budget_strict_ev_can_skip_bad_candidates(self):
        cands = [
            {"ticket_kind": "tansho", "p_hit": 0.1, "odds_est": 2.0, "ev": 0.2, "bet": 0, "horses": [1]},
        ]

        self.assertEqual(betting.allocate_budget(cands, 1000, "maxroi"), [])

    def test_rank_predictions_uses_win_for_top_and_top3_for_rest(self):
        df = pd.DataFrame(
            [
                {"race_id": 10, "horse_no": 4, "p_win": 0.10, "p_top3": 0.80, "pred_rank": 1},
                {"race_id": 10, "horse_no": 1, "p_win": 0.40, "p_top3": 0.45, "pred_rank": 2},
                {"race_id": 11, "horse_no": 2, "p_win": 0.20, "p_top3": 0.70, "pred_rank": 1},
                {"race_id": 11, "horse_no": 3, "p_win": 0.30, "p_top3": 0.60, "pred_rank": 2},
            ]
        )

        ranked = betting.rank_predictions(df, "hit_focus")

        self.assertEqual(ranked.loc[0, "horse_no"], 1)
        self.assertEqual(ranked.loc[0, "pred_rank"], 1)
        self.assertEqual(ranked.loc[2, "horse_no"], 3)
        self.assertEqual(ranked.loc[2, "pred_rank"], 1)

    def test_suggest_and_format_have_consistent_totals(self):
        result = betting.suggest(sample_predictions(), budget=3000, style="hit_focus")
        text = betting.format_suggestion(result)

        self.assertEqual(result["total_bet"], sum(b["bet"] for b in result["bets"]))
        self.assertTrue(0.0 <= result["p_any_hit"] <= 1.0)
        self.assertFalse(math.isnan(result["expected_roi"]))
        self.assertIn(f"{result['budget']:,}", text)

    def test_rank_plan_mode_allocates_from_ranked_horses(self):
        cfg = {
            "name": "rank plan",
            "desc": "rank plan",
            "stake_mode": "rank_plan",
            "rank_plan": [
                {"ticket_kind": "tansho", "ranks": [1], "weight": 2.0},
                {"ticket_kind": "fukusho", "ranks": [2], "weight": 1.0},
                {"ticket_kind": "wide", "ranks": [1, 2], "weight": 1.0},
                {"ticket_kind": "umaren", "ranks": [1, 3], "weight": 1.0},
                {"ticket_kind": "umatan", "ranks": [1, 2], "weight": 1.0},
                {"ticket_kind": "sanrenpuku", "ranks": [1, 2, 3], "weight": 1.0},
                {"ticket_kind": "bad", "ranks": [1], "weight": 1.0},
            ],
        }
        betting.STYLE_CONFIG["_test_rank_plan"] = cfg

        result = betting.suggest(sample_predictions(), 3000, "_test_rank_plan")

        self.assertTrue(result["bets"])
        self.assertLessEqual(result["total_bet"], 3000)
        self.assertIn("wide", {b["ticket_kind"] for b in result["bets"]})

    def test_allocation_modes_cover_fixed_equal_ev_base_and_kelly(self):
        base_cfg = {
            "name": "test",
            "desc": "test",
            "tickets": ["tansho", "fukusho", "wide"],
            "max_combos": {"tansho": 2, "fukusho": 2, "wide": 2},
            "min_p": {},
            "kelly_frac": 0.25,
            "min_ev": 1.0,
            "strict_ev": True,
            "min_tickets": 1,
            "max_total_tickets": 3,
        }

        betting.STYLE_CONFIG["_test_equal"] = {**base_cfg, "stake_mode": "equal"}
        equal = betting.allocate_budget(self.cands(), 900, "_test_equal")
        self.assertEqual(sum(b["bet"] for b in equal), 900)

        betting.STYLE_CONFIG["_test_fixed"] = {
            **base_cfg,
            "stake_mode": "fixed_kind_amounts",
            "fixed_amounts": {"tansho": 200, "wide": 400},
        }
        fixed = betting.allocate_budget(self.cands(), 1000, "_test_fixed")
        self.assertEqual({b["ticket_kind"] for b in fixed}, {"tansho", "wide"})

        betting.STYLE_CONFIG["_test_ev_prop"] = {**base_cfg, "stake_mode": "ev_prop"}
        ev_prop = betting.allocate_budget(self.cands(), 1000, "_test_ev_prop")
        self.assertTrue(ev_prop)

        betting.STYLE_CONFIG["_test_base_best"] = {
            **base_cfg,
            "stake_mode": "base_best",
            "base_bet": 100,
            "bonus_target": "wide",
        }
        base_best = betting.allocate_budget(self.cands(), 1000, "_test_base_best")
        self.assertEqual(max(base_best, key=lambda b: b["bet"])["ticket_kind"], "wide")

        betting.STYLE_CONFIG["_test_kelly"] = {**base_cfg, "stake_mode": "kelly"}
        kelly = betting.allocate_budget(self.cands(), 1000, "_test_kelly")
        self.assertTrue(kelly)

    def test_allocation_rejects_missing_required_kinds_and_bad_limits(self):
        cfg = {
            "name": "test",
            "desc": "test",
            "tickets": ["tansho", "wide"],
            "max_combos": {"tansho": 2, "wide": 2},
            "kelly_frac": 0.25,
            "min_ev": 1.0,
            "strict_ev": True,
            "min_kinds": 2,
            "min_tickets": 1,
            "max_pred_rank_by_kind": {"wide": 1},
            "max_odds_by_kind": {"tansho": 2.0},
        }
        betting.STYLE_CONFIG["_test_min_kinds"] = cfg

        self.assertFalse(betting.rank_allowed_for_candidate({"ticket_kind": "wide", "max_pred_rank": 3}, cfg))
        self.assertFalse(betting.odds_allowed_for_candidate({"ticket_kind": "tansho", "odds_est": 5.0}, cfg))
        self.assertEqual(betting.allocate_budget(self.cands(), 1000, "_test_min_kinds"), [])

    def test_top1_hit_and_trim_to_budget_paths(self):
        cfg = {
            "name": "test",
            "desc": "test",
            "tickets": ["tansho", "fukusho", "wide"],
            "max_combos": {"tansho": 2, "fukusho": 2, "wide": 2},
            "kelly_frac": 0.25,
            "min_ev": 1.0,
            "strict_ev": True,
            "stake_mode": "top1_hit",
            "min_tickets": 1,
            "max_total_points": 1,
        }
        betting.STYLE_CONFIG["_test_top1_hit"] = cfg

        out = betting.allocate_budget(self.cands(), 500, "_test_top1_hit")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bet"], 500)

    def test_smart_mode_and_no_result_fallback_paths(self):
        smart = betting.suggest(sample_predictions(), budget=3000, style="smart")

        self.assertIn(smart.get("chosen_style"), {"roi_focus", "hit_focus"})
        self.assertGreaterEqual(smart.get("confidence", 0.0), 0.0)

        betting.STYLE_CONFIG["_test_no_result"] = {
            "name": "no result",
            "desc": "no result",
            "tickets": ["tansho"],
            "max_combos": {"tansho": 1},
            "min_p": {"tansho": 0.99},
            "kelly_frac": 0.25,
            "min_ev": 99.0,
            "strict_ev": True,
            "min_tickets": 1,
        }
        bad = betting.suggest(sample_predictions(), budget=3000, style="_test_no_result")
        self.assertEqual(bad["total_bet"], 0)
        self.assertEqual(bad["bets"], [])

    def test_selection_sort_modes_and_exception_tolerant_limits(self):
        cfg_hit = {"selection_key": "p_hit"}
        cfg_balanced = {"selection_key": "balanced"}
        cand = {"ticket_kind": "wide", "p_hit": 0.4, "ev": 1.3, "horses": [1, 2], "order_tiebreak": [0.5]}

        self.assertGreater(betting.selection_sort_tuple(cand, cfg_hit)[0], 0.0)
        self.assertGreater(betting.selection_sort_tuple(cand, cfg_balanced)[0], 0.0)
        self.assertTrue(betting.rank_allowed_for_candidate({"ticket_kind": "wide", "max_pred_rank": "bad"}, {"max_pred_rank_by_kind": {"wide": 2}}))
        self.assertTrue(betting.odds_allowed_for_candidate({"ticket_kind": "wide", "odds_est": "bad"}, {"max_odds_by_kind": {"wide": 2.0}}))

    def test_seed_each_kind_and_candidate_odds_fallback_paths(self):
        cfg = {
            "name": "test",
            "desc": "test",
            "tickets": ["tansho", "fukusho", "wide"],
            "max_combos": {"tansho": 1, "fukusho": 1, "wide": 1},
            "kelly_frac": 0.25,
            "min_ev": 1.0,
            "strict_ev": True,
            "seed_each_kind": True,
            "min_tickets": 1,
            "max_total_tickets": 2,
            "stake_mode": "equal",
        }
        betting.STYLE_CONFIG["_test_select_p_hit"] = cfg
        out = betting.allocate_budget(self.cands(), 1000, "_test_select_p_hit")
        self.assertLessEqual(len(out), 2)

        original = sys.modules.get("race_scraper")
        sys.modules["race_scraper"] = types.SimpleNamespace(actual_odds_multiplier=lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            odds = betting.candidate_odds("wide", [1, 2], 4.5, {}, {"ok": True})
        finally:
            if original is None:
                sys.modules.pop("race_scraper", None)
            else:
                sys.modules["race_scraper"] = original
        self.assertEqual(odds, 4.5)


if __name__ == "__main__":
    unittest.main()
