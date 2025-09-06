from typing import Dict, List

import numpy as np

from ..entities.betting import BettingOpportunity
from ..entities.prediction import Prediction
from ..value_objects.odds import Odds, OddsSet
from ..value_objects.probability import Probability


class EnhancedBettingService:
    """Enhanced betting service with all original calculations"""

    def __init__(
        self, min_edge: float = 0.02, min_prob: float = 0.1, max_odds: float = 10.0
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds

    def analyze_predictions_fixed(
        self, predictions_dict: Dict[str, Prediction], odds: OddsSet
    ) -> Dict[str, any]:
        """Comprehensive betting analysis using all prediction methods"""

        # Average model probabilities
        methods = ["poisson", "zip", "mov"]
        avg_probs = [0.0, 0.0, 0.0]  # H, D, A

        valid_methods = 0
        for method in methods:
            if method in predictions_dict:
                pred = predictions_dict[method]
                avg_probs[0] += pred.home_win_prob.value
                avg_probs[1] += pred.draw_prob.value
                avg_probs[2] += pred.away_win_prob.value
                valid_methods += 1

        # Average across valid methods
        if valid_methods > 0:
            avg_probs = [p / valid_methods for p in avg_probs]
        else:
            avg_probs = [0.33, 0.33, 0.34]  # Default equal probabilities

        # Get no-vig probabilities
        fair_odds = odds.remove_vig()
        market_probs = [
            fair_odds.home.implied_probability,
            fair_odds.draw.implied_probability,
            fair_odds.away.implied_probability,
        ]

        # Combine model and market (10% model, 90% market)
        combined_probs = [0.1 * avg_probs[i] + 0.9 * market_probs[i] for i in range(3)]

        # Calculate all edge types
        soft_odds_values = [odds.home.value, odds.draw.value, odds.away.value]
        fair_odds_values = [
            fair_odds.home.value,
            fair_odds.draw.value,
            fair_odds.away.value,
        ]

        # Expected value edges
        expected_values = [
            combined_probs[i] * soft_odds_values[i] - 1 for i in range(3)
        ]

        # Probability edges
        probability_edges = [
            combined_probs[i] - 1 / soft_odds_values[i] for i in range(3)
        ]

        # Kelly edges
        kelly_edges = []
        for i in range(3):
            if soft_odds_values[i] > 1:
                kelly = (combined_probs[i] * soft_odds_values[i] - 1) / (
                    soft_odds_values[i] - 1
                )
                kelly_edges.append(kelly)
            else:
                kelly_edges.append(0.0)

        # Find best bet
        best_bet_idx = np.argmax(expected_values)
        bet_types = ["Home", "Draw", "Away"]

        result = {
            "bet_type": bet_types[best_bet_idx],
            "edge": expected_values[best_bet_idx],
            "model_prob": combined_probs[best_bet_idx],
            "market_prob": market_probs[best_bet_idx],
            "sharp_odds": fair_odds_values[best_bet_idx],
            "soft_odds": soft_odds_values[best_bet_idx],
            "fair_odds": 1 / combined_probs[best_bet_idx],
            "expected_values": expected_values,
            "probability_edges": probability_edges,
            "kelly_edges": kelly_edges,
            "all_predictions": predictions_dict,
            "avg_model_probs": avg_probs,
            "combined_probs": combined_probs,
        }

        return result

    def find_betting_opportunities_enhanced(
        self, betting_analysis: Dict[str, any]
    ) -> List[BettingOpportunity]:
        """Find betting opportunities from enhanced analysis"""
        opportunities = []

        edge = betting_analysis["edge"]
        model_prob = betting_analysis["model_prob"]
        soft_odds = betting_analysis["soft_odds"]

        if (
            edge >= self.min_edge
            and model_prob >= self.min_prob
            and soft_odds <= self.max_odds
        ):
            opportunity = BettingOpportunity(
                match_id=list(betting_analysis["all_predictions"].values())[0].match_id,
                bet_type=betting_analysis["bet_type"],
                stake=1.0,
                odds=Odds(soft_odds),
                model_probability=Probability(model_prob),
                market_probability=Probability(betting_analysis["market_prob"]),
                edge=edge,
                expected_value=edge * 1.0,
            )
            opportunities.append(opportunity)

        return opportunities
