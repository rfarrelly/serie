from typing import Dict, List, Optional

import pandas as pd
from utils.odds_helpers import get_no_vig_odds_multiway

from ..predictions.entities import Prediction
from ..shared.value_objects import Odds
from .entities import BettingOpportunity


class BettingAnalysisService:
    """Comprehensive betting analysis service with edge calculation."""

    def __init__(
        self, min_edge: float = 0.02, min_prob: float = 0.1, max_odds: float = 10.0
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds
        self.bet_types = ["Home", "Draw", "Away"]

    def analyze_predictions(
        self, predictions: List[Prediction], odds_data: dict
    ) -> List[BettingOpportunity]:
        """Analyze predictions for betting opportunities."""
        opportunities = []

        for prediction in predictions:
            match_key = f"{prediction.match.home_team.name}_{prediction.match.away_team.name}_{prediction.match.date}"

            if match_key not in odds_data:
                continue

            odds_row = odds_data[match_key]
            opportunity = self._analyze_single_prediction(prediction, odds_row)

            if opportunity and self._meets_criteria(opportunity):
                opportunities.append(opportunity)

        return opportunities

    def analyze_prediction_row(self, prediction_row: pd.Series) -> Optional[Dict]:
        """Analyze a single prediction row for betting opportunities (legacy compatibility)."""
        odds_cols = ["PSH", "PSD", "PSA"]
        if any(pd.isna(prediction_row.get(col)) for col in odds_cols):
            return None

        try:
            return self._calculate_betting_metrics(prediction_row)
        except Exception as e:
            print(f"Error in betting analysis: {e}")
            return None

    def find_betting_candidates(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Find betting candidates from predictions dataframe (legacy compatibility)."""
        if len(predictions_df) == 0:
            return pd.DataFrame()

        candidates = []
        for _, pred in predictions_df.iterrows():
            metrics = self.analyze_prediction_row(pred)
            if metrics:
                candidate_dict = pred.to_dict()
                candidate_dict.update(self._metrics_to_dict(metrics))
                candidates.append(candidate_dict)

        return pd.DataFrame(candidates)

    def _analyze_single_prediction(
        self, prediction: Prediction, odds_row: dict
    ) -> Optional[BettingOpportunity]:
        """Analyze a single prediction for betting opportunities."""
        try:
            # Extract odds
            sharp_odds = [odds_row.get("PSH"), odds_row.get("PSD"), odds_row.get("PSA")]
            soft_odds = [
                odds_row.get("B365H"),
                odds_row.get("B365D"),
                odds_row.get("B365A"),
            ]

            if any(x is None for x in sharp_odds + soft_odds):
                return None

            # Calculate no-vig probabilities
            no_vig_odds = get_no_vig_odds_multiway(sharp_odds)
            no_vig_probs = [1 / odd for odd in no_vig_odds]

            # Combine model and market probabilities (10% model, 90% market)
            model_probs = [
                float(prediction.probabilities.home),
                float(prediction.probabilities.draw),
                float(prediction.probabilities.away),
            ]

            weighted_probs = [
                model_probs[i] * 0.1 + no_vig_probs[i] * 0.9 for i in range(3)
            ]

            # Calculate edges and find best bet
            edges = [weighted_probs[i] * soft_odds[i] - 1 for i in range(3)]
            best_idx = edges.index(max(edges))

            if edges[best_idx] <= 0:
                return None

            outcomes = ["H", "D", "A"]

            market_odds = Odds(
                home=float(soft_odds[0]),
                draw=float(soft_odds[1]),
                away=float(soft_odds[2]),
            )

            return BettingOpportunity(
                prediction=prediction,
                recommended_outcome=outcomes[best_idx],
                edge=float(edges[best_idx]),
                fair_odds=float(1 / weighted_probs[best_idx]),
                market_odds=market_odds,
                expected_value=float(edges[best_idx]),
            )

        except Exception as e:
            print(f"Error analyzing prediction: {e}")
            return None

    def _calculate_betting_metrics(self, pred: pd.Series) -> Dict:
        """Calculate comprehensive betting metrics (legacy compatibility)."""
        # Extract odds
        sharp_odds = [pred["PSH"], pred["PSD"], pred["PSA"]]
        soft_odds = [pred["B365H"], pred["B365D"], pred["B365A"]]

        # Get model probabilities
        prob_sets = self._extract_model_probabilities(pred)

        # Calculate no-vig and combined probabilities
        no_vig_sharp_odds = get_no_vig_odds_multiway(sharp_odds)
        no_vig_sharp_probs = [1 / odd for odd in no_vig_sharp_odds]
        model_avg_probs = self._average_model_probabilities(prob_sets)
        weighted_probs = self._combine_probabilities(
            model_avg_probs, no_vig_sharp_probs
        )

        # Calculate edges
        edges = self._calculate_all_edges(weighted_probs, soft_odds)
        best_bet_idx = self._find_best_bet(edges["expected_values"])

        return {
            "bet_type": self.bet_types[best_bet_idx],
            "edge": edges["expected_values"][best_bet_idx],
            "model_prob": weighted_probs[best_bet_idx],
            "market_prob": no_vig_sharp_probs[best_bet_idx],
            "sharp_odds": sharp_odds[best_bet_idx],
            "soft_odds": soft_odds[best_bet_idx],
            "fair_odds": 1 / weighted_probs[best_bet_idx],
            "expected_values": edges["expected_values"],
            "kelly_edges": edges["kelly_edges"],
            "probability_edges": edges["probability_edges"],
        }

    def _extract_model_probabilities(self, pred: pd.Series) -> Dict[str, List[float]]:
        """Extract model probabilities from prediction row."""
        return {
            "poisson": [
                pred.get("Poisson_Prob_H", 0.33),
                pred.get("Poisson_Prob_D", 0.33),
                pred.get("Poisson_Prob_A", 0.34),
            ],
            "zip": [
                pred.get("ZIP_Prob_H", 0.33),
                pred.get("ZIP_Prob_D", 0.33),
                pred.get("ZIP_Prob_A", 0.34),
            ],
            "mov": [
                pred.get("MOV_Prob_H", 0.33),
                pred.get("MOV_Prob_D", 0.33),
                pred.get("MOV_Prob_A", 0.34),
            ],
        }

    def _average_model_probabilities(
        self, prob_sets: Dict[str, List[float]]
    ) -> List[float]:
        """Calculate average probabilities across models."""
        n_models = len(prob_sets)
        return [
            sum(prob_sets[model][i] for model in prob_sets) / n_models for i in range(3)
        ]

    def _combine_probabilities(
        self,
        model_probs: List[float],
        market_probs: List[float],
        model_weight: float = 0.1,
    ) -> List[float]:
        """Combine model and market probabilities."""
        market_weight = 1 - model_weight
        return [
            model_probs[i] * model_weight + market_probs[i] * market_weight
            for i in range(3)
        ]

    def _calculate_all_edges(
        self, weighted_probs: List[float], soft_odds: List[float]
    ) -> Dict[str, List[float]]:
        """Calculate all edge types."""
        probability_edges = [weighted_probs[i] - 1 / soft_odds[i] for i in range(3)]
        expected_values = [(weighted_probs[i] * soft_odds[i]) - 1 for i in range(3)]
        kelly_edges = [
            (
                (weighted_probs[i] * soft_odds[i] - 1) / (soft_odds[i] - 1)
                if soft_odds[i] > 1
                else 0
            )
            for i in range(3)
        ]

        return {
            "probability_edges": probability_edges,
            "expected_values": expected_values,
            "kelly_edges": kelly_edges,
        }

    def _find_best_bet(self, expected_values: List[float]) -> int:
        """Find index of best betting opportunity."""
        return expected_values.index(max(expected_values))

    def _metrics_to_dict(self, metrics: Dict) -> Dict:
        """Convert betting metrics to dictionary for DataFrame construction."""
        base_dict = {
            "Bet_Type": metrics["bet_type"],
            "Edge": metrics["edge"],
            "Model_Prob": metrics["model_prob"],
            "Market_Prob": metrics["market_prob"],
            "Sharp_Odds": metrics["sharp_odds"],
            "Soft_Odds": metrics["soft_odds"],
            "Fair_Odds_Selected": metrics["fair_odds"],
        }

        # Add detailed edge calculations
        for i, bet_type in enumerate(self.bet_types):
            suffix = bet_type[0]  # H, D, A
            base_dict.update(
                {
                    f"EV_{suffix}": metrics["expected_values"][i],
                    f"Prob_Edge_{suffix}": metrics["probability_edges"][i],
                    f"Kelly_{suffix}": metrics["kelly_edges"][i],
                }
            )

        return base_dict

    def _meets_criteria(self, opportunity: BettingOpportunity) -> bool:
        """Check if opportunity meets betting criteria."""
        edge = float(opportunity.edge)

        # Get the probability for the recommended outcome
        if opportunity.recommended_outcome == "H":
            model_prob = float(opportunity.prediction.probabilities.home)
        elif opportunity.recommended_outcome == "D":
            model_prob = float(opportunity.prediction.probabilities.draw)
        else:  # "A"
            model_prob = float(opportunity.prediction.probabilities.away)

        # Get the recommended odds
        if opportunity.recommended_outcome == "H":
            recommended_odds = float(opportunity.market_odds.home)
        elif opportunity.recommended_outcome == "D":
            recommended_odds = float(opportunity.market_odds.draw)
        else:  # "A"
            recommended_odds = float(opportunity.market_odds.away)

        return (
            edge >= self.min_edge
            and model_prob >= self.min_prob
            and recommended_odds <= self.max_odds
        )
