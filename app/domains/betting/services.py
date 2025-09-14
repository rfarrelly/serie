from typing import List, Optional

from utils.odds_helpers import get_no_vig_odds_multiway

from ..predictions.entities import Prediction
from ..shared.value_objects import Odds
from .entities import BettingOpportunity


class BettingAnalysisService:
    def __init__(
        self, min_edge: float = 0.02, min_prob: float = 0.1, max_odds: float = 10.0
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds

    def analyze_predictions(
        self, predictions: List[Prediction], odds_data: dict
    ) -> List[BettingOpportunity]:
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

    def _analyze_single_prediction(
        self, prediction: Prediction, odds_row: dict
    ) -> Optional[BettingOpportunity]:
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
            # Convert all to float to avoid Decimal/float mixing
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

            # Convert to Decimal for value objects, but use float for calculations
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

    def _meets_criteria(self, opportunity: BettingOpportunity) -> bool:
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
