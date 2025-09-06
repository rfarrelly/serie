from typing import List

from ..entities.betting import BettingOpportunity
from ..entities.prediction import Prediction
from ..value_objects.odds import OddsSet
from ..value_objects.probability import Probability


class BettingService:
    def __init__(
        self, min_edge: float = 0.02, min_prob: float = 0.1, max_odds: float = 10.0
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds

    def find_betting_opportunities(
        self, prediction: Prediction, odds: OddsSet
    ) -> List[BettingOpportunity]:
        """Find profitable betting opportunities for a match"""
        opportunities = []

        # Remove bookmaker margin
        fair_odds = odds.remove_vig()

        # Combine model predictions with market probabilities
        model_probs = [
            prediction.home_win_prob.value,
            prediction.draw_prob.value,
            prediction.away_win_prob.value,
        ]

        market_probs = [
            fair_odds.home.implied_probability,
            fair_odds.draw.implied_probability,
            fair_odds.away.implied_probability,
        ]

        # Weight combination (10% model, 90% market)
        weighted_probs = [
            0.1 * model_probs[i] + 0.9 * market_probs[i] for i in range(3)
        ]

        # Check each outcome for betting value
        bet_types = ["Home", "Draw", "Away"]
        soft_odds = [odds.home, odds.draw, odds.away]

        for i, (bet_type, soft_odd) in enumerate(zip(bet_types, soft_odds)):
            edge = weighted_probs[i] * soft_odd.value - 1

            if (
                edge >= self.min_edge
                and weighted_probs[i] >= self.min_prob
                and soft_odd.value <= self.max_odds
            ):
                opportunity = BettingOpportunity(
                    match_id=prediction.match_id,
                    bet_type=bet_type,
                    stake=1.0,  # Default stake
                    odds=soft_odd,
                    model_probability=Probability(weighted_probs[i]),
                    market_probability=Probability(market_probs[i]),
                    edge=edge,
                    expected_value=edge * 1.0,  # stake
                )
                opportunities.append(opportunity)

        return opportunities

    def calculate_kelly_stake(
        self, opportunity: BettingOpportunity, bankroll: float
    ) -> float:
        """Calculate optimal stake using Kelly criterion"""
        if opportunity.odds.value <= 1:
            return 0

        kelly_fraction = opportunity.edge / (opportunity.odds.value - 1)
        return min(bankroll * kelly_fraction, bankroll * 0.05)  # Max 5% of bankroll
