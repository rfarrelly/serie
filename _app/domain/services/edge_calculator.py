# domain/services/edge_calculator.py
"""
Betting Edge Calculation Service.

Extracted from analysis/betting.py - preserves your sophisticated
edge calculation logic while providing clean domain interface.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from domain.entities.betting import (
    BettingOpportunity,
    MarketOdds,
    ModelProbabilities,
    Odds,
    Probability,
)
from shared.exceptions import InvalidOddsException, InvalidProbabilityException
from shared.types.common_types import BetType, BookmakerType


@dataclass(frozen=True)
class EdgeCalculationResult:
    """Complete edge calculation result for a match"""

    home_edge: Decimal
    draw_edge: Decimal
    away_edge: Decimal
    best_bet_type: BetType
    best_edge: Decimal
    expected_values: Dict[BetType, Decimal]
    kelly_fractions: Dict[BetType, Decimal]
    probability_edges: Dict[BetType, Decimal]


@dataclass(frozen=True)
class NoVigResult:
    """Result of no-vig probability calculation"""

    original_odds: MarketOdds
    no_vig_odds: MarketOdds
    overround: Decimal
    no_vig_probabilities: Dict[BetType, Probability]


class EdgeCalculator:
    """
    Domain service for calculating betting edges and expected values.

    Implements your sophisticated edge calculation methodology:
    1. Remove bookmaker margins (no-vig)
    2. Combine model predictions with market wisdom
    3. Calculate multiple edge types (EV, probability, Kelly)
    """

    def __init__(self, model_weight: Decimal = Decimal("0.1")):
        """
        Initialize edge calculator.

        Args:
            model_weight: Weight given to model predictions vs market (0.1 = 10% model, 90% market)
        """
        self.model_weight = model_weight
        self.market_weight = Decimal("1.0") - model_weight

    def calculate_comprehensive_edge(
        self,
        model_probabilities: ModelProbabilities,
        sharp_odds: MarketOdds,
        soft_odds: MarketOdds,
    ) -> EdgeCalculationResult:
        """
        Calculate comprehensive edge analysis using your methodology.

        Args:
            model_probabilities: Probabilities from your prediction model
            sharp_odds: Sharp (Pinnacle) odds for fair probability estimation
            soft_odds: Soft (recreational) odds for betting

        Returns:
            Complete edge calculation with all metrics
        """
        # Step 1: Calculate no-vig probabilities from sharp odds
        no_vig_result = self.remove_bookmaker_margin(sharp_odds)

        # Step 2: Combine model predictions with market wisdom
        weighted_probabilities = self._combine_probabilities(
            model_probabilities, no_vig_result.no_vig_probabilities
        )

        # Step 3: Calculate all edge types
        edges = self._calculate_all_edge_types(weighted_probabilities, soft_odds)

        # Step 4: Find best betting opportunity
        best_bet_type, best_edge = self._find_best_opportunity(edges["expected_values"])

        return EdgeCalculationResult(
            home_edge=edges["expected_values"][BetType.HOME],
            draw_edge=edges["expected_values"][BetType.DRAW],
            away_edge=edges["expected_values"][BetType.AWAY],
            best_bet_type=best_bet_type,
            best_edge=best_edge,
            expected_values=edges["expected_values"],
            kelly_fractions=edges["kelly_fractions"],
            probability_edges=edges["probability_edges"],
        )

    def remove_bookmaker_margin(self, market_odds: MarketOdds) -> NoVigResult:
        """
        Remove bookmaker margin to get fair probabilities.

        Uses the iterative method from your odds_helpers.py
        """
        # Implementation of your get_no_vig_odds_multiway algorithm
        odds_list = [
            float(market_odds.home.decimal),
            float(market_odds.draw.decimal),
            float(market_odds.away.decimal),
        ]

        no_vig_odds_list = self._iterative_no_vig_calculation(odds_list)

        no_vig_odds = MarketOdds(
            home=Odds(Decimal(str(no_vig_odds_list[0]))),
            draw=Odds(Decimal(str(no_vig_odds_list[1]))),
            away=Odds(Decimal(str(no_vig_odds_list[2]))),
            bookmaker=market_odds.bookmaker,
        )

        no_vig_probs = {
            BetType.HOME: Probability(Decimal("1.0") / no_vig_odds.home.decimal),
            BetType.DRAW: Probability(Decimal("1.0") / no_vig_odds.draw.decimal),
            BetType.AWAY: Probability(Decimal("1.0") / no_vig_odds.away.decimal),
        }

        return NoVigResult(
            original_odds=market_odds,
            no_vig_odds=no_vig_odds,
            overround=market_odds.overround,
            no_vig_probabilities=no_vig_probs,
        )

    def calculate_expected_value(
        self, true_probability: Probability, betting_odds: Odds
    ) -> Decimal:
        """
        Calculate expected value of a bet.

        EV = (True_Probability × Betting_Odds) - 1

        Positive EV indicates profitable betting opportunity.
        """
        expected_return = true_probability.value * betting_odds.decimal
        return expected_return - Decimal("1.0")

    def calculate_kelly_fraction(
        self, true_probability: Probability, betting_odds: Odds
    ) -> Decimal:
        """
        Calculate Kelly Criterion stake fraction.

        Kelly% = (True_Prob × Odds - 1) / (Odds - 1)
        """
        if betting_odds.decimal <= 1:
            return Decimal("0")

        numerator = (true_probability.value * betting_odds.decimal) - Decimal("1.0")
        denominator = betting_odds.decimal - Decimal("1.0")

        kelly_fraction = numerator / denominator

        # Cap at reasonable maximum (25%)
        return min(kelly_fraction, Decimal("0.25"))

    def calculate_probability_edge(
        self, model_probability: Probability, market_probability: Probability
    ) -> Decimal:
        """
        Calculate probability edge (model prob - market prob).

        Positive value indicates model sees higher probability than market.
        """
        return model_probability.value - market_probability.value

    def _iterative_no_vig_calculation(self, odds: List[float]) -> List[float]:
        """
        Iterative no-vig calculation - your exact algorithm from odds_helpers.py
        """
        import math

        c, target_overround, accuracy, current_error = 1, 0, 3, 1000
        max_error = (10 ** (-accuracy)) / 2

        fair_odds = []
        while current_error > max_error:
            f = -1 - target_overround
            for o in odds:
                f += (1 / o) ** c

            f_dash = 0
            for o in odds:
                f_dash += ((1 / o) ** c) * (-math.log(o))

            h = -f / f_dash
            c = c + h

            t = 0
            for o in odds:
                t += (1 / o) ** c
            current_error = abs(t - 1 - target_overround)

            fair_odds = []
            for o in odds:
                fair_odds.append(round(o**c, 3))

        return fair_odds

    def _combine_probabilities(
        self, model_probs: ModelProbabilities, market_probs: Dict[BetType, Probability]
    ) -> Dict[BetType, Probability]:
        """
        Combine model and market probabilities using weighted average.

        Your methodology: 10% model weight, 90% market weight
        """
        combined = {}

        model_values = {
            BetType.HOME: model_probs.home_win.value,
            BetType.DRAW: model_probs.draw.value,
            BetType.AWAY: model_probs.away_win.value,
        }

        for bet_type in [BetType.HOME, BetType.DRAW, BetType.AWAY]:
            combined_value = (
                model_values[bet_type] * self.model_weight
                + market_probs[bet_type].value * self.market_weight
            )
            combined[bet_type] = Probability(combined_value)

        return combined

    def _calculate_all_edge_types(
        self, weighted_probs: Dict[BetType, Probability], soft_odds: MarketOdds
    ) -> Dict[str, Dict[BetType, Decimal]]:
        """Calculate all types of edges for each outcome"""

        odds_map = {
            BetType.HOME: soft_odds.home,
            BetType.DRAW: soft_odds.draw,
            BetType.AWAY: soft_odds.away,
        }

        expected_values = {}
        kelly_fractions = {}
        probability_edges = {}

        for bet_type in [BetType.HOME, BetType.DRAW, BetType.AWAY]:
            prob = weighted_probs[bet_type]
            odds = odds_map[bet_type]

            # Expected Value
            expected_values[bet_type] = self.calculate_expected_value(prob, odds)

            # Kelly Fraction
            kelly_fractions[bet_type] = self.calculate_kelly_fraction(prob, odds)

            # Probability Edge (vs implied odds probability)
            market_implied = Probability(Decimal("1.0") / odds.decimal)
            probability_edges[bet_type] = self.calculate_probability_edge(
                prob, market_implied
            )

        return {
            "expected_values": expected_values,
            "kelly_fractions": kelly_fractions,
            "probability_edges": probability_edges,
        }

    def _find_best_opportunity(
        self, expected_values: Dict[BetType, Decimal]
    ) -> Tuple[BetType, Decimal]:
        """Find the best betting opportunity by expected value"""
        best_bet_type = max(expected_values.keys(), key=lambda k: expected_values[k])
        best_edge = expected_values[best_bet_type]

        return best_bet_type, best_edge

    def create_betting_opportunity(
        self,
        edge_result: EdgeCalculationResult,
        sharp_odds: MarketOdds,
        soft_odds: MarketOdds,
        weighted_probabilities: Dict[BetType, Probability],
    ) -> BettingOpportunity:
        """
        Create a BettingOpportunity domain object from calculation results.
        """
        best_bet_type = edge_result.best_bet_type

        # Get the relevant odds and probabilities for the best bet
        soft_odds_map = {
            BetType.HOME: soft_odds.home,
            BetType.DRAW: soft_odds.draw,
            BetType.AWAY: soft_odds.away,
        }

        sharp_odds_map = {
            BetType.HOME: sharp_odds.home,
            BetType.DRAW: sharp_odds.draw,
            BetType.AWAY: sharp_odds.away,
        }

        model_prob = weighted_probabilities[best_bet_type]
        market_prob = Probability(
            Decimal("1.0") / sharp_odds_map[best_bet_type].decimal
        )
        fair_odds = Odds(Decimal("1.0") / model_prob.value)

        return BettingOpportunity(
            bet_type=best_bet_type,
            edge=edge_result.best_edge,
            model_probability=model_prob,
            market_probability=market_prob,
            recommended_odds=soft_odds_map[best_bet_type],
            fair_odds=fair_odds,
            expected_value=edge_result.expected_values[best_bet_type],
            kelly_fraction=edge_result.kelly_fractions[best_bet_type],
        )
