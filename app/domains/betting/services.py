from typing import Dict, List, Optional

import pandas as pd
from utils.odds_helpers import get_no_vig_odds_multiway

from ..predictions.entities import Prediction
from ..shared.value_objects import Odds
from .entities import BettingOpportunity


class BettingAnalysisService:
    """Enhanced betting analysis service with proper edge calculations."""

    def __init__(
        self,
        min_edge: float = 0.02,
        min_prob: float = 0.05,
        max_odds: float = 15.0,
        model_weight: float = 0.15,
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds
        self.model_weight = model_weight
        self.market_weight = 1 - model_weight
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

            if opportunity and self._meets_betting_criteria(opportunity):
                opportunities.append(opportunity)

        return opportunities

    def analyze_prediction_row(self, prediction_row: pd.Series) -> Optional[Dict]:
        """Analyze a single prediction row for betting opportunities."""
        if not self._has_required_odds(prediction_row):
            return None

        try:
            return self._calculate_comprehensive_betting_metrics(prediction_row)
        except Exception as e:
            print(f"Error in betting analysis: {e}")
            return None

    def _has_required_odds(self, pred: pd.Series) -> bool:
        """Check if prediction has required odds data."""
        required_odds = ["PSH", "PSD", "PSA", "B365H", "B365D", "B365A"]
        return all(col in pred.index and pd.notna(pred[col]) for col in required_odds)

    def _calculate_comprehensive_betting_metrics(self, pred: pd.Series) -> Dict:
        """Calculate comprehensive betting metrics with proper edge calculations."""
        # Extract odds
        sharp_odds = [pred["PSH"], pred["PSD"], pred["PSA"]]
        soft_odds = [pred["B365H"], pred["B365D"], pred["B365A"]]

        # Validate odds
        if any(x <= 1.0 for x in sharp_odds + soft_odds):
            return None

        # Calculate no-vig probabilities (true market assessment)
        try:
            no_vig_odds = get_no_vig_odds_multiway(sharp_odds)
            market_probs = [1 / odd for odd in no_vig_odds]
        except:
            # Fallback calculation
            raw_probs = [1 / odd for odd in sharp_odds]
            total = sum(raw_probs)
            market_probs = [p / total for p in raw_probs]

        # Get model probabilities
        model_probs = self._extract_model_probabilities(pred)

        # Create blended probabilities (model + market)
        blended_probs = self._blend_probabilities(model_probs, market_probs)

        # Calculate edges for each outcome
        edges = self._calculate_true_edges(blended_probs, soft_odds, market_probs)

        # Find best betting opportunity
        best_bet_idx = self._find_optimal_bet(edges)

        if edges[best_bet_idx] <= 0:
            return None

        return {
            "bet_type": self.bet_types[best_bet_idx],
            "edge": edges[best_bet_idx],
            "model_prob": blended_probs[best_bet_idx],
            "market_prob": market_probs[best_bet_idx],
            "sharp_odds": sharp_odds[best_bet_idx],
            "soft_odds": soft_odds[best_bet_idx],
            "fair_odds": 1 / blended_probs[best_bet_idx],
            "expected_value": edges[best_bet_idx] * soft_odds[best_bet_idx],
            "kelly_fraction": self._calculate_kelly_fraction(
                blended_probs[best_bet_idx], soft_odds[best_bet_idx]
            ),
            "confidence": self._calculate_bet_confidence(
                model_probs[best_bet_idx],
                market_probs[best_bet_idx],
                edges[best_bet_idx],
            ),
        }

    def _extract_model_probabilities(self, pred: pd.Series) -> List[float]:
        """Extract and validate model probabilities."""
        # Try different model probability columns
        prob_sources = [
            ["ZIP_Prob_H", "ZIP_Prob_D", "ZIP_Prob_A"],
            ["ZSD_Prob_H", "ZSD_Prob_D", "ZSD_Prob_A"],
            ["Poisson_Prob_H", "Poisson_Prob_D", "Poisson_Prob_A"],
        ]

        for cols in prob_sources:
            if all(col in pred.index for col in cols):
                probs = [pred[col] for col in cols]
                # Validate probabilities
                if all(0 <= p <= 1 for p in probs) and 0.95 <= sum(probs) <= 1.05:
                    # Normalize if needed
                    total = sum(probs)
                    return [p / total for p in probs]

        # Fallback to uniform distribution
        return [1 / 3, 1 / 3, 1 / 3]

    def _blend_probabilities(
        self, model_probs: List[float], market_probs: List[float]
    ) -> List[float]:
        """Blend model and market probabilities with confidence weighting."""
        blended = []
        for i in range(3):
            # Calculate divergence between model and market
            divergence = abs(model_probs[i] - market_probs[i])

            # Reduce model weight when there's high divergence (be more conservative)
            adjusted_model_weight = self.model_weight * (1 - divergence)
            adjusted_market_weight = 1 - adjusted_model_weight

            prob = (
                model_probs[i] * adjusted_model_weight
                + market_probs[i] * adjusted_market_weight
            )
            blended.append(prob)

        # Normalize
        total = sum(blended)
        return [p / total for p in blended]

    def _calculate_true_edges(
        self,
        blended_probs: List[float],
        soft_odds: List[float],
        market_probs: List[float],
    ) -> List[float]:
        """Calculate true betting edges accounting for market efficiency."""
        edges = []

        for i in range(3):
            # Basic expected value calculation
            ev = blended_probs[i] * soft_odds[i] - 1

            # Apply confidence adjustment based on market divergence
            market_divergence = abs(blended_probs[i] - market_probs[i])
            confidence_penalty = market_divergence * 0.5  # Penalize high divergence

            adjusted_edge = ev - confidence_penalty
            edges.append(adjusted_edge)

        return edges

    def _find_optimal_bet(self, edges: List[float]) -> int:
        """Find the optimal betting opportunity considering minimum thresholds."""
        valid_bets = []

        for i, edge in enumerate(edges):
            if edge >= self.min_edge:
                valid_bets.append((i, edge))

        if not valid_bets:
            return edges.index(max(edges))  # Return best even if below threshold

        # Return the bet with highest edge
        return max(valid_bets, key=lambda x: x[1])[0]

    def _calculate_kelly_fraction(self, prob: float, odds: float) -> float:
        """Calculate Kelly criterion fraction for optimal bet sizing."""
        if odds <= 1.0:
            return 0.0

        # Kelly formula: (bp - q) / b
        # where b = odds-1, p = probability, q = 1-p
        b = odds - 1
        kelly = (b * prob - (1 - prob)) / b

        # Cap Kelly at reasonable levels (never bet more than 25% of bankroll)
        return max(0, min(kelly, 0.25))

    def _calculate_bet_confidence(
        self, model_prob: float, market_prob: float, edge: float
    ) -> float:
        """Calculate confidence score for the betting opportunity."""
        # Base confidence on edge size
        edge_confidence = min(edge * 10, 1.0)  # Scale edge to 0-1

        # Adjust for model-market agreement
        agreement = 1 - abs(model_prob - market_prob)

        # Combine metrics
        confidence = edge_confidence * 0.7 + agreement * 0.3
        return max(0, min(confidence, 1.0))

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

            if any(x is None or x <= 1.0 for x in sharp_odds + soft_odds):
                return None

            # Calculate market probabilities
            no_vig_odds = get_no_vig_odds_multiway(sharp_odds)
            market_probs = [1 / odd for odd in no_vig_odds]

            # Get model probabilities
            model_probs = [
                float(prediction.probabilities.home),
                float(prediction.probabilities.draw),
                float(prediction.probabilities.away),
            ]

            # Blend probabilities
            blended_probs = self._blend_probabilities(model_probs, market_probs)

            # Calculate edges
            edges = self._calculate_true_edges(blended_probs, soft_odds, market_probs)
            best_idx = self._find_optimal_bet(edges)

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
                fair_odds=float(1 / blended_probs[best_idx]),
                market_odds=market_odds,
                expected_value=float(edges[best_idx] * soft_odds[best_idx]),
                kelly_fraction=self._calculate_kelly_fraction(
                    blended_probs[best_idx], soft_odds[best_idx]
                ),
            )

        except Exception as e:
            print(f"Error analyzing prediction: {e}")
            return None

    def _meets_betting_criteria(self, opportunity: BettingOpportunity) -> bool:
        """Check if opportunity meets betting criteria."""
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
            float(opportunity.edge) >= self.min_edge
            and model_prob >= self.min_prob
            and recommended_odds <= self.max_odds
            and getattr(opportunity, "kelly_fraction", 0) > 0
        )
