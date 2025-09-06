from typing import List

import numpy as np
from scipy.stats import norm, poisson

from ..entities.match import Match
from ..entities.prediction import Prediction
from ..entities.team import TeamRatings
from ..value_objects.probability import Probability


class PredictionService:
    def __init__(self, max_goals: int = 15):
        self.max_goals = max_goals

    def predict_match(
        self,
        match: Match,
        home_ratings: TeamRatings,
        away_ratings: TeamRatings,
        home_advantage: float = 0.2,
        method: str = "poisson",
    ) -> Prediction:
        """Generate prediction for a single match"""

        # Calculate expected goals
        home_strength = (
            home_advantage + home_ratings.attack_rating - away_ratings.defense_rating
        )
        away_strength = away_ratings.attack_rating - home_ratings.defense_rating

        lambda_home = max(0.1, 1.5 + 0.5 * home_strength)
        lambda_away = max(0.1, 1.2 + 0.5 * away_strength)

        # Calculate probabilities based on method
        if method == "poisson":
            probs = self._poisson_probabilities(lambda_home, lambda_away)
        elif method == "zip":
            probs = self._zip_probabilities(lambda_home, lambda_away)
        elif method == "mov":
            probs = self._mov_probabilities(lambda_home, lambda_away)
        else:
            raise ValueError(f"Unknown prediction method: {method}")

        return Prediction(
            match_id=f"{match.home_team}_{match.away_team}_{match.date.isoformat()}",
            home_team=match.home_team,
            away_team=match.away_team,
            home_win_prob=Probability(probs[0]),
            draw_prob=Probability(probs[1]),
            away_win_prob=Probability(probs[2]),
            expected_home_goals=lambda_home,
            expected_away_goals=lambda_away,
            model_type=method,
        )

    def _poisson_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> List[float]:
        """Calculate match outcome probabilities using Poisson distribution"""
        prob_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))

        for h_goals in range(self.max_goals + 1):
            for a_goals in range(self.max_goals + 1):
                prob_matrix[h_goals, a_goals] = poisson.pmf(
                    h_goals, lambda_home
                ) * poisson.pmf(a_goals, lambda_away)

        # Normalize
        prob_matrix /= prob_matrix.sum()

        # Calculate outcome probabilities
        prob_home = np.tril(prob_matrix, -1).sum()  # Home wins
        prob_draw = np.trace(prob_matrix)  # Draws
        prob_away = np.triu(prob_matrix, 1).sum()  # Away wins

        return [prob_home, prob_draw, prob_away]

    def _zip_probabilities(self, lambda_home: float, lambda_away: float) -> List[float]:
        """Zero-inflated Poisson with empirical adjustments"""
        base_probs = self._poisson_probabilities(lambda_home, lambda_away)

        # Apply zero-inflation adjustment (simplified)
        adjustment_factors = [
            0.95,
            1.1,
            0.95,
        ]  # Slightly reduce home/away, increase draw
        adjusted_probs = [p * f for p, f in zip(base_probs, adjustment_factors)]

        # Renormalize
        total = sum(adjusted_probs)
        return [p / total for p in adjusted_probs]

    def _mov_probabilities(self, lambda_home: float, lambda_away: float) -> List[float]:
        """Margin of victory based probabilities"""
        mov_prediction = lambda_home - lambda_away
        mov_std_error = np.sqrt(lambda_home + lambda_away)

        # Use normal distribution for outcomes
        prob_home = 1 - norm.cdf(0.5, mov_prediction, mov_std_error)
        prob_away = norm.cdf(-0.5, mov_prediction, mov_std_error)
        prob_draw = max(0.05, 1 - prob_home - prob_away)  # Minimum draw probability

        # Renormalize
        total = prob_home + prob_draw + prob_away
        return [prob_home / total, prob_draw / total, prob_away / total]
