from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from ..entities.match import Match
from ..entities.ppi import PPIData
from ..entities.prediction import Prediction
from ..entities.team import TeamRatings
from ..value_objects.probability import Probability
from .ppi_service import PPIService


class EnhancedPredictionService:
    """Enhanced prediction service with all original methods"""

    def __init__(self, max_goals: int = 15):
        self.max_goals = max_goals
        self.ppi_service = PPIService()
        self.adjustment_matrix = None
        self.mov_model_params = None

    def fit_adjustment_matrix(self, historical_matches: List[Match]):
        """Fit ZIP adjustment matrix from historical data"""
        recent_matches = [m for m in historical_matches if m.is_played][
            -200:
        ]  # Last 200 matches

        if len(recent_matches) < 50:
            self.adjustment_matrix = np.ones((self.max_goals + 1, self.max_goals + 1))
            return

        # Calculate observed score distribution
        home_goals = [m.home_goals for m in recent_matches]
        away_goals = [m.away_goals for m in recent_matches]

        home_dist = (
            pd.Series(home_goals)
            .value_counts(normalize=True)
            .reindex(range(self.max_goals + 1), fill_value=0)
        )
        away_dist = (
            pd.Series(away_goals)
            .value_counts(normalize=True)
            .reindex(range(self.max_goals + 1), fill_value=0)
        )

        observed_matrix = np.outer(home_dist.values, away_dist.values)

        # Calculate expected distribution
        avg_home = np.mean(home_goals)
        avg_away = np.mean(away_goals)
        expected_matrix = self._poisson_matrix(avg_home, avg_away)

        # Calculate adjustment factors
        self.adjustment_matrix = np.where(
            expected_matrix > 1e-6, observed_matrix / expected_matrix, 1.0
        )
        self.adjustment_matrix = np.clip(self.adjustment_matrix, 0.1, 5.0)

    def fit_mov_model(
        self, historical_matches: List[Match], team_ratings: Dict[str, TeamRatings]
    ):
        """Fit margin of victory model"""
        played_matches = [m for m in historical_matches if m.is_played]

        if len(played_matches) < 20:
            self.mov_model_params = [0.0, 1.0, 1.0]  # intercept, slope, std_error
            return

        # Calculate predicted MOV from team ratings
        predicted_movs = []
        actual_movs = []

        for match in played_matches:
            home_rating = team_ratings.get(match.home_team)
            away_rating = team_ratings.get(match.away_team)

            if home_rating and away_rating:
                predicted_mov = (
                    home_rating.attack_rating - away_rating.defense_rating
                ) - (away_rating.attack_rating - home_rating.defense_rating)
                actual_mov = match.home_goals - match.away_goals

                predicted_movs.append(predicted_mov)
                actual_movs.append(actual_mov)

        if len(predicted_movs) > 10:
            # Simple linear regression
            X = np.array(predicted_movs)
            y = np.array(actual_movs)

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            # Solve normal equations
            try:
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                residuals = y - X_with_intercept @ coeffs
                std_error = np.std(residuals)

                self.mov_model_params = [coeffs[0], coeffs[1], std_error]
            except:
                self.mov_model_params = [0.0, 1.0, 1.0]
        else:
            self.mov_model_params = [0.0, 1.0, 1.0]

    def predict_match_all_methods(
        self,
        match: Match,
        home_ratings: TeamRatings,
        away_ratings: TeamRatings,
        historical_matches: List[Match],
        home_advantage: float = 0.2,
        away_adjustment: float = -0.1,
    ) -> Dict[str, Prediction]:
        """Generate predictions using all methods (Poisson, ZIP, MOV)"""

        # Calculate expected goals
        home_strength = (
            home_advantage + home_ratings.attack_rating - away_ratings.defense_rating
        )
        away_strength = (
            away_adjustment + away_ratings.attack_rating - home_ratings.defense_rating
        )

        lambda_home = max(0.1, 1.5 + 0.5 * home_strength)
        lambda_away = max(0.1, 1.2 + 0.5 * away_strength)

        predictions = {}

        # Poisson prediction
        poisson_probs = self._poisson_probabilities(lambda_home, lambda_away)
        predictions["poisson"] = Prediction(
            match_id=f"{match.home_team}_{match.away_team}_{match.date.isoformat()}",
            home_team=match.home_team,
            away_team=match.away_team,
            home_win_prob=Probability(poisson_probs[0]),
            draw_prob=Probability(poisson_probs[1]),
            away_win_prob=Probability(poisson_probs[2]),
            expected_home_goals=lambda_home,
            expected_away_goals=lambda_away,
            model_type="poisson",
        )

        # ZIP prediction
        if self.adjustment_matrix is not None:
            zip_probs = self._zip_probabilities(lambda_home, lambda_away)
        else:
            zip_probs = poisson_probs

        predictions["zip"] = Prediction(
            match_id=f"{match.home_team}_{match.away_team}_{match.date.isoformat()}",
            home_team=match.home_team,
            away_team=match.away_team,
            home_win_prob=Probability(zip_probs[0]),
            draw_prob=Probability(zip_probs[1]),
            away_win_prob=Probability(zip_probs[2]),
            expected_home_goals=lambda_home,
            expected_away_goals=lambda_away,
            model_type="zip",
        )

        # MOV prediction
        mov_probs = self._mov_probabilities(lambda_home, lambda_away)
        predictions["mov"] = Prediction(
            match_id=f"{match.home_team}_{match.away_team}_{match.date.isoformat()}",
            home_team=match.home_team,
            away_team=match.away_team,
            home_win_prob=Probability(mov_probs[0]),
            draw_prob=Probability(mov_probs[1]),
            away_win_prob=Probability(mov_probs[2]),
            expected_home_goals=lambda_home,
            expected_away_goals=lambda_away,
            model_type="mov",
        )

        return predictions

    def _poisson_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Calculate Poisson probability matrix"""
        prob_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))

        for h_goals in range(self.max_goals + 1):
            for a_goals in range(self.max_goals + 1):
                prob_matrix[h_goals, a_goals] = poisson.pmf(
                    h_goals, lambda_home
                ) * poisson.pmf(a_goals, lambda_away)

        return prob_matrix / prob_matrix.sum()

    def _poisson_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> List[float]:
        """Calculate Poisson outcome probabilities"""
        prob_matrix = self._poisson_matrix(lambda_home, lambda_away)

        # Calculate outcome probabilities
        prob_home = np.tril(prob_matrix, -1).sum()  # Home wins
        prob_draw = np.trace(prob_matrix)  # Draws
        prob_away = np.triu(prob_matrix, 1).sum()  # Away wins

        return [prob_home, prob_draw, prob_away]

    def _zip_probabilities(self, lambda_home: float, lambda_away: float) -> List[float]:
        """Zero-inflated Poisson probabilities"""
        base_matrix = self._poisson_matrix(lambda_home, lambda_away)
        adjusted_matrix = base_matrix * self.adjustment_matrix
        adjusted_matrix = adjusted_matrix / adjusted_matrix.sum()

        # Calculate outcome probabilities
        prob_home = np.tril(adjusted_matrix, -1).sum()
        prob_draw = np.trace(adjusted_matrix)
        prob_away = np.triu(adjusted_matrix, 1).sum()

        return [prob_home, prob_draw, prob_away]

    def _mov_probabilities(self, lambda_home: float, lambda_away: float) -> List[float]:
        """Margin of victory based probabilities"""
        raw_mov = lambda_home - lambda_away

        if self.mov_model_params:
            intercept, slope, std_error = self.mov_model_params
            mov_prediction = intercept + slope * raw_mov
            mov_std_error = std_error
        else:
            mov_prediction = raw_mov
            mov_std_error = np.sqrt(lambda_home + lambda_away)

        # Use normal distribution for outcomes
        prob_home = 1 - norm.cdf(0.5, mov_prediction, mov_std_error)
        prob_away = norm.cdf(-0.5, mov_prediction, mov_std_error)
        prob_draw = max(0.05, 1 - prob_home - prob_away)

        # Normalize
        total = prob_home + prob_draw + prob_away
        return [prob_home / total, prob_draw / total, prob_away / total]
