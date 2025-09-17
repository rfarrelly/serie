"""
Enhanced ZSD Poisson model with improved regularization and validation.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .core import MatchPrediction, ModelConfig
from .optimizer import TeamRatingsOptimizer
from .predictors import PredictorFactory


class ZSDPoissonModel:
    """Enhanced soccer prediction model with proper regularization and validation."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.optimizer = TeamRatingsOptimizer(self.config)

        # Model components
        self.teams = []
        self.team_index = {}
        self.attack_ratings = None
        self.defense_ratings = None
        self.home_advantage = None
        self.away_adjustment = None
        self.mov_model = None
        self.predictors = {}

        # Training diagnostics
        self.convergence_info = {}
        self.data = None
        self.n_teams = 0
        self.model_quality_score = 0.0

    def fit(self, matches_df: pd.DataFrame) -> None:
        """Fit the model with enhanced validation."""
        self.data = self._prepare_data(matches_df)
        self._validate_data()
        self._initialize_teams()

        # Check for minimum data requirements
        min_matches_per_team = max(5, len(self.teams) // 4)
        team_match_counts = self._calculate_team_match_counts()

        insufficient_teams = [
            team
            for team, count in team_match_counts.items()
            if count < min_matches_per_team
        ]

        if len(insufficient_teams) > len(self.teams) * 0.3:
            raise ValueError(
                f"Too many teams with insufficient matches: {len(insufficient_teams)}"
            )

        # Optimize ratings with enhanced validation
        try:
            ratings = self.optimizer.optimize(self.data, self.team_index, self.n_teams)
            self._assign_ratings(ratings)

            # Validate ratings quality
            self._validate_ratings_quality()

        except Exception as e:
            raise RuntimeError(f"Model fitting failed: {e}")

        # Fit MOV model and initialize predictors
        self._fit_mov_model()
        self._initialize_predictors()

        # Calculate model quality score
        self.model_quality_score = self._calculate_model_quality()
        self.convergence_info = {
            "success": True,
            "quality_score": self.model_quality_score,
        }

    def predict_match(
        self, home_team: str, away_team: str, method: str = "zip"
    ) -> MatchPrediction:
        """Predict match with enhanced validation."""
        self._validate_teams(home_team, away_team)
        self._validate_method(method)

        # Calculate expected goals with bounds checking
        home_idx = self.team_index[home_team]
        away_idx = self.team_index[away_team]

        lambda_home = self._calculate_expected_goals(home_idx, away_idx, is_home=True)
        lambda_away = self._calculate_expected_goals(away_idx, home_idx, is_home=False)

        # Validate lambda values
        lambda_home = np.clip(
            lambda_home, self.config.lambda_bounds[0], self.config.lambda_bounds[1]
        )
        lambda_away = np.clip(
            lambda_away, self.config.lambda_bounds[0], self.config.lambda_bounds[1]
        )

        # Predict MOV
        predicted_mov, mov_std_error = self._predict_mov(lambda_home, lambda_away)

        # Get outcome probabilities
        kwargs = {"mov_prediction": predicted_mov, "mov_std_error": mov_std_error}
        prob_home, prob_draw, prob_away = self.predictors[
            method
        ].predict_outcome_probabilities(lambda_home, lambda_away, **kwargs)

        # Validate and normalize probabilities
        total_prob = prob_home + prob_draw + prob_away
        if total_prob <= 0:
            # Fallback to uniform distribution
            prob_home, prob_draw, prob_away = 0.33, 0.34, 0.33
        elif not (0.95 <= total_prob <= 1.05):
            # Normalize if needed
            prob_home /= total_prob
            prob_draw /= total_prob
            prob_away /= total_prob

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            mov_prediction=predicted_mov,
            mov_std_error=mov_std_error,
            prob_home_win=prob_home,
            prob_draw=prob_draw,
            prob_away_win=prob_away,
            model_type=method,
        )

    def predict_all_methods(
        self, home_team: str, away_team: str
    ) -> Dict[str, MatchPrediction]:
        """Get predictions for all available methods."""
        predictions = {}

        for method in self.predictors.keys():
            try:
                predictions[method] = self.predict_match(home_team, away_team, method)
            except Exception as e:
                print(f"Error predicting with {method}: {e}")
                continue

        return predictions

    def get_team_ratings(self) -> pd.DataFrame:
        """Return team ratings with additional metrics."""
        if self.attack_ratings is None:
            return pd.DataFrame()

        ratings_df = pd.DataFrame(
            {
                "Team": self.teams,
                "Attack_Rating": self.attack_ratings,
                "Defense_Rating": self.defense_ratings,
                "Net_Rating": self.attack_ratings - self.defense_ratings,
            }
        ).sort_values("Net_Rating", ascending=False)

        # Add percentile ranks
        ratings_df["Attack_Percentile"] = (
            ratings_df["Attack_Rating"].rank(pct=True) * 100
        )
        ratings_df["Defense_Percentile"] = (
            1 - ratings_df["Defense_Rating"].rank(pct=True)
        ) * 100

        return ratings_df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data preparation with validation."""
        df = df.copy()

        # Convert date column
        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])

        # Calculate derived features
        df["Home_MOV"] = df["FTHG"] - df["FTAG"]
        df["Total_Goals"] = df["FTHG"] + df["FTAG"]

        # Sort by date for proper temporal ordering
        df = df.sort_values("Date").reset_index(drop=True)

        # Enhanced time weighting with validation
        if self.config.decay_rate > 0:
            match_index = np.arange(len(df))[::-1]  # Most recent = 0
            weights = np.exp(-self.config.decay_rate * match_index)

            # Normalize weights to maintain sample size
            weights *= len(df) / weights.sum()

            # Ensure minimum weight for stability
            weights = np.maximum(weights, 0.1)
        else:
            weights = np.ones(len(df))

        df["Weight"] = weights

        return df

    def _validate_data(self) -> None:
        """Enhanced data validation."""
        required_cols = ["Home", "Away", "FTHG", "FTAG", "Date"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for invalid scores
        invalid_scores = (
            (self.data["FTHG"] < 0)
            | (self.data["FTAG"] < 0)
            | (self.data["FTHG"] > 20)
            | (self.data["FTAG"] > 20)
        )

        if invalid_scores.sum() > 0:
            print(f"Warning: {invalid_scores.sum()} matches have invalid scores")
            self.data = self.data[~invalid_scores].reset_index(drop=True)

        # Check for minimum data
        if len(self.data) < 50:
            raise ValueError("Insufficient data for model fitting")

    def _initialize_teams(self) -> None:
        """Enhanced team initialization."""
        all_teams = set(self.data["Home"]).union(set(self.data["Away"]))
        self.teams = sorted(list(all_teams))
        self.team_index = {team: i for i, team in enumerate(self.teams)}
        self.n_teams = len(self.teams)

        if self.n_teams < 4:
            raise ValueError("Need at least 4 teams for model fitting")

    def _calculate_team_match_counts(self) -> Dict[str, int]:
        """Calculate number of matches per team."""
        team_counts = {}
        for team in self.teams:
            home_matches = (self.data["Home"] == team).sum()
            away_matches = (self.data["Away"] == team).sum()
            team_counts[team] = home_matches + away_matches
        return team_counts

    def _assign_ratings(self, ratings: Dict) -> None:
        """Assign optimized ratings with validation."""
        self.attack_ratings = ratings["attack_ratings"]
        self.defense_ratings = ratings["defense_ratings"]
        self.home_advantage = ratings["home_advantage"]
        self.away_adjustment = ratings["away_adjustment"]

        # Validate ratings are reasonable
        if np.any(np.abs(self.attack_ratings) > 5):
            print("Warning: Extreme attack ratings detected")
        if np.any(np.abs(self.defense_ratings) > 5):
            print("Warning: Extreme defense ratings detected")

    def _validate_ratings_quality(self) -> None:
        """Validate that ratings make sense."""
        # Check for reasonable variance in ratings
        attack_std = np.std(self.attack_ratings)
        defense_std = np.std(self.defense_ratings)

        if attack_std < 0.1 or defense_std < 0.1:
            print("Warning: Low variance in team ratings - model may be underfit")
        elif attack_std > 2.0 or defense_std > 2.0:
            print("Warning: High variance in team ratings - model may be overfit")

    def _calculate_expected_goals(
        self, attacking_idx: int, defending_idx: int, is_home: bool
    ) -> float:
        """Enhanced expected goals calculation."""
        if is_home:
            strength = (
                self.home_advantage
                + self.attack_ratings[attacking_idx]
                - self.defense_ratings[defending_idx]
            )
            baseline = self.data["FTHG"].mean()
        else:
            strength = (
                self.away_adjustment
                + self.attack_ratings[attacking_idx]
                - self.defense_ratings[defending_idx]
            )
            baseline = self.data["FTAG"].mean()

        # Enhanced strength-to-goals conversion
        expected_goals = baseline * np.exp(strength * 0.1)  # Exponential scaling

        # Apply bounds
        return np.clip(
            expected_goals, self.config.lambda_bounds[0], self.config.lambda_bounds[1]
        )

    def _predict_mov(self, lambda_home: float, lambda_away: float) -> tuple:
        """Enhanced MOV prediction with validation."""
        raw_mov = lambda_home - lambda_away

        if self.mov_model is not None:
            predicted_mov = (
                self.mov_model.params[0] + self.mov_model.params[1] * raw_mov
            )
            mov_std_error = np.sqrt(self.mov_model.mse_resid)

            # Bound the standard error
            mov_std_error = np.clip(mov_std_error, 0.5, 3.0)
        else:
            # Fallback calculation
            predicted_mov = raw_mov
            mov_std_error = 1.5

        return predicted_mov, mov_std_error

    def _fit_mov_model(self) -> None:
        """Enhanced MOV model fitting."""
        try:
            home_indices = self.data["Home"].map(self.team_index).values
            away_indices = self.data["Away"].map(self.team_index).values

            # Calculate expected goals for all matches
            expected_home = np.array(
                [
                    self._calculate_expected_goals(h_idx, a_idx, is_home=True)
                    for h_idx, a_idx in zip(home_indices, away_indices)
                ]
            )

            expected_away = np.array(
                [
                    self._calculate_expected_goals(a_idx, h_idx, is_home=False)
                    for a_idx, h_idx in zip(away_indices, home_indices)
                ]
            )

            raw_mov = expected_home - expected_away
            actual_mov = self.data["Home_MOV"].values
            weights = self.data["Weight"].values

            # Fit weighted regression with validation
            X = sm.add_constant(raw_mov)

            # Remove outliers for better fit
            mov_residuals = actual_mov - raw_mov
            outlier_threshold = 3 * np.std(mov_residuals)
            non_outliers = np.abs(mov_residuals) <= outlier_threshold

            if non_outliers.sum() < len(actual_mov) * 0.8:
                # Too many outliers, use all data
                non_outliers = np.ones(len(actual_mov), dtype=bool)

            self.mov_model = sm.WLS(
                actual_mov[non_outliers], X[non_outliers], weights=weights[non_outliers]
            ).fit()

            # Validate MOV model quality
            if self.mov_model.rsquared < 0.05:
                print("Warning: MOV model has very low R-squared")

        except Exception as e:
            print(f"Error fitting MOV model: {e}")
            # Create dummy model
            self.mov_model = None

    def _initialize_predictors(self) -> None:
        """Initialize prediction methods."""
        try:
            self.predictors = PredictorFactory.create_predictors(
                self.config, self.data, self.mov_model
            )
        except Exception as e:
            print(f"Error initializing predictors: {e}")
            # Create minimal predictor set
            from .predictors import PoissonPredictor

            self.predictors = {"poisson": PoissonPredictor(self.config)}

    def _calculate_model_quality(self) -> float:
        """Calculate overall model quality score."""
        try:
            # Factors contributing to model quality
            factors = []

            # Rating variance (should be reasonable)
            attack_std = np.std(self.attack_ratings)
            defense_std = np.std(self.defense_ratings)
            variance_score = 1.0 - abs(0.5 - min(attack_std, defense_std, 1.0))
            factors.append(variance_score)

            # MOV model fit quality
            if self.mov_model:
                mov_quality = min(self.mov_model.rsquared * 2, 1.0)  # Scale R-squared
            else:
                mov_quality = 0.3  # Default for missing MOV model
            factors.append(mov_quality)

            # Data sufficiency
            avg_matches_per_team = (
                len(self.data) / len(self.teams) * 2
            )  # Each match involves 2 teams
            data_quality = min(
                avg_matches_per_team / 20, 1.0
            )  # 20 matches per team is good
            factors.append(data_quality)

            return np.mean(factors)

        except Exception:
            return 0.5  # Neutral score if calculation fails

    def _validate_teams(self, home_team: str, away_team: str) -> None:
        """Enhanced team validation."""
        if home_team not in self.team_index:
            raise ValueError(f"Unknown home team: {home_team}")
        if away_team not in self.team_index:
            raise ValueError(f"Unknown away team: {away_team}")
        if home_team == away_team:
            raise ValueError("Team cannot play against itself")

    def _validate_method(self, method: str) -> None:
        """Enhanced method validation."""
        if method not in self.predictors:
            available_methods = list(self.predictors.keys())
            raise ValueError(
                f"Unknown prediction method: {method}. Available: {available_methods}"
            )

    def get_model_diagnostics(self) -> Dict:
        """Get comprehensive model diagnostics."""
        if not self.convergence_info.get("success", False):
            return {"error": "Model not fitted successfully"}

        diagnostics = {
            "quality_score": self.model_quality_score,
            "n_teams": self.n_teams,
            "n_matches": len(self.data) if self.data is not None else 0,
            "home_advantage": float(self.home_advantage) if self.home_advantage else 0,
            "away_adjustment": (
                float(self.away_adjustment) if self.away_adjustment else 0
            ),
        }

        if self.attack_ratings is not None:
            diagnostics.update(
                {
                    "attack_rating_std": float(np.std(self.attack_ratings)),
                    "defense_rating_std": float(np.std(self.defense_ratings)),
                    "strongest_attack": self.teams[np.argmax(self.attack_ratings)],
                    "strongest_defense": self.teams[np.argmin(self.defense_ratings)],
                }
            )

        if self.mov_model:
            diagnostics["mov_r_squared"] = float(self.mov_model.rsquared)

        return diagnostics
