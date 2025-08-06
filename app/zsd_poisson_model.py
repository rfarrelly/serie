# refactored_zsd_poisson_model.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ModelConfig:
    """Enhanced configuration for the soccer prediction model with regularization."""

    # Core parameters
    decay_rate: float = 0.001
    max_goals: int = 15
    eps: float = 1e-6
    lambda_bounds: Tuple[float, float] = (0.01, 10.0)
    param_bounds: Tuple[float, float] = (-6.0, 6.0)
    min_matches_per_team: int = 5

    # Regularization parameters
    l1_reg: float = 0.0
    l2_reg: float = 0.01
    team_reg: float = 0.005
    shrink_to_mean: bool = True

    # Cross-validation and optimization
    auto_tune_regularization: bool = False
    cv_folds: int = 5
    n_optimization_starts: int = 3
    max_iter: int = 2000


@dataclass
class MatchPrediction:
    """Structured output for match predictions."""

    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    mov_prediction: float
    mov_std_error: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    model_type: str

    @property
    def probabilities(self) -> np.ndarray:
        """Return probabilities as numpy array."""
        return np.array([self.prob_home_win, self.prob_draw, self.prob_away_win])


class PredictorProtocol(Protocol):
    """Protocol for prediction methods."""

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> Tuple[float, float, float]: ...


class BasePoissonCalculator:
    """Base class for Poisson probability calculations."""

    def __init__(self, max_goals: int = 15):
        self.max_goals = max_goals

    def compute_prob_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Compute probability matrix for all score combinations."""
        home_goals = np.arange(0, self.max_goals + 1)
        away_goals = np.arange(0, self.max_goals + 1)
        home_probs = poisson.pmf(home_goals, lambda_home)
        away_probs = poisson.pmf(away_goals, lambda_away)
        return np.outer(home_probs, away_probs)

    def matrix_to_outcomes(self, prob_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Convert probability matrix to match outcome probabilities."""
        prob_matrix /= prob_matrix.sum()
        home_win = np.tril(prob_matrix, -1).sum()
        draw = np.trace(prob_matrix)
        away_win = np.triu(prob_matrix, 1).sum()
        return home_win, draw, away_win


class PoissonPredictor(BasePoissonCalculator, PredictorProtocol):
    """Standard Poisson prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__(config.max_goals)
        self.config = config

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> Tuple[float, float, float]:
        prob_matrix = self.compute_prob_matrix(lambda_home, lambda_away)
        return self.matrix_to_outcomes(prob_matrix)


class ZIPPredictor(BasePoissonCalculator, PredictorProtocol):
    """Zero-inflated Poisson with dynamic adjustment."""

    def __init__(self, config: ModelConfig, historical_data: pd.DataFrame):
        super().__init__(config.max_goals)
        self.config = config
        self.adjustment_matrix = self._compute_adjustment_matrix(historical_data)

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> Tuple[float, float, float]:
        base_matrix = self.compute_prob_matrix(lambda_home, lambda_away)
        adjusted_matrix = base_matrix * self.adjustment_matrix
        return self.matrix_to_outcomes(adjusted_matrix)

    def _compute_adjustment_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Compute adjustment matrix from recent historical data."""
        recent_data = data.tail(min(200, len(data)))

        # Compute observed distribution
        home_dist = recent_data["FTHG"].value_counts(normalize=True).sort_index()
        away_dist = recent_data["FTAG"].value_counts(normalize=True).sort_index()

        idx = np.arange(0, self.max_goals + 1)
        observed = np.outer(
            home_dist.reindex(idx, fill_value=0), away_dist.reindex(idx, fill_value=0)
        )

        # Compute expected distribution
        avg_home = recent_data["FTHG"].mean()
        avg_away = recent_data["FTAG"].mean()
        expected = self.compute_prob_matrix(avg_home, avg_away)

        # Calculate adjustment with bounds
        adjustment = np.where(expected > 1e-6, observed / expected, 1.0)
        return np.clip(adjustment, 0.1, 5.0)


class TeamRatingsOptimizer:
    """Handles the optimization of team ratings."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.regularization_path = []

    def optimize(
        self, data: pd.DataFrame, team_index: Dict[str, int], n_teams: int
    ) -> Dict:
        """Optimize team ratings with multiple starts."""
        best_result = None
        best_loss = float("inf")

        for _ in range(self.config.n_optimization_starts):
            initial_params = self._initialize_parameters(n_teams)

            try:
                result = minimize(
                    lambda p: self._objective_function(p, data, team_index, n_teams),
                    initial_params,
                    method="L-BFGS-B",
                    bounds=self._get_bounds(n_teams),
                    options={"maxiter": self.config.max_iter, "ftol": 1e-9},
                )

                if result.fun < best_loss and result.success:
                    best_result = result
                    best_loss = result.fun

            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("All optimization attempts failed")

        return self._unpack_parameters(best_result.x, n_teams)

    def _initialize_parameters(self, n_teams: int) -> np.ndarray:
        """Initialize parameters with appropriate starting values."""
        n_params = 2 * n_teams + 2

        if self.config.shrink_to_mean:
            params = np.random.normal(0, 0.05, n_params)
        else:
            params = np.random.normal(0, 0.1, n_params)

        params[-2] = 0.2  # home advantage
        params[-1] = -0.1  # away adjustment

        return params

    def _get_bounds(self, n_teams: int) -> List[Tuple[float, float]]:
        """Get parameter bounds."""
        team_bounds = [self.config.param_bounds] * (2 * n_teams)
        global_bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        return team_bounds + global_bounds

    def _objective_function(
        self,
        params: np.ndarray,
        data: pd.DataFrame,
        team_index: Dict[str, int],
        n_teams: int,
    ) -> float:
        """Calculate regularized objective function."""
        # Base loss
        base_loss = self._calculate_base_loss(params, data, team_index, n_teams)

        # Regularization terms
        attack_ratings = params[:n_teams]
        defense_ratings = params[n_teams : 2 * n_teams]
        home_adv, away_adj = params[-2:]

        l1_penalty = self.config.l1_reg * (
            np.sum(np.abs(attack_ratings)) + np.sum(np.abs(defense_ratings))
        )

        l2_penalty = self.config.l2_reg * (
            np.sum(attack_ratings**2)
            + np.sum(defense_ratings**2)
            + home_adv**2
            + away_adj**2
        )

        team_penalty = 0.0
        if self.config.team_reg > 0:
            imbalances = (attack_ratings - defense_ratings) ** 2
            team_penalty = self.config.team_reg * np.sum(imbalances)

        shrinkage_penalty = 0.0
        if self.config.shrink_to_mean:
            mean_attack = np.mean(attack_ratings)
            mean_defense = np.mean(defense_ratings)
            shrinkage_penalty = 0.001 * (
                np.sum((attack_ratings - mean_attack) ** 2)
                + np.sum((defense_ratings - mean_defense) ** 2)
            )

        total_loss = (
            base_loss + l1_penalty + l2_penalty + team_penalty + shrinkage_penalty
        )

        self.regularization_path.append(
            {
                "base_loss": base_loss,
                "l1_penalty": l1_penalty,
                "l2_penalty": l2_penalty,
                "team_penalty": team_penalty,
                "shrinkage_penalty": shrinkage_penalty,
                "total_loss": total_loss,
            }
        )

        return total_loss

    def _calculate_base_loss(
        self,
        params: np.ndarray,
        data: pd.DataFrame,
        team_index: Dict[str, int],
        n_teams: int,
    ) -> float:
        """Calculate base prediction loss."""
        attack_ratings = params[:n_teams]
        defense_ratings = params[n_teams : 2 * n_teams]
        home_adv, away_adj = params[-2:]

        # Get team indices
        home_indices = data["Home"].map(team_index).values
        away_indices = data["Away"].map(team_index).values

        # Compute strengths
        home_strength = (
            home_adv + attack_ratings[home_indices] - defense_ratings[away_indices]
        )
        away_strength = (
            away_adj + attack_ratings[away_indices] - defense_ratings[home_indices]
        )

        # Convert to goals
        home_goals_pred = self._strength_to_goals(home_strength, data["FTHG"])
        away_goals_pred = self._strength_to_goals(away_strength, data["FTAG"])

        # Weighted squared error
        home_error = (home_goals_pred - data["FTHG"].values) ** 2
        away_error = (away_goals_pred - data["FTAG"].values) ** 2
        weights = data["Weight"].values

        return np.sum(weights * (home_error + away_error))

    def _strength_to_goals(
        self, strength: np.ndarray, actual_goals: pd.Series
    ) -> np.ndarray:
        """Convert team strength to expected goals."""
        baseline = actual_goals.mean()
        std_dev = actual_goals.std()

        # Sigmoid transformation
        prob = 1 / (1 + np.exp(-np.clip(strength, *self.config.param_bounds)))
        prob = np.clip(prob, self.config.eps, 1 - self.config.eps)
        z_score = norm.ppf(prob)

        return baseline + z_score * std_dev

    def _unpack_parameters(self, params: np.ndarray, n_teams: int) -> Dict:
        """Unpack optimized parameters."""
        return {
            "attack_ratings": params[:n_teams],
            "defense_ratings": params[n_teams : 2 * n_teams],
            "home_advantage": params[-2],
            "away_adjustment": params[-1],
        }


class ZSDPoissonModel:
    """Enhanced soccer prediction model with regularization and improved structure."""

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

    def fit(self, matches_df: pd.DataFrame) -> None:
        """Fit the model to historical match data."""
        self.data = self._prepare_data(matches_df)
        self._validate_data()
        self._initialize_teams()

        # Optimize ratings
        ratings = self.optimizer.optimize(self.data, self.team_index, self.n_teams)
        self.attack_ratings = ratings["attack_ratings"]
        self.defense_ratings = ratings["defense_ratings"]
        self.home_advantage = ratings["home_advantage"]
        self.away_adjustment = ratings["away_adjustment"]

        # Fit MOV model and initialize predictors
        self._fit_mov_model()
        self._initialize_predictors()

        self.convergence_info = {"success": True}

    def predict_match(
        self, home_team: str, away_team: str, method: str = "zip"
    ) -> MatchPrediction:
        """Predict a single match outcome."""
        if home_team not in self.team_index or away_team not in self.team_index:
            raise ValueError(f"Unknown team(s): {home_team}, {away_team}")

        if method not in self.predictors:
            raise ValueError(f"Unknown prediction method: {method}")

        # Calculate expected goals
        home_idx = self.team_index[home_team]
        away_idx = self.team_index[away_team]

        lambda_home = self._calculate_expected_goals(home_idx, away_idx, is_home=True)
        lambda_away = self._calculate_expected_goals(away_idx, home_idx, is_home=False)

        # Predict MOV
        raw_mov = lambda_home - lambda_away
        predicted_mov = self.mov_model.params[0] + self.mov_model.params[1] * raw_mov

        # Get outcome probabilities
        prob_home, prob_draw, prob_away = self.predictors[
            method
        ].predict_outcome_probabilities(lambda_home, lambda_away)

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            mov_prediction=predicted_mov,
            mov_std_error=np.sqrt(self.mov_model.mse_resid),
            prob_home_win=prob_home,
            prob_draw=prob_draw,
            prob_away_win=prob_away,
            model_type=method,
        )

    def get_team_ratings(self) -> pd.DataFrame:
        """Return current team ratings."""
        return pd.DataFrame(
            {
                "Team": self.teams,
                "Attack_Rating": self.attack_ratings,
                "Defense_Rating": self.defense_ratings,
                "Net_Rating": self.attack_ratings - self.defense_ratings,
            }
        ).sort_values("Net_Rating", ascending=False)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the match data."""
        df = df.copy()

        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])

        df["Home_MOV"] = df["FTHG"] - df["FTAG"]
        df["Total_Goals"] = df["FTHG"] + df["FTAG"]

        df = df.sort_values("Date").reset_index(drop=True)

        # Add time weights
        match_index = np.arange(len(df))[::-1]
        df["Weight"] = np.exp(-self.config.decay_rate * match_index)
        df["Weight"] *= len(df) / df["Weight"].sum()

        return df

    def _validate_data(self) -> None:
        """Validate data quality."""
        required_cols = ["Home", "Away", "FTHG", "FTAG", "Date"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _initialize_teams(self) -> None:
        """Initialize team indexing."""
        self.teams = sorted(list(set(self.data["Home"]).union(set(self.data["Away"]))))
        self.team_index = {team: i for i, team in enumerate(self.teams)}
        self.n_teams = len(self.teams)

    def _calculate_expected_goals(
        self, attacking_idx: int, defending_idx: int, is_home: bool
    ) -> float:
        """Calculate expected goals for a team."""
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

        # Simple transformation for expected goals
        return max(self.config.lambda_bounds[0], baseline + 0.5 * strength)

    def _fit_mov_model(self) -> None:
        """Fit margin of victory regression model."""
        # Compute predictions for all matches
        home_indices = self.data["Home"].map(self.team_index).values
        away_indices = self.data["Away"].map(self.team_index).values

        home_goals = np.array(
            [
                self._calculate_expected_goals(h_idx, a_idx, is_home=True)
                for h_idx, a_idx in zip(home_indices, away_indices)
            ]
        )

        away_goals = np.array(
            [
                self._calculate_expected_goals(a_idx, h_idx, is_home=False)
                for a_idx, h_idx in zip(away_indices, home_indices)
            ]
        )

        raw_mov = home_goals - away_goals
        actual_mov = self.data["Home_MOV"].values

        # Fit weighted regression
        X = sm.add_constant(raw_mov)
        weights = self.data["Weight"].values
        self.mov_model = sm.WLS(actual_mov, X, weights=weights).fit()

    def _initialize_predictors(self) -> None:
        """Initialize prediction methods."""
        self.predictors = {
            "poisson": PoissonPredictor(self.config),
            "zip": ZIPPredictor(self.config, self.data),
        }
