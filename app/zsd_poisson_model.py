from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ModelConfig:
    """Enhanced configuration for the soccer prediction model with regularization."""

    decay_rate: float = 0.001
    max_goals: int = 15
    eps: float = 1e-6
    lambda_bounds: Tuple[float, float] = (0.01, 10.0)
    param_bounds: Tuple[float, float] = (-6.0, 6.0)
    min_matches_per_team: int = 5

    # Regularization parameters
    l1_reg: float = 0.0  # Lasso regularization (sparsity)
    l2_reg: float = 0.01  # Ridge regularization (stability) - RECOMMENDED DEFAULT
    team_reg: float = 0.005  # Team-specific regularization (balance attack/defense)
    shrink_to_mean: bool = True  # Shrink ratings toward league average

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


class BasePredictor(ABC):
    """Abstract base class for different prediction methods."""

    @abstractmethod
    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> Tuple[float, float, float]:
        """Return (P(Home), P(Draw), P(Away))"""
        pass


class PoissonPredictor(BasePredictor):
    """Standard Poisson prediction."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> Tuple[float, float, float]:
        prob_matrix = self._poisson_prob_matrix(lambda_home, lambda_away)
        prob_matrix /= prob_matrix.sum()

        home_win = np.tril(prob_matrix, -1).sum()
        draw = np.trace(prob_matrix)
        away_win = np.triu(prob_matrix, 1).sum()

        return home_win, draw, away_win

    def _poisson_prob_matrix(
        self, lambda_home: float, lambda_away: float
    ) -> np.ndarray:
        home_goals = np.arange(0, self.config.max_goals + 1)
        away_goals = np.arange(0, self.config.max_goals + 1)
        home_probs = poisson.pmf(home_goals, lambda_home)
        away_probs = poisson.pmf(away_goals, lambda_away)
        return np.outer(home_probs, away_probs)


class ZIPPredictor(BasePredictor):
    """Zero-inflated Poisson with dynamic adjustment."""

    def __init__(self, config: ModelConfig, historical_data: pd.DataFrame):
        self.config = config
        self.adjustment_matrix = self._compute_adjustment_matrix(historical_data)

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float
    ) -> Tuple[float, float, float]:
        base_poisson = PoissonPredictor(self.config)._poisson_prob_matrix(
            lambda_home, lambda_away
        )

        # Apply adjustment matrix
        adjusted_matrix = base_poisson * self.adjustment_matrix
        adjusted_matrix /= adjusted_matrix.sum()

        home_win = np.tril(adjusted_matrix, -1).sum()
        draw = np.trace(adjusted_matrix)
        away_win = np.triu(adjusted_matrix, 1).sum()

        return home_win, draw, away_win

    def _compute_adjustment_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Compute adjustment matrix from recent historical data."""
        # Use more recent data for adjustment (e.g., last 200 matches)
        recent_data = data.tail(min(200, len(data)))

        home_dist = recent_data["FTHG"].value_counts(normalize=True).sort_index()
        away_dist = recent_data["FTAG"].value_counts(normalize=True).sort_index()

        idx = np.arange(0, self.config.max_goals + 1)
        observed = np.outer(
            home_dist.reindex(idx, fill_value=0), away_dist.reindex(idx, fill_value=0)
        )

        avg_home = recent_data["FTHG"].mean()
        avg_away = recent_data["FTAG"].mean()
        expected = PoissonPredictor(self.config)._poisson_prob_matrix(
            avg_home, avg_away
        )

        # Smooth the adjustment to avoid extreme values
        adjustment = np.where(expected > 1e-6, observed / expected, 1.0)
        return np.clip(adjustment, 0.1, 5.0)  # Prevent extreme adjustments


class ZSDPoissonModel:
    """Enhanced soccer prediction model with regularization and improved statistical foundation."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

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
        self.regularization_path = []
        self.cv_results = None

        # Data storage
        self.data = None
        self.latest_date = None
        self.n_teams = 0

    def fit(self, matches_df: pd.DataFrame) -> None:
        """Fit the model to historical match data with optional parameter tuning."""
        self.data = self._prepare_data(matches_df)
        self._validate_data()
        self._initialize_teams()

        # Auto-tune regularization if requested
        if self.config.auto_tune_regularization:
            print("Auto-tuning regularization parameters...")
            self._tune_regularization_parameters()

        # print(
        #     f"Fitting model with regularization: L1={self.config.l1_reg:.4f}, "
        #     f"L2={self.config.l2_reg:.4f}, Team={self.config.team_reg:.4f}"
        # )

        # Fit the model
        self._fit_ratings()
        self._fit_mov_model()
        self._initialize_predictors()

        # print(
        #     f"Model fitted successfully. Convergence: {self.convergence_info.get('success', 'Unknown')}"
        # )

    def _tune_regularization_parameters(self):
        """Use time-series cross-validation to find optimal regularization parameters."""
        param_grid = {
            "l1_reg": [0.0, 0.001, 0.005, 0.01],
            "l2_reg": [0.001, 0.005, 0.01, 0.05, 0.1],
            "team_reg": [0.0, 0.001, 0.005, 0.01],
            "decay_rate": [0.0005, 0.001, 0.002],
        }

        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)

        # Generate parameter combinations (sample subset to avoid excessive computation)
        param_combinations = self._generate_param_combinations(
            param_grid, max_combinations=20
        )

        best_score = float("inf")
        best_params = None
        cv_results = []

        print(f"Testing {len(param_combinations)} parameter combinations...")

        for i, params in enumerate(param_combinations):
            if i % 5 == 0:
                print(f"Progress: {i + 1}/{len(param_combinations)}")

            fold_scores = []
            for train_idx, val_idx in tscv.split(self.data):
                try:
                    score = self._evaluate_params_on_fold(params, train_idx, val_idx)
                    fold_scores.append(score)
                except Exception as e:
                    fold_scores.append(float("inf"))

            avg_score = np.mean(fold_scores)
            cv_results.append(
                {
                    "params": params.copy(),
                    "mean_score": avg_score,
                    "std_score": np.std(fold_scores),
                    "scores": fold_scores.copy(),
                }
            )

            if avg_score < best_score:
                best_score = avg_score
                best_params = params.copy()

        # Update config with best parameters
        for key, value in best_params.items():
            setattr(self.config, key, value)

        self.cv_results = cv_results
        print(f"Best CV score: {best_score:.4f}")
        print(f"Optimal parameters: {best_params}")

    def _generate_param_combinations(
        self, param_grid: Dict, max_combinations: int = 20
    ) -> List[Dict]:
        """Generate a reasonable subset of parameter combinations."""
        import random
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        all_combinations = list(product(*param_values))

        # If too many combinations, sample randomly
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)

        return [dict(zip(param_names, combo)) for combo in all_combinations]

    def _evaluate_params_on_fold(
        self, params: Dict, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> float:
        """Evaluate parameter set on a single fold."""
        # Create temporary model with these parameters
        temp_config = ModelConfig(**{**self.config.__dict__, **params})
        temp_model = ZSDPoissonModel(temp_config)

        # Fit on training fold
        train_data = self.data.iloc[train_idx]
        temp_model.data = temp_model._prepare_data(train_data)
        temp_model._initialize_teams()
        temp_model._fit_ratings()

        # Evaluate on validation fold
        val_data = self.data.iloc[val_idx]
        predictions = []
        actuals = []

        for _, match in val_data.iterrows():
            home_team = match["Home"]
            away_team = match["Away"]

            if (
                home_team in temp_model.team_index
                and away_team in temp_model.team_index
            ):
                try:
                    pred_home, pred_away = temp_model._predict_goals_simple(
                        home_team, away_team
                    )
                    predictions.append([pred_home, pred_away])
                    actuals.append([match["FTHG"], match["FTAG"]])
                except Exception:
                    continue

        if not predictions:
            return float("inf")

        # Calculate MSE
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        mse = np.mean((predictions - actuals) ** 2)

        return mse

    def _predict_goals_simple(
        self, home_team: str, away_team: str
    ) -> Tuple[float, float]:
        """Simple goal prediction for cross-validation."""
        home_idx = self.team_index[home_team]
        away_idx = self.team_index[away_team]

        # Predict expected goals using current ratings
        home_strength = (
            self.home_advantage
            + self.attack_ratings[home_idx]
            - self.defense_ratings[away_idx]
        )
        away_strength = (
            self.away_adjustment
            + self.attack_ratings[away_idx]
            - self.defense_ratings[home_idx]
        )

        # Simple conversion to goals (more robust than full _strength_to_goals)
        home_goals = max(0.1, 1.5 + 0.5 * home_strength)
        away_goals = max(0.1, 1.2 + 0.5 * away_strength)

        return home_goals, away_goals

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the match data."""
        df = df.copy()

        # Ensure date column
        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])

        # Add derived columns
        df["Home_MOV"] = df["FTHG"] - df["FTAG"]
        df["Total_Goals"] = df["FTHG"] + df["FTAG"]

        # Sort by date and add time weights
        df = df.sort_values("Date").reset_index(drop=True)
        self.latest_date = df["Date"].max()

        # Exponential decay weighting (more recent = higher weight)
        match_index = np.arange(len(df))[::-1]
        df["Weight"] = np.exp(-self.config.decay_rate * match_index)
        df["Weight"] *= len(df) / df["Weight"].sum()  # Normalize

        return df

    def _validate_data(self) -> None:
        """Validate that we have sufficient data quality."""
        required_cols = ["Home", "Away", "FTHG", "FTAG", "Date"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for teams with insufficient matches
        all_teams = list(set(self.data["Home"]).union(set(self.data["Away"])))
        for team in all_teams:
            team_matches = len(
                self.data[(self.data["Home"] == team) | (self.data["Away"] == team)]
            )
            if team_matches < self.config.min_matches_per_team:
                print(f"Warning: {team} has only {team_matches} matches")

    def _initialize_teams(self) -> None:
        """Initialize team indexing."""
        self.teams = sorted(list(set(self.data["Home"]).union(set(self.data["Away"]))))
        self.team_index = {team: i for i, team in enumerate(self.teams)}
        self.n_teams = len(self.teams)

    def _fit_ratings(self) -> None:
        """Fit attack/defense ratings using regularized optimization with multiple starts."""
        best_result = None
        best_loss = float("inf")

        # Clear regularization path
        self.regularization_path = []

        for start in range(self.config.n_optimization_starts):
            # Initialize parameters with small random values
            initial_params = self._initialize_parameters()

            try:
                result = minimize(
                    self._regularized_objective_function,
                    initial_params,
                    method="L-BFGS-B",
                    bounds=self._get_parameter_bounds(),
                    options={"maxiter": self.config.max_iter, "ftol": 1e-9},
                )

                if result.fun < best_loss and result.success:
                    best_result = result
                    best_loss = result.fun

            except Exception as e:
                print(f"Optimization start {start} failed: {str(e)}")
                continue

        if best_result is None:
            raise RuntimeError("All optimization attempts failed")

        # Unpack optimized parameters
        self.attack_ratings = best_result.x[: self.n_teams]
        self.defense_ratings = best_result.x[self.n_teams : 2 * self.n_teams]
        self.home_advantage = best_result.x[-2]
        self.away_adjustment = best_result.x[-1]

        # Store convergence information
        self.convergence_info = {
            "success": best_result.success,
            "iterations": best_result.nit,
            "final_loss": best_result.fun,
            "message": best_result.message,
        }

        if not best_result.success:
            print(
                f"Warning: Optimization may not have converged: {best_result.message}"
            )

    def _initialize_parameters(self) -> np.ndarray:
        """Initialize parameters with shrinkage toward sensible priors."""
        n_params = 2 * self.n_teams + 2

        if self.config.shrink_to_mean:
            # Initialize ratings closer to zero (league average)
            params = np.random.normal(0, 0.05, n_params)
        else:
            # Standard initialization
            params = np.random.normal(0, 0.1, n_params)

        # Set reasonable initial values for global parameters
        params[-2] = 0.2  # home advantage
        params[-1] = -0.1  # away adjustment

        return params

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for all parameters."""
        team_bounds = [self.config.param_bounds] * (2 * self.n_teams)
        global_bounds = [(-2.0, 2.0), (-2.0, 2.0)]  # home/away adjustments
        return team_bounds + global_bounds

    def _regularized_objective_function(self, params: np.ndarray) -> float:
        """Enhanced objective function with comprehensive regularization."""
        # Base prediction loss
        base_loss = self._base_prediction_loss(params)

        # Unpack parameters for regularization
        attack_ratings = params[: self.n_teams]
        defense_ratings = params[self.n_teams : 2 * self.n_teams]
        home_adv, away_adj = params[-2:]

        # L1 regularization (promotes sparsity)
        l1_penalty = self.config.l1_reg * (
            np.sum(np.abs(attack_ratings)) + np.sum(np.abs(defense_ratings))
        )

        # L2 regularization (promotes stability)
        l2_penalty = self.config.l2_reg * (
            np.sum(attack_ratings**2)
            + np.sum(defense_ratings**2)
            + home_adv**2
            + away_adj**2
        )

        # Team-specific regularization (balance attack/defense)
        team_penalty = 0.0
        if self.config.team_reg > 0:
            for i in range(self.n_teams):
                # Penalize extreme imbalances between attack and defense
                imbalance = (attack_ratings[i] - defense_ratings[i]) ** 2
                team_penalty += self.config.team_reg * imbalance

        # Shrinkage toward league average
        shrinkage_penalty = 0.0
        if self.config.shrink_to_mean:
            # Shrink ratings toward zero (league average)
            mean_attack = np.mean(attack_ratings)
            mean_defense = np.mean(defense_ratings)

            shrinkage_penalty = 0.001 * (
                np.sum((attack_ratings - mean_attack) ** 2)
                + np.sum((defense_ratings - mean_defense) ** 2)
            )

        total_loss = (
            base_loss + l1_penalty + l2_penalty + team_penalty + shrinkage_penalty
        )

        # Store regularization components for analysis
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

    def _base_prediction_loss(self, params: np.ndarray) -> float:
        """Base prediction loss (weighted squared error)."""
        attack_ratings = params[: self.n_teams]
        defense_ratings = params[self.n_teams : 2 * self.n_teams]
        home_adv, away_adj = params[-2:]

        # Get team indices for all matches
        home_indices = self.data["Home"].map(self.team_index).values
        away_indices = self.data["Away"].map(self.team_index).values

        # Compute expected goals
        home_strength = (
            home_adv + attack_ratings[home_indices] - defense_ratings[away_indices]
        )
        away_strength = (
            away_adj + attack_ratings[away_indices] - defense_ratings[home_indices]
        )

        # Clip to prevent extreme values
        home_strength = np.clip(home_strength, *self.config.param_bounds)
        away_strength = np.clip(away_strength, *self.config.param_bounds)

        # Convert to expected goals using consistent transformation
        home_goals_pred = self._strength_to_goals(home_strength, is_home=True)
        away_goals_pred = self._strength_to_goals(away_strength, is_home=False)

        # Weighted squared error
        home_error = (home_goals_pred - self.data["FTHG"].values) ** 2
        away_error = (away_goals_pred - self.data["FTAG"].values) ** 2
        weights = self.data["Weight"].values

        return np.sum(weights * (home_error + away_error))

    def _strength_to_goals(self, strength: np.ndarray, is_home: bool) -> np.ndarray:
        """Convert team strength to expected goals."""
        if is_home:
            baseline = self.data["FTHG"].mean()
            std_dev = self.data["FTHG"].std()
        else:
            baseline = self.data["FTAG"].mean()
            std_dev = self.data["FTAG"].std()

        # Use sigmoid to map strength to probability, then inverse normal
        prob = self._sigmoid(strength)
        prob = np.clip(prob, self.config.eps, 1 - self.config.eps)
        z_score = norm.ppf(prob)

        return baseline + z_score * std_dev

    def _fit_mov_model(self) -> None:
        """Fit margin of victory regression model."""
        # Compute raw MOV predictions from ratings
        raw_mov = self._compute_raw_mov()
        actual_mov = self.data["Home_MOV"].values

        # Fit linear regression
        X = sm.add_constant(raw_mov)
        weights = self.data["Weight"].values
        self.mov_model = sm.WLS(actual_mov, X, weights=weights).fit()

        self.mov_intercept = self.mov_model.params[0]
        self.mov_slope = self.mov_model.params[1]
        self.mov_std_error = np.sqrt(self.mov_model.mse_resid)

    def _compute_raw_mov(self) -> np.ndarray:
        """Compute raw margin of victory from current ratings."""
        home_indices = self.data["Home"].map(self.team_index).values
        away_indices = self.data["Away"].map(self.team_index).values

        home_goals = self._predict_goals(home_indices, away_indices, is_home=True)
        away_goals = self._predict_goals(away_indices, home_indices, is_home=False)

        return home_goals - away_goals

    def _predict_goals(
        self,
        attacking_indices: np.ndarray,
        defending_indices: np.ndarray,
        is_home: bool,
    ) -> np.ndarray:
        """Predict goals for given team matchups."""
        if is_home:
            strength = (
                self.home_advantage
                + self.attack_ratings[attacking_indices]
                - self.defense_ratings[defending_indices]
            )
        else:
            strength = (
                self.away_adjustment
                + self.attack_ratings[attacking_indices]
                - self.defense_ratings[defending_indices]
            )

        strength = np.clip(strength, *self.config.param_bounds)
        return self._strength_to_goals(strength, is_home)

    def _initialize_predictors(self) -> None:
        """Initialize different prediction methods."""
        self.predictors = {
            "poisson": PoissonPredictor(self.config),
            "zip": ZIPPredictor(self.config, self.data),
        }

    def predict_match(
        self, home_team: str, away_team: str, method: str = "zip"
    ) -> MatchPrediction:
        """Predict a single match outcome."""
        if home_team not in self.team_index or away_team not in self.team_index:
            raise ValueError(f"Unknown team(s): {home_team}, {away_team}")

        if method not in self.predictors:
            raise ValueError(f"Unknown prediction method: {method}")

        # Get team indices
        home_idx = self.team_index[home_team]
        away_idx = self.team_index[away_team]

        # Predict expected goals (lambdas)
        lambda_home = self._predict_goals(
            np.array([home_idx]), np.array([away_idx]), is_home=True
        )[0]
        lambda_away = self._predict_goals(
            np.array([away_idx]), np.array([home_idx]), is_home=False
        )[0]

        # Ensure positive lambdas
        lambda_home = max(lambda_home, self.config.lambda_bounds[0])
        lambda_away = max(lambda_away, self.config.lambda_bounds[0])

        # Predict MOV
        raw_mov = lambda_home - lambda_away
        predicted_mov = self.mov_intercept + self.mov_slope * raw_mov

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
            mov_std_error=self.mov_std_error,
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

    def get_regularization_summary(self) -> Dict:
        """Get summary of regularization effects and model diagnostics."""
        if not self.regularization_path:
            return {"error": "No regularization path available"}

        final_reg = self.regularization_path[-1]

        summary = {
            "regularization_config": {
                "l1_reg": self.config.l1_reg,
                "l2_reg": self.config.l2_reg,
                "team_reg": self.config.team_reg,
                "shrink_to_mean": self.config.shrink_to_mean,
            },
            "final_loss_components": final_reg,
            "regularization_ratio": (
                final_reg.get("l1_penalty", 0)
                + final_reg.get("l2_penalty", 0)
                + final_reg.get("team_penalty", 0)
            )
            / max(final_reg.get("base_loss", 1), 1e-10),
            "convergence_info": self.convergence_info,
            "model_complexity": {
                "n_teams": self.n_teams,
                "n_parameters": 2 * self.n_teams + 2,
                "n_matches": len(self.data) if self.data is not None else 0,
            },
        }

        # Add cross-validation results if available
        if self.cv_results is not None:
            best_cv = min(self.cv_results, key=lambda x: x["mean_score"])
            summary["cross_validation"] = {
                "best_score": best_cv["mean_score"],
                "best_params": best_cv["params"],
                "n_param_combinations_tested": len(self.cv_results),
            }

        # Add rating statistics
        if self.attack_ratings is not None:
            summary["rating_statistics"] = {
                "attack_rating_range": (
                    float(np.min(self.attack_ratings)),
                    float(np.max(self.attack_ratings)),
                ),
                "defense_rating_range": (
                    float(np.min(self.defense_ratings)),
                    float(np.max(self.defense_ratings)),
                ),
                "rating_std": float(
                    np.std(np.concatenate([self.attack_ratings, self.defense_ratings]))
                ),
                "home_advantage": float(self.home_advantage),
                "away_adjustment": float(self.away_adjustment),
            }

        return summary

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Stable sigmoid function."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    teams = [f"Team_{i}" for i in range(8)]
    n_matches = 200

    sample_data = []
    for i in range(n_matches):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])

        # Simulate realistic team strengths
        team_strengths = {
            "Team_0": 1.8,
            "Team_1": 1.6,
            "Team_2": 1.4,
            "Team_3": 1.2,
            "Team_4": 1.0,
            "Team_5": 0.9,
            "Team_6": 0.8,
            "Team_7": 0.7,
        }

        home_goals = np.random.poisson(team_strengths[home] * 1.2)  # Add home advantage
        away_goals = np.random.poisson(team_strengths[away])

        sample_data.append(
            {
                "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "Home": home,
                "Away": away,
                "FTHG": home_goals,
                "FTAG": away_goals,
            }
        )

    df = pd.DataFrame(sample_data)

    # Test the model
    config = ModelConfig(
        decay_rate=0.001, l2_reg=0.01, team_reg=0.005, auto_tune_regularization=False
    )

    model = ZSDPoissonModel(config)
    model.fit(df)

    # Make a prediction
    prediction = model.predict_match("Team_0", "Team_7")
    print(f"Prediction: {prediction}")

    # Show team ratings
    print("\nTeam Ratings:")
    print(model.get_team_ratings())

    # Show regularization summary
    print("\nRegularization Summary:")
    print(model.get_regularization_summary())
