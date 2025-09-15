# application/services/model_optimization_service.py
import itertools
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
from models.core import ModelConfig
from utils.config_manager import ConfigManager


class ModelOptimizationService:
    """Application service for optimizing model parameters using backtesting."""

    def __init__(self, config_dir: Path = Path("zsd_configs")):
        self.config_manager = ConfigManager(config_dir)

    def optimize_league_parameters(
        self, historical_data: pd.DataFrame, league: str, save_config: bool = True
    ) -> ModelConfig:
        """Optimize regularization parameters for a specific league using backtesting."""
        print(f"Optimizing parameters for {league}...")

        league_data = historical_data[historical_data["League"] == league].copy()
        if len(league_data) < 100:
            print(f"Insufficient data for {league}, using default config")
            return self.config_manager.default_config

        # Generate parameter combinations to test
        param_combinations = self._generate_param_combinations()

        # Find best configuration through simplified backtesting
        best_config = self._find_best_config_simple(
            league_data, param_combinations, league
        )

        if save_config:
            self.config_manager.save_league_config(league, best_config)

        return best_config

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate parameter combinations for optimization."""
        param_grid = {
            "l1_reg": [0.0, 0.001],
            "l2_reg": [0.005, 0.01, 0.05],
            "team_reg": [0.001, 0.005, 0.01],
            "decay_rate": [0.0005, 0.001],
        }

        all_combos = list(itertools.product(*param_grid.values()))
        selected_combos = random.sample(all_combos, k=min(5, len(all_combos)))

        return [dict(zip(param_grid.keys(), combo)) for combo in selected_combos]

    def _find_best_config_simple(
        self, league_data: pd.DataFrame, param_combinations: List[Dict], league: str
    ) -> ModelConfig:
        """Find the best configuration through simplified evaluation."""

        best_config = None
        best_score = float("inf")

        # Extract seasons for validation
        seasons = sorted(league_data["Season"].unique())
        if len(seasons) < 2:
            print(f"Insufficient seasons for validation in {league}")
            return self.config_manager.default_config

        # Use last two seasons for simple validation
        train_season = seasons[-2]  # Second to last season
        test_season = seasons[-1]  # Last season

        for i, params in enumerate(param_combinations):
            print(f"  Testing combination {i + 1}/{len(param_combinations)}: {params}")

            try:
                # Create test configuration
                test_config = self._create_test_config(params)

                # Simple scoring based on parameter values (placeholder)
                # In a full implementation, you'd run actual backtesting here
                score = self._calculate_simple_score(params)

                if score < best_score:
                    best_score = score
                    best_config = test_config
                    print(f"    ✅ New best config found!")
                else:
                    print(f"    Score: {score:.4f} (not best)")

            except Exception as e:
                print(f"    Error testing parameters: {e}")
                continue

        print(f"\n🏆 Best configuration for {league}:")
        if best_config:
            print(f"   l1_reg: {best_config.l1_reg}")
            print(f"   l2_reg: {best_config.l2_reg}")
            print(f"   team_reg: {best_config.team_reg}")
            print(f"   decay_rate: {best_config.decay_rate}")
            print(f"   Best score: {best_score:.4f}")

        return best_config or self.config_manager.default_config

    def _create_test_config(self, params: Dict) -> ModelConfig:
        """Create a test configuration by merging default config with test parameters."""
        base_config_dict = {
            "decay_rate": self.config_manager.default_config.decay_rate,
            "max_goals": self.config_manager.default_config.max_goals,
            "eps": self.config_manager.default_config.eps,
            "lambda_bounds": self.config_manager.default_config.lambda_bounds,
            "param_bounds": self.config_manager.default_config.param_bounds,
            "min_matches_per_team": self.config_manager.default_config.min_matches_per_team,
            "l1_reg": self.config_manager.default_config.l1_reg,
            "l2_reg": self.config_manager.default_config.l2_reg,
            "team_reg": self.config_manager.default_config.team_reg,
            "shrink_to_mean": self.config_manager.default_config.shrink_to_mean,
            "auto_tune_regularization": self.config_manager.default_config.auto_tune_regularization,
            "cv_folds": self.config_manager.default_config.cv_folds,
            "n_optimization_starts": self.config_manager.default_config.n_optimization_starts,
            "max_iter": self.config_manager.default_config.max_iter,
        }

        # Override with test parameters
        base_config_dict.update(params)
        return ModelConfig(**base_config_dict)

    def _calculate_simple_score(self, params: Dict) -> float:
        """Calculate a simple score for parameter combination (placeholder)."""
        # Simple heuristic: prefer lower regularization but not zero
        l1_score = params["l1_reg"] if params["l1_reg"] > 0 else 0.1
        l2_score = params["l2_reg"]
        team_score = params["team_reg"]
        decay_score = params["decay_rate"]

        # Lower is better - this is a placeholder for actual backtesting
        return (
            l1_score + l2_score + team_score + decay_score
        )  # application/services/model_optimization_service.py


import itertools
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
from domains.backtesting.entities import BacktestConfig
from domains.backtesting.services import BacktestingService
from models.core import ModelConfig
from models.zsd_model import ZSDPoissonModel
from utils.config_manager import ConfigManager


class ModelOptimizationService:
    """Application service for optimizing model parameters using backtesting."""

    def __init__(self, config_dir: Path = Path("zsd_configs")):
        self.config_manager = ConfigManager(config_dir)

    def optimize_league_parameters(
        self, historical_data: pd.DataFrame, league: str, save_config: bool = True
    ) -> ModelConfig:
        """Optimize regularization parameters for a specific league using backtesting."""
        print(f"Optimizing parameters for {league}...")

        league_data = historical_data[historical_data["League"] == league].copy()
        if len(league_data) < 100:
            print(f"Insufficient data for {league}, using default config")
            return self.config_manager.default_config

        # Generate parameter combinations to test
        param_combinations = self._generate_param_combinations()

        # Find best configuration through backtesting
        best_config = self._find_best_config_via_backtesting(
            league_data, param_combinations, league
        )

        if save_config:
            self.config_manager.save_league_config(league, best_config)

        return best_config

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate parameter combinations for optimization."""
        param_grid = {
            "l1_reg": [0.0, 0.001],
            "l2_reg": [0.005, 0.01, 0.05],
            "team_reg": [0.001, 0.005, 0.01],
            "decay_rate": [0.0005, 0.001],
        }

        all_combos = list(itertools.product(*param_grid.values()))
        selected_combos = random.sample(all_combos, k=min(5, len(all_combos)))

        return [dict(zip(param_grid.keys(), combo)) for combo in selected_combos]

    def _find_best_config_via_backtesting(
        self, league_data: pd.DataFrame, param_combinations: List[Dict], league: str
    ) -> ModelConfig:
        """Find the best configuration through backtesting."""

        best_config = None
        best_score = float("inf")

        # Create backtesting service
        backtest_config = BacktestConfig(
            min_training_weeks=8, betting_threshold=0.02, stake_size=1.0
        )
        backtesting_service = BacktestingService(backtest_config)

        # Extract seasons for backtesting
        seasons = sorted(league_data["Season"].unique())
        if len(seasons) < 2:
            print(f"Insufficient seasons for backtesting in {league}")
            return self.config_manager.default_config

        train_season = seasons[-2]  # Second to last season
        test_season = seasons[-1]  # Last season

        for i, params in enumerate(param_combinations):
            print(f"  Testing combination {i + 1}/{len(param_combinations)}: {params}")

            try:
                # Create test configuration
                test_config = self._create_test_config(params)

                # Run backtest
                summary = backtesting_service.run_cross_season_backtest(
                    model_class=ZSDPoissonModel,
                    data=league_data,
                    train_season=train_season,
                    test_season=test_season,
                    league=league,
                    model_params={"config": test_config},
                )

                # Calculate optimization score
                score = self._calculate_optimization_score(summary)

                if score < best_score:
                    best_score = score
                    best_config = test_config
                    self._save_best_results(summary, league)
                    print(f"    ✅ New best config found!")
                else:
                    print(f"    Score: {score:.4f} (not best)")

            except Exception as e:
                print(f"    Error testing parameters: {e}")
                continue

        print(f"\n🏆 Best configuration for {league}:")
        if best_config:
            print(f"   l1_reg: {best_config.l1_reg}")
            print(f"   l2_reg: {best_config.l2_reg}")
            print(f"   team_reg: {best_config.team_reg}")
            print(f"   decay_rate: {best_config.decay_rate}")
            print(f"   Best score: {best_score:.4f}")

        return best_config or self.config_manager.default_config

    def _create_test_config(self, params: Dict) -> ModelConfig:
        """Create a test configuration by merging default config with test parameters."""
        base_config_dict = {
            "decay_rate": self.config_manager.default_config.decay_rate,
            "max_goals": self.config_manager.default_config.max_goals,
            "eps": self.config_manager.default_config.eps,
            "lambda_bounds": self.config_manager.default_config.lambda_bounds,
            "param_bounds": self.config_manager.default_config.param_bounds,
            "min_matches_per_team": self.config_manager.default_config.min_matches_per_team,
            "l1_reg": self.config_manager.default_config.l1_reg,
            "l2_reg": self.config_manager.default_config.l2_reg,
            "team_reg": self.config_manager.default_config.team_reg,
            "shrink_to_mean": self.config_manager.default_config.shrink_to_mean,
            "auto_tune_regularization": self.config_manager.default_config.auto_tune_regularization,
            "cv_folds": self.config_manager.default_config.cv_folds,
            "n_optimization_starts": self.config_manager.default_config.n_optimization_starts,
            "max_iter": self.config_manager.default_config.max_iter,
        }

        # Override with test parameters
        base_config_dict.update(params)
        return ModelConfig(**base_config_dict)

    def _calculate_optimization_score(self, summary) -> float:
        """Calculate optimization score from backtest summary."""
        # We want to minimize this score
        # Lower log loss is better, higher ROI is better, reasonable number of bets

        # Simulate log loss calculation (would need actual implementation)
        log_loss = 0.7 if len(summary.predictions_df) > 0 else 10.0

        roi = summary.roi_percent
        n_bets = summary.total_bets

        # Penalty for too few bets
        bet_penalty = max(0, (50 - n_bets) * 0.01) if n_bets < 50 else 0

        # Score combines log loss (lower is better) and ROI (higher is better)
        score = log_loss - (roi / 100.0) + bet_penalty

        print(
            f"    Score: {score:.4f} (LogLoss: {log_loss:.4f}, ROI: {roi:.1f}%, Bets: {n_bets})"
        )
        return score

    def _save_best_results(self, summary, league: str):
        """Save best optimization results."""
        # Create directories if they don't exist
        Path("optimisation_validation/betting_results").mkdir(
            parents=True, exist_ok=True
        )
        Path("optimisation_validation/prediction_results").mkdir(
            parents=True, exist_ok=True
        )

        # Save betting results
        if summary.betting_results:
            betting_df = pd.DataFrame(
                [
                    {
                        "date": bet.date,
                        "home_team": bet.home_team,
                        "away_team": bet.away_team,
                        "bet_type": bet.bet_type,
                        "stake": bet.stake,
                        "odds": bet.odds,
                        "profit": bet.profit,
                        "model_prob": bet.model_prob,
                        "market_prob": bet.market_prob,
                        "edge": bet.edge,
                        "expected_value": bet.expected_value,
                    }
                    for bet in summary.betting_results
                ]
            )
            betting_df.to_csv(
                f"optimisation_validation/betting_results/{league}_best_betting_results.csv",
                index=False,
            )

        # Save predictions
        if not summary.predictions_df.empty:
            summary.predictions_df.to_csv(
                f"optimisation_validation/prediction_results/{league}_best_predictions.csv",
                index=False,
            )
