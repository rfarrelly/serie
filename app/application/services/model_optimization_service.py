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
    """Enhanced model optimization service with proper validation."""

    def __init__(self, config_dir: Path = Path("zsd_configs")):
        self.config_manager = ConfigManager(config_dir)

    def optimize_league_parameters(
        self, historical_data: pd.DataFrame, league: str, save_config: bool = True
    ) -> ModelConfig:
        """Optimize parameters using proper cross-validation."""
        print(f"Optimizing parameters for {league}...")

        league_data = historical_data[historical_data["League"] == league].copy()
        if len(league_data) < 200:  # Need substantial data for optimization
            print(f"Insufficient data for {league}, using default config")
            return self.config_manager.default_config

        # Generate parameter combinations
        param_combinations = self._generate_smart_param_combinations()

        # Find best configuration through proper backtesting
        best_config = self._find_best_config_via_backtesting(
            league_data, param_combinations, league
        )

        if save_config:
            self.config_manager.save_league_config(league, best_config)

        return best_config

    def _generate_smart_param_combinations(self) -> List[Dict]:
        """Generate smart parameter combinations based on betting literature."""
        # Conservative parameter ranges based on sports betting research
        param_grid = {
            "l1_reg": [0.0, 0.001, 0.005],  # L1 for feature selection
            "l2_reg": [0.001, 0.005, 0.01, 0.02],  # L2 for regularization
            "team_reg": [0.001, 0.005, 0.01],  # Team-specific regularization
            "decay_rate": [0.0002, 0.0005, 0.001, 0.002],  # Temporal decay
        }

        # Generate all combinations
        all_combos = list(itertools.product(*param_grid.values()))

        # Sample intelligently - include extremes and middle values
        if len(all_combos) > 12:  # Limit for computational efficiency
            # Always include default-like parameters
            selected_combos = [
                (0.0, 0.005, 0.005, 0.001),  # Conservative default
                (0.001, 0.01, 0.01, 0.0005),  # Moderate regularization
                (0.0, 0.001, 0.001, 0.002),  # Light regularization
            ]
            # Add random samples from remaining
            remaining = [c for c in all_combos if c not in selected_combos]
            selected_combos.extend(random.sample(remaining, min(9, len(remaining))))
        else:
            selected_combos = all_combos

        return [dict(zip(param_grid.keys(), combo)) for combo in selected_combos]

    def _find_best_config_via_backtesting(
        self, league_data: pd.DataFrame, param_combinations: List[Dict], league: str
    ) -> ModelConfig:
        """Find best configuration using proper temporal validation."""

        best_config = None
        best_score = float("inf")  # Lower is better for our scoring
        best_results = None

        # Setup temporal validation
        seasons = sorted(league_data["Season"].unique())
        if len(seasons) < 2:
            print(f"Insufficient seasons for validation in {league}")
            return self.config_manager.default_config

        # Use the two most recent complete seasons
        train_season = seasons[-2]
        test_season = seasons[-1]

        # Ensure we have enough data in both seasons
        train_data = league_data[league_data["Season"] == train_season]
        test_data = league_data[league_data["Season"] == test_season]

        if len(train_data) < 100 or len(test_data) < 100:
            print(f"Insufficient data per season for {league}")
            return self.config_manager.default_config

        print(
            f"Validation setup: {train_season} ({len(train_data)}) -> {test_season} ({len(test_data)})"
        )

        # Create backtesting service
        backtest_config = BacktestConfig(
            min_training_weeks=10,
            betting_threshold=0.03,  # Conservative threshold
            stake_size=1.0,
            max_stake_fraction=0.05,  # Max 5% of bankroll per bet
        )
        backtesting_service = BacktestingService(backtest_config)

        for i, params in enumerate(param_combinations):
            print(f"  Testing combination {i + 1}/{len(param_combinations)}: {params}")

            try:
                # Create test configuration
                test_config = self._create_test_config(params)

                # Run backtest with proper temporal validation
                summary = backtesting_service.run_cross_season_backtest(
                    model_class=ZSDPoissonModel,
                    data=league_data,
                    train_season=train_season,
                    test_season=test_season,
                    league=league,
                    model_params={"config": test_config},
                )

                # Calculate optimization score (lower is better)
                score = self._calculate_optimization_score(summary)

                print(
                    f"    Score: {score:.4f} (ROI: {summary.roi_percent:.1f}%, Bets: {summary.total_bets})"
                )

                if score < best_score:
                    best_score = score
                    best_config = test_config
                    best_results = summary
                    print(f"    ✅ New best configuration!")

            except Exception as e:
                print(f"    Error testing parameters: {e}")
                continue

        # Log best results
        if best_config and best_results:
            print(f"\n🏆 Best configuration for {league}:")
            print(f"   l1_reg: {best_config.l1_reg}")
            print(f"   l2_reg: {best_config.l2_reg}")
            print(f"   team_reg: {best_config.team_reg}")
            print(f"   decay_rate: {best_config.decay_rate}")
            print(f"   Score: {best_score:.4f}")
            print(f"   ROI: {best_results.roi_percent:.1f}%")
            print(f"   Bets: {best_results.total_bets}")
            print(f"   Win Rate: {best_results.win_rate:.1f}%")

            # Save optimization results
            self._save_optimization_results(best_results, league)

        return best_config or self.config_manager.default_config

    def _create_test_config(self, params: Dict) -> ModelConfig:
        """Create test configuration with validated parameters."""
        base_config = self.config_manager.default_config

        config_dict = {
            "decay_rate": params.get("decay_rate", base_config.decay_rate),
            "max_goals": base_config.max_goals,
            "eps": base_config.eps,
            "lambda_bounds": base_config.lambda_bounds,
            "param_bounds": base_config.param_bounds,
            "min_matches_per_team": base_config.min_matches_per_team,
            "l1_reg": params.get("l1_reg", base_config.l1_reg),
            "l2_reg": params.get("l2_reg", base_config.l2_reg),
            "team_reg": params.get("team_reg", base_config.team_reg),
            "shrink_to_mean": base_config.shrink_to_mean,
            "auto_tune_regularization": False,  # We're doing manual tuning
            "cv_folds": base_config.cv_folds,
            "n_optimization_starts": base_config.n_optimization_starts,
            "max_iter": base_config.max_iter,
        }

        return ModelConfig(**config_dict)

    def _calculate_optimization_score(self, summary) -> float:
        """Calculate optimization score (lower is better)."""
        # Base score components
        roi = summary.roi_percent
        n_bets = summary.total_bets
        win_rate = summary.win_rate

        # Penalties for poor performance
        roi_penalty = 0
        if roi < 0:
            roi_penalty = abs(roi) * 2  # Heavy penalty for negative ROI

        # Penalty for too few bets (insufficient data)
        bet_penalty = max(0, (30 - n_bets) * 0.1) if n_bets < 30 else 0

        # Penalty for unrealistic win rates
        winrate_penalty = 0
        if win_rate > 70:  # Suspicious high win rate
            winrate_penalty = (win_rate - 70) * 0.05
        elif win_rate < 35:  # Very low win rate
            winrate_penalty = (35 - win_rate) * 0.02

        # Reward for positive ROI, penalize for negative
        roi_score = -roi / 100.0  # Convert to decimal, negate so lower is better

        # Final score (lower is better)
        total_score = roi_score + roi_penalty + bet_penalty + winrate_penalty

        return total_score

    def _save_optimization_results(self, summary, league: str):
        """Save optimization results for analysis."""
        # Create directories
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
