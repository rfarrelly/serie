import itertools
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from backtesting import BackTestConfig, ImprovedBacktester
from zsd_poisson_model import ModelConfig, ZSDPoissonModel

warnings.filterwarnings("ignore")


class ZSDModelManager:
    """Manages ZSD Poisson models for multiple leagues with optimized parameters."""

    def __init__(self, global_config, zsd_config_dir: Path = Path("zsd_configs")):
        self.global_config = global_config
        self.config_dir = zsd_config_dir
        self.config_dir.mkdir(exist_ok=True)
        self.models = {}
        self.league_opt_configs = {}
        self.calibrated_models = {}

        # Default fallback configuration
        self.default_config = ModelConfig(
            decay_rate=0.001,
            max_goals=15,
            l1_reg=0.0,
            l2_reg=0.01,
            team_reg=0.005,
            auto_tune_regularization=False,
            min_matches_per_team=5,
        )

    def optimize_league_parameters(
        self,
        historical_data: pd.DataFrame,
        league: str,
        save_config: bool = True,
    ) -> ModelConfig:
        """
        Optimize regularization parameters for a specific league using cross-validation.
        This should be run periodically (e.g., start of each season).
        """
        print(f"Optimizing parameters for {league}...")

        # Filter data for the league
        league_data = historical_data[historical_data["League"] == league].copy()

        if len(league_data) < 100:  # Minimum data requirement
            print(f"Insufficient data for {league}, using default config")
            return self.default_config

        l1 = [0.0, 0.001]
        l2 = [0.005, 0.01, 0.05]
        team = [0.001, 0.005, 0.01]
        decay = [0.0005, 0.001]

        all_combos = list(itertools.product(l1, l2, team, decay))

        all_param_combinations = [
            {"l1_reg": l1, "l2_reg": l2, "team_reg": team, "decay_rate": decay}
            for (l1, l2, team, decay) in all_combos
        ]

        # (Optional) Random sample if you want to reduce size
        param_combinations = random.sample(all_param_combinations, k=20)

        # Parameter grid for optimization
        # param_combinations = [
        #     {"l1_reg": 0.0, "l2_reg": 0.005, "team_reg": 0.001, "decay_rate": 0.0005},
        #     {"l1_reg": 0.0, "l2_reg": 0.01, "team_reg": 0.005, "decay_rate": 0.001},
        #     {"l1_reg": 0.0, "l2_reg": 0.02, "team_reg": 0.01, "decay_rate": 0.001},
        #     {"l1_reg": 0.001, "l2_reg": 0.01, "team_reg": 0.005, "decay_rate": 0.001},
        #     {"l1_reg": 0.0, "l2_reg": 0.05, "team_reg": 0.01, "decay_rate": 0.002},
        #     {"l1_reg": 0.0, "l2_reg": 0.1, "team_reg": 0.01, "decay_rate": 0.001},
        # ]

        best_config = None
        best_score = float("inf")
        best_results = None

        # Set up backtesting for parameter optimization
        backtest_config = BackTestConfig(
            min_training_weeks=8, betting_threshold=0.02, stake_size=1.0
        )

        backtester = ImprovedBacktester(backtest_config)

        for i, params in enumerate(param_combinations):
            # print(
            #     f"  Testing combination {i+1}/{len(param_combinations)}: {params}"
            # )

            try:
                # Create model config
                test_config = ModelConfig(**{**self.default_config.__dict__, **params})

                # Create model factory
                def model_factory(**kwargs):
                    return ZSDPoissonModel(test_config)

                # Run backtest
                results = backtester.backtest_cross_season(
                    data=league_data,
                    model_class=model_factory,
                    train_season=self.global_config.previous_season,
                    test_season=self.global_config.current_season,
                    league=league,
                )

                # Score based on combined metric (log loss + betting performance)
                log_loss = results.metrics.get("log_loss", 10.0)
                roi = results.metrics.get("roi_percent", -100.0)
                n_bets = results.metrics.get("total_bets", 0)

                # Penalize configs that don't generate enough bets
                bet_penalty = max(0, (50 - n_bets) * 0.01) if n_bets < 50 else 0

                # Combined score (lower is better)
                score = log_loss - (roi / 100.0) + bet_penalty

                # print(
                #     f"    Score: {score:.4f} (LogLoss: {log_loss:.4f}, ROI: {roi:.1f}%, Bets: {n_bets})"
                # )

                if score < best_score:
                    best_score = score
                    best_config = test_config
                    best_results = results

            except Exception as e:
                print(f"    Error testing parameters: {e}")
                continue

        if best_config is None:
            print(f"  No successful parameter optimization for {league}, using default")
            best_config = self.default_config
        else:
            print(
                f"  Best config for {league}: L1={best_config.l1_reg:.4f}, L2={best_config.l2_reg:.4f}, "
                f"Team={best_config.team_reg:.4f}, Decay={best_config.decay_rate:.4f}"
            )
            print(f"  Best score: {best_score:.4f}")

            if best_results:
                print(
                    f"  Performance: {best_results.metrics.get('accuracy', 0):.3f} accuracy, "
                    f"{best_results.metrics.get('roi_percent', 0):.1f}% ROI"
                )

        # Save configuration
        if save_config:
            self.league_opt_configs[league] = best_config
            self._save_league_config(league, best_config)

        return best_config

    def _save_league_config(self, league: str, config: ModelConfig):
        """Save league-specific configuration to file."""
        config_file = (
            self.config_dir
            / f"{league.replace(' ', '_').replace('-', '_')}_config.json"
        )

        config_dict = {
            "decay_rate": config.decay_rate,
            "max_goals": config.max_goals,
            "l1_reg": config.l1_reg,
            "l2_reg": config.l2_reg,
            "team_reg": config.team_reg,
            "min_matches_per_team": config.min_matches_per_team,
            "auto_tune_regularization": config.auto_tune_regularization,
            "optimized_date": datetime.now().isoformat(),
        }

        import json

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    def load_league_config(self, league: str) -> ModelConfig:
        """Load league-specific configuration from file."""
        config_file = (
            self.config_dir
            / f"{league.replace(' ', '_').replace('-', '_')}_config.json"
        )

        if not config_file.exists():
            print(f"No saved config for {league}, using default")
            return self.default_config

        try:
            import json

            with open(config_file, "r") as f:
                config_dict = json.load(f)

            # Remove non-ModelConfig fields
            config_dict.pop("optimized_date", None)

            return ModelConfig(**config_dict)

        except Exception as e:
            print(f"Error loading config for {league}: {e}, using default")
            return self.default_config

    def fit_league_model(
        self,
        historical_data: pd.DataFrame,
        league: str,
        min_training_matches: int = 200,
    ) -> Optional[ZSDPoissonModel]:
        """
        Fit a ZSD Poisson model for a specific league using optimized parameters.
        """
        print(f"Fitting ZSD model for {league}...")

        # Filter and prepare data
        league_data = historical_data[historical_data["League"] == league].copy()
        league_data = league_data.sort_values("Date").reset_index(drop=True)

        if len(league_data) < min_training_matches:
            print(
                f"Insufficient data for {league}: {len(league_data)} matches (need {min_training_matches})"
            )
            return None

        # Load or use default configuration
        config = self.league_opt_configs.get(league) or self.load_league_config(league)

        # Create and fit model
        try:
            model = ZSDPoissonModel(config)
            model.fit(league_data)

            # Store the fitted model
            self.models[league] = model

            print(f"  Successfully fitted model for {league}")
            print(f"  Training data: {len(league_data)} matches")
            print(f"  Teams: {len(model.teams)}")
            print(f"  Convergence: {model.convergence_info.get('success', 'Unknown')}")

            return model

        except Exception as e:
            print(f"Error fitting model for {league}: {e}")
            return None

    def predict_matches(
        self, fixtures_df: pd.DataFrame, method: str = "zip"
    ) -> pd.DataFrame:
        """
        Generate predictions for upcoming matches using fitted models.
        """
        predictions = []

        for _, match in fixtures_df.iterrows():
            league = match.get("League")
            home_team = match["Home"]
            away_team = match["Away"]

            # Get the appropriate model
            model = self.models.get(league)

            if model is None:
                print(f"No model available for {league}")
                continue

            try:
                # Generate prediction
                prediction = model.predict_match(home_team, away_team, method=method)

                # Convert to dictionary and add match info
                pred_dict = {
                    "Date": match["Date"],
                    "League": league,
                    "Home": home_team,
                    "Away": away_team,
                    "ZSD_Prob_H": prediction.prob_home_win,
                    "ZSD_Prob_D": prediction.prob_draw,
                    "ZSD_Prob_A": prediction.prob_away_win,
                    "ZSD_Lambda_H": prediction.lambda_home,
                    "ZSD_Lambda_A": prediction.lambda_away,
                    "ZSD_MOV": prediction.mov_prediction,
                    "ZSD_MOV_SE": prediction.mov_std_error,
                    "Model_Type": prediction.model_type,
                }

                # Add any additional columns from the original fixture
                for col in ["Wk", "PSH", "PSD", "PSA", "PPI_Diff", "hPPI", "aPPI"]:
                    if col in match.index:
                        pred_dict[col] = match[col]

                predictions.append(pred_dict)

            except Exception as e:
                print(f"Error predicting {home_team} vs {away_team} in {league}: {e}")
                continue

        return pd.DataFrame(predictions)

    def get_betting_candidates(
        self,
        predictions_df: pd.DataFrame,
        edge_threshold: float = 0.02,
        max_candidates: int = 20,
    ) -> pd.DataFrame:
        """
        Identify betting candidates based on model predictions vs market odds.
        """
        if len(predictions_df) == 0:
            return pd.DataFrame()

        # Calculate fair probabilities from odds if available
        betting_candidates = []

        for _, pred in predictions_df.iterrows():
            odds_cols = ["PSH", "PSD", "PSA"]

            # Skip if odds not available
            if any(pd.isna(pred.get(col)) for col in odds_cols):
                continue

            odds = [pred["PSH"], pred["PSD"], pred["PSA"]]
            model_probs = [pred["ZSD_Prob_H"], pred["ZSD_Prob_D"], pred["ZSD_Prob_A"]]

            # Calculate implied probabilities (simplified, you might want to use your odds helpers)
            implied_probs = [1 / odd for odd in odds]
            total_implied = sum(implied_probs)
            fair_probs = [p / total_implied for p in implied_probs]  # Remove vig

            # Calculate edges
            edges = [model_probs[i] - fair_probs[i] for i in range(3)]
            max_edge = max(edges)

            if max_edge >= edge_threshold:
                bet_idx = edges.index(max_edge)
                bet_types = ["Home", "Draw", "Away"]

                candidate = {
                    **pred.to_dict(),
                    "Bet_Type": bet_types[bet_idx],
                    "Edge": max_edge,
                    "Model_Prob": model_probs[bet_idx],
                    "Fair_Prob": fair_probs[bet_idx],
                    "Odds": odds[bet_idx],
                }

                betting_candidates.append(candidate)

        # Sort by edge and return top candidates
        candidates_df = pd.DataFrame(betting_candidates)
        if len(candidates_df) > 0:
            candidates_df = candidates_df.sort_values("Edge", ascending=False).head(
                max_candidates
            )

        return candidates_df


class ZSDIntegratedProcessor:
    """
    Integrates ZSD Poisson model into your existing pipeline.
    """

    def __init__(self, config):
        self.zsd_manager = ZSDModelManager(global_config=config)
        self.last_optimization_date = {}

    def should_reoptimize_parameters(
        self, league: str, days_threshold: int = 90
    ) -> bool:
        """
        Check if parameters should be re-optimized for a league.
        """
        config_file = (
            self.zsd_manager.config_dir
            / f"{league.replace(' ', '_').replace('-', '_')}_config.json"
        )

        if not config_file.exists():
            return True

        try:
            import json

            with open(config_file, "r") as f:
                config_dict = json.load(f)

            if "optimized_date" not in config_dict:
                return True

            opt_date = datetime.fromisoformat(config_dict["optimized_date"])
            days_since = (datetime.now() - opt_date).days

            return days_since > days_threshold

        except Exception:
            return True

    def optimize_all_league_parameters(self, historical_data: pd.DataFrame):
        """
        Optimize parameters for all leagues with sufficient data.
        Run this periodically (e.g., monthly or start of season).
        """
        print("Starting parameter optimization for all leagues...")

        available_leagues = historical_data["League"].unique()

        for league in available_leagues:
            league_data = historical_data[historical_data["League"] == league]

            # Only optimize if we have sufficient data and it's been a while
            if len(league_data) > 200 and self.should_reoptimize_parameters(league):
                self.zsd_manager.optimize_league_parameters(
                    historical_data=historical_data,
                    league=league,
                    save_config=True,
                )
            else:
                print(
                    f"Skipping optimization for {league} (insufficient data or recently optimized)"
                )

    def fit_all_models(self, historical_data: pd.DataFrame):
        """
        Fit ZSD models for all leagues.
        Run this before generating predictions.
        """
        print("Fitting ZSD models for all leagues...")

        available_leagues = historical_data["League"].unique()

        for league in available_leagues:
            self.zsd_manager.fit_league_model(historical_data, league)

    def get_zsd_predictions(self, fixtures_df: pd.DataFrame) -> List[Dict]:
        """
        Get ZSD Poisson predictions for upcoming fixtures.
        This replaces/augments your existing get_zsd_poisson method.
        """
        if len(fixtures_df) == 0:
            return []

        # Generate predictions
        predictions_df = self.zsd_manager.predict_matches(fixtures_df)

        if len(predictions_df) == 0:
            return []

        # Get betting candidates
        betting_candidates = self.zsd_manager.get_betting_candidates(
            predictions_df,
            edge_threshold=0.015,  # Slightly lower threshold for more candidates
            max_candidates=50,
        )

        print(f"Generated {len(predictions_df)} ZSD predictions")
        print(f"Found {len(betting_candidates)} betting candidates")

        # Return all predictions, but mark betting candidates
        all_predictions = predictions_df.to_dict("records")

        # Add betting flag
        candidate_matches = set()
        if len(betting_candidates) > 0:
            for _, candidate in betting_candidates.iterrows():
                match_id = (
                    f"{candidate['Date']}_{candidate['Home']}_{candidate['Away']}"
                )
                candidate_matches.add(match_id)

        for pred in all_predictions:
            match_id = f"{pred['Date']}_{pred['Home']}_{pred['Away']}"
            pred["Is_Betting_Candidate"] = match_id in candidate_matches

            # Add edge info if it's a betting candidate
            if pred["Is_Betting_Candidate"]:
                candidate_row = betting_candidates[
                    (betting_candidates["Home"] == pred["Home"])
                    & (betting_candidates["Away"] == pred["Away"])
                    & (betting_candidates["Date"] == pred["Date"])
                ]
                if len(candidate_row) > 0:
                    pred["Edge"] = candidate_row.iloc[0]["Edge"]
                    pred["Bet_Type"] = candidate_row.iloc[0]["Bet_Type"]

        return all_predictions


# Integration functions for your main.py


def setup_zsd_integration(config):
    """
    Initialize ZSD integration. Call this once when setting up your pipeline.
    """
    return ZSDIntegratedProcessor(config)


def periodic_parameter_optimization(processor: ZSDIntegratedProcessor):
    """
    Run parameter optimization. Call this periodically (e.g., monthly).
    """
    try:
        # Load historical data
        historical_data = pd.read_csv("historical_ppi_and_odds.csv")

        # Optimize parameters for all leagues
        processor.optimize_all_league_parameters(historical_data)

        print("Parameter optimization completed successfully")

    except Exception as e:
        print(f"Error in parameter optimization: {e}")


def daily_model_fitting(processor: ZSDIntegratedProcessor):
    """
    Fit models with latest data. Call this daily or when you want fresh predictions.
    """
    try:
        # Load historical data (including any new results)
        historical_data = pd.read_csv("historical_ppi_and_odds.csv")

        # Fit all models with latest data
        processor.fit_all_models(historical_data)

        print("Model fitting completed successfully")

    except Exception as e:
        print(f"Error in model fitting: {e}")


def generate_zsd_predictions(
    processor: ZSDIntegratedProcessor, fixtures_df: pd.DataFrame
) -> List[Dict]:
    """
    Generate ZSD predictions for upcoming fixtures.
    """
    try:
        return processor.get_zsd_predictions(fixtures_df)

    except Exception as e:
        print(f"Error generating ZSD predictions: {e}")
        return []


# Example usage patterns:


def run_parameter_optimization_example():
    """
    Example of how to run parameter optimization (do this monthly/seasonally).
    """
    from config import DEFAULT_CONFIG

    # Setup
    processor = setup_zsd_integration(DEFAULT_CONFIG)

    # Run optimization
    periodic_parameter_optimization(processor)


def run_daily_prediction_example():
    """
    Example of daily prediction workflow.
    """
    from config import DEFAULT_CONFIG

    # Setup
    processor = setup_zsd_integration(DEFAULT_CONFIG)

    # Fit models with latest data
    daily_model_fitting(processor)

    # Load upcoming fixtures (your existing pipeline data)
    try:
        fixtures_df = pd.read_csv("latest_ppi_and_odds.csv")

        # Generate ZSD predictions
        zsd_predictions = generate_zsd_predictions(processor, fixtures_df)

        # Save results
        if zsd_predictions:
            zsd_df = pd.DataFrame(zsd_predictions)
            zsd_df.to_csv("latest_zsd_enhanced.csv", index=False)

            # Show betting candidates
            candidates = zsd_df[zsd_df["Is_Betting_Candidate"] == True]
            if len(candidates) > 0:
                print(f"\nFound {len(candidates)} ZSD betting candidates:")
                for _, candidate in candidates.head(10).iterrows():
                    print(
                        f"  {candidate['Home']} vs {candidate['Away']} ({candidate['League']})"
                    )
                    print(
                        f"    Bet: {candidate.get('Bet_Type', 'N/A')}, Edge: {candidate.get('Edge', 0):.3f}"
                    )
                    print(
                        f"    ZSD Probs: H={candidate['ZSD_Prob_H']:.3f}, D={candidate['ZSD_Prob_D']:.3f}, A={candidate['ZSD_Prob_A']:.3f}"
                    )
            else:
                print("No ZSD betting candidates found")

    except FileNotFoundError:
        print("No fixtures file found")


if __name__ == "__main__":
    # Run parameter optimization (do this periodically)
    print("Running parameter optimization example...")
    run_parameter_optimization_example()

    print("\n" + "=" * 50)

    # Run daily prediction example
    print("Running daily prediction example...")
    run_daily_prediction_example()
