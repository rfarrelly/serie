import itertools
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from backtesting import BackTestConfig, ImprovedBacktester
from utils.odds_helpers import get_no_vig_odds_multiway
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
        """Optimize regularization parameters for a specific league using cross-validation."""
        print(f"Optimizing parameters for {league}...")

        league_data = historical_data[historical_data["League"] == league].copy()
        if len(league_data) < 100:
            print(f"Insufficient data for {league}, using default config")
            return self.default_config

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()

        best_config = self._find_best_config(league_data, param_combinations, league)

        if save_config:
            self.league_opt_configs[league] = best_config
            self._save_league_config(league, best_config)

        return best_config

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate parameter combinations for optimization."""
        l1 = [0.0, 0.001]
        l2 = [0.005, 0.01, 0.05]
        team = [0.001, 0.005, 0.01]
        decay = [0.0005, 0.001]

        all_combos = list(itertools.product(l1, l2, team, decay))
        return random.sample(
            [
                {"l1_reg": l1, "l2_reg": l2, "team_reg": team, "decay_rate": decay}
                for (l1, l2, team, decay) in all_combos
            ],
            k=5,
        )

    def _find_best_config(
        self, league_data: pd.DataFrame, param_combinations: List[Dict], league: str
    ) -> ModelConfig:
        """Find the best configuration through backtesting."""
        best_config, best_score = None, float("inf")

        backtest_config = BackTestConfig(
            min_training_weeks=8, betting_threshold=0.02, stake_size=1.0
        )
        backtester = ImprovedBacktester(backtest_config)

        for i, params in enumerate(param_combinations):
            print(f"  Testing combination {i + 1}/{len(param_combinations)}: {params}")

            try:
                test_config = ModelConfig(**{**self.default_config.__dict__, **params})
                model_factory = lambda **kwargs: ZSDPoissonModel(test_config)

                results = backtester.backtest_cross_season(
                    data=league_data,
                    model_class=model_factory,
                    train_season=self.global_config.previous_season,
                    test_season=self.global_config.current_season,
                    league=league,
                )

                score = self._calculate_score(results)

                if score < best_score:
                    best_score = score
                    best_config = test_config
                    self._save_best_results(results, league)

            except Exception as e:
                print(f"    Error testing parameters: {e}")
                continue

        return best_config or self.default_config

    def _calculate_score(self, results) -> float:
        """Calculate optimization score from backtest results."""
        log_loss = results.metrics.get("log_loss", 10.0)
        roi = results.metrics.get("roi_percent", -100.0)
        n_bets = results.metrics.get("total_bets", 0)

        bet_penalty = max(0, (50 - n_bets) * 0.01) if n_bets < 50 else 0
        score = log_loss - (roi / 100.0) + bet_penalty

        print(
            f"    Score: {score:.4f} (LogLoss: {log_loss:.4f}, ROI: {roi:.1f}%, Bets: {n_bets})"
        )
        return score

    def _save_best_results(self, results, league: str):
        """Save best optimization results to CSV files."""
        backtester = ImprovedBacktester(BackTestConfig())
        backtester.save_betting_results_to_csv(
            results,
            f"optimisation_validation/betting_results/{league}_best_betting_results.csv",
        )
        backtester.save_predictions_to_csv(
            results,
            f"optimisation_validation/prediction_results/{league}_best_predictions.csv",
        )

    def _save_league_config(self, league: str, config: ModelConfig):
        """Save league-specific configuration to file."""
        config_file = (
            self.config_dir
            / f"{league.replace(' ', '_').replace('-', '_')}_config.json"
        )
        config_dict = {
            **{k: v for k, v in config.__dict__.items() if not k.startswith("_")},
            "optimized_date": datetime.now().isoformat(),
        }

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
            with open(config_file, "r") as f:
                config_dict = json.load(f)

            config_dict.pop("optimized_date", None)
            return ModelConfig(**config_dict)

        except Exception as e:
            print(f"Error loading config for {league}: {e}, using default")
            return self.default_config

    def fit_league_model(
        self,
        historical_data: pd.DataFrame,
        league: str,
        min_training_matches: int = 180,
    ) -> Optional[ZSDPoissonModel]:
        """Fit a ZSD Poisson model for a specific league using optimized parameters."""
        print(f"Fitting ZSD model for {league}...")

        league_data = historical_data[historical_data["League"] == league].copy()
        league_data = league_data.sort_values("Date").reset_index(drop=True)

        if len(league_data) < min_training_matches:
            print(
                f"Insufficient data for {league}: {len(league_data)} matches (need {min_training_matches})"
            )
            return None

        config = self.league_opt_configs.get(league) or self.load_league_config(league)

        try:
            model = ZSDPoissonModel(config)
            model.fit(league_data)
            self.models[league] = model

            print(f"  Successfully fitted model for {league}")
            print(
                f"  Training data: {len(league_data)} matches, Teams: {len(model.teams)}"
            )
            print(f"  Convergence: {model.convergence_info.get('success', 'Unknown')}")
            return model

        except Exception as e:
            print(f"Error fitting model for {league}: {e}")
            return None

    def predict_matches(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for upcoming matches using fitted models."""
        predictions = []

        for _, match in fixtures_df.iterrows():
            league = match.get("League")
            model = self.models.get(league)

            if model is None:
                print(f"No model available for {league}")
                continue

            try:
                pred_dict = self._generate_match_predictions(match, model)
                predictions.append(pred_dict)

            except Exception as e:
                print(
                    f"Error predicting {match['Home']} vs {match['Away']} in {league}: {e}"
                )
                continue

        return pd.DataFrame(predictions)

    def _generate_match_predictions(self, match, model) -> Dict:
        """Generate all prediction types for a single match."""
        home_team, away_team = match["Home"], match["Away"]

        # Get predictions for all methods
        methods = ["poisson", "zip", "mov"]
        predictions = {
            method: model.predict_match(home_team, away_team, method=method)
            for method in methods
        }

        # Build prediction dictionary
        pred_dict = {
            "Date": match["Date"],
            "League": match.get("League"),
            "Home": home_team,
            "Away": away_team,
        }

        # Add predictions for each method
        method_names = ["Poisson", "ZIP", "MOV"]
        for method, name in zip(methods, method_names):
            pred = predictions[method]
            pred_dict.update(
                {
                    f"{name}_Prob_H": pred.prob_home_win,
                    f"{name}_Prob_D": pred.prob_draw,
                    f"{name}_Prob_A": pred.prob_away_win,
                }
            )

        # Add backward compatibility fields
        zip_pred = predictions["zip"]
        pred_dict.update(
            {
                "ZSD_Prob_H": zip_pred.prob_home_win,
                "ZSD_Prob_D": zip_pred.prob_draw,
                "ZSD_Prob_A": zip_pred.prob_away_win,
                "ZSD_Lambda_H": zip_pred.lambda_home,
                "ZSD_Lambda_A": zip_pred.lambda_away,
                "ZSD_MOV": zip_pred.mov_prediction,
                "ZSD_MOV_SE": zip_pred.mov_std_error,
                "Model_Type": zip_pred.model_type,
            }
        )

        # Add additional columns from fixture
        for col in ["Wk", "PSH", "PSD", "PSA", "PPI_Diff", "hPPI", "aPPI"]:
            if col in match.index:
                pred_dict[col] = match[col]

        return pred_dict

    def get_betting_candidates(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Identify betting candidates based on model predictions vs market odds."""
        if len(predictions_df) == 0:
            return pd.DataFrame()

        betting_candidates = []

        for _, pred in predictions_df.iterrows():
            odds_cols = ["PSH", "PSD", "PSA"]
            if any(pd.isna(pred.get(col)) for col in odds_cols):
                continue

            try:
                candidate = self._calculate_betting_metrics(pred)
                if candidate:
                    betting_candidates.append(candidate)
            except Exception as e:
                print(
                    f"Error calculating betting metrics for {pred['Home']} vs {pred['Away']}: {e}"
                )
                continue

        return pd.DataFrame(betting_candidates)

    def _calculate_betting_metrics(self, pred) -> Optional[Dict]:
        """Calculate all betting metrics for a single prediction."""
        market_odds = [pred["PSH"], pred["PSD"], pred["PSA"]]

        # Get model probabilities
        prob_sets = {
            "poisson": [
                pred["Poisson_Prob_H"],
                pred["Poisson_Prob_D"],
                pred["Poisson_Prob_A"],
            ],
            "zip": [pred["ZIP_Prob_H"], pred["ZIP_Prob_D"], pred["ZIP_Prob_A"]],
            "mov": [pred["MOV_Prob_H"], pred["MOV_Prob_D"], pred["MOV_Prob_A"]],
        }

        # Calculate no-vig and model averages
        no_vig_odds = get_no_vig_odds_multiway(market_odds)
        no_vig_probs = [1 / odd for odd in no_vig_odds]
        model_avg_probs = [
            (sum(prob_sets[m][i] for m in prob_sets) / 3) for i in range(3)
        ]
        weighted_probs = [
            model_avg_probs[i] * 0.1 + no_vig_probs[i] * 0.9 for i in range(3)
        ]

        # Calculate edges
        edges = self._calculate_all_edges(weighted_probs, no_vig_probs, market_odds)
        max_edge = max(edges["expected_values"])
        bet_idx = edges["expected_values"].index(max_edge)

        # Build candidate dictionary
        candidate = pred.to_dict()
        candidate.update(
            self._build_betting_dict(
                edges,
                no_vig_odds,
                no_vig_probs,
                model_avg_probs,
                weighted_probs,
                market_odds,
                bet_idx,
            )
        )

        return candidate

    def _calculate_all_edges(self, weighted_probs, no_vig_probs, market_odds) -> Dict:
        """Calculate all types of edges."""
        return {
            "probability_edges": [
                weighted_probs[i] - no_vig_probs[i] for i in range(3)
            ],
            "expected_values": [
                (weighted_probs[i] * market_odds[i]) - 1 for i in range(3)
            ],
            "kelly_edges": [
                (
                    (weighted_probs[i] * market_odds[i] - 1) / (market_odds[i] - 1)
                    if market_odds[i] > 1
                    else 0
                )
                for i in range(3)
            ],
        }

    def _build_betting_dict(
        self,
        edges,
        no_vig_odds,
        no_vig_probs,
        model_avg_probs,
        weighted_probs,
        market_odds,
        bet_idx,
    ) -> Dict:
        """Build the betting metrics dictionary."""
        bet_types = ["Home", "Draw", "Away"]
        fair_odds = [1 / prob for prob in weighted_probs]

        return {
            "Bet_Type": bet_types[bet_idx],
            "Edge": edges["expected_values"][bet_idx],
            # Edge calculations
            **{
                f"EV_{t[0]}": edges["expected_values"][i]
                for i, t in enumerate(bet_types)
            },
            **{
                f"Prob_Edge_{t[0]}": edges["probability_edges"][i]
                for i, t in enumerate(bet_types)
            },
            **{
                f"Kelly_{t[0]}": edges["kelly_edges"][i]
                for i, t in enumerate(bet_types)
            },
            # Market calculations
            **{f"NoVig_Odds_{t[0]}": no_vig_odds[i] for i, t in enumerate(bet_types)},
            **{f"NoVig_Prob_{t[0]}": no_vig_probs[i] for i, t in enumerate(bet_types)},
            **{
                f"ModelAvg_Prob_{t[0]}": model_avg_probs[i]
                for i, t in enumerate(bet_types)
            },
            **{
                f"Weighted_Prob_{t[0]}": weighted_probs[i]
                for i, t in enumerate(bet_types)
            },
            **{f"Fair_Odds_{t[0]}": fair_odds[i] for i, t in enumerate(bet_types)},
            # Selected bet details
            "Model_Prob": weighted_probs[bet_idx],
            "Market_Prob": no_vig_probs[bet_idx],
            "Market_Odds": market_odds[bet_idx],
            "Fair_Odds_Selected": fair_odds[bet_idx],
        }


class ZSDIntegratedProcessor:
    """Integrates ZSD Poisson model into your existing pipeline."""

    def __init__(self, config):
        self.zsd_manager = ZSDModelManager(global_config=config)

    def should_reoptimize_parameters(
        self, league: str, days_threshold: int = 90
    ) -> bool:
        """Check if parameters should be re-optimized for a league."""
        config_file = (
            self.zsd_manager.config_dir
            / f"{league.replace(' ', '_').replace('-', '_')}_config.json"
        )

        if not config_file.exists():
            return True

        try:
            with open(config_file, "r") as f:
                config_dict = json.load(f)

            if "optimized_date" not in config_dict:
                return True

            opt_date = datetime.fromisoformat(config_dict["optimized_date"])
            return (datetime.now() - opt_date).days > days_threshold

        except Exception:
            return True

    def optimize_all_league_parameters(self, historical_data: pd.DataFrame):
        """Optimize parameters for all leagues with sufficient data."""
        print("Starting parameter optimization for all leagues...")

        for league in ["Belgian-Pro-League"]:  # historical_data["League"].unique()
            league_data = historical_data[historical_data["League"] == league]

            if len(league_data) > 179 and self.should_reoptimize_parameters(league):
                self.zsd_manager.optimize_league_parameters(
                    historical_data, league, save_config=True
                )
            else:
                print(
                    f"Skipping optimization for {league} (insufficient data or recently optimized)"
                )

    def fit_all_models(self, historical_data: pd.DataFrame):
        """Fit ZSD models for all leagues."""
        print("Fitting ZSD models for all leagues...")

        for league in ["Belgian-Pro-League"]:  # historical_data["League"].unique()
            self.zsd_manager.fit_league_model(historical_data, league)

    def get_zsd_predictions(self, fixtures_df: pd.DataFrame) -> List[Dict]:
        """Get ZSD Poisson predictions for upcoming fixtures with enhanced features."""
        if len(fixtures_df) == 0:
            return []

        predictions_df = self.zsd_manager.predict_matches(fixtures_df)
        if len(predictions_df) == 0:
            return []

        betting_candidates = self.zsd_manager.get_betting_candidates(predictions_df)

        print(f"Generated {len(predictions_df)} ZSD predictions")
        print(f"Found {len(betting_candidates)} betting candidates")

        all_betting_candidates = betting_candidates.to_dict("records")

        # Add betting flag
        candidate_matches = {
            f"{pred['Date']}_{pred['Home']}_{pred['Away']}"
            for _, pred in predictions_df.iterrows()
        }

        for candidate in all_betting_candidates:
            match_id = f"{candidate['Date']}_{candidate['Home']}_{candidate['Away']}"
            candidate["Is_Betting_Candidate"] = match_id in candidate_matches

        return all_betting_candidates


# Integration functions for main.py - keeping original function signatures
def setup_zsd_integration(config):
    """Initialize ZSD integration."""
    return ZSDIntegratedProcessor(config)


def periodic_parameter_optimization(processor: ZSDIntegratedProcessor):
    """Run parameter optimization."""
    try:
        historical_data = pd.read_csv("historical_ppi_and_odds.csv")
        processor.optimize_all_league_parameters(historical_data)
        print("Parameter optimization completed successfully")
    except Exception as e:
        print(f"Error in parameter optimization: {e}")


def daily_model_fitting(processor: ZSDIntegratedProcessor):
    """Fit models with latest data."""
    try:
        historical_data = pd.read_csv("historical_ppi_and_odds.csv")
        processor.fit_all_models(historical_data)
        print("Model fitting completed successfully")
    except Exception as e:
        print(f"Error in model fitting: {e}")


def generate_zsd_predictions(
    processor: ZSDIntegratedProcessor, fixtures_df: pd.DataFrame
) -> List[Dict]:
    """Generate ZSD predictions for upcoming fixtures."""
    try:
        return processor.get_zsd_predictions(fixtures_df)
    except Exception as e:
        print(f"Error generating ZSD predictions: {e}")
        return []


# Example usage functions
def run_parameter_optimization_example():
    """Example of how to run parameter optimization."""
    from config import DEFAULT_CONFIG

    processor = setup_zsd_integration(DEFAULT_CONFIG)
    periodic_parameter_optimization(processor)


def run_daily_prediction_example():
    """Example of daily prediction workflow."""
    from config import DEFAULT_CONFIG

    processor = setup_zsd_integration(DEFAULT_CONFIG)
    daily_model_fitting(processor)

    try:
        fixtures_df = pd.read_csv("fixtures_ppi_and_odds.csv")
        zsd_predictions = generate_zsd_predictions(processor, fixtures_df)

        if zsd_predictions:
            zsd_df = pd.DataFrame(zsd_predictions)
            zsd_df.to_csv("latest_zsd_enhanced.csv", index=False)

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
            else:
                print("No ZSD betting candidates found")

    except FileNotFoundError:
        print("No fixtures file found")


if __name__ == "__main__":
    print("Running parameter optimization example...")
    run_parameter_optimization_example()
    print("\n" + "=" * 50)
    print("Running daily prediction example...")
    run_daily_prediction_example()
