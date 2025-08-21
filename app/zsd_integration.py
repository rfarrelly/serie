"""
Simplified ZSD integration layer for main.py compatibility.
"""

import warnings
from typing import Dict, List

import pandas as pd
from analysis.betting import BettingCalculator, BettingFilter
from managers.model_manager import ModelManager

warnings.filterwarnings("ignore")


class ZSDIntegratedProcessor:
    """Simplified processor that integrates ZSD models into existing pipeline."""

    def __init__(self, config):
        self.model_manager = ModelManager(global_config=config)
        self.betting_calculator = BettingCalculator()
        self.betting_filter = BettingFilter(min_edge=0.02)

    def should_reoptimize_parameters(
        self, league: str, days_threshold: int = 90
    ) -> bool:
        """Check if parameters should be re-optimized for a league."""
        return self.model_manager.config_manager.should_reoptimize(
            league, days_threshold
        )

    def optimize_all_league_parameters(self, historical_data: pd.DataFrame):
        """Optimize parameters for all leagues with sufficient data."""
        print("Starting parameter optimization for all leagues...")

        for league in historical_data["League"].unique():
            league_data = historical_data[historical_data["League"] == league]

            if len(league_data) > 179 and self.should_reoptimize_parameters(league):
                self.model_manager.optimize_league_parameters(
                    historical_data, league, save_config=True
                )
            else:
                print(
                    f"Skipping optimization for {league} (insufficient data or recently optimized)"
                )

    def fit_all_models(self, historical_data: pd.DataFrame):
        """Fit ZSD models for all leagues."""
        print("Fitting ZSD models for all leagues...")

        for league in historical_data["League"].unique():
            self.model_manager.fit_league_model(historical_data, league)

    def get_zsd_predictions(self, fixtures_df: pd.DataFrame) -> List[Dict]:
        """Get ZSD predictions for upcoming fixtures with betting analysis."""
        if len(fixtures_df) == 0:
            return []

        # Generate predictions (now includes betting analysis)
        predictions_df = self.model_manager.predict_matches(fixtures_df)
        if len(predictions_df) == 0:
            return []

        print(f"Generated {len(predictions_df)} ZSD predictions")

        # Convert to list of dicts
        all_predictions = predictions_df.to_dict("records")

        # Identify betting candidates based on edge threshold
        betting_candidates = []
        for prediction in all_predictions:
            edge = prediction.get("Edge", 0.0)
            model_prob = prediction.get("Model_Prob", 0.0)
            soft_odds = prediction.get("Soft_Odds", 0.0)

            # Apply betting filter criteria
            is_candidate = (
                edge >= self.betting_filter.min_edge
                and model_prob >= self.betting_filter.min_prob
                and 0 < soft_odds <= self.betting_filter.max_odds
            )

            prediction["Is_Betting_Candidate"] = is_candidate

            if is_candidate:
                betting_candidates.append(prediction)

        print(f"Found {len(betting_candidates)} betting candidates")

        return all_predictions


# Integration functions for main.py compatibility
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
    print(f"{'=' * 60}\r\nRunning daily prediction example...\r\n{'=' * 60}\r\n")
    run_daily_prediction_example()
