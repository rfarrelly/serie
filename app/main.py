import sys
from pathlib import Path

from application.services.league_processing_service import LeagueProcessingService
from application.services.pipeline_orchestrator import PipelineOrchestrator
from application.use_cases.calculate_betting_edges import CalculateBettingEdgesUseCase
from application.use_cases.generate_predictions import GeneratePredictionsUseCase
from application.use_cases.ingest_data import IngestDataUseCase
from application.use_cases.optimize_models import OptimizeModelsUseCase
from backtesting_validator import validate_zsd_backtest_results
from config import DEFAULT_CONFIG, AppConfig, Leagues
from domains.betting.services import BettingAnalysisService
from domains.data.services import PPICalculationService
from domains.predictions.entities import Prediction
from domains.shared.value_objects import Probabilities
from infrastructure.adapters.file_adapter import CSVFileAdapter
from infrastructure.repositories.csv_betting_repository import CSVBettingRepository
from infrastructure.repositories.csv_match_repository import CSVMatchRepository
from infrastructure.repositories.csv_prediction_repository import (
    CSVPredictionRepository,
)
from managers.model_manager import ModelManager
from utils.data_merging import merge_future_odds_data, merge_historical_odds_data
from utils.team_name_dict_builder import build_team_name_dictionary


class DependencyContainer:
    """Simple dependency injection container"""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

        # Infrastructure
        self.file_adapter = CSVFileAdapter()

        # Repositories
        self.match_repository = CSVMatchRepository(self.file_adapter, self.config)
        self.prediction_repository = CSVPredictionRepository(self.file_adapter)
        self.betting_repository = CSVBettingRepository(self.file_adapter)

        # Domain services
        self.ppi_service = PPICalculationService(self.match_repository)
        self.betting_service = BettingAnalysisService(min_edge=0.02)

        # Application services
        self.league_processing_service = LeagueProcessingService(
            self.ppi_service, self.file_adapter, self.config
        )

        # Legacy components for model management
        self.model_manager = ModelManager(self.config)

        # Use cases
        self.generate_predictions_uc = (
            None  # Will be initialized when model is available
        )
        self.calculate_edges_uc = CalculateBettingEdgesUseCase(
            self.betting_service, self.betting_repository, self.file_adapter
        )
        self.optimize_models_uc = OptimizeModelsUseCase(
            self.model_manager, self.match_repository
        )
        self.ingest_data_uc = IngestDataUseCase(
            self.match_repository,
            self.ppi_service,
            self.file_adapter,
            self.league_processing_service,
        )

        # Pipeline orchestrator
        self.pipeline = None  # Will be initialized when use cases are ready


class BettingPipeline:
    """Main pipeline orchestrator using DDD architecture"""

    def __init__(self, config=None):
        self.container = DependencyContainer(config)
        self.config = config or DEFAULT_CONFIG

    def run_zsd_predictions(self) -> bool:
        """Generate ZSD predictions with enhanced features"""
        print(f"{'=' * 60}\r\nRUNNING ENHANCED ZSD PREDICTIONS\r\n{'=' * 60}\r\n")

        try:
            # Check required files
            if not self.container.file_adapter.file_exists("fixtures_ppi_and_odds.csv"):
                print(
                    "Error: fixtures_ppi_and_odds.csv not found. Please run 'latest_ppi' first."
                )
                return False

            # Initialize models for all leagues
            self._initialize_models()

            # Load fixtures
            fixtures_df = self.container.file_adapter.read_csv(
                "fixtures_ppi_and_odds.csv"
            )
            print(f"Loaded {len(fixtures_df)} fixtures")

            # Generate predictions using the orchestrator
            results = self.container.pipeline.run_full_prediction_pipeline()

            if not results["predictions"]:
                print("No predictions generated")
                return False

            # Display results
            self._display_results(results)
            return True

        except Exception as e:
            print(f"Error in ZSD predictions: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_parameter_optimization(self) -> bool:
        """Run ZSD parameter optimization for all leagues"""
        print(f"{'=' * 60}\r\nRUNNING ZSD PARAMETER OPTIMIZATION\r\n{'=' * 60}\r\n")

        try:
            if not self.container.file_adapter.file_exists(
                "historical_ppi_and_odds.csv"
            ):
                print(
                    "Error: historical_ppi_and_odds.csv not found. Please run 'historical_ppi' first."
                )
                return False

            # Get all available leagues
            historical_df = self.container.file_adapter.read_csv(
                "historical_ppi_and_odds.csv"
            )
            leagues = historical_df["League"].unique().tolist()

            self.container.optimize_models_uc.execute(leagues)
            print("Parameter optimization completed successfully")
            return True

        except Exception as e:
            print(f"Error in parameter optimization: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_latest_ppi(self) -> bool:
        """Generate latest PPI data and merge with odds using the new service"""
        print(f"{'=' * 60}\r\nGENERATING LATEST PPI PREDICTIONS\r\n{'=' * 60}\r\n")

        try:
            # Use the new league processing service
            ppi_all_leagues = self.container.league_processing_service.process_latest_ppi_for_all_leagues()

            if not ppi_all_leagues:
                print("No PPI data generated")
                return False

            # Save PPI data
            import pandas as pd

            ppi_latest = pd.DataFrame(ppi_all_leagues).sort_values(by="PPI_Diff")
            self.container.file_adapter.write_csv(ppi_latest, "latest_ppi.csv")
            print(f"Saved {len(ppi_latest)} PPI records to latest_ppi.csv")

            # Merge with odds data
            merge_future_odds_data()
            print("Successfully merged PPI data with odds")

            return True

        except Exception as e:
            print(f"Error generating latest PPI: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_historical_ppi(self) -> bool:
        """Generate historical PPI data and merge with odds using the new service"""
        print(f"{'=' * 60}\r\nGENERATING HISTORICAL PPI DATA\r\n{'=' * 60}\r\n")

        try:
            # Use the new league processing service
            historical_ppi = self.container.league_processing_service.process_historical_ppi_for_all_leagues()

            self.container.file_adapter.write_csv(historical_ppi, "historical_ppi.csv")
            print(f"Saved {len(historical_ppi)} historical PPI records")

            # Merge with odds data
            merge_historical_odds_data()
            print("Successfully merged historical PPI data with odds")
            return True

        except Exception as e:
            print(f"Error generating historical PPI: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_get_data(self, season: str) -> bool:
        """Download data for all leagues for a specific season"""
        print(f"Downloading data for season: {season}")

        if not season or len(season.split("-")) != 2:
            print("Error: Please provide a valid season format (e.g., '2023-24')")
            return False

        config = AppConfig(season)
        failed_leagues = []
        successful_leagues = []

        # Import here to avoid circular dependency
        from ingestion import DataIngestion

        data_ingestion = DataIngestion(config)

        for league in Leagues:
            try:
                print(f"Processing {league.name}...")
                data_ingestion.get_fbref_data(league, season)
                data_ingestion.get_fbduk_data(league, season)
                successful_leagues.append(league.name)
            except Exception as e:
                print(f"Error getting data for {league.name} {season}: {e}")
                failed_leagues.append(league.name)
                continue

        print(f"\nData download summary:")
        print(f"  Successful: {len(successful_leagues)} leagues")
        print(f"  Failed: {len(failed_leagues)} leagues")

        if failed_leagues:
            print(f"  Failed leagues: {', '.join(failed_leagues)}")

        return len(failed_leagues) == 0

    def run_backtest_validation(self, betting_csv: str, predictions_csv: str) -> bool:
        """Run comprehensive backtest validation"""
        print("Running comprehensive backtest validation...")

        if not Path(betting_csv).exists():
            print(f"Error: Betting results file not found: {betting_csv}")
            return False

        if not Path(predictions_csv).exists():
            print(f"Error: Predictions file not found: {predictions_csv}")
            return False

        try:
            # Use legacy validation system for now
            from utils.pipeline_config import PipelineConfig

            pipeline_config = PipelineConfig(self.config)

            validation_result = validate_zsd_backtest_results(
                betting_csv, predictions_csv, pipeline_config
            )

            if validation_result:
                print(f"\nValidation complete!")
                print(f"Valid: {validation_result.is_valid}")
                print(f"Confidence: {validation_result.confidence_score:.2f}")

            return True

        except Exception as e:
            print(f"Error in backtest validation: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _initialize_models(self):
        """Initialize prediction models for all leagues"""
        # Load historical data
        historical_df = self.container.file_adapter.read_csv(
            "historical_ppi_and_odds.csv"
        )

        # Fit models for all leagues
        print("Fitting models for all leagues...")
        for league in historical_df["League"].unique():
            self.container.model_manager.fit_league_model(historical_df, league)

        # Create a unified prediction service that works with the model manager
        prediction_service = ModelManagerPredictionService(self.container.model_manager)

        # Initialize the generate predictions use case
        self.container.generate_predictions_uc = GeneratePredictionsUseCase(
            prediction_service,
            self.container.prediction_repository,
            self.container.match_repository,
        )

        # Initialize the pipeline orchestrator
        self.container.pipeline = PipelineOrchestrator(
            self.container.generate_predictions_uc,
            self.container.calculate_edges_uc,
            self.container.optimize_models_uc,
            self.container.ingest_data_uc,
        )

    def _display_results(self, results):
        """Display prediction and betting results"""
        predictions = results["predictions"]
        betting_candidates = results["betting_candidates"]

        print(f"\nZSD PREDICTION SUMMARY:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Betting candidates: {len(betting_candidates)}")

        if betting_candidates:
            print(f"\nFound {len(betting_candidates)} enhanced ZSD betting candidates:")
            print("-" * 100)

            # Sort by edge (highest first)
            sorted_candidates = sorted(
                betting_candidates, key=lambda x: x["Edge"], reverse=True
            )

            for candidate in sorted_candidates[: min(20, len(sorted_candidates))]:
                print(
                    f"{candidate['Home']} vs {candidate['Away']} ({candidate['League']})"
                )
                print(f"  Date: {candidate['Date']}")
                print(
                    f"  Recommended Bet: {candidate['Bet_Type']} at {candidate.get('Soft_Odds', 0):.2f} (Edge: {candidate['Edge']:.3f})"
                )
                print()


class ModelManagerPredictionService:
    """Adapter to make ModelManager compatible with PredictionService interface"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def generate_predictions(self, matches):
        """Generate predictions for matches using the model manager"""
        import pandas as pd

        # Convert matches to DataFrame format expected by model manager
        match_data = []
        for match in matches:
            match_data.append(
                {
                    "Date": match.date,
                    "League": match.home_team.league,
                    "Home": match.home_team.name,
                    "Away": match.away_team.name,
                    "Wk": match.week,
                }
            )

        fixtures_df = pd.DataFrame(match_data)

        # Use model manager to generate predictions
        predictions_df = self.model_manager.predict_matches(fixtures_df)

        if len(predictions_df) == 0:
            print("No predictions generated by model manager")
            return []

        # Convert back to domain objects
        predictions = []
        for _, row in predictions_df.iterrows():
            # Find corresponding match
            match = next(
                (
                    m
                    for m in matches
                    if m.home_team.name == row["Home"]
                    and m.away_team.name == row["Away"]
                ),
                None,
            )

            if match is None:
                continue

            probabilities = Probabilities(
                home=float(row.get("ZIP_Prob_H", 0.33)),
                draw=float(row.get("ZIP_Prob_D", 0.33)),
                away=float(row.get("ZIP_Prob_A", 0.34)),
            )

            # Extract all method predictions for metadata
            all_methods = {}
            for method in ["Poisson", "Zip", "Mov"]:
                all_methods[method.lower()] = {
                    "prob_home": row.get(f"{method}_Prob_H", 0.33),
                    "prob_draw": row.get(f"{method}_Prob_D", 0.33),
                    "prob_away": row.get(f"{method}_Prob_A", 0.34),
                }

            prediction = Prediction(
                match=match,
                probabilities=probabilities,
                lambda_home=row.get("ZSD_Lambda_H", 1.5),
                lambda_away=row.get("ZSD_Lambda_A", 1.2),
                model_type="ZSD",
                created_at=match.date,
                metadata={"all_methods": all_methods},
            )
            predictions.append(prediction)

        return predictions

    def is_fitted(self) -> bool:
        return len(self.model_manager.models) > 0


def main():
    """Main entry point for the betting pipeline"""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        from utils.pipeline_config import print_pipeline_help

        print_pipeline_help()
        return

    pipeline = BettingPipeline()

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "status":
            from utils.pipeline_config import PipelineConfig

            config = PipelineConfig(pipeline.config)
            config.print_status_report()

        elif mode == "get_data":
            if len(sys.argv) > 2:
                season = sys.argv[2]
                pipeline.run_get_data(season)
            else:
                print("Usage: python main.py get_data <season>")

        elif mode == "latest_ppi":
            pipeline.run_latest_ppi()

        elif mode == "historical_ppi":
            pipeline.run_historical_ppi()

        elif mode == "update_teams":
            build_team_name_dictionary()

        elif mode == "optimize":
            pipeline.run_parameter_optimization()

        elif mode == "validate":
            if len(sys.argv) > 3:
                betting_results_directory = "optimisation_validation/betting_results"
                prediction_results_directory = (
                    "optimisation_validation/prediction_results"
                )
                betting_filename = f"{betting_results_directory}/{sys.argv[2]}"
                prediction_filename = f"{prediction_results_directory}/{sys.argv[3]}"
                pipeline.run_backtest_validation(betting_filename, prediction_filename)
            else:
                print(
                    "Usage: python main.py validate <betting_file> <predictions_file>"
                )

        elif mode == "predict":
            pipeline.run_zsd_predictions()

        else:
            print(f"Unknown mode: {mode}")
            from utils.pipeline_config import print_pipeline_help

            print_pipeline_help()


if __name__ == "__main__":
    main()
