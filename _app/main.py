import sys
from pathlib import Path

from application.services.application_service import ApplicationService
from application.use_cases.generate_predictions import GeneratePredictionsUseCase
from application.use_cases.optimize_parameters import OptimizeParametersUseCase
from application.use_cases.update_data import UpdateDataUseCase
from application.use_cases.validate_backtest import ValidateBacktestUseCase
from domain.services.betting_service import BettingService
from domain.services.model_service import ModelService
from domain.services.prediction_service import PredictionService
from infrastructure.data_sources.fbduk_client import FBDukClient
from infrastructure.data_sources.fbref_client import FBRefClient
from infrastructure.storage.config_storage import ConfigStorage

# Dependency injection setup
from infrastructure.storage.csv_storage import (
    CSVMatchRepository,
    CSVPredictionRepository,
    CSVTeamRepository,
)
from presentation.cli.main import CLIApplication
from presentation.formatters.output_formatter import OutputFormatter


def create_application() -> CLIApplication:
    """Factory function to create the application with all dependencies"""

    # Infrastructure layer
    data_dir = Path("data")
    config_dir = Path("config")
    output_dir = Path("output")

    match_repository = CSVMatchRepository(data_dir / "fbref")
    team_repository = CSVTeamRepository(data_dir / "fbref")
    prediction_repository = CSVPredictionRepository(output_dir)
    config_storage = ConfigStorage(config_dir)

    fbref_client = FBRefClient("https://fbref.com/en/comps")
    fbduk_client = FBDukClient(
        "https://www.football-data.co.uk/mmz4281", "https://www.football-data.co.uk/new"
    )

    # Domain services
    prediction_service = PredictionService()
    betting_service = BettingService()
    model_service = ModelService()

    # Use cases
    generate_predictions_use_case = GeneratePredictionsUseCase(
        match_repository,
        team_repository,
        prediction_service,
        betting_service,
        model_service,
    )
    optimize_parameters_use_case = OptimizeParametersUseCase(
        match_repository, model_service, config_storage
    )
    update_data_use_case = UpdateDataUseCase(
        fbref_client, fbduk_client, match_repository
    )
    validate_backtest_use_case = ValidateBacktestUseCase()

    # Application service
    application_service = ApplicationService(
        generate_predictions_use_case,
        optimize_parameters_use_case,
        update_data_use_case,
        validate_backtest_use_case,
    )

    # Presentation layer
    output_formatter = OutputFormatter()
    cli_application = CLIApplication(application_service, output_formatter)

    return cli_application


def main():
    """Main entry point"""
    try:
        app = create_application()
        exit_code = app.run(sys.argv)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
