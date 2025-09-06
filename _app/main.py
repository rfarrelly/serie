from pathlib import Path

# Enhanced dependency injection setup
from application.services.application_service import ApplicationService
from application.use_cases.enhanced_generate_predictions import (
    EnhancedGeneratePredictionsUseCase,
)
from application.use_cases.optimize_parameters import OptimizeParametersUseCase
from application.use_cases.update_data import UpdateDataUseCase
from application.use_cases.validate_backtest import ValidateBacktestUseCase
from domain.services.enhanced_betting_service import EnhancedBettingService
from domain.services.enhanced_prediction_service import EnhancedPredictionService
from domain.services.model_service import ModelService
from domain.services.ppi_service import PPIService
from infrastructure.data_sources.fbduk_client import FBDukClient
from infrastructure.data_sources.fbref_client import FBRefClient
from infrastructure.storage.config_storage import ConfigStorage
from infrastructure.storage.csv_storage import (
    CSVMatchRepository,
    CSVPredictionRepository,
    CSVTeamRepository,
)
from presentation.cli.main import EnhancedCLIApplication
from presentation.formatters.output_formatter import OutputFormatter


def create_enhanced_application() -> EnhancedCLIApplication:
    """Factory function to create the enhanced application with all original features"""

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
    prediction_service = EnhancedPredictionService()
    betting_service = EnhancedBettingService()
    model_service = ModelService()
    ppi_service = PPIService()

    # Use cases
    enhanced_generate_predictions_use_case = EnhancedGeneratePredictionsUseCase(
        match_repository,
        team_repository,
        prediction_service,
        betting_service,
        model_service,
        ppi_service,
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
        enhanced_generate_predictions_use_case,
        optimize_parameters_use_case,
        update_data_use_case,
        validate_backtest_use_case,
    )

    # Presentation layer
    output_formatter = OutputFormatter()
    cli_application = EnhancedCLIApplication(application_service, output_formatter)

    return cli_application
