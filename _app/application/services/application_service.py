from datetime import date
from typing import Any, Dict, List

from ..dto.betting_dto import BettingOpportunityDTO
from ..dto.prediction_dto import PredictionRequest, PredictionResponse
from ..use_cases.generate_predictions import GeneratePredictionsUseCase
from ..use_cases.optimize_parameters import OptimizeParametersUseCase
from ..use_cases.update_data import UpdateDataUseCase
from ..use_cases.validate_backtest import ValidateBacktestUseCase


class ApplicationService:
    def __init__(
        self,
        generate_predictions_use_case: GeneratePredictionsUseCase,
        optimize_parameters_use_case: OptimizeParametersUseCase,
        update_data_use_case: UpdateDataUseCase,
        validate_backtest_use_case: ValidateBacktestUseCase,
    ):
        self.generate_predictions_use_case = generate_predictions_use_case
        self.optimize_parameters_use_case = optimize_parameters_use_case
        self.update_data_use_case = update_data_use_case
        self.validate_backtest_use_case = validate_backtest_use_case

    def generate_predictions(
        self,
        league: str = None,
        season: str = None,
        start_date: date = None,
        end_date: date = None,
        method: str = "zip",
    ) -> tuple[List[PredictionResponse], List[BettingOpportunityDTO]]:
        request = PredictionRequest(
            league=league,
            season=season,
            start_date=start_date,
            end_date=end_date,
            prediction_method=method,
        )
        return self.generate_predictions_use_case.execute(request)

    def optimize_model_parameters(self, league: str = None) -> Dict[str, Any]:
        return self.optimize_parameters_use_case.execute(league)

    def update_league_data(self, leagues: List[Dict], season: str) -> Dict[str, bool]:
        return self.update_data_use_case.execute(leagues, season)

    def validate_backtest_results(
        self, betting_file: str, predictions_file: str
    ) -> Dict[str, Any]:
        return self.validate_backtest_use_case.execute(betting_file, predictions_file)
